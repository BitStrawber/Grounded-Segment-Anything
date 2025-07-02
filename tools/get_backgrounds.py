import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def extract_frames(video_path, output_dir, target_fps, gpu_id=None):
    """
    视频抽帧函数（支持GPU加速和CPU回退）
    :param gpu_id: 为None时使用CPU处理
    """
    try:
        if gpu_id is not None:
            # GPU加速方案
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            cmd = [
                'ffmpeg',
                '-hwaccel', 'cuda',
                '-hwaccel_device', str(gpu_id),
                '-i', video_path,
                '-vf', f'fps={target_fps},format=nv12,hwupload_cuda',
                '-c:v', 'h264_nvenc',  # 使用NVIDIA编码器
                '-q:v', '2',
                '-f', 'image2',
                '-loglevel', 'error',
                f'{output_dir}/{os.path.splitext(os.path.basename(video_path))[0]}_%05d.jpg'
            ]
        else:
            # CPU回退方案
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vf', f'fps={target_fps}',
                '-q:v', '2',
                '-f', 'image2',
                '-loglevel', 'error',
                f'{output_dir}/{os.path.splitext(os.path.basename(video_path))[0]}_%05d.jpg'
            ]

        subprocess.run(cmd, check=True)
        generated = len([f for f in os.listdir(output_dir)
                         if f.startswith(os.path.splitext(os.path.basename(video_path))[0])])
        return (video_path, generated, gpu_id if gpu_id is not None else 'CPU')

    except subprocess.CalledProcessError as e:
        print(f"处理失败 {video_path} (GPU {gpu_id}): {str(e)}")
        if gpu_id is not None:
            print("尝试回退到CPU处理...")
            return extract_frames(video_path, output_dir, target_fps, gpu_id=None)
        return (video_path, 0, 'Failed')


def batch_processor(video_list, output_dir, target_fps=23, gpu_ids=None):
    """
    批量处理器（支持指定GPU或自动回退到CPU）
    :param gpu_ids: None表示使用CPU，列表表示使用指定GPU
    """
    os.makedirs(output_dir, exist_ok=True)
    total_videos = len(video_list)

    # 确定处理模式
    use_gpu = gpu_ids is not None and len(gpu_ids) > 0
    worker_count = len(gpu_ids) if use_gpu else min(4, os.cpu_count())  # CPU模式限制线程数

    # 任务分配
    tasks = []
    for i, video_path in enumerate(video_list):
        gpu_id = gpu_ids[i % len(gpu_ids)] if use_gpu else None
        tasks.append((video_path, output_dir, target_fps, gpu_id))

    # 多进程处理
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(extract_frames, *task) for task in tasks]

        with tqdm(total=total_videos, desc="视频处理进度") as pbar:
            stats = {'GPU': {}, 'CPU': 0, 'Failed': 0}

            for future in futures:
                video_path, frame_count, device = future.result()

                # 更新统计
                if isinstance(device, int):
                    stats['GPU'].setdefault(device, 0)
                    stats['GPU'][device] += frame_count
                else:
                    stats[device] += frame_count

                pbar.update(1)
                pbar.set_postfix_str(
                    f"当前设备: {device} | 生成帧: {frame_count} | "
                    f"总进度: {pbar.n}/{total_videos}"
                )

    # 最终统计
    total_frames = sum(stats['GPU'].values()) + stats['CPU']
    print(f"\n✅ 完成! 共处理 {total_videos} 视频, 生成 {total_frames} 帧")

    if use_gpu:
        for gpu, frames in stats['GPU'].items():
            print(f"  GPU{gpu}: {frames}帧")
    if stats['CPU'] > 0:
        print(f"  CPU: {stats['CPU']}帧 (回退处理)")
    if stats['Failed'] > 0:
        print(f"  ❗ 失败: {stats['Failed']}视频")


if __name__ == "__main__":
    # 配置参数
    input_folder = "/media/HDD0/XCX/UVEB/test/blur/cv_1000.mp4"
    output_folder = "/media/HDD0/XCX/background0"
    target_fps = 23
    specified_gpus = [4, 5]  # 指定要使用的GPU ID列表，设置为None则使用CPU

    # 扫描视频文件
    video_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(input_folder)
        for f in files
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ]

    print(f"▶ 开始处理 {len(video_files)} 个视频 (目标帧率: {target_fps}FPS)")
    batch_processor(video_files, output_folder, target_fps, specified_gpus)