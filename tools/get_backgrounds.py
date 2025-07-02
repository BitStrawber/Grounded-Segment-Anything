import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def extract_frames_ffmpeg(video_path, output_dir, target_fps, gpu_id):
    """使用指定GPU进行视频抽帧"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    cmd = [
        'ffmpeg',
        '-hwaccel', 'cuda',  # 启用CUDA硬件加速
        '-hwaccel_device', str(gpu_id),  # 指定GPU设备
        '-i', video_path,
        '-vf', f'fps={target_fps},hwupload_cuda',  # 上传到GPU处理
        '-c:v', 'mjpeg_cuvid',  # 使用NVIDIA的MJPEG解码器
        '-q:v', '2',
        '-f', 'image2',
        '-loglevel', 'error',
        f'{output_dir}/{os.path.splitext(os.path.basename(video_path))[0]}_%05d.jpg'
    ]

    try:
        subprocess.run(cmd, check=True)
        generated = len([f for f in os.listdir(output_dir)
                         if f.startswith(os.path.splitext(os.path.basename(video_path))[0])])
        return (video_path, generated, gpu_id)
    except subprocess.CalledProcessError as e:
        print(f"处理失败 {video_path} (GPU {gpu_id}): {str(e)}")
        return (video_path, 0, gpu_id)


def batch_processor(video_list, output_dir, target_fps=23, gpu_ids=[0]):
    """使用指定GPU列表进行批量处理"""
    os.makedirs(output_dir, exist_ok=True)
    total_videos = len(video_list)

    # 任务分配（轮询指定的GPU）
    tasks = []
    for i, video_path in enumerate(video_list):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        tasks.append((video_path, output_dir, target_fps, gpu_id))

    # 多进程处理（每个GPU一个进程）
    with ProcessPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = [executor.submit(extract_frames_ffmpeg, *task) for task in tasks]

        with tqdm(total=total_videos, desc="视频处理进度") as pbar:
            gpu_stats = {gpu: {"completed": 0, "frames": 0} for gpu in gpu_ids}

            for future in futures:
                video_path, frame_count, gpu_id = future.result()
                gpu_stats[gpu_id]["completed"] += 1
                gpu_stats[gpu_id]["frames"] += frame_count

                pbar.update(1)
                pbar.set_postfix_str(
                    f"GPU{gpu_id}: {frame_count}帧 | "
                    f"总进度: {pbar.n}/{total_videos}"
                )

    # 最终统计
    total_frames = sum(stat["frames"] for stat in gpu_stats.values())
    print(f"\n✅ 完成! 共处理 {total_videos} 视频, 生成 {total_frames} 帧")
    for gpu, stat in gpu_stats.items():
        print(f"  GPU{gpu}: {stat['completed']}视频/{stat['frames']}帧")


if __name__ == "__main__":
    # 配置参数
    input_folder = "/media/HDD0/XCX/UVEB/test/blur"
    output_folder = "/media/HDD0/XCX/background"
    target_fps = 23  # 每秒1帧
    specified_gpus = [3,4,5]  # 指定要使用的GPU ID列表，例如[0]或[0,1]

    # 扫描视频文件
    video_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(input_folder)
        for f in files
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ]

    print(f"▶ 开始处理 {len(video_files)} 个视频 (目标帧率: {target_fps}FPS)")
    print(f"▶ 指定GPU: {specified_gpus}")
    batch_processor(video_files, output_folder, target_fps, specified_gpus)