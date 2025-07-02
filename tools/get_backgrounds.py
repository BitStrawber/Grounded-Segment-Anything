import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pynvml


def get_available_gpus(min_memory_free=2048):
    """获取可用GPU列表（显存大于指定值）"""
    pynvml.nvmlInit()
    available = []
    for i in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if mem.free / 1024 ** 2 >= min_memory_free:
            available.append(i)
    return available


def extract_frames_ffmpeg(video_path, output_dir, target_fps, gpu_id):
    """GPU加速抽帧（隔离到指定GPU）"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    cmd = [
        'ffmpeg',
        '-hwaccel', 'cuda',
        '-hwaccel_device', str(gpu_id),
        '-i', video_path,
        '-vf', f'fps={target_fps}',
        '-q:v', '2',
        '-f', 'image2',
        '-loglevel', 'error',
        f'{output_dir}/{os.path.splitext(os.path.basename(video_path))[0]}_%05d.jpg'
    ]
    try:
        subprocess.run(cmd, check=True)
        # 返回处理成功的视频路径和帧数（通过输出文件统计）
        generated = len([f for f in os.listdir(output_dir)
                         if f.startswith(os.path.splitext(os.path.basename(video_path))[0])])
        return (video_path, generated, gpu_id)
    except subprocess.CalledProcessError as e:
        return (video_path, 0, gpu_id)


def batch_processor(video_list, output_dir, target_fps=23):
    """带GPU负载均衡的批量处理器"""
    available_gpus = get_available_gpus()
    if not available_gpus:
        raise RuntimeError("没有可用的GPU资源！")

    os.makedirs(output_dir, exist_ok=True)
    total_videos = len(video_list)

    # 任务分配（轮询GPU）
    tasks = []
    for i, video_path in enumerate(video_list):
        gpu_id = available_gpus[i % len(available_gpus)]
        tasks.append((video_path, output_dir, target_fps, gpu_id))

    # 多进程处理
    with ProcessPoolExecutor(max_workers=len(available_gpus)) as executor:
        futures = []
        for task in tasks:
            futures.append(executor.submit(extract_frames_ffmpeg, *task))

        # 全局进度条
        with tqdm(total=total_videos, desc="视频处理进度", unit="视频",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [GPU:{postfix[0]}|剩余:{postfix[1]}]") as pbar:
            completed = 0
            gpu_usage = {gpu: 0 for gpu in available_gpus}

            for future in futures:
                video_path, frame_count, gpu_id = future.result()
                completed += 1
                gpu_usage[gpu_id] += frame_count

                # 更新进度条
                pbar.set_postfix([
                    f"{','.join(map(str, available_gpus))}",
                    f"{total_videos - completed}"
                ])
                pbar.update(1)

                # 打印单任务结果
                pbar.write(
                    f"GPU{gpu_id}: {os.path.basename(video_path)} → {frame_count}帧 "
                    f"(总进度: {completed}/{total_videos})"
                )

    # 最终统计
    total_frames = len(os.listdir(output_dir))
    print(f"\n✅ 全部完成！共处理 {total_videos} 个视频，生成 {total_frames} 张图片")
    print(f"GPU负载统计: {gpu_usage}")


if __name__ == "__main__":
    # 配置参数
    input_folder = "/media/HDD0/XCX/UVEB/test/blur/cv_151.mp4"
    output_folder = "/media/HDD0/XCX/background"
    target_fps = 23  # 每秒1帧

    # 扫描视频文件
    video_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(input_folder)
        for f in files
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ]

    # 启动处理
    print(f"▶ 开始处理 {len(video_files)} 个视频 → 目标帧率: {target_fps}FPS")
    print(f"▶ 可用GPU: {get_available_gpus()}")
    batch_processor(video_files, output_folder, target_fps)