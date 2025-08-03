import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from functools import partial
import threading
import subprocess


def extract_frames_optimized(video_path, output_dir, target_fps, progress=None):
    """
    CPU优化版抽帧（关键帧感知+时间戳跳转）
    :param progress: 多线程共享进度字典（线程安全）
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(video_path))[0]
        frame_count = 0

        # 使用FFMPEG解码（优先尝试）
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(video_path)  # 回退到默认解码器

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, int(fps / target_fps))

        # 时间戳跳转法（减少无效解码）
        for frame_id in range(0, total_frames, interval):
            target_time = frame_id * (1000 / fps)  # 毫秒单位
            cap.set(cv2.CAP_PROP_POS_MSEC, target_time)

            ret, frame = cap.read()
            if not ret:
                break

            cv2.imwrite(
                f"{output_dir}/{basename}_{frame_count:05d}.jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 85]  # 压缩质量优化
            )
            frame_count += 1

        cap.release()

        # 校验帧数是否达标（误差超过20%则调用FFmpeg补足）
        expected_frames = int(total_frames / interval)
        if frame_count < expected_frames * 0.9:
            ffmpeg_fallback(video_path, output_dir, target_fps, basename)
            frame_count = len(os.listdir(output_dir))  # 更新实际帧数

        # 更新进度（线程安全）
        if progress is not None:
            with threading.Lock():
                progress[video_path] = 1
        return (video_path, frame_count, 'CPU')

    except Exception as e:
        print(f"处理失败 {video_path}: {str(e)}")
        return (video_path, 0, 'Failed')


def ffmpeg_fallback(video_path, output_dir, target_fps, basename):
    """FFmpeg兜底方案（解决关键帧限制问题）"""
    cmd = f"ffmpeg -i {video_path} -vf fps={target_fps} {output_dir}/{basename}_%05d.jpg -hide_banner -loglevel error"
    subprocess.run(cmd, shell=True)


def batch_processor(video_list, output_dir, target_fps=23, max_workers=8):
    """
    批量处理器（动态负载均衡）
    :param max_workers: 线程数（建议≤CPU逻辑核心数×1.5）
    """
    # 多线程共享进度（线程安全字典）
    progress = {}
    lock = threading.Lock()

    # 按视频时长预估排序（大文件优先）
    sorted_videos = sorted(
        video_list,
        key=lambda x: cv2.VideoCapture(x).get(cv2.CAP_PROP_FRAME_COUNT),
        reverse=True
    )
    total_videos = len(sorted_videos)

    # 任务包装函数
    task_func = partial(
        extract_frames_optimized,
        output_dir=output_dir,
        target_fps=target_fps,
        progress=progress
    )

    # 动态调整线程数（避免超额订阅）
    cpu_count = os.cpu_count()
    adjusted_workers = min(max_workers, int(cpu_count * 1.5)) if cpu_count else max_workers

    with ThreadPoolExecutor(max_workers=adjusted_workers) as executor:
        futures = [
            executor.submit(task_func, video_path=video)
            for video in sorted_videos
        ]

        # 进度条监控（实时更新）
        with tqdm(total=total_videos, desc="视频处理进度") as pbar:
            stats = {'CPU': 0, 'Failed': 0}
            last_progress = 0

            while True:
                done_count = len(progress)
                if done_count > last_progress:
                    pbar.update(done_count - last_progress)
                    last_progress = done_count
                    pbar.set_postfix_str(f"已完成 {done_count}/{total_videos}")

                if done_count >= total_videos:
                    break

            # 收集结果
            for future in futures:
                video_path, count, device = future.result()
                stats[device] += count

    # 打印统计
    print(f"\n✅ 完成! 共处理 {total_videos} 视频, 生成 {stats['CPU']} 帧")
    if stats['Failed'] > 0:
        print(f"  ❗ 失败: {stats['Failed']}视频")


if __name__ == "__main__":
    # 配置参数
    input_folder = "/media/HDD0/XCX/Selected-UVEB"
    output_folder = "/media/HDD0/XCX/selected_backgrounds"
    target_fps = 15
    max_threads = 100  # 根据实际CPU核心数调整

    # 扫描视频文件
    video_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(input_folder)
        for f in files
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ]

    print(f"▶ 开始处理 {len(video_files)} 个视频 (目标帧率: {target_fps}FPS)")
    batch_processor(video_files, output_folder, target_fps, max_threads)


