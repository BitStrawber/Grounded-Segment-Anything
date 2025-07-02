import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor  # 改为线程池
from tqdm import tqdm
from functools import partial
import threading

def extract_frames_cpu(video_path, output_dir, target_fps, progress=None):
    """
    CPU多线程抽帧（兼容性更强）
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
        interval = max(1, int(fps / target_fps))

        while cap.isOpened():
            # 跳帧优化（减少解码压力）
            for _ in range(interval - 1):
                cap.grab()

            ret, frame = cap.retrieve()
            if not ret:
                break

            # 直接保存帧（无GPU中转）
            cv2.imwrite(
                f"{output_dir}/{basename}_{frame_count:05d}.jpg",
                frame
            )
            frame_count += 1

        cap.release()

        # 更新进度（线程安全）
        if progress is not None:
            with threading.Lock():  # 加锁避免竞争
                progress[video_path] = 1
        return (video_path, frame_count, 'CPU')

    except Exception as e:
        print(f"处理失败 {video_path}: {str(e)}")
        return (video_path, 0, 'Failed')

def batch_processor(video_list, output_dir, target_fps=23, max_workers=8):
    """
    批量处理器（多线程动态调度）
    :param max_workers: 线程数（建议≤CPU核心数×2）
    """
    # 多线程共享进度（线程安全字典）
    progress = {}
    lock = threading.Lock()

    # 按视频大小排序优化负载（大文件优先）
    sorted_videos = sorted(video_list, key=lambda x: os.path.getsize(x), reverse=True)
    total_videos = len(sorted_videos)

    # 任务包装函数
    task_func = partial(
        extract_frames_cpu,
        output_dir=output_dir,
        target_fps=target_fps,
        progress=progress
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(task_func, video_path=video)
            for video in sorted_videos
        ]

        # 进度条监控
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
    input_folder = "/media/HDD0/XCX/UVEB/"
    output_folder = "/media/HDD0/XCX/background"
    target_fps = 1
    max_threads = 60  # 根据CPU核心数调整（建议：物理核心数×1.5）

    # 扫描视频文件
    video_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(input_folder)
        for f in files
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ]

    print(f"▶ 开始处理 {len(video_files)} 个视频 (目标帧率: {target_fps}FPS)")
    batch_processor(video_files, output_folder, target_fps, max_threads)