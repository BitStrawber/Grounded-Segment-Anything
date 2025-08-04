import os
import cv2
import numpy as np
import random
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from functools import partial


# ==============================================================================
# 方案一：提取所有帧 (已按您的要求修改)
# ==============================================================================

def extract_all_frames_ffmpeg(video_path, output_dir, progress=None):
    """
    【已修改】使用 FFmpeg 提取视频中的所有帧。
    所有视频的帧都会直接保存在指定的 output_dir 中，并以视频名作为前缀。

    :param video_path: 视频文件路径。
    :param output_dir: 主输出目录。
    :param progress: 多线程共享进度字典（线程安全）。
    """
    # 获取不含扩展名的视频文件名，作为图片前缀
    basename = os.path.splitext(os.path.basename(video_path))[0]

    # 【修改点】不再为每个视频创建独立的子目录
    # os.makedirs(os.path.join(output_dir, basename), exist_ok=True)

    try:
        # 【修改点】构建新的 FFmpeg 命令，文件名包含视频basename前缀
        # 这可以防止不同视频的同序号帧相互覆盖
        cmd = (
            f'ffmpeg -i "{video_path}" '
            f'-q:v 2 '
            f'"{output_dir}/{basename}_%06d.jpg" '  # 文件名格式: a_video_000001.jpg
            f'-hide_banner -loglevel error'
        )

        subprocess.run(cmd, shell=True, check=True)

        # 【修改点】更新统计逻辑，只计算属于当前视频的帧数
        frame_count = len([f for f in os.listdir(output_dir) if f.startswith(f"{basename}_")])

        # 更新进度（线程安全）
        if progress is not None:
            with threading.Lock():
                progress[video_path] = 1

        return (video_path, frame_count, 'FFMPEG')

    except subprocess.CalledProcessError as e:
        print(f"处理失败 (FFmpeg error) {video_path}: {e}")
        return (video_path, 0, 'Failed')
    except Exception as e:
        print(f"处理失败 {video_path}: {str(e)}")
        return (video_path, 0, 'Failed')


def batch_processor_all_frames(video_list, output_dir, max_workers=8):
    """
    批量处理器，用于提取所有视频的所有帧。
    (此函数无需修改)
    """
    progress = {}
    sorted_videos = sorted(
        video_list,
        key=lambda x: os.path.getsize(x),
        reverse=True
    )
    total_videos = len(sorted_videos)
    task_func = partial(
        extract_all_frames_ffmpeg,
        output_dir=output_dir,
        progress=progress
    )

    cpu_count = os.cpu_count()
    adjusted_workers = min(max_workers, int(cpu_count * 1.5)) if cpu_count else max_workers

    with ThreadPoolExecutor(max_workers=adjusted_workers) as executor:
        futures = [executor.submit(task_func, video_path=video) for video in sorted_videos]

        with tqdm(total=total_videos, desc="提取所有帧") as pbar:
            stats = {'FFMPEG': 0, 'Failed': 0}
            last_progress = 0
            while len(progress) < total_videos:
                done_count = len(progress)
                if done_count > last_progress:
                    pbar.update(done_count - last_progress)
                    last_progress = done_count
                pbar.set_postfix_str(f"已完成 {done_count}/{total_videos}")
                threading.Event().wait(0.1)
            pbar.update(total_videos - last_progress)
            pbar.set_postfix_str(f"已完成 {total_videos}/{total_videos}")

            for future in futures:
                try:
                    _, count, method = future.result()
                    if method == 'FFMPEG':
                        stats[method] += count
                    else:
                        stats['Failed'] += 1
                except Exception as e:
                    print(f"获取任务结果时出错: {e}")
                    stats['Failed'] += 1

    total_frames = stats['FFMPEG']
    print(f"\n✅ 完成! 共处理 {total_videos} 个视频, 成功生成 {total_frames:,} 帧。")
    if stats['Failed'] > 0:
        print(f"  ❗ 失败: {stats['Failed']} 个视频")


# ==============================================================================
# 方案二：按目标FPS提取 (无需修改，其逻辑已符合要求)
# ==============================================================================

def extract_frames_optimized(video_path, output_dir, target_fps, progress=None):
    """CPU优化版抽帧（关键帧感知+时间戳跳转）"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(video_path))[0]
        frame_count = 0

        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened(): cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total_frames <= 0:
            raise ValueError("无法读取视频的FPS或总帧数。")

        interval = max(1, int(fps / target_fps))
        for frame_id in range(0, total_frames, interval):
            target_time = frame_id * (1000 / fps)
            cap.set(cv2.CAP_PROP_POS_MSEC, target_time)
            ret, frame = cap.read()
            if not ret: break
            cv2.imwrite(
                f"{output_dir}/{basename}_{frame_count:05d}.jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 85]
            )
            frame_count += 1
        cap.release()

        expected_frames = int(total_frames / interval)
        if frame_count < expected_frames * 0.9:
            print(f"帧数不足，为 {video_path} 调用FFmpeg补足...")
            ffmpeg_fallback(video_path, output_dir, target_fps, basename)
            frame_count = len([f for f in os.listdir(output_dir) if f.startswith(basename)])

        if progress is not None:
            with threading.Lock(): progress[video_path] = 1
        return (video_path, frame_count, 'CPU')
    except Exception as e:
        print(f"处理失败 {video_path}: {str(e)}")
        return (video_path, 0, 'Failed')


def ffmpeg_fallback(video_path, output_dir, target_fps, basename):
    """FFmpeg兜底方案（解决关键帧限制问题）"""
    for f in os.listdir(output_dir):
        if f.startswith(basename):
            os.remove(os.path.join(output_dir, f))
    cmd = f'ffmpeg -i "{video_path}" -vf fps={target_fps} "{output_dir}/{basename}_%05d.jpg" -hide_banner -loglevel error'
    subprocess.run(cmd, shell=True)


def batch_processor_fps(video_list, output_dir, target_fps=23, max_workers=8):
    """批量处理器（动态负载均衡），按指定FPS抽帧"""
    progress = {}
    sorted_videos = sorted(video_list, key=lambda x: os.path.getsize(x), reverse=True)
    total_videos = len(sorted_videos)
    task_func = partial(extract_frames_optimized, output_dir=output_dir, target_fps=target_fps, progress=progress)
    cpu_count = os.cpu_count()
    adjusted_workers = min(max_workers, int(cpu_count * 1.5)) if cpu_count else max_workers

    with ThreadPoolExecutor(max_workers=adjusted_workers) as executor:
        futures = [executor.submit(task_func, video_path=video) for video in sorted_videos]
        with tqdm(total=total_videos, desc=f"按{target_fps}FPS抽帧") as pbar:
            stats = {'CPU': 0, 'Failed': 0}
            last_progress = 0
            while len(progress) < total_videos:
                done_count = len(progress)
                if done_count > last_progress:
                    pbar.update(done_count - last_progress)
                    last_progress = done_count
                pbar.set_postfix_str(f"已完成 {done_count}/{total_videos}")
                threading.Event().wait(0.1)
            pbar.update(total_videos - last_progress)
            pbar.set_postfix_str(f"已完成 {total_videos}/{total_videos}")
            for future in futures:
                try:
                    _, count, method = future.result()
                    if method == 'CPU':
                        stats[method] += count
                    else:
                        stats['Failed'] += 1
                except Exception as e:
                    print(f"获取任务结果时出错: {e}")
                    stats['Failed'] += 1
    total_frames = stats['CPU']
    print(f"\n✅ 完成! 共处理 {total_videos} 个视频, 生成 {total_frames:,} 帧")
    if stats['Failed'] > 0: print(f"  ❗ 失败: {stats['Failed']} 个视频")


if __name__ == "__main__":
    # --- 公共配置 ---
    input_folder = "/media/HDD0/XCX/Selected-UVEB"
    video_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(input_folder)
        for f in files
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ]
    # 线程数可以根据您的硬件进行调整
    max_threads = 100

    # --- 选择一个方案来运行 ---
    # 【注意】请只取消注释一个方案来运行，避免混淆输出

    # 方案一：提取所有帧
    print("=" * 50)
    print("▶️ 开始方案一：提取所有视频的所有帧 (无子文件夹)...")
    print("=" * 50)
    output_folder_all_frames = "/media/HDD0/XCX/selected_backgrounds"
    # 确保主输出目录存在
    os.makedirs(output_folder_all_frames, exist_ok=True)
    if video_files:
        print(f"▶ 开始处理 {len(video_files)} 个视频")
        batch_processor_all_frames(video_files, output_folder_all_frames, max_threads)
    else:
        print("在输入目录中未找到视频文件。")

    # # 方案二：按目标FPS提取（原始功能）
    # print("\n\n")
    # print("="*50)
    # print("▶️ 开始方案二：按指定FPS提取帧...")
    # print("="*50)
    # output_folder_by_fps = "/media/HDD0/XCX/selected_backgrounds"
    # # 确保主输出目录存在
    # os.makedirs(output_folder_by_fps, exist_ok=True)
    # target_fps = 30
    # if video_files:
    #     print(f"▶ 开始处理 {len(video_files)} 个视频 (目标帧率: {target_fps}FPS)")
    #     batch_processor_fps(video_files, output_folder_by_fps, target_fps, max_threads)
    # else:
    #      print("在输入目录中未找到视频文件。")