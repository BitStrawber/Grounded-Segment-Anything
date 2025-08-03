import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from functools import partial
import threading
import subprocess


# ==============================================================================
# 方案一：提取所有帧 (新功能)
# ==============================================================================

def extract_all_frames_ffmpeg(video_path, output_dir, progress=None):
    """
    使用 FFmpeg 提取视频中的所有帧。
    每个视频的帧会保存在 output_dir 下的一个与视频同名的子目录中。
    :param video_path: 视频文件路径。
    :param output_dir: 主输出目录。
    :param progress: 多线程共享进度字典（线程安全）。
    """
    basename = os.path.splitext(os.path.basename(video_path))[0]
    # 为每个视频创建一个独立的子目录
    video_output_dir = os.path.join(output_dir, basename)
    os.makedirs(video_output_dir, exist_ok=True)

    try:
        # 构建 FFmpeg 命令
        # -i: 输入文件
        # -q:v 2: 设置输出JPEG的质量，2是非常高的质量（范围1-31，越小越好）
        # {video_output_dir}/%06d.jpg: 输出文件模式，%06d表示6位补零的数字序列
        # -hide_banner -loglevel error: 隐藏不必要的输出信息
        cmd = (
            f'ffmpeg -i "{video_path}" '
            f'-q:v 2 '
            f'"{video_output_dir}/%06d.jpg" '
            f'-hide_banner -loglevel error'
        )

        # 使用 check=True，如果ffmpeg执行失败会抛出异常
        subprocess.run(cmd, shell=True, check=True)

        # 统计成功提取的帧数
        frame_count = len(os.listdir(video_output_dir))

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
    :param video_list: 视频文件路径列表。
    :param output_dir: 主输出目录。
    :param max_workers: 最大线程数。
    """
    # 多线程共享进度（线程安全字典）
    progress = {}

    # 按视频时长预估排序（大文件优先），减少短任务等待时间
    sorted_videos = sorted(
        video_list,
        key=lambda x: os.path.getsize(x),  # 按文件大小排序作为时长的代理
        reverse=True
    )
    total_videos = len(sorted_videos)

    # 任务包装函数
    task_func = partial(
        extract_all_frames_ffmpeg,
        output_dir=output_dir,
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
        with tqdm(total=total_videos, desc="提取所有帧") as pbar:
            stats = {'FFMPEG': 0, 'Failed': 0}
            last_progress = 0

            while len(progress) < total_videos:
                done_count = len(progress)
                if done_count > last_progress:
                    pbar.update(done_count - last_progress)
                    last_progress = done_count
                pbar.set_postfix_str(f"已完成 {done_count}/{total_videos}")
                # 短暂休眠避免CPU空转
                threading.Event().wait(0.1)

            # 确保进度条达到100%
            pbar.update(total_videos - last_progress)
            pbar.set_postfix_str(f"已完成 {total_videos}/{total_videos}")

            # 收集结果
            for future in futures:
                try:
                    video_path, count, method = future.result()
                    if method == 'FFMPEG':
                        stats[method] += count
                    else:  # Failed
                        stats['Failed'] += 1
                except Exception as e:
                    print(f"获取任务结果时出错: {e}")
                    stats['Failed'] += 1

    total_frames = stats['FFMPEG']
    # 打印统计
    print(f"\n✅ 完成! 共处理 {total_videos} 个视频, 成功生成 {total_frames:,} 帧。")
    if stats['Failed'] > 0:
        print(f"  ❗ 失败: {stats['Failed']} 个视频")


# ==============================================================================
# 方案二：按目标FPS提取 (原始脚本功能)
# ==============================================================================

def extract_frames_optimized(video_path, output_dir, target_fps, progress=None):
    """
    CPU优化版抽帧（关键帧感知+时间戳跳转）
    :param progress: 多线程共享进度字典（线程安全）
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(video_path))[0]
        frame_count = 0

        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total_frames <= 0:
            raise ValueError("无法读取视频的FPS或总帧数。")

        interval = max(1, int(fps / target_fps))

        for frame_id in range(0, total_frames, interval):
            target_time = frame_id * (1000 / fps)
            cap.set(cv2.CAP_PROP_POS_MSEC, target_time)

            ret, frame = cap.read()
            if not ret:
                break

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
            # 注意：这里的fallback会覆盖之前生成的所有帧
            ffmpeg_fallback(video_path, output_dir, target_fps, basename)
            frame_count = len([f for f in os.listdir(output_dir) if f.startswith(basename)])

        if progress is not None:
            with threading.Lock():
                progress[video_path] = 1
        return (video_path, frame_count, 'CPU')

    except Exception as e:
        print(f"处理失败 {video_path}: {str(e)}")
        return (video_path, 0, 'Failed')


def ffmpeg_fallback(video_path, output_dir, target_fps, basename):
    """FFmpeg兜底方案（解决关键帧限制问题）"""
    # 先删除可能已生成的旧文件，避免混淆
    for f in os.listdir(output_dir):
        if f.startswith(basename):
            os.remove(os.path.join(output_dir, f))

    cmd = f'ffmpeg -i "{video_path}" -vf fps={target_fps} "{output_dir}/{basename}_%05d.jpg" -hide_banner -loglevel error'
    subprocess.run(cmd, shell=True)


def batch_processor_fps(video_list, output_dir, target_fps=23, max_workers=8):
    """
    批量处理器（动态负载均衡），按指定FPS抽帧
    :param max_workers: 线程数（建议≤CPU逻辑核心数×1.5）
    """
    progress = {}

    sorted_videos = sorted(
        video_list,
        key=lambda x: os.path.getsize(x),
        reverse=True
    )
    total_videos = len(sorted_videos)

    task_func = partial(
        extract_frames_optimized,
        output_dir=output_dir,
        target_fps=target_fps,
        progress=progress
    )

    cpu_count = os.cpu_count()
    adjusted_workers = min(max_workers, int(cpu_count * 1.5)) if cpu_count else max_workers

    with ThreadPoolExecutor(max_workers=adjusted_workers) as executor:
        futures = [
            executor.submit(task_func, video_path=video)
            for video in sorted_videos
        ]

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
                    video_path, count, method = future.result()
                    if method == 'CPU':
                        stats[method] += count
                    else:  # Failed
                        stats['Failed'] += 1
                except Exception as e:
                    print(f"获取任务结果时出错: {e}")
                    stats['Failed'] += 1

    total_frames = stats['CPU']
    print(f"\n✅ 完成! 共处理 {total_videos} 个视频, 生成 {total_frames:,} 帧")
    if stats['Failed'] > 0:
        print(f"  ❗ 失败: {stats['Failed']} 个视频")


if __name__ == "__main__":
    # --- 公共配置 ---
    input_folder = "/media/HDD0/XCX/Selected-UVEB"
    video_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(input_folder)
        for f in files
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ]
    max_threads = os.cpu_count()  # 默认使用CPU核心数作为最大线程数

    # --- 选择一个方案来运行 ---

    # 方案一：提取所有帧
    print("=" * 50)
    print("▶️ 开始方案一：提取所有视频的所有帧...")
    print("=" * 50)
    output_folder_all_frames = "/media/HDD0/XCX/selected_background"
    if video_files:
        batch_processor_all_frames(video_files, output_folder_all_frames, max_threads)
    else:
        print("在输入目录中未找到视频文件。")

    print("\n\n")

    # # 方案二：按目标FPS提取（原始功能）
    # print("="*50)
    # print("▶️ 开始方案二：按指定FPS提取帧...")
    # print("="*50)
    # output_folder_by_fps = "/media/HDD0/XCX/selected_backgrounds"
    # target_fps = 30
    # if video_files:
    #     print(f"▶ 开始处理 {len(video_files)} 个视频 (目标帧率: {target_fps}FPS)")
    #     batch_processor_fps(video_files, output_folder_by_fps, target_fps, max_threads)
    # else:
    #      print("在输入目录中未找到视频文件。")