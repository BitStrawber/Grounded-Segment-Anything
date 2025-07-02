import os
import cv2
import av
import avcuda
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from multiprocessing import Manager
from functools import partial


def extract_frames(video_path, output_dir, target_fps, gpu_id=None, progress=None):
    """
    视频抽帧函数（支持PyAV-CUDA/OpenCV硬件加速）
    :param gpu_id: 为None时使用CPU处理，否则使用指定GPU
    :param progress: 多进程共享进度字典
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(video_path))[0]
        frame_count = 0

        if gpu_id is not None:
            # 方案1：PyAV-CUDA硬件解码（需安装avcuda）
            container = av.open(video_path)
            stream = container.streams.video[0]
            avcuda.init_hwcontext(stream.codec_context, gpu_id)

            # 计算跳帧间隔
            fps = stream.average_rate
            interval = max(1, int(fps / target_fps))

            for i, frame in enumerate(container.decode(stream)):
                if i % interval == 0:
                    img = frame.to_image()
                    img.save(f"{output_dir}/{basename}_{frame_count:05d}.jpg")
                    frame_count += 1
        else:
            # 方案2：OpenCV硬件解码（兼容性更好）
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            interval = max(1, int(fps / target_fps))

            while cap.isOpened():
                # 跳帧优化：只解码目标帧
                for _ in range(interval - 1):
                    cap.grab()

                ret, frame = cap.retrieve()
                if not ret:
                    break

                cv2.imwrite(f"{output_dir}/{basename}_{frame_count:05d}.jpg", frame)
                frame_count += 1
            cap.release()

        # 更新进度
        if progress is not None:
            progress[video_path] = 1
        return (video_path, frame_count, gpu_id if gpu_id is not None else 'CPU')

    except Exception as e:
        print(f"处理失败 {video_path} (GPU {gpu_id}): {str(e)}")
        if gpu_id is not None:
            print("尝试回退到CPU处理...")
            return extract_frames(video_path, output_dir, target_fps, gpu_id=None, progress=progress)
        return (video_path, 0, 'Failed')


def batch_processor(video_list, output_dir, target_fps=23, gpu_ids=None):
    """
    批量处理器（支持动态负载均衡）
    :param gpu_ids: None表示使用CPU，列表表示使用指定GPU
    """
    total_videos = len(video_list)
    use_gpu = gpu_ids is not None and len(gpu_ids) > 0

    # 动态分配GPU（轮询方式）
    def get_gpu_id(index):
        return gpu_ids[index % len(gpu_ids)] if use_gpu else None

    # 多进程共享进度
    with Manager() as manager:
        progress = manager.dict()
        worker_count = len(gpu_ids) if use_gpu else min(8, os.cpu_count())

        # 任务包装函数
        task_func = partial(extract_frames,
                            output_dir=output_dir,
                            target_fps=target_fps,
                            progress=progress)

        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            # 提交任务（按视频大小排序优化负载）
            futures = []
            sorted_videos = sorted(video_list, key=lambda x: os.path.getsize(x), reverse=True)
            for i, video in enumerate(sorted_videos):
                futures.append(executor.submit(task_func,
                                               video_path=video,
                                               gpu_id=get_gpu_id(i)))

            # 进度条监控
            with tqdm(total=total_videos, desc="视频处理进度") as pbar:
                stats = {'GPU': {}, 'CPU': 0, 'Failed': 0}
                last_progress = 0

                while True:
                    done_count = sum(progress.values())
                    if done_count > last_progress:
                        pbar.update(done_count - last_progress)
                        last_progress = done_count
                        pbar.set_postfix_str(f"已完成 {done_count}/{total_videos}")

                    if done_count >= total_videos:
                        break

                # 收集结果
                for future in futures:
                    video_path, count, device = future.result()
                    if isinstance(device, int):
                        stats['GPU'].setdefault(device, 0)
                        stats['GPU'][device] += count
                    else:
                        stats[device] += count

        # 打印统计
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
    input_folder = "/media/HDD0/XCX/UVEB/test/blur"
    output_folder = "/media/HDD0/XCX/background"
    target_fps = 23
    specified_gpus = [4, 5]  # 指定GPU ID列表，None则使用CPU

    # 扫描视频文件
    video_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(input_folder)
        for f in files
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ]

    print(f"▶ 开始处理 {len(video_files)} 个视频 (目标帧率: {target_fps}FPS)")
    batch_processor(video_files, output_folder, target_fps, specified_gpus)