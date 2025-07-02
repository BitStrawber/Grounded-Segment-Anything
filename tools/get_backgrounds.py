import os
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from multiprocessing import Manager
from functools import partial


def init_cuda():
    """初始化CUDA环境"""
    cv2.cuda.setDevice(0)  # 默认设备，实际使用时会被覆盖
    if not cv2.cuda.getCudaEnabledDeviceCount():
        raise RuntimeError("No CUDA-capable devices detected")


def extract_frames_cuda(video_path, output_dir, target_fps, gpu_id=None, progress=None):
    """
    OpenCV CUDA硬件加速抽帧
    :param gpu_id: 指定GPU设备ID（None时回退到CPU）
    :param progress: 多进程共享进度字典
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(video_path))[0]
        frame_count = 0

        if gpu_id is not None:
            # 设置当前进程使用的GPU设备
            cv2.cuda.setDevice(gpu_id)

            # 方案1：使用cudacodec模块硬解码（需要NVIDIA Video Codec SDK）
            try:
                # 初始化CUDA视频读取器
                decoder = cv2.cudacodec.createVideoReader(video_path)
                gpu_frame = cv2.cuda_GpuMat()

                # 获取视频属性
                fps = decoder.get(cv2.cudacodec.VideoReaderProperties_PROP_FPS)
                interval = max(1, int(fps / target_fps))

                while True:
                    # 硬件解码到GPU内存
                    ret, gpu_frame = decoder.nextFrame(gpu_frame)
                    if not ret:
                        break

                    # 跳帧逻辑
                    if frame_count % interval == 0:
                        # 下载到CPU内存并保存
                        cpu_frame = gpu_frame.download()
                        cv2.imwrite(f"{output_dir}/{basename}_{frame_count:05d}.jpg", cpu_frame)

                    frame_count += 1

            except cv2.error as e:
                print(f"CUDA解码失败 {video_path} (GPU {gpu_id}): {str(e)}")
                # 回退到常规CUDA加速方案
                return extract_frames_cuda(video_path, output_dir, target_fps, gpu_id=None, progress=progress)

        else:
            # 方案2：常规OpenCV + CUDA处理（兼容性更好）
            cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap = cv2.VideoCapture(video_path)  # 回退到默认解码器

            fps = cap.get(cv2.CAP_PROP_FPS)
            interval = max(1, int(fps / target_fps))
            gpu_frame = cv2.cuda_GpuMat()

            while cap.isOpened():
                # 跳帧优化
                for _ in range(interval - 1):
                    cap.grab()

                ret, cpu_frame = cap.retrieve()
                if not ret:
                    break

                # 上传到GPU处理（可选加速）
                gpu_frame.upload(cpu_frame)
                # 此处可添加CUDA处理（如色彩空间转换、缩放等）
                processed_frame = gpu_frame.download()

                cv2.imwrite(f"{output_dir}/{basename}_{frame_count:05d}.jpg", processed_frame)
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
            return extract_frames_cuda(video_path, output_dir, target_fps, gpu_id=None, progress=progress)
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
        task_func = partial(extract_frames_cuda,
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
    specified_gpus = [4, 5]  # 指定可用的GPU ID列表

    # 初始化CUDA环境检查
    try:
        init_cuda()
        print(f"检测到 {cv2.cuda.getCudaEnabledDeviceCount()} 个CUDA设备")
    except Exception as e:
        print(str(e))
        specified_gpus = None  # 强制回退到CPU模式

    # 扫描视频文件
    video_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(input_folder)
        for f in files
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ]

    print(f"▶ 开始处理 {len(video_files)} 个视频 (目标帧率: {target_fps}FPS)")
    batch_processor(video_files, output_folder, target_fps, specified_gpus)