import os
import cv2


def video_to_frames(video_path, output_folder, fps=1.0):
    """
    按秒抽帧（默认每秒1帧）
    :param video_path: 视频路径
    :param output_folder: 输出文件夹
    :param fps: 每秒抽帧数（1.0表示每秒1帧）
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    # 获取视频原始帧率
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps) if fps > 0 else 1

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # 生成时间戳格式文件名（秒.毫秒）
            timestamp = frame_count / video_fps
            output_path = os.path.join(
                output_folder,
                f"{video_name}_{timestamp:.3f}s.jpg"  # 例如: video1_12.345s.jpg
            )
            cv2.imwrite(output_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"已从 {video_name} 抽取 {saved_count} 帧 (间隔 {1 / fps:.1f} 秒)")


def batch_process(input_dir, output_dir, fps=1.0):
    """批量处理所有子文件夹中的视频"""
    os.makedirs(output_dir, exist_ok=True)
    video_exts = ('.mp4', '.avi', '.mov', '.mkv', '.flv')

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(video_exts):
                video_path = os.path.join(root, file)
                video_to_frames(video_path, output_dir, fps)


if __name__ == "__main__":
    # 参数设置
    input1_folder = "/media/HDD0/XCX/UVEB/test"  # 包含子文件夹的视频目录
    input2_folder = "/media/HDD0/XCX/UVEB/train/blur"  # 包含子文件夹的视频目录
    output_folder = "/media/HDD0/XCX/backgrounds"  # 所有帧直接输出到这里
    frames_per_second = 1.0  # 每秒抽帧数

    # 执行处理
    batch_process(input1_folder, output_folder, frames_per_second)
    batch_process(input2_folder, output_folder, frames_per_second)