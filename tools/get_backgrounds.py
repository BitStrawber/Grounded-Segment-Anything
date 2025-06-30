import os
import cv2
from tqdm import tqdm


def extract_frames(video_path, output_dir, target_fps=23):
    """从视频中按指定帧率抽帧"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: 无法打开视频 {video_path}")
        return

    # 获取视频的原始帧率
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 1  # 默认值

    # 计算帧间隔（每隔多少帧取一帧）
    frame_interval = max(1, int(round(original_fps / target_fps)))

    os.makedirs(output_dir, exist_ok=True)
    frame_count = 0
    saved_count = 0
    success = True

    with tqdm(desc=f"处理 {os.path.basename(video_path)}") as pbar:
        while success:
            success, frame = cap.read()
            if not success:
                break

            # 按计算出的间隔抽帧
            if frame_count % frame_interval == 0:
                frame_filename = os.path.join(
                    output_dir,
                    f"frame_{saved_count:05d}.jpg"
                )
                cv2.imwrite(frame_filename, frame)
                saved_count += 1
                pbar.update(1)

            frame_count += 1

    cap.release()
    return saved_count


def process_videos(input_root, output_root, target_fps=23):
    """递归处理所有视频文件"""
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm')

    for root, _, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(video_extensions):
                video_path = os.path.join(root, file)

                # 保持原始目录结构
                relative_path = os.path.relpath(root, input_root)
                output_dir = os.path.join(output_root, relative_path, os.path.splitext(file)[0])

                print(f"\n开始处理: {video_path}")
                extracted = extract_frames(video_path, output_dir, target_fps)
                print(f"完成! 共提取 {extracted} 帧 -> {output_dir}")


if __name__ == "__main__":
    # 配置参数
    input_folder = "/media/HDD0/XCX/UVEB"  # 替换为你的视频文件夹路径
    output_folder = "/media/HDD0/XCX/backgrounds"  # 替换为输出文件夹路径
    target_fps = 1  # 目标帧率（每秒23帧）

    # 开始处理
    print(f"开始从 {input_folder} 递归提取视频帧...")
    process_videos(input_folder, output_folder, target_fps)
    print("\n所有视频处理完成！")