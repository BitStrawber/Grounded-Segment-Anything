import os
import cv2
from tqdm import tqdm


def extract_frames(video_path, output_dir, target_fps=23, video_index=0):
    """从视频中抽帧，直接保存到目标文件夹"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: 无法打开视频 {video_path}")
        return 0

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 1  # 防止除以零

    frame_interval = max(1, int(round(original_fps / target_fps)))
    os.makedirs(output_dir, exist_ok=True)

    saved_count = 0
    frame_count = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # 文件名格式：视频名_帧序号.jpg（例如：video1_00042.jpg）
            cv2.imwrite(
                os.path.join(output_dir, f"{video_name}_{saved_count:05d}.jpg"),
                frame
            )
            saved_count += 1
        frame_count += 1

    cap.release()
    return saved_count


def process_videos(input_root, output_root, target_fps=23):
    """处理所有视频，直接输出到目标文件夹"""
    # 扫描所有视频文件
    video_files = []
    for root, _, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm')):
                video_files.append(os.path.join(root, file))

    if not video_files:
        print("未找到任何视频文件！")
        return

    # 全局进度条
    with tqdm(video_files, desc="整体进度", unit="视频") as pbar:
        for i, video_path in enumerate(pbar):
            pbar.set_postfix_str(os.path.basename(video_path))
            extracted = extract_frames(video_path, output_root, target_fps, i)
            pbar.write(f"{os.path.basename(video_path)}: 提取 {extracted} 帧")


if __name__ == "__main__":
    # 配置参数
    input_folder = "/media/HDD0/XCX/UVEB"
    output_folder = "/media/HDD0/XCX/backgrounds"
    target_fps = 1  # 每秒1帧

    # 开始处理
    print(f"▶ 开始从 {input_folder} 提取视频帧（目标帧率: {target_fps}FPS）")
    print(f"▶ 输出模式: 所有图片直接保存到 {output_folder}")
    process_videos(input_folder, output_folder, target_fps)
    print("✅ 所有视频处理完成！")