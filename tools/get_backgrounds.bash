#!/bin/bash

# 输入文件夹（包含子文件夹）
input_dir="/media/HDD0/XCX/UVEB"  # 替换为你的视频文件夹路径

# 输出文件夹（自动创建相同目录结构）
output_dir="/media/HDD0/XCX/UVEB/backgrounds"  # 替换为输出文件夹路径

# 目标帧率（每秒抽 23 帧）
target_fps=1

# 递归查找所有视频文件（支持 MP4/AVI/MOV/MKV）
find "$input_dir" -type f \( -iname "*.mp4" -o -iname "*.avi" -o -iname "*.mov" -o -iname "*.mkv" \) | while read -r video; do
    # 获取相对路径（保持目录结构）
    relative_path="${video#$input_dir/}"
    video_name="$(basename "$video")"
    video_name_noext="${video_name%.*}"

    # 输出子目录（如输出文件夹 + 原路径 + 视频文件名）
    output_subdir="$output_dir/$(dirname "$relative_path")/$video_name_noext"

    # 创建输出目录
    mkdir -p "$output_subdir"

    # 使用 FFmpeg 抽帧（每秒 target_fps 帧）
    ffmpeg -i "$video" \
           -vf "fps=$target_fps" \
           -q:v 2 \
           "$output_subdir/frame_%04d.jpg" \
           -hide_banner -loglevel error

    echo "已处理: $video → $output_subdir"
done

echo "所有视频抽帧完成！输出目录: $output_dir"