import os
import json
import random
from sklearn.model_selection import train_test_split


def split_coco_dataset(ann_file, output_dir, part1_ratio=0.2, test_ratio=0.25, seed=42):
    """
    划分COCO数据集：
    1. 随机划分为两部分（1:4比例）
    2. 将两部分都分别划分为训练集和测试集

    Args:
        ann_file: COCO标注文件路径
        output_dir: 输出目录
        part1_ratio: 第一部分占比（默认0.2，即1:4）
        test_ratio: 测试集占每部分的比率（默认0.25）
        seed: 随机种子
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载原始标注
    with open(ann_file) as f:
        ann_data = json.load(f)

    # 获取所有图像ID并随机打乱
    all_images = ann_data['images']
    random.seed(seed)
    random.shuffle(all_images)

    # 计算划分点
    split_idx = int(len(all_images) * part1_ratio)

    # 划分两部分
    part1_images = all_images[:split_idx]  # 20%数据
    part2_images = all_images[split_idx:]  # 80%数据

    # 在第一部分中划分训练集和测试集
    part1_train_images, part1_test_images = train_test_split(
        part1_images,
        test_size=test_ratio,
        random_state=seed
    )

    # 在第二部分中也划分训练集和测试集
    part2_train_images, part2_test_images = train_test_split(
        part2_images,
        test_size=test_ratio,
        random_state=seed
    )

    # 构建标注子集函数
    def build_subset(images):
        image_ids = {img['id'] for img in images}
        return {
            'images': images,
            'annotations': [ann for ann in ann_data['annotations'] if ann['image_id'] in image_ids],
            'categories': ann_data['categories'],
            'info': ann_data.get('info', {}),
            'licenses': ann_data.get('licenses', [])
        }

    # 构建各数据集
    part2_full_ann = build_subset(part2_images)  # 完整的part2数据集
    part1_train_ann = build_subset(part1_train_images)
    part1_test_ann = build_subset(part1_test_images)
    part2_train_ann = build_subset(part2_train_images)
    part2_test_ann = build_subset(part2_test_images)

    # 保存结果
    with open(os.path.join(output_dir, 'part2.json'), 'w') as f:
        json.dump(part2_full_ann, f)
    with open(os.path.join(output_dir, 'part1_train.json'), 'w') as f:
        json.dump(part1_train_ann, f)
    with open(os.path.join(output_dir, 'part1_test.json'), 'w') as f:
        json.dump(part1_test_ann, f)
    with open(os.path.join(output_dir, 'part2_train.json'), 'w') as f:
        json.dump(part2_train_ann, f)
    with open(os.path.join(output_dir, 'part2_test.json'), 'w') as f:
        json.dump(part2_test_ann, f)

    # 打印统计信息
    print(f"划分完成！结果保存在 {output_dir}")
    print(f"总图像数: {len(all_images)}")
    print(f"Part1 (20%): {len(part1_images)}张")
    print(f"  ├─ 训练集: {len(part1_train_images)}张")
    print(f"  └─ 测试集: {len(part1_test_images)}张")
    print(f"Part2 (80%): {len(part2_images)}张")
    print(f"  ├─ 训练集: {len(part2_train_images)}张")
    print(f"  └─ 测试集: {len(part2_test_images)}张")
    print(f"完整Part2数据集已保存为 part2_full.json")


if __name__ == '__main__':
    # 使用示例
    split_coco_dataset(
        ann_file='/home/xcx/桌面/synthetic_dataset/annotations/instances_all.json',
        output_dir='/home/xcx/桌面/synthetic_dataset/annotations/split_results',
        part1_ratio=0.2,
        test_ratio=0.25,
        seed=42
    )