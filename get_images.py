import os
import shutil
import scipy.io as sio


def extract_imagenet_train_subset(
        imagenet_train_dir,  # ImageNet训练集路径（如包含n01440764/的目录）
        target_ids,  # 目标ID列表，如["n01440764", "n02119789"]
        output_dir,  # 输出目录
        meta_path=None  # 元数据文件路径（默认从devkit读取）
):
    # 1. 加载元数据（映射ID到类别名）
    if meta_path is None:
        # 默认从devkit目录查找meta.mat
        devkit_dir = os.path.join(os.path.dirname(imagenet_train_dir), 'devkit')
        meta_path = os.path.join(devkit_dir, 'data', 'meta.mat')

    meta = sio.loadmat(meta_path, squeeze_me=True)['synsets']
    id_to_name = {}
    for item in meta:
        synset_id = item['synset_id']
        class_name = item['words'].split(',')[0].strip()  # 取第一个名称（如"tench"）
        id_to_name[synset_id] = class_name

    # 2. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 3. 复制目标类别的图像
    for synset_id in target_ids:
        if synset_id not in id_to_name:
            print(f"警告：跳过未知的类别ID {synset_id}")
            continue

        # 源目录（如/path/to/imagenet/train/n01440764/）
        src_dir = os.path.join(imagenet_train_dir, synset_id)
        if not os.path.exists(src_dir):
            print(f"警告：跳过不存在的目录 {src_dir}")
            continue

        # 目标目录（如output_dir/tench/）
        class_name = id_to_name[synset_id]
        dst_dir = os.path.join(output_dir, class_name)
        os.makedirs(dst_dir, exist_ok=True)

        # 复制所有图像
        for filename in os.listdir(src_dir):
            src_path = os.path.join(src_dir, filename)
            dst_path = os.path.join(dst_dir, filename)
            shutil.copy2(src_path, dst_path)

        print(f"已复制 {synset_id} ({class_name})：{len(os.listdir(src_dir))} 张图像")

    print(f"完成！子数据集已保存到 {output_dir}")


# 使用示例
if __name__ == "__main__":
    extract_imagenet_train_subset(
        imagenet_train_dir="/path/to/imagenet/train/",  # 替换为你的路径
        target_ids=["n01440764", "n02119789"],  # 目标类别ID
        output_dir="./imagenet_subset"  # 输出目录
    )