import os
import shutil
import scipy.io
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def load_imagenet_meta(devkit_path):
    """加载ImageNet元数据，返回WNID到(ILSVRC_ID, 类别名)的映射字典"""
    meta_path = os.path.join(devkit_path, 'data', 'meta.mat')
    synsets = scipy.io.loadmat(meta_path)['synsets']
    return {
        s[0][1][0]: (int(s[0][0][0][0]), s[0][2][0].replace(', ', '_'))
        for s in synsets if int(s[0][0][0][0]) <= 1000  # 仅保留1-1000的类别
    }


def process_single_class(wnid, src_root, target_root, wnid_to_info):
    """处理单个WNID类别的文件（仅处理目标WNID）"""
    # 检查是否为需要处理的WNID
    if wnid not in wnid_to_info:
        return wnid, 0

    src_folder = os.path.join(src_root, wnid)
    if not os.path.isdir(src_folder):
        return wnid, 0

    # 获取类别信息
    ilsvrc_id, class_name = wnid_to_info[wnid]

    # 创建目标文件夹（格式：ILSVRCID_类别名）
    target_folder = os.path.join(target_root, f"{class_name}")
    os.makedirs(target_folder, exist_ok=True)

    # 复制所有图片
    copied = 0
    for img in os.listdir(src_folder):
        shutil.copy2(
            os.path.join(src_folder, img),
            os.path.join(target_folder, img)
        )
        copied += 1

    return wnid, copied


def organize_imagenet_by_id(
        src_dir,
        target_dir,
        devkit_path,
        target_wnids=None,  # 新增：指定目标WNID列表
        num_workers=8
):
    """
    主函数：按类别ID重组ImageNet训练集（支持指定目标WNID）
    :param target_wnids: 需要处理的WNID列表（如["n01440764", "n02119789"]）
    """
    # 加载元数据
    wnid_to_info = load_imagenet_meta(devkit_path)

    # 如果指定了target_wnids，则过滤元数据
    if target_wnids is not None:
        wnid_to_info = {k: v for k, v in wnid_to_info.items() if k in target_wnids}
        print(f"✅ 已过滤，待处理类别数: {len(wnid_to_info)}")

    # 获取所有WNID文件夹（如果指定了target_wnids，则只处理这些）
    wnids = target_wnids if target_wnids is not None else [
        d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))
    ]

    # 多进程处理
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for wnid in wnids:
            futures.append(executor.submit(
                process_single_class,
                wnid, src_dir, target_dir, wnid_to_info
            ))

        # 进度监控
        total = len(wnids)
        pbar = tqdm(total=total, desc="处理进度", unit="类别")

        results = {}
        for future in futures:
            wnid, count = future.result()
            results[wnid] = count
            pbar.set_postfix_str(f"最新: {wnid} ({count}张)")
            pbar.update(1)

        pbar.close()

    # 统计报告
    total_copied = sum(results.values())
    print(f"\n✅ 完成！共处理 {len(results)} 个类别，复制 {total_copied} 张图片")
    print(f"目标路径: {target_dir}")


if __name__ == "__main__":
    # 配置路径
    TRAIN_SRC_DIR = "/path/to/ILSVRC2012_img_train"  # 原始训练集路径
    TARGET_DIR = "/path/to/organized_imagenet"  # 目标路径
    DEVKIT_PATH = "/path/to/ILSVRC2012_devkit_t12"  # devkit解压路径

    # 指定需要处理的WNID列表（示例）
    TARGET_WNIDS = ["n01440764", "n02119789", "n03000684"]  # 修改为你需要的WNID

    # 执行重组（仅处理指定WNID）
    organize_imagenet_by_id(
        src_dir=TRAIN_SRC_DIR,
        target_dir=TARGET_DIR,
        devkit_path=DEVKIT_PATH,
        target_wnids=TARGET_WNIDS,  # 传入目标WNID列表
        num_workers=os.cpu_count()
    )