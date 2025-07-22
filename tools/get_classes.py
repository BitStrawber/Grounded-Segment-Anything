import os
import tarfile
import scipy.io
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def load_imagenet_meta(devkit_path):
    """
    【已修改】加载ImageNet元数据，返回 WNID 到 “核心英文名” 的映射。
    它会从标签列表（如 "tench, Tinca tinca"）中提取第一个名字（"tench"），
    并将其中的空格替换为下划线，使其成为一个安全、规范的文件夹名。
    """
    meta_path = os.path.join(devkit_path, 'data', 'meta.mat')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"错误：在路径 '{devkit_path}' 下找不到 meta.mat 文件。")

    synsets = scipy.io.loadmat(meta_path)['synsets']

    wnid_to_name = {}
    for s in synsets:
        wnid = s[0][1][0]
        # 完整标签，例如:"great white shark, white shark, man-eater..."
        full_label = s[0][2][0]
        # 1. 用逗号分割，只取第一部分
        primary_name = full_label.split(',')[0].strip()
        # 2. 将这部分的空格替换为下划线，作为最终名称
        safe_name = primary_name.replace(' ', '_')
        wnid_to_name[wnid] = safe_name

    return wnid_to_name


def extract_single_class(wnid, class_name, src_dir, target_dir):
    """
    处理单个WNID：找到对应的.tar文件并将其解压到指定的目标子文件夹。
    """
    src_archive_path = os.path.join(src_dir, f"{wnid}.tar")

    # 检查源文件是否存在
    if not os.path.exists(src_archive_path):
        return wnid, 0, f"源文件未找到: {src_archive_path}"

    # 定义并创建目标文件夹
    class_target_dir = os.path.join(target_dir, class_name)
    os.makedirs(class_target_dir, exist_ok=True)

    try:
        # 直接解压到目标文件夹
        with tarfile.open(src_archive_path, 'r') as tar:
            # 获取成员数量用于返回
            members = tar.getmembers()
            tar.extractall(path=class_target_dir)
            return wnid, len(members), "成功"
    except Exception as e:
        return wnid, 0, f"解压失败: {e}"


def extract_imagenet_subclasses(
        src_dir,
        target_dir,
        devkit_path,
        target_wnids,
        num_workers=8
):
    """
    主函数：按给定的WNID列表，查找、创建并解压ImageNet子类别。
    """
    print("1. 正在加载ImageNet元数据...")
    try:
        wnid_to_name = load_imagenet_meta(devkit_path)
    except FileNotFoundError as e:
        print(e)
        return

    # 过滤出我们需要处理的类别信息
    tasks = []
    for wnid in target_wnids:
        if wnid in wnid_to_name:
            tasks.append((wnid, wnid_to_name[wnid]))
        else:
            print(f"⚠️ 警告：在meta.mat中未找到WNID '{wnid}'，将跳过。")

    print(f"2. 准备处理 {len(tasks)} 个目标子类别...")

    # 多进程处理
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        futures = [
            executor.submit(extract_single_class, wnid, class_name, src_dir, target_dir)
            for wnid, class_name in tasks
        ]

        # 使用tqdm监控进度
        pbar = tqdm(total=len(tasks), desc="解压进度", unit="类别")

        success_count = 0
        for future in futures:
            wnid, num_files, status = future.result()
            if status == "成功":
                success_count += 1
            else:
                pbar.write(f"❌ 错误 ({wnid}): {status}")
            pbar.set_postfix_str(f"最新: {wnid} ({status})")
            pbar.update(1)

        pbar.close()

    print(f"\n✅ 全部完成！")
    print(f"成功处理并解压了 {success_count} / {len(tasks)} 个子类别。")
    print(f"数据已存放在目标路径: {target_dir}")


if __name__ == "__main__":
    # 1. ===== 配置您的路径 =====
    # 包含所有 nXXXX.tar 文件的原始ImageNet数据集路径
    TRAIN_SRC_DIR = "/media/HDD0/XCX/IMAGENET"
    # 您希望存放提取出的子类别文件夹的目标路径
    TARGET_DIR = "/media/HDD0/XCX/classes/images"
    # devkit解压后的路径
    DEVKIT_PATH = "/media/HDD0/XCX/IMAGENET/ILSVRC2012_devkit_t12"

    # 2. ===== 定义需要提取的所有子类别WNID =====
    # 这是根据我们之前讨论的所有海洋生物子类别整理的完整列表
    TARGET_WNIDS = [
        'n01968897','n01950731'
    ]

    # 3. ===== 执行脚本 =====
    # 使用与CPU核心数匹配的进程数以获得最佳性能
    cpu_cores = os.cpu_count() or 8

    extract_imagenet_subclasses(
        src_dir=TRAIN_SRC_DIR,
        target_dir=TARGET_DIR,
        devkit_path=DEVKIT_PATH,
        target_wnids=TARGET_WNIDS,
        num_workers=cpu_cores
    )