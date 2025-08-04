import cv2
import numpy as np
import random
import os
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor  # 1. 引入多进程池
from functools import partial  # 用于包装函数参数，方便传递给map


# ==============================================================================
# 1. 全局配置与辅助函数
# ==============================================================================

class Config:
    """集中管理所有配置参数，方便修改。"""
    # ... (配置内容与之前相同，此处省略以保持简洁) ...
    # 输入路径
    background_dir = "/media/HDD0/XCX/selected_backgrounds"
    object_root_dir = "/media/HDD0/XCX/classes/masks"
    # 输出路径
    output_root = "/media/HDD0/XCX/fusions"
    blended_dir = os.path.join(output_root, "images")
    annotations_dir = os.path.join(output_root, "annotations")
    visualization_dir = os.path.join(output_root, "visualization")
    # 数据集参数
    min_objects_per_image = 1
    max_objects_per_image = 3
    min_instances_per_class = 1
    max_instances_per_class = 3
    test_size = 0.3
    random_seed = 42
    max_overlap_ratio = 0.3
    max_placement_attempts = 50
    class_dict = None


# 确保输出目录存在
os.makedirs(Config.blended_dir, exist_ok=True)
os.makedirs(Config.annotations_dir, exist_ok=True)
os.makedirs(Config.visualization_dir, exist_ok=True)


# --- 原始辅助函数 (功能不变，注释已添加) ---
def calculate_overlap_ratio(bbox1, bbox2):
    """计算两个边界框的重叠比例。"""
    box1 = [bbox1[0], bbox1[1], bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]]
    box2 = [bbox2[0], bbox2[1], bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]]
    x_left, y_top = max(box1[0], box2[0]), max(box1[1], box2[1])
    x_right, y_bottom = min(box1[2], box2[2]), min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top: return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area1, area2 = bbox1[2] * bbox1[3], bbox2[2] * bbox2[3]
    min_area = min(area1, area2)
    return intersection_area / min_area if min_area > 0 else 0.0


def find_valid_position(target_shape, obj_size, existing_bboxes):
    """为新物体寻找一个与已存在物体不严重重叠的随机位置。"""
    target_h, target_w = target_shape[:2];
    obj_w, obj_h = obj_size
    max_x, max_y = max(0, target_w - obj_w - 1), max(0, target_h - obj_h - 1)
    if not existing_bboxes: return random.randint(0, max_x), random.randint(0, max_y)
    for _ in range(Config.max_placement_attempts):
        x, y = random.randint(0, max_x), random.randint(0, max_y)
        new_bbox = [x, y, obj_w, obj_h]
        if all(calculate_overlap_ratio(new_bbox, bbox) <= Config.max_overlap_ratio for bbox in
               existing_bboxes): return x, y
    return None


def get_tight_bounding_box_from_mask(mask):
    """从二值掩码计算紧密的边界框。"""
    rows, cols = np.nonzero(mask)
    if not len(rows): return None
    y_min, y_max, x_min, x_max = np.min(rows), np.max(rows), np.min(cols), np.max(cols)
    return [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]


def resize_object_to_fit(object_image, mask, target_shape, min_scale=0.2, max_scale=1.0):
    """随机缩放物体以适应背景尺寸。"""
    target_h, target_w = target_shape[:2];
    obj_h, obj_w = object_image.shape[:2]
    max_possible_scale = min(target_w / obj_w, target_h / obj_h)
    scale_factor = max_possible_scale * random.uniform(min_scale, max_scale)
    new_w, new_h = int(obj_w * scale_factor), int(obj_h * scale_factor)
    resized_obj = cv2.resize(object_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return resized_obj, resized_mask, (new_w, new_h)


def smooth_blend_object(target_img, obj_img, mask, position, alpha=1.0):
    """将物体图像根据掩码融合到背景图上。"""
    x, y = position;
    obj_h, obj_w = obj_img.shape[:2]
    target_region = target_img[y:y + obj_h, x:x + obj_w]
    blended = np.where(mask[..., None], alpha * obj_img + (1 - alpha) * target_region, target_region)
    target_img[y:y + obj_h, x:x + obj_w] = blended.astype(np.uint8)
    return target_img


def load_object_data(object_root_dir):
    """加载所有物体的图像和掩码路径。"""
    object_data = {}
    class_dirs = [d for d in os.listdir(object_root_dir) if os.path.isdir(os.path.join(object_root_dir, d))]
    Config.class_dict = {name: name for name in class_dirs}
    for class_name in class_dirs:
        images_dir = os.path.join(object_root_dir, class_name, "image")
        masks_dir = os.path.join(object_root_dir, class_name, "mask")
        if not (os.path.exists(images_dir) and os.path.exists(masks_dir)): continue
        obj_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg')) and "_obj_" in f]
        object_data[class_name] = []
        for obj_file in obj_files:
            mask_file = obj_file.replace("_obj_", "_mask_")
            obj_path, mask_path = os.path.join(images_dir, obj_file), os.path.join(masks_dir, mask_file)
            if os.path.exists(mask_path): object_data[class_name].append((obj_path, mask_path))
    return object_data


# ==============================================================================
# 2. 【新功能】用于多进程的任务函数
# ==============================================================================
def process_single_background(task_args):
    """
    处理单个背景图片，生成合成图和标注。
    这是每个子进程要执行的核心任务，它接收一个包含所有必要参数的元组。
    """
    # 解包任务参数
    img_idx, bg_file, object_data, category_id_map = task_args
    available_classes = list(object_data.keys())

    # 读取背景图片
    bg_path = os.path.join(Config.background_dir, bg_file)
    bg_img = cv2.imread(bg_path)
    if bg_img is None:
        return None  # 如果背景图无法读取，返回None，主进程会忽略这个结果

    # --- 核心合成逻辑 (与原单线程版本一致) ---
    synthetic_img = bg_img.copy()
    bg_h, bg_w = bg_img.shape[:2]

    num_classes = random.randint(Config.min_objects_per_image,
                                 min(Config.max_objects_per_image, len(available_classes)))
    selected_classes = random.sample(available_classes, num_classes)

    placed_bboxes, annotations = [], []

    for class_name in selected_classes:
        num_instances = random.randint(Config.min_instances_per_class, Config.max_instances_per_class)
        available_instances = object_data.get(class_name, [])
        if not available_instances: continue
        selected_instances = random.sample(available_instances, min(num_instances, len(available_instances)))

        for obj_path, mask_path in selected_instances:
            obj_img, mask = cv2.imread(obj_path), cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if obj_img is None or mask is None: continue

            resized_obj, resized_mask, obj_size = resize_object_to_fit(obj_img, mask, bg_img.shape)
            position = find_valid_position(bg_img.shape, obj_size, placed_bboxes)
            if position is None: continue

            synthetic_img = smooth_blend_object(synthetic_img, resized_obj, resized_mask, position)
            bbox = get_tight_bounding_box_from_mask(resized_mask)
            if bbox is None: continue

            bbox[0] += position[0];
            bbox[1] += position[1]
            bbox[0] = max(0, bbox[0]);
            bbox[1] = max(0, bbox[1])
            bbox[2] = min(bbox[2], bg_w - bbox[0]);
            bbox[3] = min(bbox[3], bg_h - bbox[1])
            placed_bboxes.append(bbox.copy())

            # 创建标注信息，注意image_id使用的是传入的唯一ID
            annotations.append({
                "image_id": img_idx,
                "category_id": category_id_map[class_name],
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0
            })

    # 保存最终合成的图像
    output_img_path = os.path.join(Config.blended_dir, bg_file)
    cv2.imwrite(output_img_path, synthetic_img)

    # 准备这张图的COCO信息
    image_info = {"id": img_idx, "file_name": bg_file, "width": bg_w, "height": bg_h}

    # 返回这张图的信息和它所有的标注，由主进程统一收集
    return image_info, annotations


# ==============================================================================
# 3. 【已修改】使用多进程加速的主函数
# ==============================================================================
def generate_synthetic_images_mp():
    """使用多进程池来并行生成所有合成图像。"""
    # 步骤1: 在主进程中提前加载所有需要共享的数据。
    # 这样做可以避免每个子进程都重复加载一遍，提高效率。
    print("正在加载背景和物体数据...")
    background_files = [f for f in os.listdir(Config.background_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    object_data = load_object_data(Config.object_root_dir)
    available_classes = list(object_data.keys())

    # 检查是否有足够的输入数据
    if not background_files or not available_classes:
        print("错误：背景或物体文件夹为空，程序无法继续。")
        return None

    print(f"找到 {len(background_files)} 个背景和 {len(available_classes)} 个物体类别。")
    category_id_map = {name: i + 1 for i, name in enumerate(available_classes)}

    # 步骤2: 准备所有任务的参数列表。每个元素都是一个元组，包含处理一张背景图所需的所有信息。
    tasks = [(i + 1, bg_file, object_data, category_id_map) for i, bg_file in enumerate(background_files)]

    # 步骤3: 初始化最终的COCO数据结构
    coco_data = {"images": [], "annotations": [],
                 "categories": [{"id": cat_id, "name": name, "supercategory": "object"} for name, cat_id in
                                category_id_map.items()]}
    annotation_id_counter = 1

    # 步骤4: 创建并运行进程池
    num_workers = os.cpu_count() or 4  # 自动获取CPU核心数作为进程数，如果失败则默认为4
    print(f"启动 {num_workers} 个进程进行并行处理...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # executor.map 会自动将tasks列表中的每个元素分配给一个子进程，并调用process_single_background函数。
        # tqdm在这里用于创建一个进度条，实时显示任务完成情况。
        results = list(tqdm(executor.map(process_single_background, tasks), total=len(tasks), desc="生成合成图像"))

    # 步骤5: 在主进程中安全地聚合所有子进程返回的结果
    print("所有子进程已完成，正在聚合结果...")
    for result in results:
        if result is None: continue  # 忽略处理失败的背景图

        image_info, annotations_for_image = result
        coco_data["images"].append(image_info)

        # 为每个标注分配一个全局唯一的、递增的ID
        for ann in annotations_for_image:
            ann["id"] = annotation_id_counter
            coco_data["annotations"].append(ann)
            annotation_id_counter += 1

    return coco_data


# 数据集划分和可视化函数 (保持不变)
def split_and_save_dataset(coco_data):
    """划分数据集为训练集和测试集并保存。"""
    if not coco_data or not coco_data["images"]:
        print("错误：没有生成任何数据，无法进行划分。")
        return
    train_images, test_images = train_test_split(coco_data["images"], test_size=Config.test_size,
                                                 random_state=Config.random_seed)
    train_ids, test_ids = {img["id"] for img in train_images}, {img["id"] for img in test_images}
    train_annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] in train_ids]
    test_annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] in test_ids]

    output_files = {"all": coco_data, "train": {"images": train_images, "annotations": train_annotations,
                                                "categories": coco_data["categories"]},
                    "test": {"images": test_images, "annotations": test_annotations,
                             "categories": coco_data["categories"]}}
    for name, data in output_files.items():
        with open(os.path.join(Config.annotations_dir, f"instances_{name}.json"), "w") as f: json.dump(data, f,
                                                                                                       indent=2)

    print(f"数据集已保存到: {Config.output_root}")
    print(f"训练集: {len(train_images)} 张图像, {len(train_annotations)} 个标注")
    print(f"测试集: {len(test_images)} 张图像, {len(test_annotations)} 个标注")


def visualize_annotations():
    """生成带标注的可视化图像，用于检查。"""
    annotation_path = os.path.join(Config.annotations_dir, "instances_all.json")
    if not os.path.exists(annotation_path): return
    with open(annotation_path, "r") as f:
        coco_data = json.load(f)
    id_to_name = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
    for img_info in tqdm(coco_data["images"], desc="可视化标注"):
        img_path = os.path.join(Config.blended_dir, img_info["file_name"])
        img = cv2.imread(img_path)
        if img is None: continue
        annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] == img_info["id"]]
        for ann in annotations:
            bbox = list(map(int, ann["bbox"]))
            class_name = id_to_name.get(ann["category_id"], "Unknown")
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
            cv2.putText(img, class_name, (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(Config.visualization_dir, img_info["file_name"]), img)


# ==============================================================================
# 4. 【重要】主执行入口
# ==============================================================================
if __name__ == "__main__":
    # 多进程代码必须放在 `if __name__ == "__main__":` 块下。
    # 这是为了防止子进程在启动时再次导入并执行主模块的代码，从而导致无限创建子进程的循环。

    # 步骤一: 使用新的多进程函数生成数据
    print("--- 步骤 1/3: 开始生成合成数据集 (多进程加速)... ---")
    coco_data = generate_synthetic_images_mp()

    # 步骤二: 划分并保存数据集
    print("\n--- 步骤 2/3: 开始划分并保存数据集... ---")
    split_and_save_dataset(coco_data)

    # 步骤三: 生成可视化结果以供检查
    print("\n--- 步骤 3/3: 开始生成可视化结果... ---")
    visualize_annotations()

    print("\n所有任务已成功完成!")