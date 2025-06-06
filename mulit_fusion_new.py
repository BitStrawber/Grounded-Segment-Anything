import cv2
import numpy as np
import random
import os
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# 配置参数
class Config:
    # 输入路径
    background_dir = "/media/HDD0/XCX/backgrounds"  # 背景图像目录
    object_root_dir = "/media/HDD0/XCX/sam"  # 物体图像根目录

    # 输出路径
    output_root = "/media/HDD0/XCX/synthetic_dataset"  # 输出根目录
    blended_dir = os.path.join(output_root, "images")  # 融合后的图像
    annotations_dir = os.path.join(output_root, "annotations")  # 标注文件
    visualization_dir = os.path.join(output_root, "visualization")  # 可视化结果

    # 数据集参数
    min_objects_per_image = 1  # 每张图像最少物体数
    max_objects_per_image = 3  # 每张图像最多物体数
    min_instances_per_class = 1  # 每个类别最少实例数
    max_instances_per_class = 3  # 每个类别最多实例数
    test_size = 0.3  # 测试集比例
    random_seed = 42  # 随机种子
    max_overlap_ratio = 0.3  # 最大允许遮挡比例
    max_placement_attempts = 50  # 最大放置尝试次数

    # 类别字典将自动生成
    class_dict = None


# 确保输出目录存在
os.makedirs(Config.blended_dir, exist_ok=True)
os.makedirs(Config.annotations_dir, exist_ok=True)
os.makedirs(Config.visualization_dir, exist_ok=True)


def calculate_overlap_ratio(bbox1, bbox2):
    """计算两个边界框的重叠比例(相对于bbox1和bbox2的面积，取较小值)"""
    # 转换bbox格式为(x1, y1, x2, y2)
    box1 = [bbox1[0], bbox1[1], bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]]
    box2 = [bbox2[0], bbox2[1], bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]]

    # 计算交集区域
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # 计算交集面积和两个框的面积
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算重叠比例（相对于较小物体的面积）
    min_area = min(area1, area2)
    overlap_ratio = intersection_area / min_area if min_area > 0 else 0.0
    return overlap_ratio


def find_valid_position(target_shape, obj_size, existing_bboxes):
    """寻找有效位置，确保新物体与已放置物体的遮挡不超过阈值"""
    target_h, target_w = target_shape[:2]
    obj_w, obj_h = obj_size
    max_x = max(0, target_w - obj_w - 1)
    max_y = max(0, target_h - obj_h - 1)

    # 如果没有已放置的物体，直接返回随机位置
    if not existing_bboxes:
        return random.randint(0, max_x), random.randint(0, max_y)

    # 尝试找到满足条件的位置
    for _ in range(Config.max_placement_attempts):
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        new_bbox = [x, y, obj_w, obj_h]

        # 检查与所有现有边界框的重叠
        valid_position = True
        for bbox in existing_bboxes:
            overlap = calculate_overlap_ratio(new_bbox, bbox)
            if overlap > Config.max_overlap_ratio:
                valid_position = False
                break

        if valid_position:
            return x, y

    # 如果没有找到合适位置
    return None


def get_tight_bounding_box_from_mask(mask):
    """从掩码图像中获取紧贴目标物体的边界框"""
    rows, cols = np.nonzero(mask)
    if len(rows) == 0 or len(cols) == 0:
        return None
    y_min, y_max = np.min(rows), np.max(rows)
    x_min, x_max = np.min(cols), np.max(cols)
    return [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]


def resize_object_to_fit(object_image, mask, target_shape, min_scale=0.2, max_scale=1.0):
    """调整目标物体的尺寸以适应目标图像"""
    target_h, target_w = target_shape[:2]
    obj_h, obj_w = object_image.shape[:2]

    # 计算最大可能的缩放比例
    max_possible_scale = min(target_w / obj_w, target_h / obj_h)

    # 应用随机缩放
    scale_factor = max_possible_scale * random.uniform(min_scale, max_scale)
    new_w, new_h = int(obj_w * scale_factor), int(obj_h * scale_factor)

    # 缩放图像和掩码
    resized_obj = cv2.resize(object_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    return resized_obj, resized_mask, (new_w, new_h)


def smooth_blend_object(target_img, obj_img, mask, position, alpha=1.0):
    """将目标物体融合到背景图像中"""
    x, y = position
    obj_h, obj_w = obj_img.shape[:2]

    # 获取目标区域
    target_region = target_img[y:y + obj_h, x:x + obj_w]

    # 使用掩码进行融合
    blended = np.where(mask[..., None],
                       alpha * obj_img + (1 - alpha) * target_region,
                       target_region)

    # 更新目标图像
    target_img[y:y + obj_h, x:x + obj_w] = blended.astype(np.uint8)
    return target_img


def load_object_data(object_root_dir):
    """加载所有类别的物体数据和掩码"""
    object_data = {}

    # 获取所有类别目录
    class_dirs = [d for d in os.listdir(object_root_dir)
                  if os.path.isdir(os.path.join(object_root_dir, d))]

    # 自动生成类别字典
    Config.class_dict = {name: name for name in class_dirs}

    for class_name in class_dirs:
        class_dir = os.path.join(object_root_dir, class_name)
        images_dir = os.path.join(class_dir, "image")
        masks_dir = os.path.join(class_dir, "mask")

        # 检查目录是否存在
        if not os.path.exists(images_dir):
            print(f"Warning: Missing images directory for class {class_name}")
            continue
        if not os.path.exists(masks_dir):
            print(f"Warning: Missing masks directory for class {class_name}")
            continue

        # 获取该类别所有物体图像
        obj_files = [f for f in os.listdir(images_dir)
                     if f.endswith(('.png', '.jpg', '.jpeg')) and "_obj_" in f]

        # 存储物体和掩码路径
        object_data[class_name] = []
        for obj_file in obj_files:
            # 按照命名规则转换：将_obj_替换为_mask_
            mask_file = obj_file.replace("_obj_", "_mask_")
            obj_path = os.path.join(images_dir, obj_file)
            mask_path = os.path.join(masks_dir, mask_file)

            if os.path.exists(mask_path):
                object_data[class_name].append((obj_path, mask_path))
            else:
                print(f"Warning: Expected mask not found at {mask_path}")

    return object_data


def generate_synthetic_images():
    """生成合成图像和标注"""
    # 加载背景图像
    background_files = [f for f in os.listdir(Config.background_dir)
                        if f.endswith(('.png', '.jpg', '.jpeg'))]

    # 加载物体数据
    object_data = load_object_data(Config.object_root_dir)
    available_classes = list(object_data.keys())

    # 初始化COCO数据结构
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i + 1, "name": name, "supercategory": "object"}
                       for i, name in enumerate(available_classes)]
    }
    category_id_map = {name: i + 1 for i, name in enumerate(available_classes)}
    annotation_id = 1

    # 处理每张背景图像
    for img_idx, bg_file in enumerate(tqdm(background_files, desc="Processing images")):
        bg_path = os.path.join(Config.background_dir, bg_file)
        bg_img = cv2.imread(bg_path)
        if bg_img is None:
            continue

        # 复制原始背景用于合成
        synthetic_img = bg_img.copy()
        bg_h, bg_w = bg_img.shape[:2]

        # 随机选择1-3个类别
        num_classes = random.randint(Config.min_objects_per_image,
                                     min(Config.max_objects_per_image, len(available_classes)))
        selected_classes = random.sample(available_classes, num_classes)

        # 存储已放置物体的边界框
        placed_bboxes = []
        annotations = []

        # 为每个选中的类别添加1-3个实例
        for class_name in selected_classes:
            # 随机选择1-3个该类的实例
            num_instances = random.randint(Config.min_instances_per_class,
                                           Config.max_instances_per_class)

            # 确保不超过该类可用实例数
            available_instances = object_data[class_name]
            num_instances = min(num_instances, len(available_instances))
            selected_instances = random.sample(available_instances, num_instances)

            for obj_path, mask_path in selected_instances:
                # 加载物体和掩码
                obj_img = cv2.imread(obj_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if obj_img is None or mask is None:
                    continue

                # 调整物体大小
                resized_obj, resized_mask, obj_size = resize_object_to_fit(
                    obj_img, mask, bg_img.shape)

                # 寻找有效位置
                position = find_valid_position(bg_img.shape, obj_size, placed_bboxes)
                if position is None:
                    continue  # 没有找到合适位置，跳过该物体

                # 融合物体到背景
                synthetic_img = smooth_blend_object(
                    synthetic_img, resized_obj, resized_mask, position)

                # 计算边界框
                bbox = get_tight_bounding_box_from_mask(resized_mask)
                if bbox is None:
                    continue

                # 调整边界框坐标
                bbox[0] += position[0]
                bbox[1] += position[1]

                # 确保边界框在图像范围内
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])
                bbox[2] = min(bbox[2], bg_w - bbox[0])
                bbox[3] = min(bbox[3], bg_h - bbox[1])

                # 添加到已放置边界框列表
                placed_bboxes.append(bbox.copy())

                # 添加到标注
                annotations.append({
                    "id": annotation_id,
                    "image_id": img_idx + 1,
                    "category_id": category_id_map[class_name],
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0
                })
                annotation_id += 1

        # 保存合成图像
        output_img_path = os.path.join(Config.blended_dir, bg_file)
        cv2.imwrite(output_img_path, synthetic_img)

        # 添加图像信息到COCO数据
        coco_data["images"].append({
            "id": img_idx + 1,
            "file_name": bg_file,
            "width": bg_w,
            "height": bg_h
        })

        # 添加所有标注
        coco_data["annotations"].extend(annotations)

    return coco_data


def split_and_save_dataset(coco_data):
    """划分并保存数据集"""
    # 划分数据集
    train_images, test_images = train_test_split(
        coco_data["images"], test_size=Config.test_size, random_state=Config.random_seed)

    # 获取对应的标注
    train_ids = {img["id"] for img in train_images}
    test_ids = {img["id"] for img in test_images}

    train_annotations = [ann for ann in coco_data["annotations"]
                         if ann["image_id"] in train_ids]
    test_annotations = [ann for ann in coco_data["annotations"]
                        if ann["image_id"] in test_ids]

    # 保存完整数据集
    full_output = os.path.join(Config.annotations_dir, "instances_all.json")
    with open(full_output, "w") as f:
        json.dump(coco_data, f, indent=2)

    # 保存训练集
    train_output = os.path.join(Config.annotations_dir, "instances_train.json")
    with open(train_output, "w") as f:
        json.dump({
            "images": train_images,
            "annotations": train_annotations,
            "categories": coco_data["categories"]
        }, f, indent=2)

    # 保存测试集
    test_output = os.path.join(Config.annotations_dir, "instances_test.json")
    with open(test_output, "w") as f:
        json.dump({
            "images": test_images,
            "annotations": test_annotations,
            "categories": coco_data["categories"]
        }, f, indent=2)

    print(f"数据集已保存到: {Config.output_root}")
    print(f"训练集: {len(train_images)} 张图像, {len(train_annotations)} 个标注")
    print(f"测试集: {len(test_images)} 张图像, {len(test_annotations)} 个标注")


def visualize_annotations():
    """可视化标注结果"""
    # 加载完整标注
    annotation_path = os.path.join(Config.annotations_dir, "instances_all.json")
    with open(annotation_path, "r") as f:
        coco_data = json.load(f)

    # 创建类别ID到名称的映射
    id_to_name = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

    # 处理每张图像
    for img_info in tqdm(coco_data["images"], desc="Visualizing annotations"):
        img_path = os.path.join(Config.blended_dir, img_info["file_name"])
        img = cv2.imread(img_path)
        if img is None:
            continue

        # 获取该图像的所有标注
        annotations = [ann for ann in coco_data["annotations"]
                       if ann["image_id"] == img_info["id"]]

        # 绘制每个标注
        for ann in annotations:
            bbox = list(map(int, ann["bbox"]))
            class_name = id_to_name[ann["category_id"]]

            # 绘制边界框
            cv2.rectangle(img, (bbox[0], bbox[1]),
                          (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                          (0, 255, 0), 2)

            # 添加类别标签
            cv2.putText(img, class_name, (bbox[0], bbox[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 保存可视化结果
        vis_path = os.path.join(Config.visualization_dir, img_info["file_name"])
        cv2.imwrite(vis_path, img)


if __name__ == "__main__":
    # 生成合成数据和标注
    coco_data = generate_synthetic_images()

    # 划分并保存数据集
    split_and_save_dataset(coco_data)

    # 可视化标注结果
    visualize_annotations()

    print("数据集生成和划分完成!")