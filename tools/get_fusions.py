import cv2
import numpy as np
import random
import os
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# ==============================================================================
# 1. 全局配置参数
# ==============================================================================
class Config:
    """
    该类用于集中管理脚本的所有配置参数，方便修改和维护。
    """
    # --- 输入路径 ---
    background_dir = "/media/HDD0/XCX/selected_backgrounds"  # 背景图像所在的目录
    object_root_dir = "/media/HDD0/XCX/classes/masks"  # 所有物体（带掩码）的根目录

    # --- 输出路径 ---
    output_root = "/media/HDD0/XCX/fusions"  # 所有生成结果的根目录
    blended_dir = os.path.join(output_root, "images")  # 用于存放最终合成的图像
    annotations_dir = os.path.join(output_root, "annotations")  # 用于存放生成的COCO格式标注文件
    visualization_dir = os.path.join(output_root, "visualization")  # 用于存放带标注框的可视化结果图

    # --- 数据集生成参数 ---
    min_objects_per_image = 1  # 每张背景图上最少贴几个物体
    max_objects_per_image = 3  # 每张背景图上最多贴几个物体
    min_instances_per_class = 1  # 对于选定的每个类别，最少贴几个该类的实例
    max_instances_per_class = 3  # 对于选定的每个类别，最多贴几个该类的实例
    test_size = 0.3  # 测试集在总数据中的比例 (例如0.3代表30%)
    random_seed = 42  # 随机种子，用于确保每次划分数据集的结果一致
    max_overlap_ratio = 0.3  # 两个物体之间允许的最大遮挡比例，以较小物体面积为基准
    max_placement_attempts = 50  # 为一个物体寻找不重叠位置的最大尝试次数

    # --- 类别信息 ---
    class_dict = None  # 类别字典，将在加载物体时自动生成


# 在脚本开始时，确保所有输出目录都已存在，如果不存在则创建
os.makedirs(Config.blended_dir, exist_ok=True)
os.makedirs(Config.annotations_dir, exist_ok=True)
os.makedirs(Config.visualization_dir, exist_ok=True)


# ==============================================================================
# 2. 核心辅助函数
# ==============================================================================

def calculate_overlap_ratio(bbox1, bbox2):
    """计算两个边界框（格式为[x, y, w, h]）的重叠比例。"""
    # 将 [x, y, w, h] 格式转换为 [x1, y1, x2, y2] 格式，方便计算
    box1 = [bbox1[0], bbox1[1], bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]]
    box2 = [bbox2[0], bbox2[1], bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]]

    # 计算交集矩形的左上角和右下角坐标
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # 如果没有重叠区域，交集面积为0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # 计算交集面积
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    # 计算两个边界框各自的面积
    area1 = bbox1[2] * bbox1[3]
    area2 = bbox2[2] * bbox2[3]

    # 重叠比例定义为：交集面积 / 两个框中较小的那个的面积
    min_area = min(area1, area2)
    overlap_ratio = intersection_area / min_area if min_area > 0 else 0.0
    return overlap_ratio


def find_valid_position(target_shape, obj_size, existing_bboxes):
    """在背景图上为新物体寻找一个有效（低遮挡）的放置位置。"""
    target_h, target_w = target_shape[:2]
    obj_w, obj_h = obj_size
    # 计算物体可以放置的最大x, y坐标
    max_x = max(0, target_w - obj_w - 1)
    max_y = max(0, target_h - obj_h - 1)

    # 如果背景上还没有其他物体，直接随机返回一个位置
    if not existing_bboxes:
        return random.randint(0, max_x), random.randint(0, max_y)

    # 尝试有限次数来寻找一个好位置
    for _ in range(Config.max_placement_attempts):
        # 随机生成一个候选位置
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        new_bbox = [x, y, obj_w, obj_h]

        # 检查这个新位置是否与所有已存在的物体边界框重叠过多
        is_valid = True
        for bbox in existing_bboxes:
            if calculate_overlap_ratio(new_bbox, bbox) > Config.max_overlap_ratio:
                is_valid = False
                break  # 只要有一个重叠超标，就立即放弃这个位置

        # 如果找到了一个有效位置，立即返回
        if is_valid:
            return x, y

    # 如果尝试了很多次都找不到合适的位置，则返回None
    return None


def get_tight_bounding_box_from_mask(mask):
    """从二值掩码图像中计算出紧贴物体的边界框 [x, y, w, h]。"""
    # np.nonzero找到所有非零（白色）像素的坐标
    rows, cols = np.nonzero(mask)
    # 如果掩码是全黑的，没有物体，返回None
    if len(rows) == 0 or len(cols) == 0:
        return None
    # 找到物体的上下左右边界
    y_min, y_max = np.min(rows), np.max(rows)
    x_min, x_max = np.min(cols), np.max(cols)
    # 计算宽度和高度并返回
    return [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]


def resize_object_to_fit(object_image, mask, target_shape, min_scale=0.2, max_scale=1.0):
    """根据背景尺寸，随机缩放物体图像及其掩码。"""
    target_h, target_w = target_shape[:2]
    obj_h, obj_w = object_image.shape[:2]

    # 计算物体能被缩放的最大比例，以确保它不会超出背景边界
    max_possible_scale = min(target_w / obj_w, target_h / obj_h)

    # 在允许的范围内，生成一个随机的缩放因子
    scale_factor = max_possible_scale * random.uniform(min_scale, max_scale)
    new_w, new_h = int(obj_w * scale_factor), int(obj_h * scale_factor)

    # 使用计算出的新尺寸来缩放物体图像和掩码
    resized_obj = cv2.resize(object_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    return resized_obj, resized_mask, (new_w, new_h)


def smooth_blend_object(target_img, obj_img, mask, position, alpha=1.0):
    """使用泊松融合或简单的Alpha融合将物体平滑地贴到背景上。"""
    x, y = position
    obj_h, obj_w = obj_img.shape[:2]

    # 从背景图中提取出要贴物体的区域
    target_region = target_img[y:y + obj_h, x:x + obj_w]

    # 使用掩码进行融合：掩码为白色的区域显示物体，黑色的区域显示原背景
    # alpha参数可以控制物体的透明度
    blended_region = np.where(mask[..., None],  # where条件，mask需要扩展一个维度以匹配彩色图像
                              alpha * obj_img + (1 - alpha) * target_region,  # True时执行
                              target_region)  # False时执行

    # 将融合好的区域放回原图
    target_img[y:y + obj_h, x:x + obj_w] = blended_region.astype(np.uint8)
    return target_img


def load_object_data(object_root_dir):
    """从磁盘加载所有物体的图像和掩码路径，并按类别组织。"""
    object_data = {}  # 用于存储最终数据的字典，格式：{'类别名': [(图像路径, 掩码路径), ...]}

    # 获取所有类别文件夹的名称
    class_dirs = [d for d in os.listdir(object_root_dir)
                  if os.path.isdir(os.path.join(object_root_dir, d))]

    # 根据文件夹名自动生成类别字典
    Config.class_dict = {name: name for name in class_dirs}

    # 遍历每个类别文件夹
    for class_name in class_dirs:
        class_dir = os.path.join(object_root_dir, class_name)
        images_dir = os.path.join(class_dir, "image")
        masks_dir = os.path.join(class_dir, "mask")

        # 确保image和mask子文件夹都存在
        if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
            print(f"警告: 类别 {class_name} 的 image 或 mask 文件夹不存在，已跳过。")
            continue

        # 获取该类别下所有以 "_obj_" 结尾的物体图像文件
        obj_files = [f for f in os.listdir(images_dir)
                     if f.endswith(('.png', '.jpg', '.jpeg')) and "_obj_" in f]

        object_data[class_name] = []
        # 为每个物体图像找到其对应的掩码图像
        for obj_file in obj_files:
            # 命名规则：掩码文件名是将物体文件名中的 "_obj_" 替换为 "_mask_"
            mask_file = obj_file.replace("_obj_", "_mask_")
            obj_path = os.path.join(images_dir, obj_file)
            mask_path = os.path.join(masks_dir, mask_file)

            # 如果对应的掩码文件存在，则将这对路径存入字典
            if os.path.exists(mask_path):
                object_data[class_name].append((obj_path, mask_path))
            else:
                print(f"警告: 未找到 {obj_path} 对应的掩码文件: {mask_path}")

    return object_data


# ==============================================================================
# 3. 主逻辑函数
# ==============================================================================

def generate_synthetic_images():
    """主函数：生成所有合成图像和对应的COCO格式标注。"""
    # 步骤1: 加载所有背景图片和物体数据
    background_files = [f for f in os.listdir(Config.background_dir)
                        if f.endswith(('.png', '.jpg', '.jpeg'))]
    object_data = load_object_data(Config.object_root_dir)
    available_classes = list(object_data.keys())

    # 步骤2: 初始化COCO标注数据结构
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i + 1, "name": name, "supercategory": "object"}
                       for i, name in enumerate(available_classes)]
    }
    category_id_map = {name: i + 1 for i, name in enumerate(available_classes)}
    annotation_id = 1  # COCO标注的ID必须是唯一的，从1开始

    # 步骤3: 遍历每张背景图片，生成合成图
    for img_idx, bg_file in enumerate(tqdm(background_files, desc="生成合成图像")):
        bg_path = os.path.join(Config.background_dir, bg_file)
        bg_img = cv2.imread(bg_path)
        if bg_img is None: continue  # 如果图片读取失败，则跳过

        synthetic_img = bg_img.copy()  # 复制背景图，用于在其上进行合成
        bg_h, bg_w = bg_img.shape[:2]

        # 步骤3.1: 随机决定在这张背景上贴几个类别的物体
        num_classes_to_place = random.randint(Config.min_objects_per_image,
                                              min(Config.max_objects_per_image, len(available_classes)))
        selected_classes = random.sample(available_classes, num_classes_to_place)

        placed_bboxes = []  # 存储这张图上已经放置的物体的边界框
        annotations_for_this_image = []  # 存储这张图上所有的标注信息

        # 步骤3.2: 遍历选中的每个类别
        for class_name in selected_classes:
            # 随机决定贴几个这个类别的实例
            num_instances = random.randint(Config.min_instances_per_class, Config.max_instances_per_class)
            available_instances = object_data[class_name]
            num_instances = min(num_instances, len(available_instances))  # 确保不超过可用实例数
            selected_instances = random.sample(available_instances, num_instances)

            # 步骤3.3: 遍历选中的每个物体实例
            for obj_path, mask_path in selected_instances:
                obj_img = cv2.imread(obj_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if obj_img is None or mask is None: continue

                # 随机缩放物体并找到一个不重叠的位置
                resized_obj, resized_mask, obj_size = resize_object_to_fit(obj_img, mask, bg_img.shape)
                position = find_valid_position(bg_img.shape, obj_size, placed_bboxes)
                if position is None: continue  # 找不到好位置，放弃这个物体

                # 将物体贴到背景上
                synthetic_img = smooth_blend_object(synthetic_img, resized_obj, resized_mask, position)

                # 计算并记录标注信息
                bbox = get_tight_bounding_box_from_mask(resized_mask)
                if bbox is None: continue
                bbox[0] += position[0];
                bbox[1] += position[1]  # 调整为在背景图上的绝对坐标

                # 更新已放置物体的边界框列表
                placed_bboxes.append(bbox.copy())

                # 创建一条COCO格式的标注
                annotations_for_this_image.append({
                    "id": annotation_id,
                    "image_id": img_idx + 1,
                    "category_id": category_id_map[class_name],
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0
                })
                annotation_id += 1  # 确保下一个标注ID是唯一的

        # 步骤4: 保存最终合成的图片
        output_img_path = os.path.join(Config.blended_dir, bg_file)
        cv2.imwrite(output_img_path, synthetic_img)

        # 步骤5: 将这张图片的信息和它上面的所有标注添加到总的COCO数据中
        coco_data["images"].append({
            "id": img_idx + 1,
            "file_name": bg_file,
            "width": bg_w,
            "height": bg_h
        })
        coco_data["annotations"].extend(annotations_for_this_image)

    return coco_data


def split_and_save_dataset(coco_data):
    """将生成的COCO数据集划分为训练集和测试集，并分别保存。"""
    # 使用sklearn的工具，根据图像列表进行划分
    train_images, test_images = train_test_split(
        coco_data["images"], test_size=Config.test_size, random_state=Config.random_seed)

    # 获取训练集和测试集各自包含的图像ID
    train_image_ids = {img["id"] for img in train_images}
    test_image_ids = {img["id"] for img in test_images}

    # 根据图像ID筛选出对应的标注
    train_annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] in train_image_ids]
    test_annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] in test_image_ids]

    # --- 保存文件 ---
    # 保存包含所有数据的完整标注文件
    full_output_path = os.path.join(Config.annotations_dir, "instances_all.json")
    with open(full_output_path, "w", encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)

    # 创建并保存训练集标注文件
    train_data = {"images": train_images, "annotations": train_annotations, "categories": coco_data["categories"]}
    train_output_path = os.path.join(Config.annotations_dir, "instances_train.json")
    with open(train_output_path, "w", encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    # 创建并保存测试集标注文件
    test_data = {"images": test_images, "annotations": test_annotations, "categories": coco_data["categories"]}
    test_output_path = os.path.join(Config.annotations_dir, "instances_test.json")
    with open(test_output_path, "w", encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    # 打印最终统计信息
    print(f"数据集已成功保存到: {Config.output_root}")
    print(f"训练集: {len(train_images)} 张图像, {len(train_annotations)} 个标注")
    print(f"测试集: {len(test_images)} 张图像, {len(test_annotations)} 个标注")


def visualize_annotations():
    """读取已生成的标注文件，并将边界框和类别名绘制到合成图像上，用于检查。"""
    annotation_path = os.path.join(Config.annotations_dir, "instances_all.json")
    with open(annotation_path, "r", encoding='utf-8') as f:
        coco_data = json.load(f)

    # 创建一个从类别ID到类别名的映射，方便后续查找
    id_to_name_map = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

    # 遍历每张图像
    for img_info in tqdm(coco_data["images"], desc="可视化标注"):
        img_path = os.path.join(Config.blended_dir, img_info["file_name"])
        img = cv2.imread(img_path)
        if img is None: continue

        # 找到属于这张图像的所有标注
        annotations_for_img = [ann for ann in coco_data["annotations"] if ann["image_id"] == img_info["id"]]

        # 在图像上绘制每个标注框和类别名
        for ann in annotations_for_img:
            bbox = list(map(int, ann["bbox"]))
            class_name = id_to_name_map.get(ann["category_id"], "Unknown")  # 使用.get避免未知ID报错

            # 绘制绿色矩形框
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
            # 在框的上方写上类别名
            cv2.putText(img, class_name, (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 保存这张绘制了标注的图像
        visualization_path = os.path.join(Config.visualization_dir, img_info["file_name"])
        cv2.imwrite(visualization_path, img)


# ==============================================================================
# 4. 脚本执行入口
# ==============================================================================
if __name__ == "__main__":
    # 步骤一: 生成合成数据和完整的COCO标注
    print("--- 步骤 1/3: 开始生成合成数据集... ---")
    coco_data = generate_synthetic_images()

    # 步骤二: 将数据集划分为训练集和测试集并保存
    print("\n--- 步骤 2/3: 开始划分并保存数据集... ---")
    split_and_save_dataset(coco_data)

    # 步骤三: 生成可视化结果以供检查
    print("\n--- 步骤 3/3: 开始生成可视化结果... ---")
    visualize_annotations()

    print("\n所有任务已成功完成!")