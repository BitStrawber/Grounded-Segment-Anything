import json
from typing import Dict, List
import os

# ==============================================================================
# 1. 权威的类别定义：基于ImageNet 2012官方标准的完整映射
# ==============================================================================

# 描述10个最终大类与官方31个子类之间的关系
OFFICIAL_CATEGORY_MAPPING: Dict[str, List[str]] = {
    'fish': [
        'anemone_fish', 'barracouta', 'coho', 'eel', 'electric_ray', 'gar',
        'goldfish', 'great_white_shark', 'hammerhead', 'lionfish', 'puffer',
        'rock_beauty', 'stingray', 'sturgeon', 'tench', 'tiger_shark'
    ],
    'turtle': [
        'box_turtle', 'leatherback_turtle', 'loggerhead', 'mud_turtle', 'terrapin'
    ],
    'corals': ['corals'],
    'holothurian': ['sea_cucumber'],
    'echinus': ['sea_urchin'],
    'starfish': ['starfish'],
    'jellyfish': ['jellyfish'],
    'diver': ['scuba_diver'],
    'scallop': ['scallop'],
    'cuttlefish': ['cuttlefish'],
}

# 定义输出的COCO文件中最终的10个类别
NEW_COCO_CATEGORIES: List[Dict] = [
    {"id": 1, "name": "holothurian", "supercategory": "marine_life"},
    {"id": 2, "name": "echinus", "supercategory": "marine_life"},
    {"id": 3, "name": "scallop", "supercategory": "marine_life"},
    {"id": 4, "name": "starfish", "supercategory": "marine_life"},
    {"id": 5, "name": "fish", "supercategory": "marine_life"},
    {"id": 6, "name": "corals", "supercategory": "marine_life"},
    {"id": 7, "name": "diver", "supercategory": "human"},
    {"id": 8, "name": "cuttlefish", "supercategory": "marine_life"},
    {"id": 9, "name": "turtle", "supercategory": "marine_life"},
    {"id": 10, "name": "jellyfish", "supercategory": "marine_life"}
]


# ==============================================================================
# 2. 核心功能函数
# ==============================================================================

def merge_coco_categories(input_path: str, output_path: str) -> None:
    """
    根据预定义的官方映射关系，合并COCO标注文件中的类别。

    Args:
        input_path (str): 输入的原始COCO anootation文件路径。
        output_path (str): 合并后要保存的新COCO anootation文件路径。
    """
    print(f"--- 开始处理文件: {os.path.basename(input_path)} ---")

    # 1. 构建从 “旧类别名” 到 “新类别ID” 的查找字典
    #    例如: {'great_white_shark': 5, 'box_turtle': 9, ...}
    old_name_to_new_id_map: Dict[str, int] = {}
    for new_category in NEW_COCO_CATEGORIES:
        new_id = new_category['id']
        new_name = new_category['name']
        if new_name in OFFICIAL_CATEGORY_MAPPING:
            for old_name in OFFICIAL_CATEGORY_MAPPING[new_name]:
                old_name_to_new_id_map[old_name] = new_id

    # 2. 读取原始COCO文件
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 错误: 输入文件未找到 -> {input_path}")
        return
    except json.JSONDecodeError:
        print(f"❌ 错误: 文件不是有效的JSON格式 -> {input_path}")
        return

    # 3. 构建从 “旧类别ID” 到 “新类别ID” 的映射
    #    例如: {34: 5, 72: 9, ...}
    old_id_to_new_id_map: Dict[int, int] = {}
    original_categories_found = 0
    for old_cat in coco_data.get('categories', []):
        # 标准化名称（将空格替换为下划线）以进行稳健匹配
        normalized_name = old_cat['name'].replace(' ', '_')
        if normalized_name in old_name_to_new_id_map:
            new_id = old_name_to_new_id_map[normalized_name]
            old_id_to_new_id_map[old_cat['id']] = new_id
            original_categories_found += 1

    if not old_id_to_new_id_map:
        print("⚠️ 警告: 在输入文件中没有找到任何需要合并的类别。输出文件将为空标注。")
    else:
        print(f"✅ 在原文件中找到 {original_categories_found} 个可映射的类别。")

    # 4. 处理标注（annotations），更新category_id
    new_annotations = []
    for ann in coco_data.get('annotations', []):
        old_cat_id = ann['category_id']
        if old_cat_id in old_id_to_new_id_map:
            # 创建标注的副本进行修改，而不是直接修改原始数据
            new_ann = ann.copy()
            new_ann['category_id'] = old_id_to_new_id_map[old_cat_id]
            new_annotations.append(new_ann)

    print(f"📊 处理了 {len(coco_data.get('annotations', []))} 条原始标注，")
    print(f"   保留并转换了 {len(new_annotations)} 条标注。")

    # 5. 构建新的COCO数据结构
    new_coco_data = {
        "info": coco_data.get("info", {"description": "Merged COCO dataset"}),
        "licenses": coco_data.get("licenses", []),
        "images": coco_data.get("images", []),
        "annotations": new_annotations,
        "categories": NEW_COCO_CATEGORIES
    }

    # 6. 保存新的COCO文件
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_coco_data, f, indent=4, ensure_ascii=False)

    print(f"✅ 成功！合并后的文件已保存到: {output_path}\n")


# ==============================================================================
# 3. 执行脚本
# ==============================================================================

if __name__ == '__main__':
    # ----- 在这里配置您的文件路径 -----

    # 示例1: 处理训练集
    input_train_path = '/media/HDD0/XCX/fusions/annotations/annotations/split_results/part2_train.json'
    output_train_path = '/media/HDD0/XCX/fusions/annotations/annotations/split_results/part2_train_merged.json'
    merge_coco_categories(input_train_path, output_train_path)

    # 示例2: 处理验证集
    input_val_path = '/media/HDD0/XCX/fusions/annotations/annotations/split_results/part2_test.json'
    output_val_path = '/media/HDD0/XCX/fusions/annotations/annotations/split_results/part2_test_merged.json'
    merge_coco_categories(input_val_path, output_val_path)

    # 如果还有测试集，可以按相同方式添加
    # input_test_path = 'path/to/your/test.json'
    # output_test_path = 'path/to/your/test_merged.json'
    # merge_coco_categories(input_test_path, output_test_path)