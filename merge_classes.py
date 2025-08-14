import json
from typing import Dict, List


def merge_coco_categories(input_path: str, output_path: str) -> None:
    """
    合并COCO标注文件中的类别并保存为新的COCO格式文件

    Args:
        input_path: 输入COCO文件路径
        output_path: 输出COCO文件路径
    """
    # 定义类别映射关系
    category_mapping = {
        # 新类别ID: [原类别名称列表]
        5: ['anemone fish', 'coho', 'gar', 'rock beauty', 'barracouta',
            'goldfish', 'lionfish', 'sturgeon', 'tench', 'great white shark',
            'hammerhead', 'tiger shark', 'puffer', 'electric ray'],
        9: ['leatherback turtle', 'box turtle', 'mud turtle', 'loggerhead', 'terrapin'],
        4: ['starfish'],
        10: ['jellyfish'],
        6: ['coral reef', 'brain coral']
    }

    # 新类别定义 (完全符合COCO格式)
    new_categories = [
        {"id": 1, "name": "holothurian", "supercategory": "marine life"},
        {"id": 2, "name": "echinus", "supercategory": "marine life"},
        {"id": 3, "name": "scallop", "supercategory": "marine life"},
        {"id": 4, "name": "starfish", "supercategory": "marine life"},
        {"id": 5, "name": "fish", "supercategory": "marine life"},
        {"id": 6, "name": "corals", "supercategory": "marine life"},
        {"id": 7, "name": "diver", "supercategory": "human"},
        {"id": 8, "name": "cuttlefish", "supercategory": "marine life"},
        {"id": 9, "name": "turtle", "supercategory": "marine life"},
        {"id": 10, "name": "jellyfish", "supercategory": "marine life"}
    ]

    # 创建原始类别名到新类别ID的映射
    reverse_mapping = {}
    for new_id, old_names in category_mapping.items():
        for name in old_names:
            reverse_mapping[name] = new_id

    # 读取原始COCO文件
    with open(input_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # 验证输入文件是否包含必要字段
    required_fields = ['images', 'annotations', 'categories']
    for field in required_fields:
        if field not in coco_data:
            raise ValueError(f"Input file missing required field: {field}")

    # 创建原始类别ID到新类别ID的映射
    old_id_to_new_id = {}
    for old_cat in coco_data['categories']:
        if old_cat['name'] in reverse_mapping:
            old_id_to_new_id[old_cat['id']] = reverse_mapping[old_cat['name']]

    # 处理annotations
    new_annotations = []
    for ann in coco_data['annotations']:
        old_cat_id = ann['category_id']
        if old_cat_id in old_id_to_new_id:
            # 创建新的annotation对象，避免修改原始数据
            new_ann = ann.copy()
            new_ann['category_id'] = old_id_to_new_id[old_cat_id]
            new_annotations.append(new_ann)
        # 其他情况跳过(不包含在new_annotations中)

    # 构建新的COCO数据
    new_coco_data = {
        "info": coco_data.get("info", {}),
        "licenses": coco_data.get("licenses", []),
        "images": coco_data["images"],
        "annotations": new_annotations,
        "categories": new_categories
    }

    # 添加缺失的标准字段(如果原文件没有)
    if "info" not in new_coco_data:
        new_coco_data["info"] = {
            "description": "Modified COCO dataset",
            "version": "1.0",
            "year": 2023,
            "contributor": "",
            "date_created": ""
        }

    # 保存新的COCO文件(保持缩进格式，便于阅读)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_coco_data, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    input_path1 = '/media/HDD0/XCX/synthetic_dataset/annotations/split_results/part2_train.json'
    output_path1 = '/media/HDD0/XCX/synthetic_dataset/annotations/split_results/part2_train_merge.json'
    input_path2 = '/media/HDD0/XCX/synthetic_dataset/annotations/split_results/part2_val.json'
    output_path2 = '/media/HDD0/XCX/synthetic_dataset/annotations/split_results/part2_val_merge.json'
    merge_coco_categories(input_path1, output_path1)
    merge_coco_categories(input_path2, output_path2)