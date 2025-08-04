# -*- coding: utf-8 -*-

import os
import sys
import argparse
import json
import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm
import torch.multiprocessing as mp
import torch.distributed as dist

# ==============================================================================
# 动态路径设置 (最可靠的方法)
# ==============================================================================
try:
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    if project_root not in sys.path:
        sys.path.append(project_root)
    print(f"项目根目录已添加至 sys.path: {project_root}")
except NameError:
    # 在交互式环境中运行（如Jupyter Notebook）
    script_dir = os.getcwd()
    if script_dir not in sys.path:
        sys.path.append(script_dir)
    print(f"在交互式环境中运行，当前工作目录已添加至 sys.path: {script_dir}")

# ==============================================================================
# 导入模型相关组件
# ==============================================================================
try:
    import GroundingDINO.groundingdino.datasets.transforms as T
    from GroundingDINO.groundingdino.models import build_model
    from GroundingDINO.groundingdino.util.slconfig import SLConfig
    from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
    from segment_anything import sam_model_registry, SamPredictor

    print("成功导入 GroundingDINO 和 segment_anything 模块。")
except ImportError as e:
    print("\n" + "=" * 80)
    print("错误：无法导入模型库。请检查 'GroundingDINO' 和 'segment_anything' 是否在项目根目录中。")
    print(f"详细导入错误: {e}")
    sys.exit(1)


class CategorizedSegmenter:
    """
    使用 GroundingDINO 和 SAM 的分布式类别分割器。
    集成了双重过滤机制和精细化结果保存，以产出高质量的分割结果。
    """

    def __init__(self, rank, world_size, args):
        self.rank = rank
        self.world_size = world_size
        self.args = args
        self.device = f"cuda:{rank}"
        torch.cuda.set_device(self.device)
        self._init_models()
        os.makedirs(self.args.output_root, exist_ok=True)

    def _init_models(self):
        """为当前进程初始化模型。"""
        print(f"[Rank {self.rank}] 正在初始化模型...")
        gd_args = SLConfig.fromfile(self.args.grounding_config)
        gd_args.device = self.device
        if hasattr(gd_args, 'bert_base_uncased_path'):
            gd_args.bert_base_uncased_path = self.args.bert_base_uncased_path
        self.grounding_model = build_model(gd_args).to(self.device)
        checkpoint = torch.load(self.args.grounding_checkpoint, map_location=self.device)
        self.grounding_model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        self.grounding_model.eval()
        self.sam = sam_model_registry[self.args.sam_version](checkpoint=self.args.sam_checkpoint).to(self.device)
        self.sam_predictor = SamPredictor(self.sam)
        print(f"[Rank {self.rank}] 模型初始化完成。")

    def _get_category_prompt(self, category_name):
        """
        生成标准的文本提示，将下划线替换为空格以适应自然语言模型。
        这是提升检测精度的关键步骤。
        """
        return f"{category_name.replace('_', ' ').lower()}."

    def _load_image(self, image_path):
        """加载并预处理图像以适应GroundingDINO。"""
        image_pil = Image.open(image_path).convert("RGB")
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_tensor, _ = transform(image_pil, None)
        return image_pil, image_tensor.to(self.device)

    @torch.no_grad()
    def _get_grounding_output(self, image_tensor, caption):
        """
        从GroundingDINO获取边界框和短语，使用双重过滤机制以提高精度。
        """
        caption = caption.lower().strip()
        if not caption.endswith("."):
            caption += "."

        with torch.cuda.amp.autocast(enabled=True):
            outputs = self.grounding_model(image_tensor[None], captions=[caption])

        logits = outputs["pred_logits"].sigmoid()[0]
        boxes = outputs["pred_boxes"][0]

        # --- 关键优化：采用双重过滤机制 ---
        # 1. 关联度过滤：检测框与文本的关联度要高于 box_threshold
        association_mask = logits.max(dim=1)[0] > self.args.box_threshold
        # 2. 置信度过滤：检测框本身的绝对置信度要高于 confidence_threshold
        confidence_mask = logits.max(dim=1)[0] > self.args.confidence_threshold
        # 最终掩码是两者的交集
        final_mask = association_mask & confidence_mask

        logits_filt, boxes_filt = logits[final_mask], boxes[final_mask]

        tokenizer = self.grounding_model.tokenizer
        tokenized = tokenizer(caption)
        pred_phrases = []
        for logit in logits_filt:
            pred_phrase = get_phrases_from_posmap(logit > self.args.text_threshold, tokenized, tokenizer)
            pred_phrases.append(f"{pred_phrase}({logit.max().item():.2f})")

        return boxes_filt, pred_phrases

    def _process_image(self, image_path, category_name):
        """处理单张图像的完整流程：加载 -> 检测 -> 分割。"""
        try:
            text_prompt = self._get_category_prompt(category_name)
            image_pil, image_tensor = self._load_image(image_path)
            # 使用cv2加载原始图像以进行保存和可视化，避免PIL和CV2转换问题
            image_cv = cv2.imread(image_path)
            if image_cv is None:
                raise IOError(f"OpenCV无法读取图像: {image_path}")

            boxes_filt, pred_phrases = self._get_grounding_output(image_tensor, text_prompt)
            if boxes_filt.size(0) == 0:
                # 在rank 0打印信息，避免多进程重复打印
                if self.rank == 0:
                    tqdm.write(f"在 {os.path.basename(image_path)} 中未找到 '{text_prompt}' 的高质量目标。")
                return None

            image_rgb_for_sam = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            self.sam_predictor.set_image(image_rgb_for_sam)

            H, W, _ = image_cv.shape
            boxes_xyxy = boxes_filt * torch.tensor([W, H, W, H], device=self.device)
            # 从中心点+宽高格式(cx,cy,w,h)转换到(x1,y1,x2,y2)格式
            boxes_xyxy[:, :2] -= boxes_xyxy[:, 2:] / 2
            boxes_xyxy[:, 2:] += boxes_xyxy[:, :2]

            transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_cv.shape[:2]).to(
                self.device)

            masks, _, _ = self.sam_predictor.predict_torch(
                point_coords=None, point_labels=None, boxes=transformed_boxes, multimask_output=False
            )
            return {
                'image_cv': image_cv,
                'boxes': boxes_xyxy.cpu(),
                'phrases': pred_phrases,
                'masks': masks.cpu().squeeze(1),
                'text_prompt': text_prompt
            }
        finally:
            # 清理缓存，对长时间运行的任务很重要
            torch.cuda.empty_cache()

    def _save_cropped_objects(self, image_cv, masks, boxes, phrases, output_dir, base_name):
        """
        为每个检测到的对象保存抠图、掩码和精细的可视化结果。
        """
        image_out_dir = os.path.join(output_dir, "image")
        mask_out_dir = os.path.join(output_dir, "mask")
        visible_out_dir = os.path.join(output_dir, "visible")
        os.makedirs(image_out_dir, exist_ok=True)
        os.makedirs(mask_out_dir, exist_ok=True)
        os.makedirs(visible_out_dir, exist_ok=True)

        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

        for i, (mask_tensor, box_tensor, phrase) in enumerate(zip(masks, boxes, phrases)):
            mask_np = mask_tensor.numpy().astype(np.uint8) * 255
            obj_suffix = f"{base_name}_{i:02d}"

            # 1. 保存二值掩码 (mask/)
            mask_filename = f"{obj_suffix}_mask.png"
            cv2.imwrite(os.path.join(mask_out_dir, mask_filename), mask_np)

            # 2. 保存抠图后的RGBA目标 (image/)
            rgba = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2RGBA)
            rgba[:, :, 3] = mask_np
            obj_filename = f"{obj_suffix}_obj.png"
            cv2.imwrite(os.path.join(image_out_dir, obj_filename), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

            # 3. 保存精细的可视化图像 (visible/)
            vis_image = image_cv.copy()
            # 绘制半透明蒙版
            color = np.array([0, 255, 0], dtype=np.uint8)  # 绿色
            vis_image[mask_np > 0] = cv2.addWeighted(vis_image[mask_np > 0], 0.5, color, 0.5, 0)

            # 绘制边界框
            box = box_tensor.numpy().astype(int)
            cv2.rectangle(vis_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

            # 准备并绘制带背景的标签
            label_text = phrase
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_pos = (box[0], box[1] - text_h - 10)
            # 绘制背景矩形
            cv2.rectangle(vis_image, (label_pos[0], label_pos[1] + text_h + baseline),
                          (label_pos[0] + text_w, label_pos[1] - baseline), (0, 255, 0), -1)
            # 绘制文字（黑色，更清晰）
            cv2.putText(vis_image, label_text, (label_pos[0], label_pos[1] + text_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            vis_filename = f"{obj_suffix}_visible.jpg"
            cv2.imwrite(os.path.join(visible_out_dir, vis_filename), vis_image)

    def _save_results(self, results, src_path, category_name):
        """
        根据新的文件结构要求保存所有结果，包括详细的元数据。
        """
        category_output_dir = os.path.join(self.args.output_root, category_name)
        base_name = os.path.splitext(os.path.basename(src_path))[0]

        # 1. 调用辅助函数保存每个对象的抠图、掩码和可视化图
        self._save_cropped_objects(
            image_cv=results['image_cv'],
            masks=results['masks'],
            boxes=results['boxes'],
            phrases=results['phrases'],
            output_dir=category_output_dir,
            base_name=base_name
        )

        # 2. 保存该图像的综合元数据 (metadata.json)
        metadata = {
            'source_image': os.path.abspath(src_path),
            'text_prompt': results['text_prompt'],
            'detections': []
        }
        for i, (p, b) in enumerate(zip(results['phrases'], results['boxes'])):
            obj_suffix = f"{base_name}_{i:02d}"
            try:
                label = p.split('(')[0]
                confidence = float(p.split('(')[1][:-1])
            except (IndexError, ValueError):
                label = p
                confidence = -1.0  # 表示未知

            detection_info = {
                'label': label,
                'confidence': confidence,
                'box_xyxy': [round(coord, 2) for coord in b.tolist()],
                'mask_file': os.path.join("mask", f"{obj_suffix}_mask.png"),
                'object_file': os.path.join("image", f"{obj_suffix}_obj.png"),
                'visualization_file': os.path.join("visible", f"{obj_suffix}_visible.jpg"),
            }
            metadata['detections'].append(detection_info)

        # 元数据保存在该图片专属的visible文件夹中，便于管理
        meta_dir = os.path.join(category_output_dir, "visible")
        meta_filename = os.path.join(meta_dir, f"{base_name}_meta.json")
        with open(meta_filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

    def run(self, all_categories):
        """主执行循环。"""
        for category in all_categories:
            category_dir = os.path.join(self.args.input_root, category)
            if not os.path.isdir(category_dir):
                if self.rank == 0:
                    print(f"警告：找不到类别目录 {category_dir}，已跳过。")
                continue

            image_files = sorted([f for f in os.listdir(category_dir)
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))])

            if not image_files:
                if self.rank == 0:
                    print(f"警告：类别目录 {category_dir} 为空，已跳过。")
                continue

            # 根据rank在类别内部对图像列表进行分片
            files_per_proc = len(image_files) // self.world_size
            start_index = self.rank * files_per_proc
            end_index = (self.rank + 1) * files_per_proc if self.rank != self.world_size - 1 else len(image_files)
            my_files_to_process = image_files[start_index:end_index]

            if not my_files_to_process:
                continue

            if self.rank == 0:
                print(f"\n开始处理类别 '{category}'...")

            pbar = tqdm(my_files_to_process, desc=f"[Rank {self.rank}] {category}", position=self.rank, file=sys.stdout)
            for img_file in pbar:
                img_path = os.path.join(category_dir, img_file)
                try:
                    results = self._process_image(img_path, category)
                    if results is not None:
                        self._save_results(results, img_path, category)
                except Exception as e:
                    pbar.write(f"\n[Rank {self.rank}] 处理 {img_path} 时发生严重错误: {e}")
                    import traceback
                    traceback.print_exc(file=sys.stderr)

            # 等待所有进程完成当前类别的处理
            dist.barrier()
            if self.rank == 0:
                print(f"类别 '{category}' 处理完成。")


def worker(rank, world_size, args, all_categories):
    """每个分布式进程的入口函数。"""
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    try:
        segmenter = CategorizedSegmenter(rank, world_size, args)
        segmenter.run(all_categories)
    finally:
        dist.destroy_process_group()
        print(f"[Rank {rank}] 任务完成并已清理。")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))

    # --- 2. 基于项目根目录构建默认路径 ---
    # 这样做的好处是，无论你在哪里运行脚本，路径总是正确的
    default_grounding_config = os.path.join(project_root,
                                            "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    default_grounding_checkpoint = os.path.join(project_root, "groundingdino_swint_ogc.pth")
    default_sam_checkpoint = os.path.join(project_root, "sam_vit_h_4b8939.pth")
    # 对于输入输出，我们通常期望用户提供绝对路径，但也可以提供一个合理的默认值
    default_input_root = '/media/HDD0/XCX/classes/images'
    default_output_root = '/media/HDD0/XCX/classes/masks'

    parser = argparse.ArgumentParser("优化后的分布式Grounded-SAM自动目标提取脚本")

    # --- 路径和目录 ---
    parser.add_argument("--input_root", type=str, default=default_input_root, help="包含分类图像文件夹的根目录。")
    parser.add_argument("--output_root", type=str, default=default_output_root, help="保存掩码和结果的根目录。")
    parser.add_argument("--grounding_config", type=str, default=default_grounding_config, help="GroundingDINO模型配置文件路径。")
    parser.add_argument("--grounding_checkpoint", type=str, default=default_grounding_checkpoint, help="GroundingDINO模型权重路径。")
    parser.add_argument("--sam_checkpoint", type=str, default=default_sam_checkpoint, help="SAM模型权重路径。")
    parser.add_argument("--bert_base_uncased_path", type=str, default="bert-base-uncased",
                        help="BERT模型路径（如果需要本地加载）。")

    # --- 模型和处理参数 ---
    parser.add_argument("--sam_version", type=str, default="vit_h", choices=['vit_h', 'vit_l', 'vit_b'],
                        help="要使用的SAM模型版本。")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="GroundingDINO检测框与文本的关联度阈值。")
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                        help="GroundingDINO检测框的绝对置信度阈值 (关键优化点)。")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="GroundingDINO文本关联置信度阈值。")

    # --- 分布式配置 ---
    parser.add_argument("--master_addr", type=str, default="localhost", help="主节点地址。")
    parser.add_argument("--master_port", type=str, default="12355", help="主节点端口。")

    args = parser.parse_args()

    # --- 检查路径是否存在 ---
    required_paths = [args.input_root, args.grounding_config, args.grounding_checkpoint, args.sam_checkpoint]
    for path in required_paths:
        if not os.path.exists(path):
            print(f"错误：必需的路径或文件未找到: {path}")
            print("请确保所有路径参数都正确。")
            sys.exit(1)

    try:
        categories = sorted([d for d in os.listdir(args.input_root) if os.path.isdir(os.path.join(args.input_root, d))])
        if not categories:
            print(f"错误：在 '{args.input_root}' 中未找到任何类别子目录。")
            sys.exit(1)
    except FileNotFoundError:
        print(f"错误：输入目录 '{args.input_root}' 不存在。")
        sys.exit(1)

    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("错误：未检测到任何可用的GPU。此脚本需要GPU。")
        sys.exit(1)

    print(f"发现 {world_size} 个GPU。将为每个类别内的图像进行分布式处理。")
    print(f"待处理的类别 ({len(categories)}个): {categories}")
    print("-" * 50)
    print("运行参数:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("-" * 50)

    # 推荐使用 'spawn' 启动方法以避免CUDA在fork中的问题
    mp.set_start_method('spawn', force=True)
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=worker, args=(rank, world_size, args, categories))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\n所有任务处理完成！")