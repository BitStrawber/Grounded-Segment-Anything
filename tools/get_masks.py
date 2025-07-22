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

# 动态添加模型库路径
# 建议在运行脚本的根目录下创建 GroundingDINO 和 segment_anything 文件夹
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

# 导入模型相关组件
try:
    import GroundingDINO.groundingdino.datasets.transforms as T
    from GroundingDINO.groundingdino.models import build_model
    from GroundingDINO.groundingdino.util.slconfig import SLConfig
    from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
    from segment_anything import sam_model_registry, SamPredictor
except ImportError as e:
    print("错误：无法导入模型库。请确保 'GroundingDINO' 和 'segment_anything' 目录存在于当前工作目录中。")
    print(f"详细错误: {e}")
    sys.exit(1)


class CategorizedSegmenter:
    """
    使用 GroundingDINO 和 SAM 的分布式类别分割器。
    每个进程（GPU）加载独立的模型实例，并协作处理每个类别中的一部分图像。
    """

    def __init__(self, rank, world_size, args):
        """
        初始化分割器。

        Args:
            rank (int): 当前进程的排名。
            world_size (int): 总进程数。
            args (argparse.Namespace): 包含所有配置的命令行参数。
        """
        self.rank = rank
        self.world_size = world_size
        self.args = args
        self.device = f"cuda:{rank}"

        # 设置当前进程使用的GPU
        torch.cuda.set_device(self.device)

        # 初始化模型
        self._init_models()

        # 确保输出根目录存在 (由所有进程执行是安全的)
        os.makedirs(self.args.output_root, exist_ok=True)

    def _init_models(self):
        """
        为当前进程初始化模型，无需DDP，因为这是纯推理。
        """
        print(f"[Rank {self.rank}] 正在初始化模型...")

        # 初始化GroundingDINO
        gd_args = SLConfig.fromfile(self.args.grounding_config)
        gd_args.device = self.device
        # 如果模型需要 bert-base-uncased，确保路径正确
        if hasattr(gd_args, 'bert_base_uncased_path'):
            gd_args.bert_base_uncased_path = self.args.bert_base_uncased_path

        self.grounding_model = build_model(gd_args).to(self.device)
        checkpoint = torch.load(self.args.grounding_checkpoint, map_location=self.device)
        self.grounding_model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        self.grounding_model.eval()  # 切换到评估模式

        # 初始化SAM
        self.sam = sam_model_registry[self.args.sam_version](
            checkpoint=self.args.sam_checkpoint
        ).to(self.device)
        self.sam_predictor = SamPredictor(self.sam)

        print(f"[Rank {self.rank}] 模型初始化完成。")

    def _get_category_prompt(self, category_name):
        """生成标准的文本提示。"""
        # 您可以根据需要自定义更复杂的提示生成逻辑
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
        """从GroundingDINO获取边界框和短语。"""
        caption = caption.lower().strip()
        if not caption.endswith("."):
            caption += "."

        # 使用混合精度以提高性能
        with torch.cuda.amp.autocast(enabled=True):
            outputs = self.grounding_model(image_tensor[None], captions=[caption])

        logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)

        # 过滤掉低置信度的检测结果
        mask = logits.max(dim=1)[0] > self.args.box_threshold
        logits_filt = logits[mask]
        boxes_filt = boxes[mask]

        # 从位置图中提取预测短语
        tokenizer = self.grounding_model.tokenizer
        tokenized = tokenizer(caption)
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(
                logit > self.args.text_threshold, tokenized, tokenizer
            )
            # 添加置信度分数到短语
            pred_phrases.append(pred_phrase + f"({logit.max().item():.2f})")

        return boxes_filt, pred_phrases

    def _process_image(self, image_path, category_name):
        """
        处理单张图像的完整流程：加载 -> 检测 -> 分割。
        """
        try:
            # 1. 加载和准备数据
            text_prompt = self._get_category_prompt(category_name)
            image_pil, image_tensor = self._load_image(image_path)
            # 使用OpenCV加载图像以用于SAM和可视化
            image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

            # 2. GroundingDINO检测
            boxes_filt, pred_phrases = self._get_grounding_output(image_tensor, text_prompt)
            if boxes_filt.size(0) == 0:
                print(f"[Rank {self.rank}] 在 {os.path.basename(image_path)} 中未找到 '{text_prompt}' 的目标。")
                return None  # 如果没有检测到任何物体，则跳过

            # 3. SAM分割
            self.sam_predictor.set_image(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

            # 将GroundingDINO的 [cx, cy, w, h] 格式的归一化box转换为 [x1, y1, x2, y2] 格式的像素坐标box
            H, W, _ = image_cv.shape
            boxes_xyxy = boxes_filt * torch.tensor([W, H, W, H], device=self.device)
            boxes_xyxy[:, :2] -= boxes_xyxy[:, 2:] / 2
            boxes_xyxy[:, 2:] += boxes_xyxy[:, :2]

            transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
                boxes_xyxy, image_cv.shape[:2]
            ).to(self.device)

            # 使用转换后的box进行预测
            masks, _, _ = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

            return {
                'image_cv': image_cv,
                'boxes': boxes_xyxy.cpu(),
                'phrases': pred_phrases,
                'masks': masks.cpu().squeeze(1),  # (N, H, W)
                'text_prompt': text_prompt
            }
        finally:
            # 清理CUDA缓存，对长时间运行的任务很重要
            torch.cuda.empty_cache()

    def _save_results(self, results, src_path, category_name):
        """
        将处理结果保存到文件。每个进程独立执行此操作。
        """
        # 创建特定于该图像的输出目录
        base_name = os.path.splitext(os.path.basename(src_path))[0]
        output_dir = os.path.join(self.args.output_root, category_name, base_name)
        os.makedirs(output_dir, exist_ok=True)

        # 1. 保存每个目标的二进制掩码
        for i, mask_tensor in enumerate(results['masks']):
            mask_np = mask_tensor.numpy().astype(np.uint8) * 255
            phrase = results['phrases'][i].replace(' ', '_').replace('.', '')
            # 清理文件名中的非法字符
            safe_phrase = "".join([c for c in phrase if c.isalpha() or c.isdigit() or c in ('_', '-')]).rstrip()
            mask_filename = f"{i:03d}_{safe_phrase}.png"
            cv2.imwrite(os.path.join(output_dir, mask_filename), mask_np)

        # 2. 如果需要，保存可视化结果
        if self.args.save_visualization:
            image_vis = results['image_cv'].copy()
            # 绘制掩码
            for mask_tensor in results['masks']:
                mask_np_bool = mask_tensor.numpy().astype(bool)
                color = np.random.randint(0, 255, size=3, dtype=np.uint8)
                image_vis[mask_np_bool] = image_vis[mask_np_bool] * 0.5 + color * 0.5
            # 绘制边界框和标签
            for box, phrase in zip(results['boxes'], results['phrases']):
                box = box.numpy().astype(int)
                cv2.rectangle(image_vis, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(image_vis, phrase, (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            vis_filename = os.path.join(output_dir, "visualization.jpg")
            cv2.imwrite(vis_filename, image_vis)

        # 3. 保存元数据
        metadata = {
            'source_image': src_path,
            'text_prompt': results['text_prompt'],
            'detections': [
                {'phrase': p, 'box': b.tolist()} for p, b in zip(results['phrases'], results['boxes'])
            ]
        }
        meta_filename = os.path.join(output_dir, "metadata.json")
        with open(meta_filename, 'w') as f:
            json.dump(metadata, f, indent=4)

    def run(self, all_categories):
        """
        主执行循环，遍历所有被分配的类别。
        """
        # 每个进程都处理所有类别
        for category in all_categories:
            category_dir = os.path.join(self.args.input_root, category)
            if not os.path.isdir(category_dir):
                if self.rank == 0:
                    print(f"警告：找不到类别目录 {category_dir}，已跳过。")
                continue

            image_files = [f for f in os.listdir(category_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

            # 关键：根据rank在类别内部对图像列表进行分片
            files_per_proc = len(image_files) // self.world_size
            start_index = self.rank * files_per_proc
            # 最后一个进程处理余下的所有文件
            end_index = (self.rank + 1) * files_per_proc if self.rank != self.world_size - 1 else len(image_files)
            my_files_to_process = image_files[start_index:end_index]

            if not my_files_to_process:
                continue

            print(f"[Rank {self.rank}] 将处理类别 '{category}' 中的 {len(my_files_to_process)} 张图像。")

            # 使用tqdm显示进度条
            for img_file in tqdm(my_files_to_process, desc=f"[Rank {self.rank}] {category}", position=self.rank):
                img_path = os.path.join(category_dir, img_file)
                try:
                    results = self._process_image(img_path, category)
                    if results is not None:
                        self._save_results(results, img_path, category)
                except Exception as e:
                    print(f"\n[Rank {self.rank}] 处理 {img_path} 时发生严重错误: {e}")
                    import traceback
                    traceback.print_exc()  # 打印详细的错误堆栈

            # 等待所有进程完成当前类别的处理
            dist.barrier()


def worker(rank, world_size, args, all_categories):
    """
    每个分布式进程的入口函数。
    """
    # 初始化分布式环境
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # 创建并运行分割器
    segmenter = CategorizedSegmenter(rank, world_size, args)
    segmenter.run(all_categories)

    # 清理
    dist.destroy_process_group()
    print(f"[Rank {self.rank}] 任务完成并已清理。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("分布式Grounded-SAM自动目标提取脚本")

    INPUT_ROOT = '/media/HDD0/XCX/classes/images'
    outPUT_ROOT = '/media/HDD0/XCX/classes/masks'

    # --- 路径和目录 ---
    parser.add_argument("--input_root", type=str, default=INPUT_ROOT, help="包含分类图像文件夹的根目录。")
    parser.add_argument("--output_root", type=str, default=OUTPUT_ROOT, help="保存掩码和结果的根目录。")
    parser.add_argument("--grounding_config", type=str,
                        default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                        help="GroundingDINO模型配置文件路径。")
    parser.add_argument("--grounding_checkpoint", type=str, default="groundingdino_swint_ogc.pth",
                        help="GroundingDINO模型权重路径。")
    parser.add_argument("--sam_checkpoint", type=str, default="sam_vit_h_4b8939.pth", help="SAM模型权重路径。")
    parser.add_argument("--bert_base_uncased_path", type=str, default="bert-base-uncased",
                        help="BERT模型路径（如果需要本地加载）。")

    # --- 模型和处理参数 ---
    parser.add_argument("--sam_version", type=str, default="vit_h", choices=['vit_h', 'vit_l', 'vit_b'],
                        help="要使用的SAM模型版本。")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="GroundingDINO检测框置信度阈值。")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="GroundingDINO文本关联置信度阈值。")
    parser.add_argument("--save_visualization", action='store_true', help="如果设置，将保存带有掩码和框的可视化图像。")

    # --- 分布式配置 ---
    parser.add_argument("--master_addr", type=str, default="localhost", help="主节点地址。")
    parser.add_argument("--master_port", type=str, default="12355", help="主节点端口。")

    args = parser.parse_args()

    # 获取所有类别目录
    try:
        categories = [d for d in os.listdir(args.input_root) if os.path.isdir(os.path.join(args.input_root, d))]
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
    print(f"待处理的类别: {categories}")

    # 启动多进程
    # 'spawn' 是在CUDA上使用多处理的推荐和最安全的方式
    mp.set_start_method('spawn', force=True)
    processes = []
    for rank in range(world_size):
        # 注意：每个worker进程都接收完整的类别列表
        p = mp.Process(target=worker, args=(rank, world_size, args, categories))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\n所有任务处理完成！")