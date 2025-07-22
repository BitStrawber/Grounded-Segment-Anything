import os
import torch
from PIL import Image
import numpy as np
import cv2
import json
from tqdm import tqdm
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 设置环境路径
import sys

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

# 导入模型相关组件
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from segment_anything import sam_model_registry, SamPredictor


class CategorizedSegmenter:
    def __init__(self, rank=0, world_size=1):
        """初始化分割器，支持多GPU"""
        # ============= 配置区域 =============
        self.rank = rank
        self.world_size = world_size
        self.input_root = "/media/HDD0/XCX/classes/images"
        self.output_root = "/media/HDD0/XCX/classes/masks"

        # 模型配置
        self.grounding_config = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.grounding_checkpoint = "groundingdino_swint_ogc.pth"
        self.sam_version = "vit_h"
        self.sam_checkpoint = "sam_vit_h_4b8939.pth"
        self.bert_base_uncased_path = "bert-base-uncased"

        # 处理参数
        self.box_threshold = 0.3
        self.text_threshold = 0.25
        self.device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        self.use_sam_hq = False
        self.save_visualization = True
        self.probability = 0.5
        # ============= 配置结束 =============

        # 初始化设备
        self.device = torch.device(self.device)
        torch.cuda.set_device(self.device)

        # 初始化模型
        self._init_models()

        # 确保输出目录存在
        os.makedirs(self.output_root, exist_ok=True)

    def _init_models(self):
        """初始化模型，支持DDP"""
        print(f"[Rank {self.rank}] 正在初始化模型...")

        # 初始化GroundingDINO
        args = SLConfig.fromfile(self.grounding_config)
        args.device = self.device.type
        args.bert_base_uncased_path = self.bert_base_uncased_path

        self.grounding_model = build_model(args).to(self.device)
        checkpoint = torch.load(self.grounding_checkpoint, map_location=self.device)
        self.grounding_model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)

        # 使用DDP包装模型
        if self.world_size > 1:
            self.grounding_model = DDP(self.grounding_model, device_ids=[self.rank])

        self.grounding_model.eval()

        # 初始化SAM (SAM不支持DDP，每个进程独立实例)
        self.sam = sam_model_registry[self.sam_version](
            checkpoint=self.sam_checkpoint
        ).to(self.device)
        self.sam_predictor = SamPredictor(self.sam)

        print(f"[Rank {self.rank}] 模型初始化完成")

    def _get_category_prompt(self, category_name):
        return f"{category_name.lower()}."

    def _load_image(self, image_path):
        """加载并预处理图像"""
        image_pil = Image.open(image_path).convert("RGB")
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image, _ = transform(image_pil, None)
        return image_pil, image.to(self.device)

    @torch.no_grad()
    def _get_grounding_output(self, image, caption):
        """从GroundingDINO获取输出"""
        caption = caption.lower().strip()
        if not caption.endswith("."):
            caption += "."

        image = image.to(self.device)

        with torch.cuda.amp.autocast():  # 混合精度
            outputs = self.grounding_model(image[None], captions=[caption])

        logits = outputs["pred_logits"].sigmoid()[0]
        boxes = outputs["pred_boxes"][0]

        # 双重过滤
        confidence_mask = logits.max(dim=1)[0] > self.probability
        filt_mask = (logits.max(dim=1)[0] > self.box_threshold) & confidence_mask
        logits_filt = logits[filt_mask]
        boxes_filt = boxes[filt_mask]

        # 获取短语
        tokenizer = self.grounding_model.tokenizer
        tokenized = tokenizer(caption)
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(
                logit > self.text_threshold, tokenized, tokenizer
            )
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")

        return boxes_filt, pred_phrases

    def _process_image(self, image_path, category_name):
        """处理单张图像"""
        try:
            # 1. 加载图像
            text_prompt = self._get_category_prompt(category_name)
            image_pil, image_tensor = self._load_image(image_path)
            image_cv = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

            # 2. GroundingDINO检测
            boxes_filt, pred_phrases = self._get_grounding_output(image_tensor, text_prompt)
            if boxes_filt.size(0) == 0:
                return None

            # 3. SAM分割
            self.sam_predictor.set_image(image_cv)
            H, W = image_pil.size[1], image_pil.size[0]

            # 转换边界框
            size = torch.tensor([W, H, W, H], device=self.device)
            boxes_filt = boxes_filt * size
            boxes_filt[:, :2] -= boxes_filt[:, 2:] / 2
            boxes_filt[:, 2:] += boxes_filt[:, :2]
            boxes_filt = boxes_filt.to(self.device)

            transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
                boxes_filt, image_cv.shape[:2]
            ).to(self.device)

            # 获取掩码
            masks, _, _ = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False
            )

            return {
                'image_pil': image_pil,
                'image_cv': image_cv,
                'boxes': boxes_filt.cpu(),
                'phrases': pred_phrases,
                'masks': masks.cpu(),
                'text_prompt': text_prompt
            }
        finally:
            torch.cuda.empty_cache()

    def _save_results(self, results, src_path, category_name):
        """保存处理结果（仅由rank 0执行）"""
        if self.rank != 0:
            return

        # ... (保持原有_save_results实现不变) ...

    def process_category(self, category):
        """处理单个类别"""
        category_dir = os.path.join(self.input_root, category)
        image_files = [f for f in os.listdir(category_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        # 分配当前GPU应该处理的图像
        chunk_size = len(image_files) // self.world_size
        start = self.rank * chunk_size
        end = (self.rank + 1) * chunk_size if self.rank != self.world_size - 1 else len(image_files)
        my_files = image_files[start:end]

        for img_file in tqdm(my_files, desc=f"[Rank {self.rank}] 处理 {category}"):
            img_path = os.path.join(category_dir, img_file)
            try:
                results = self._process_image(img_path, category)
                if results is not None:
                    self._save_results(results, img_path, category)
            except Exception as e:
                print(f"[Rank {self.rank}] 处理 {img_path} 时出错: {str(e)}")


def worker(rank, world_size, categories):
    """工作进程函数"""
    # 初始化分布式环境
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # 创建分割器实例
    segmenter = CategorizedSegmenter(rank, world_size)

    # 处理分配的类别
    for category in categories:
        segmenter.process_category(category)

    # 清理
    dist.destroy_process_group()


if __name__ == "__main__":
    # 获取所有类别
    input_root = "/media/HDD0/XCX/classes/images"
    categories = [d for d in os.listdir(input_root)
                  if os.path.isdir(os.path.join(input_root, d))]

    # 配置GPU数量
    world_size = torch.cuda.device_count()
    print(f"发现 {world_size} 个GPU，开始分布式处理...")

    # 均衡分配类别到GPU
    chunk_size = len(categories) // world_size
    tasks = []
    for i in range(world_size):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i != world_size - 1 else len(categories)
        tasks.append(categories[start:end])

    # 启动进程
    mp.set_start_method('spawn', force=True)
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=worker, args=(rank, world_size, tasks[rank]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()