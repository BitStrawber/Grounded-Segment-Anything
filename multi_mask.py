import os
import argparse
import torch
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
from tqdm import tqdm  # 用于显示进度条

# 设置环境路径（根据实际需要调整）
import sys

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

# 导入模型相关组件
from GroundingDINO.groundingdino.datasets.transforms import T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from segment_anything import sam_model_registry, SamPredictor


class BatchImageSegmenter:
    def __init__(self, config):
        """
        初始化分割器
        :param config: 包含所有配置参数的字典
        """
        self.config = config
        self.device = torch.device(config['device'])

        # 初始化GroundingDINO
        self.grounding_model = self._load_grounding_model(
            config['grounding_config'],
            config['grounding_checkpoint'],
            config['bert_base_uncased_path']
        )

        # 初始化SAM
        self.sam_predictor = self._load_sam_predictor(
            config['sam_version'],
            config['sam_checkpoint'],
            config['use_sam_hq']
        )

        # 创建输出目录
        os.makedirs(config['output_dir'], exist_ok=True)

    def _load_grounding_model(self, config_path, checkpoint_path, bert_path):
        """加载GroundingDINO模型"""
        args = SLConfig.fromfile(config_path)
        args.device = self.device.type
        args.bert_base_uncased_path = bert_path
        model = build_model(args)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        return model

    def _load_sam_predictor(self, version, checkpoint_path, use_hq):
        """加载SAM预测器"""
        if use_hq:
            sam = sam_model_registry[version](checkpoint=checkpoint_path)
        else:
            sam = sam_model_registry[version](checkpoint=checkpoint_path)
        sam.to(device=self.device)
        return SamPredictor(sam)

    def _process_single_image(self, image_path):
        """
        处理单张图像
        :return: 包含处理结果的字典
        """
        # 加载图像
        image_pil, image_tensor = self.load_image(image_path)
        image_cv = cv2.imread(image_path)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

        # 使用GroundingDINO获取边界框
        boxes_filt, pred_phrases = self.get_grounding_output(
            image_tensor,
            self.config['text_prompt'],
            self.config['box_threshold'],
            self.config['text_threshold']
        )

        if boxes_filt.size(0) == 0:
            return None  # 没有检测到目标

        # 准备SAM输入
        self.sam_predictor.set_image(image_cv)
        H, W = image_pil.size[1], image_pil.size[0]

        # 转换边界框格式
        boxes_filt = boxes_filt * torch.Tensor([W, H, W, H])
        boxes_filt[:, :2] -= boxes_filt[:, 2:] / 2
        boxes_filt[:, 2:] += boxes_filt[:, :2]
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            boxes_filt, image_cv.shape[:2]).to(self.device)

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
            'boxes': boxes_filt,
            'phrases': pred_phrases,
            'masks': masks
        }

    def load_image(self, image_path):
        """加载并预处理图像"""
        image_pil = Image.open(image_path).convert("RGB")
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image, _ = transform(image_pil, None)
        return image_pil, image

    def get_grounding_output(self, image, caption, box_threshold, text_threshold):
        """从GroundingDINO获取输出"""
        caption = caption.lower().strip()
        if not caption.endswith("."):
            caption += "."

        with torch.no_grad():
            outputs = self.grounding_model(image[None].to(self.device), captions=[caption])

        logits = outputs["pred_logits"].cpu().sigmoid()[0]
        boxes = outputs["pred_boxes"].cpu()[0]

        # 过滤输出
        filt_mask = logits.max(dim=1)[0] > box_threshold
        logits_filt = logits[filt_mask]
        boxes_filt = boxes[filt_mask]

        # 获取短语
        tokenizer = self.grounding_model.tokenizer
        tokenized = tokenizer(caption)
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")

        return boxes_filt, pred_phrases

    def save_results(self, results, image_path):
        """保存处理结果"""
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_subdir = os.path.join(self.config['output_dir'], base_name)
        os.makedirs(output_subdir, exist_ok=True)

        # 保存原始图像
        results['image_pil'].save(os.path.join(output_subdir, "original.jpg"))

        # 保存可视化结果
        plt.figure(figsize=(10, 10))
        plt.imshow(results['image_cv'])
        for mask in results['masks']:
            self._show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box, label in zip(results['boxes'], results['phrases']):
            self._show_box(box.numpy(), plt.gca(), label)
        plt.axis('off')
        plt.savefig(
            os.path.join(output_subdir, "segmentation.jpg"),
            bbox_inches="tight", dpi=300, pad_inches=0.0
        )
        plt.close()

        # 保存掩码数据
        self._save_mask_data(
            output_subdir,
            results['masks'],
            results['boxes'],
            results['phrases']
        )

        # 保存纯掩码图像
        mask_img = torch.zeros(results['masks'].shape[-2:])
        for idx, mask in enumerate(results['masks']):
            mask_img[mask.cpu().numpy()[0] == True] = idx + 1
        plt.figure(figsize=(10, 10))
        plt.imshow(mask_img.numpy())
        plt.axis('off')
        plt.savefig(
            os.path.join(output_subdir, "mask.jpg"),
            bbox_inches="tight", dpi=300, pad_inches=0.0
        )
        plt.close()

        # 保存元数据
        metadata = {
            'image_path': image_path,
            'text_prompt': self.config['text_prompt'],
            'detected_objects': [
                {
                    'label': label.split('(')[0],
                    'confidence': float(label.split('(')[1][:-1]),
                    'box': box.numpy().tolist()
                }
                for label, box in zip(results['phrases'], results['boxes'])
            ]
        }
        with open(os.path.join(output_subdir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)

    def _show_mask(self, mask, ax, random_color=False):
        """显示掩码"""
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def _show_box(self, box, ax, label):
        """显示边界框"""
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
        ax.text(x0, y0, label)

    def _save_mask_data(self, output_dir, masks, boxes, labels):
        """保存掩码数据"""
        value = 0
        json_data = [{'value': value, 'label': 'background'}]

        for label, box in zip(labels, boxes):
            value += 1
            name, logit = label.split('(')
            logit = logit[:-1]
            json_data.append({
                'value': value,
                'label': name,
                'logit': float(logit),
                'box': box.numpy().tolist(),
            })

        with open(os.path.join(output_dir, 'mask_data.json'), 'w') as f:
            json.dump(json_data, f, indent=4)

    def process_batch(self):
        """批量处理图像"""
        image_files = [
            f for f in os.listdir(self.config['input_dir'])
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]

        print(f"发现 {len(image_files)} 张图像需要处理")
        print(f"使用提示文本: '{self.config['text_prompt']}'")
        print(f"输出目录: {self.config['output_dir']}")

        for image_file in tqdm(image_files, desc="处理进度"):
            image_path = os.path.join(self.config['input_dir'], image_file)
            try:
                results = self._process_single_image(image_path)
                if results is not None:
                    self.save_results(results, image_path)
            except Exception as e:
                print(f"处理 {image_file} 时出错: {str(e)}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="批量图像分割处理器")

    # 必需参数
    parser.add_argument("--input_dir", type=str, required=True,
                        help="包含输入图像的目录")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="保存输出结果的目录")
    parser.add_argument("--text_prompt", type=str, required=True,
                        help="用于目标检测的文本提示，例如: 'dog. cat. chair.'")

    # 模型参数
    parser.add_argument("--grounding_config", type=str,
                        default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                        help="GroundingDINO配置文件路径")
    parser.add_argument("--grounding_checkpoint", type=str,
                        default="groundingdino_swint_ogc.pth",
                        help="GroundingDINO检查点路径")
    parser.add_argument("--sam_version", type=str, default="vit_h",
                        help="SAM模型版本: vit_b/vit_l/vit_h")
    parser.add_argument("--sam_checkpoint", type=str,
                        default="sam_vit_h_4b8939.pth",
                        help="SAM检查点路径")
    parser.add_argument("--bert_base_uncased_path", type=str,
                        default="bert-base-uncased",
                        help="BERT基础模型路径")

    # 处理参数
    parser.add_argument("--box_threshold", type=float, default=0.3,
                        help="边界框置信度阈值")
    parser.add_argument("--text_threshold", type=float, default=0.25,
                        help="文本匹配阈值")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="计算设备 (cpu/cuda)")
    parser.add_argument("--use_sam_hq", action="store_true",
                        help="使用SAM-HQ模型")

    return parser.parse_args()


def main():
    # 解析命令行参数
    args = parse_args()

    # 准备配置
    config = {
        'input_dir': args.input_dir,
        'output_dir': args.output_dir,
        'text_prompt': args.text_prompt,
        'grounding_config': args.grounding_config,
        'grounding_checkpoint': args.grounding_checkpoint,
        'sam_version': args.sam_version,
        'sam_checkpoint': args.sam_checkpoint,
        'bert_base_uncased_path': args.bert_base_uncased_path,
        'box_threshold': args.box_threshold,
        'text_threshold': args.text_threshold,
        'device': args.device,
        'use_sam_hq': args.use_sam_hq
    }

    # 初始化并运行处理器
    segmenter = BatchImageSegmenter(config)
    segmenter.process_batch()


if __name__ == "__main__":
    main()