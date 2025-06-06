import os
import torch
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

# 设置环境路径（根据实际安装位置调整）
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
    def __init__(self):
        """初始化分割器，直接在代码中配置参数"""
        # ============= 配置区域 =============
        self.input_root = "/media/HDD0/XCX/imagenet/imagenet_images"  # 输入根目录（包含分类子文件夹）
        self.output_root = "/media/HDD0/XCX/sam"  # 输出根目录

        # 模型文件路径配置
        self.grounding_config = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.grounding_checkpoint = "groundingdino_swint_ogc.pth"
        self.sam_version = "vit_h"
        self.sam_checkpoint = "sam_vit_h_4b8939.pth"
        self.bert_base_uncased_path = "bert-base-uncased"

        # 处理参数配置
        self.box_threshold = 0.3
        self.text_threshold = 0.25
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_sam_hq = False
        self.save_visualization = True  # 是否保存可视化结果
        self.probability = 0.5  # 调整生成对应目标的置信度下限
        # ============= 配置结束 =============

        # 初始化设备
        self.device = torch.device(self.device)

        # 初始化模型
        self._init_models()

        # 确保输出目录存在
        os.makedirs(self.output_root, exist_ok=True)

    def _init_models(self):
        """初始化GroundingDINO和SAM模型"""
        print("正在初始化模型...")
        print(f"Using device: {self.device}")

        # 初始化GroundingDINO
        args = SLConfig.fromfile(self.grounding_config)
        args.device = self.device.type
        args.bert_base_uncased_path = self.bert_base_uncased_path
        self.grounding_model = build_model(args).to(self.device)

        # 确保模型权重在正确设备上
        checkpoint = torch.load(self.grounding_checkpoint, map_location=self.device)
        self.grounding_model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        self.grounding_model.eval()
        print(f"GroundingDINO model device: {next(self.grounding_model.parameters()).device}")

        # 初始化SAM
        sam = sam_model_registry[self.sam_version](checkpoint=self.sam_checkpoint).to(self.device)
        self.sam_predictor = SamPredictor(sam)
        print(f"SAM model device: {next(sam.parameters()).device}")
        print("模型初始化完成")

    def _get_category_prompt(self, category_name):
        """生成类别特定的提示文本：'类别名. fish.'"""
        return f"{category_name.lower()}." #不用fish
        #return f"{category_name.lower()}. fish."

    def _load_image(self, image_path):
        """加载并预处理图像"""
        image_pil = Image.open(image_path).convert("RGB")
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)
        return image_pil, image.to(self.device)

    @torch.no_grad()  # 禁用梯度
    def _get_grounding_output(self, image, caption):
        """从GroundingDINO获取输出，要求置信度足够大"""
        caption = caption.lower().strip()
        if not caption.endswith("."):
            caption += "."

        # 确保输入在正确设备上
        if image.device != self.device:
            image = image.to(self.device)

        with torch.no_grad():
            outputs = self.grounding_model(image[None], captions=[caption])

        logits = outputs["pred_logits"].sigmoid()[0]  # 保持在模型设备上
        boxes = outputs["pred_boxes"][0]  # 保持在模型设备上

        # 双重过滤：既要大于box_threshold，又要置信度>0.5
        confidence_mask = logits.max(dim=1)[0] > self.probability  # 新增置信度过滤
        filt_mask = (logits.max(dim=1)[0] > self.box_threshold) & confidence_mask
        logits_filt = logits[filt_mask]
        boxes_filt = boxes[filt_mask]

        # 获取短语
        tokenizer = self.grounding_model.tokenizer
        tokenized = tokenizer(caption)
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(
                logit > self.text_threshold,
                tokenized,
                tokenizer
            )
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")

        return boxes_filt, pred_phrases

    def _process_image(self, image_path, category_name):
        """处理单张图像（完整修正版）"""
        # 1. 生成提示文本
        text_prompt = self._get_category_prompt(category_name)

        # 2. 加载图像
        image_pil, image_tensor = self._load_image(image_path)
        image_cv = cv2.imread(image_path)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

        # 3. GroundingDINO检测
        boxes_filt, pred_phrases = self._get_grounding_output(image_tensor, text_prompt)
        if boxes_filt.size(0) == 0:
            return None

        # 4. SAM分割
        self.sam_predictor.set_image(image_cv)
        H, W = image_pil.size[1], image_pil.size[0]

        # 5. 转换边界框格式（全部在GPU上完成）
        size = torch.tensor([W, H, W, H], device=self.device)
        boxes_filt = boxes_filt * size
        boxes_filt[:, :2] -= boxes_filt[:, 2:] / 2
        boxes_filt[:, 2:] += boxes_filt[:, :2]

        # 确保boxes_filt在正确设备上
        if boxes_filt.device != self.device:
            boxes_filt = boxes_filt.to(self.device)

        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            boxes_filt, image_cv.shape[:2]
        ).to(self.device)

        # 6. 获取掩码
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False
        )

        # 7. 返回结果（所有张量移回CPU）
        return {
            'image_pil': image_pil,
            'image_cv': image_cv,
            'boxes': boxes_filt.cpu(),
            'phrases': pred_phrases,
            'masks': masks.cpu(),
            'text_prompt': text_prompt
        }

    def _save_cropped_objects(self, image_cv, masks, pred_phrases, output_dir, base_name):
        """保存抠出的目标图像，可视化结果附带边界框、类别标签和置信度"""
        # 创建子文件夹
        image_dir = os.path.join(output_dir, "image")
        mask_dir = os.path.join(output_dir, "mask")
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        if self.save_visualization:
            visible_dir = os.path.join(output_dir, "visible")
            os.makedirs(visible_dir, exist_ok=True)

        for idx, (mask, phrase) in enumerate(zip(masks, pred_phrases)):
            mask_np = mask.numpy()[0].astype(np.uint8) * 255

            # 1. 保存mask图像
            cv2.imwrite(os.path.join(mask_dir, f"{base_name}_mask_{idx}.png"), mask_np)

            # 2. 抠出目标图像（保持原始尺寸）
            # 创建透明背景的RGBA图像
            rgba = cv2.cvtColor(image_cv, cv2.COLOR_RGB2RGBA)
            # 将mask区域外的像素设为透明
            rgba[:, :, 3] = mask_np
            # 保存为PNG以保留透明度
            cv2.imwrite(os.path.join(image_dir, f"{base_name}_obj_{idx}.png"),
                        cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

            """"下面是调整扣除图像尺寸的功能"""
            # contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # if contours:
            #     x, y, w, h = cv2.boundingRect(contours[0])
            #     cropped = masked_image[y:y + h, x:x + w]
            #     cv2.imwrite(os.path.join(image_dir, f"{base_name}_obj_{idx}.png"),
            #                 cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))

            # 3. 可视化效果（带完整标签信息）
            if self.save_visualization:
                # 解析标签信息（示例格式："tench(0.72)"）
                label = phrase.split("(")[0]  # 获取类别名
                confidence = float(phrase.split("(")[-1][:-1])  # 获取置信度

                # 创建带mask的可视化图像
                color_mask = np.zeros_like(image_cv)
                color_mask[mask_np == 255] = [0, 255, 0]  # 绿色mask
                visible = cv2.addWeighted(image_cv, 0.7, color_mask, 0.3, 0)

                # 绘制边界框
                x, y, w, h = cv2.boundingRect(mask_np)
                cv2.rectangle(visible, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 组合标签文本
                display_text = f"{label}: {confidence:.2f}"

                # 计算文本大小和位置
                (text_width, text_height), _ = cv2.getTextSize(
                    display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                # 绘制标签背景框
                cv2.rectangle(visible,
                              (x, y - text_height - 10),
                              (x + text_width, y),
                              (0, 255, 0), -1)  # 实心绿色背景

                # 添加标签文本（白色文字）
                cv2.putText(visible, display_text,
                            (x, y - 5),  # 位置微调
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 1)  # 白色文字

                cv2.imwrite(os.path.join(visible_dir, f"{base_name}_visible_{idx}.png"),
                            cv2.cvtColor(visible, cv2.COLOR_RGB2BGR))


    def _save_results(self, results, src_path, category_name):
        """保存处理结果"""
        # 确保所有张量都在CPU上
        boxes = results['boxes'].cpu() if isinstance(results['boxes'], torch.Tensor) else torch.tensor(results['boxes'])
        masks = results['masks'].cpu() if isinstance(results['masks'], torch.Tensor) else torch.tensor(results['masks'])

        # 创建输出子目录
        output_dir = os.path.join(self.output_root, category_name)
        os.makedirs(output_dir, exist_ok=True)

        # 获取原始文件名(不带扩展名)
        base_name = os.path.splitext(os.path.basename(src_path))[0]

        # 保存抠出的目标图像（现在传递所有必需参数）
        self._save_cropped_objects(
            image_cv=results['image_cv'],
            masks=masks,
            pred_phrases=results['phrases'],
            output_dir=output_dir,
            base_name=base_name
        )

        # 保存元数据到visible文件夹（或其他子文件夹，这里选择visible）
        metadata_dir = os.path.join(output_dir, "visible")
        os.makedirs(metadata_dir, exist_ok=True)

        metadata = {
            'original_path': src_path,
            'category': category_name,
            'text_prompt': results['text_prompt'],
            'detected_objects': [
                {
                    'label': label.split('(')[0],
                    'confidence': float(label.split('(')[1][:-1]),
                    'box': box.numpy().tolist()
                }
                for label, box in zip(results['phrases'], boxes)
            ]
        }
        metadata_path = os.path.join(metadata_dir, f"{base_name}_meta.json")
        with open(metadata_path, 'w') as f:
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

    def process_all_categories(self):
        """处理所有分类目录中的图像"""
        # 获取所有类别子目录
        categories = [d for d in os.listdir(self.input_root)
                      if os.path.isdir(os.path.join(self.input_root, d))]

        print(f"发现 {len(categories)} 个类别目录:")
        print("\n".join(f"- {c}" for c in categories))
        print(f"输出将保存到: {self.output_root}")

        # 处理每个类别
        for category in tqdm(categories, desc="处理类别"):
            category_dir = os.path.join(self.input_root, category)

            # 获取该类别下所有图像
            image_files = [f for f in os.listdir(category_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

            print(f"\n正在处理类别: {category} (共 {len(image_files)} 张图像)")
            print(f"使用提示文本: '{self._get_category_prompt(category)}'")

            # 处理每张图像
            for img_file in tqdm(image_files, desc=f"处理 {category} 图像"):
                img_path = os.path.join(category_dir, img_file)
                try:
                    results = self._process_image(img_path, category)
                    if results is not None:
                        self._save_results(results, img_path, category)
                except Exception as e:
                    print(f"处理 {img_path} 时出错: {str(e)}")


if __name__ == "__main__":
    # 创建并运行分割器
    segmenter = CategorizedSegmenter()
    segmenter.process_all_categories()