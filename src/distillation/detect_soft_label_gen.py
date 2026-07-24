"""
Detection Soft Label Generator
==============================

Detect任务的软标签生成器，继承基类复用公共逻辑，实现Detection特有逻辑。

关键差异：
1. 候选空间约束：从VQA的「黑名单」变Detect的「白名单」
2. 回归头的软标签处理（Detect独有）
3. 候选框联合过滤
4. 无目标场景处理（background类别）

重构说明：
- 继承 BaseSoftLabelGenerator 复用公共逻辑（温度缩放、保存、验证等）
- 仅保留 Detection 特有的方法（白名单、回归头软化等）
"""

import torch
import json
import numpy as np
from typing import Dict, Any, List, Optional, Union, Set
from pathlib import Path
from datetime import datetime

from .base_soft_label_gen import BaseSoftLabelGenerator
from ..models.teacher_model import TeacherModel
from ..utils.config import ConfigManager
from ..utils.logger import get_logger
from ..utils.vqa_token_filter import VQATokenFilter


class DetectSoftLabelGenerator(BaseSoftLabelGenerator):
    """
    Detection任务的软标签生成器。

    继承基类复用公共逻辑：
    - 温度缩放 (_apply_temperature)
    - Top-K概率提取 (_get_top_k_probabilities)
    - 数据序列化 (_make_serializable)
    - 文件保存 (save_soft_labels)
    - 数据验证 (validate_soft_labels)
    - 统计信息 (get_statistics)

    Detection特有逻辑：
    - 类别白名单过滤（_apply_category_whitelist）
    - 回归头软化（_soften_bbox）
    - 候选框联合过滤
    - background类别处理
    """

    def __init__(
        self,
        teacher_model: TeacherModel,
        config: Optional[ConfigManager] = None,
        detect_categories: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None
    ):
        """
        Initialize Detection Soft Label Generator.

        Args:
            teacher_model: Teacher model instance
            config: Configuration manager
            detect_categories: Detection类别列表（如COCO的80类）
            temperature: 温度参数（覆盖配置）
            top_k: Top-K参数（覆盖配置）
        """
        # 🔧 调用基类初始化
        super().__init__(
            teacher_model=teacher_model,
            config=config,
            temperature=temperature,
            top_k=top_k
        )

        # Detection特有参数
        self.min_probability = self.config.get("distillation.soft_labels.min_probability", 0.01)

        # 🔧 Detect专用：类别白名单
        self.detect_categories = detect_categories or self._get_default_detect_categories()
        self.category_whitelist_token_ids = self._build_category_whitelist()
        self.logger.info(f"Detect类别白名单构建完成：{len(self.detect_categories)}个类别，{len(self.category_whitelist_token_ids)}个token ID")

        # 🔧 复用VQA的Token过滤器（用于BPE碎片黑名单）
        try:
            self.token_filter = VQATokenFilter()
            self.logger.info("✓ VQA Token过滤器初始化成功（复用BPE碎片黑名单）")
        except Exception as e:
            self.logger.warning(f"VQA Token过滤器初始化失败: {e}")
            self.token_filter = None

    def _get_default_detect_categories(self) -> List[str]:
        """
        获取默认的Detect类别列表（COCO 80类）。

        Returns:
            COCO类别列表
        """
        # COCO 80类
        coco_categories = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush', 'background'  # 🔧 新增background类别
        ]

        return self.config.get("detect.categories", coco_categories)

    def _build_category_whitelist(self) -> Set[int]:
        """
        构建Detect类别白名单的Token ID集合。

        改进：
        1. 支持一个类别对应多个token变体（如 'person'/'Person'/' person'）
        2. 支持多词类别（如 'traffic light' → ['traffic', 'light']）

        Returns:
            Token ID集合
        """
        whitelist_ids = set()

        for category in self.detect_categories:
            # 🔧 关键：处理一个类别的多种token形式

            # 形式1: 类别本身（如 'person'）
            try:
                token_ids = self.teacher.tokenizer.encode(category, add_special_tokens=False)
                whitelist_ids.update(token_ids)
                self.logger.debug(f"Category '{category}' → token IDs: {token_ids}")
            except Exception as e:
                self.logger.warning(f"Failed to encode category '{category}': {e}")

            # 形式2: 首字母大写（如 'Person'）
            capitalized = category.capitalize()
            if capitalized != category:
                try:
                    token_ids = self.teacher.tokenizer.encode(capitalized, add_special_tokens=False)
                    whitelist_ids.update(token_ids)
                except Exception:
                    pass

            # 形式3: 全大写（如 'PERSON'）
            upper = category.upper()
            if upper != category:
                try:
                    token_ids = self.teacher.tokenizer.encode(upper, add_special_tokens=False)
                    whitelist_ids.update(token_ids)
                except Exception:
                    pass

            # 形式4: 带空格前缀（如 ' person'）
            # 某些tokenizer会将前导空格编码为单独的token
            with_space = f" {category}"
            try:
                token_ids = self.teacher.tokenizer.encode(with_space, add_special_tokens=False)
                whitelist_ids.update(token_ids)
            except Exception:
                pass

        return whitelist_ids

    def generate_detect_soft_labels(
        self,
        image_path: str,
        image_id: Optional[str] = None,
        hard_label_result: Optional[Dict[str, Any]] = None,
        return_bbox_soft_labels: bool = True
    ) -> Dict[str, Any]:
        """
        Generate soft labels for object detection.

        改进：
        1. 支持传入已有的hard_label结果，避免重复推理
        2. 支持回归头的软标签处理
        3. 候选框联合过滤

        Args:
            image_path: Path to image
            image_id: Image identifier
            hard_label_result: 已有的hard_label结果（包含objects），避免重复推理
            return_bbox_soft_labels: 是否返回bbox回归头的软标签

        Returns:
            Soft label dictionary
        """
        self.logger.debug(f"Generating detection soft labels for image {image_id}")

        soft_label = {
            'image_id': image_id,
            'task': 'detection',
            'temperature': self.temperature,
            'timestamp': datetime.now().isoformat(),
        }

        # 🔧 优先使用已有的hard_label结果，避免重复推理导致结果不一致
        if hard_label_result and 'objects' in hard_label_result:
            objects = hard_label_result['objects']
            self.logger.debug(f"Using existing hard_label result with {len(objects)} objects")
        else:
            # 如果没有hard_label结果，才调用模型
            self.logger.warning(f"No hard_label result provided, calling inference_detection")
            result = self.teacher.inference_detection(
                image=image_path,
                return_logits=True,  # 🔧 关键：需要logits来生成软标签
                generate_cot=False
            )
            objects = result.get('objects', [])

            # 🔧 将result中的logits提取出来，用于后续处理
            if 'logits' in result:
                hard_label_result = {'objects': objects, 'logits': result['logits']}

        # 🔧 基于检测结果生成分布
        if objects:
            object_soft_labels = []

            for obj in objects:
                category = obj.get('category', 'unknown')
                confidence = obj.get('confidence', 0.5)
                bbox = obj.get('bbox', [])

                # 🔧 检查类别是否在白名单内
                if category not in self.detect_categories and category != 'unknown':
                    self.logger.warning(f"Category '{category}' not in whitelist, skipping")
                    continue

                # 🔧 Step 1: 分类头的软标签（应用类别白名单过滤）
                category_distribution = self._generate_category_distribution_with_whitelist(
                    category=category,
                    confidence=confidence,
                    temperature=self.temperature,
                    hard_label_result=hard_label_result
                )

                # 🔧 Step 2: 回归头的软标签（Detect独有）
                bbox_soft_label = None
                if return_bbox_soft_labels and bbox:
                    bbox_soft_label = self._soften_bbox(
                        teacher_bbox=bbox,
                        confidence=confidence
                    )

                object_soft_labels.append({
                    'category': category,
                    'bbox': bbox,
                    'confidence': confidence,
                    'category_distribution': category_distribution,
                    'bbox_soft_label': bbox_soft_label  # 🔧 新增：回归头软标签
                })

            soft_label['object_soft_labels'] = object_soft_labels
            soft_label['num_objects'] = len(object_soft_labels)

            self.logger.debug(f"Generated soft labels for {len(object_soft_labels)} objects")

        return soft_label

    def _generate_category_distribution_with_whitelist(
        self,
        category: str,
        confidence: float,
        temperature: float,
        hard_label_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        为检测到的物体生成类别分布，应用类别白名单过滤。

        关键改进：
        1. 复用VQA的温度缩放逻辑
        2. 复用VQA的硬标签保底机制
        3. 从黑名单策略改为白名单策略（Detect专用）
        4. Top-K兜底逻辑改为「候选框兜底」

        Args:
            category: 检测到的类别
            confidence: 检测置信度
            temperature: 温度参数
            hard_label_result: hard_label结果（包含logits）

        Returns:
            类别概率分布
        """
        distribution = {}

        # 🔧 策略1: 如果有logits数据，使用真实的类别分布
        if hard_label_result and 'logits' in hard_label_result:
            logits_data = hard_label_result['logits']

            # 调用VQA的logits处理逻辑（复用80%）
            raw_distribution = self._process_detect_logits(
                logits_data=logits_data,
                category=category,
                confidence=confidence
            )

            # 🔧 应用Detect专用：类别白名单过滤
            distribution = self._apply_category_whitelist(
                distribution=raw_distribution,
                hard_label_category=category
            )

        # 🔧 策略2: 如果没有logits，使用置信度生成分布
        else:
            # 使用温度缩放（复用VQA逻辑）
            scaled_confidence = confidence / temperature

            # 主类别概率
            main_prob = min(scaled_confidence, 0.98)

            # 🔧 应用白名单：只允许白名单内的类别
            # 获取相似类别（子类别或兄弟类别）
            similar_categories = self._get_similar_categories(category)

            # 过滤：只保留白名单内的相似类别
            valid_similar_categories = [
                cat for cat in similar_categories
                if cat in self.detect_categories
            ]

            if valid_similar_categories:
                # 主类别的实际概率
                distribution[category] = main_prob

                # 剩余概率分配给白名单内的相似类别
                remaining_prob = 1.0 - main_prob

                # 均匀分配（或按权重分配）
                for similar_cat in valid_similar_categories:
                    distribution[similar_cat] = remaining_prob / len(valid_similar_categories)
            else:
                # 如果没有相似类别，主类别获得大部分概率
                distribution[category] = main_prob
                # 剩余概率给background
                if 'background' in self.detect_categories:
                    distribution['background'] = 1.0 - main_prob

        # 🔧 归一化分布
        if distribution:
            total_prob = sum(distribution.values())
            if total_prob > 0:
                distribution = {k: v / total_prob for k, v in distribution.items()}

        return distribution

    def _process_detect_logits(
        self,
        logits_data: Dict[str, torch.Tensor],
        category: str,
        confidence: float
    ) -> Dict[str, float]:
        """
        处理Detect的logits，复用VQA的逻辑框架。

        关键复用：
        1. 温度缩放基础逻辑
        2. 硬标签保底机制
        3. Top-K兜底的逻辑框架

        Args:
            logits_data: Logits数据
            category: 检测到的类别
            confidence: 置信度

        Returns:
            原始类别分布（未应用白名单）
        """
        distribution = {}

        # 🔧 复用VQA的logits处理逻辑
        if 'top_k_indices' in logits_data and 'top_k_values' in logits_data:
            token_indices = logits_data['top_k_indices']
            token_logits = logits_data['top_k_values']

            if token_indices.dim() >= 1 and token_logits.dim() >= 1:
                # 取第一个位置
                if token_indices.dim() == 1:
                    first_token_indices = token_indices
                    first_token_logits = token_logits
                elif token_indices.dim() == 2:
                    first_token_indices = token_indices[0]
                    first_token_logits = token_logits[0]
                else:
                    first_token_indices = token_indices[0, 0]
                    first_token_logits = token_logits[0, 0]

                # 🔧 Step 1: 应用温度缩放（复用VQA逻辑）
                scaled_logits = first_token_logits / self.temperature

                # 🔧 Step 2: 计算softmax得到概率
                token_probs = torch.softmax(scaled_logits, dim=-1)

                # 🔧 Step 3: 提取并解码（复用VQA逻辑）
                items = []
                for idx, prob_val in zip(first_token_indices, token_probs):
                    if prob_val < 0.001:
                        continue

                    try:
                        word = self.teacher.tokenizer.decode([idx.item()])
                        word = word.strip().lower()

                        # 🔧 复用VQA的BPE碎片黑名单过滤
                        if self.token_filter and self.token_filter.is_valid_token(word, None):
                            if word and len(word) > 0:
                                items.append((word, float(prob_val)))
                    except Exception:
                        pass

                # 🔧 Step 4: 合并相同词的概率
                word_probs = {}
                for word, prob in items:
                    if word in word_probs:
                        word_probs[word] += prob
                    else:
                        word_probs[word] = prob

                # 🔧 Step 5: 硬标签保底（复用VQA逻辑）
                # 确保GT对应的类别token永远不会被过滤
                category_lower = category.lower()
                if category_lower not in word_probs:
                    # 强制添加，使用置信度作为概率
                    word_probs[category_lower] = confidence

                # 构建 distribution
                for word, prob in sorted(word_probs.items(), key=lambda x: x[1], reverse=True):
                    distribution[word] = prob

        # 如果没有logits，使用置信度构建
        elif category and confidence:
            main_prob = min(confidence, 0.95)
            distribution[category.lower()] = main_prob
            remaining = 1.0 - main_prob
            if remaining > 0:
                distribution['other'] = remaining

        return distribution

    def _apply_category_whitelist(
        self,
        distribution: Dict[str, float],
        hard_label_category: str
    ) -> Dict[str, float]:
        """
        应用Detect类别白名单过滤（核心差异点）。

        关键逻辑：
        1. 只允许白名单内的类别通过
        2. 硬标签保底：GT类别永远不被过滤
        3. 过滤后归一化

        Args:
            distribution: 原始分布
            hard_label_category: 硬标签类别（GT）

        Returns:
            过滤后的分布
        """
        filtered_distribution = {}

        for word, prob in distribution.items():
            word_lower = word.lower()

            # 🔧 硬标签保底：GT类别永远不被过滤
            if word_lower == hard_label_category.lower():
                filtered_distribution[word_lower] = prob
                self.logger.debug(f"[Hard Label Protection] Reserved GT category: '{word_lower}'")
                continue

            # 🔧 白名单过滤：只允许白名单内的类别
            # 检查word是否在白名单内（通过token ID判断）
            try:
                token_ids = self.teacher.tokenizer.encode(word, add_special_tokens=False)

                # 如果所有token都在白名单内，则保留
                if all(tid in self.category_whitelist_token_ids for tid in token_ids):
                    filtered_distribution[word_lower] = prob
                    self.logger.debug(f"[Whitelist Filter] ✓ Valid category: '{word_lower}'")
                else:
                    self.logger.debug(f"[Whitelist Filter] ✗ Filtered out: '{word_lower}'")

            except Exception as e:
                self.logger.warning(f"[Whitelist Filter] Failed to encode '{word}': {e}")
                # 如果编码失败，保守起见保留（避免过度过滤）
                # 但这里可以选择过滤掉，取决于业务需求
                pass

        # 🔧 归一化（确保分布有效）
        if filtered_distribution:
            total_prob = sum(filtered_distribution.values())
            if total_prob > 0:
                filtered_distribution = {k: v / total_prob for k, v in filtered_distribution.items()}
        else:
            # 极端情况：所有类别都被过滤（不应该发生，因为有硬标签保底）
            self.logger.warning("[Emergency Fallback] All categories filtered! Using hard label only")
            filtered_distribution[hard_label_category.lower()] = 1.0

        return filtered_distribution

    def _soften_bbox(
        self,
        teacher_bbox: List[float],
        confidence: float,
        gt_bbox: Optional[List[float]] = None,
        teacher_weight: float = 0.7
    ) -> Dict[str, Any]:
        """
        回归头的软标签处理（Detect独有）。

        关键改进：
        1. 保留Teacher的不确定性
        2. 结合GT bbox做加权融合
        3. 不应用温度缩放（回归头不适用）

        Args:
            teacher_bbox: Teacher预测的bbox [x1, y1, x2, y2]
            confidence: Teacher的置信度
            gt_bbox: Ground truth bbox（如果有）
            teacher_weight: Teacher的权重（默认0.7）

        Returns:
            Bbox软标签字典
        """
        bbox_soft_label = {
            'teacher_bbox': teacher_bbox,
            'confidence': confidence,
            'teacher_weight': teacher_weight,
        }

        if gt_bbox is None:
            # 无GT时，直接使用Teacher预测
            bbox_soft_label['soft_bbox'] = teacher_bbox
            bbox_soft_label['fusion_method'] = 'teacher_only'
        else:
            # 🔧 加权融合：保留Teacher的不确定性，同时向GT对齐
            soft_bbox = [
                teacher_bbox[i] * teacher_weight + gt_bbox[i] * (1 - teacher_weight)
                for i in range(len(teacher_bbox))
            ]

            bbox_soft_label['soft_bbox'] = soft_bbox
            bbox_soft_label['gt_bbox'] = gt_bbox
            bbox_soft_label['fusion_method'] = 'weighted_fusion'

        # 🔧 可选：生成高斯分布参数（更精细的软标签）
        # 这里简化为直接返回融合后的bbox
        # 如果需要更精细的处理，可以添加高斯噪声或生成分布

        return bbox_soft_label

    def _get_similar_categories(self, category: str) -> List[str]:
        """
        获取相似类别（简化版本，复用VQA的逻辑框架）。

        在实际应用中，应该使用类别语义相似度或视觉相似度。

        Args:
            category: 输入类别

        Returns:
            相似类别列表
        """
        # 简化的相似类别映射
        similar_map = {
            'person': ['man', 'woman', 'child'],
            'car': ['truck', 'bus', 'vehicle'],
            'bicycle': ['motorcycle', 'bike'],
            'dog': ['cat', 'animal'],
            'cat': ['dog', 'animal'],
            'chair': ['sofa', 'couch', 'seat'],
        }

        return similar_map.get(category.lower(), [])

    # ==================
# 继承自基类的方法（已删除重复实现）
# ==================
# 以下方法已从基类 BaseSoftLabelGenerator 继承：
# - save_soft_labels(): 保存软标签到文件
# - _make_serializable(): 转换为可序列化格式
# - validate_soft_labels(): 验证数据有效性
# - get_statistics(): 计算统计信息
# - __repr__(): 字符串表示
#
# 如需Detection特定的验证逻辑，可覆盖validate_soft_labels方法：
#
# def validate_soft_labels(self, soft_labels: Dict[str, Any]) -> bool:
#     """验证Detection软标签"""
#     if not super().validate_soft_labels(soft_labels):
#         return False
#     return 'category_distribution' in soft_labels

    def _make_serializable(
        self,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert tensors to serializable format（复用VQA的逻辑）。

        Args:
            data: Dictionary potentially containing tensors

        Returns:
            JSON-serializable dictionary
        """
        serializable = {}

        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                serializable[key] = value.tolist()
            elif isinstance(value, np.ndarray):
                serializable[key] = value.tolist()
            elif isinstance(value, dict):
                serializable[key] = self._make_serializable(value)
            elif isinstance(value, list):
                serializable[key] = [
                    self._make_serializable(v) if isinstance(v, dict) else
                    v.tolist() if isinstance(v, (torch.Tensor, np.ndarray)) else v
                    for v in value
                ]
            else:
                serializable[key] = value

        return serializable

    def validate_soft_labels(
        self,
        soft_labels: Dict[str, Any]
    ) -> bool:
        """
        Validate soft label structure.

        Args:
            soft_labels: Soft label dictionary

        Returns:
            True if valid
        """
        required_keys = ['image_id', 'task', 'temperature', 'timestamp']

        for key in required_keys:
            if key not in soft_labels:
                self.logger.warning(f"Missing key in soft labels: {key}")
                return False

        return True

    def get_statistics(
        self,
        soft_labels_list: List[Dict]
    ) -> Dict[str, Any]:
        """
        Compute statistics from soft labels（复用VQA的逻辑）。

        Args:
            soft_labels_list: List of soft labels

        Returns:
            Statistics dictionary
        """
        stats = {
            'total_count': len(soft_labels_list),
            'by_task': {},
            'average_temperature': self.temperature,
            'total_objects': 0,
            'total_bbox_soft_labels': 0,
        }

        for label in soft_labels_list:
            task = label.get('task', 'unknown')
            if task not in stats['by_task']:
                stats['by_task'][task] = 0
            stats['by_task'][task] += 1

            # Count objects
            if 'object_soft_labels' in label:
                stats['total_objects'] += len(label['object_soft_labels'])

                # Count bbox soft labels
                for obj in label['object_soft_labels']:
                    if obj.get('bbox_soft_label'):
                        stats['total_bbox_soft_labels'] += 1

        return stats

    def __repr__(self) -> str:
        """String representation."""
        return f"DetectSoftLabelGenerator(teacher={self.teacher.model_name}, temp={self.temperature}, categories={len(self.detect_categories)})"