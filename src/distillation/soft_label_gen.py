"""
Soft Label Generator
====================

Generates soft labels (probability distributions) from teacher model.
"""

import torch
import json
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime

from ..models.teacher_model import TeacherModel
from ..utils.config import ConfigManager
from ..utils.logger import get_logger


class SoftLabelGenerator:
    """
    Generates soft labels (probability distributions) from teacher model.

    Soft labels provide richer information for knowledge distillation:
    - Probability distributions over possible answers
    - Temperature-scaled logits
    - Top-k probabilities for storage efficiency
    """

    def __init__(
        self,
        teacher_model: TeacherModel,
        config: Optional[ConfigManager] = None
    ):
        """
        Initialize Soft Label Generator.

        Args:
            teacher_model: Teacher model instance
            config: Configuration manager
        """
        self.teacher = teacher_model
        self.config = config or ConfigManager()
        self.logger = get_logger()

        # Settings
        self.temperature = self.config.get("distillation.soft_labels.temperature", 2.0)
        self.top_k = self.config.get("distillation.soft_labels.top_k", 100)
        self.min_probability = self.config.get("distillation.soft_labels.min_probability", 0.01)

    def generate_vqa_soft_labels(
        self,
        image_path: str,
        question: str,
        image_id: Optional[str] = None,
        answer_candidates: Optional[List[str]] = None,
        hard_label_result: Optional[Dict[str, Any]] = None,
        cot_result: Optional[Dict[str, Any]] = None  # 保留参数兼容，但不再使用
    ) -> Dict[str, Any]:
        """
        Generate soft labels for VQA.

        新方案：使用 hard_label 中的真实 logits，不从 CoT 获取

        Args:
            image_path: Path to image
            question: Question text
            image_id: Image identifier
            answer_candidates: List of possible answer candidates (optional)
            hard_label_result: hard_label 结果（包含 answer 和 logits）
            cot_result: 保留参数兼容，但不再使用

        Returns:
            Soft label dictionary
        """
        self.logger.debug(f"Generating VQA soft labels for image {image_id}")

        soft_label = {
            'image_id': image_id,
            'task': 'vqa',
            'question': question,
            'temperature': self.temperature,
            'timestamp': datetime.now().isoformat(),
        }

        # 🔧 从 hard_label 获取 logits
        if hard_label_result and 'logits' in hard_label_result:
            logits_data = hard_label_result['logits']
            distribution = self._process_vqa_logits(logits_data, answer_candidates)
            soft_label['answer_distribution'] = distribution
            soft_label['primary_answer'] = hard_label_result.get('answer', '')
            soft_label['confidence'] = hard_label_result.get('confidence', 0.0)
            soft_label['source'] = 'hard_label_logits'
            return soft_label

        # 如果没有 logits，调用模型获取
        if hard_label_result and 'answer' in hard_label_result:
            primary_answer = hard_label_result['answer']
            confidence = hard_label_result.get('confidence', 0.5)

            soft_label['primary_answer'] = primary_answer
            soft_label['confidence'] = confidence
            soft_label['source'] = 'hard_label_derived'

            # 简化分布
            main_prob = min(confidence, 0.98)
            soft_label['answer_distribution'] = {
                primary_answer.lower(): main_prob,
                'other': 1.0 - main_prob
            }
            return soft_label

        # 如果没有 hard_label，调用模型
        self.logger.warning(f"No hard_label result provided, calling inference_vqa")
        result = self.teacher.inference_vqa(
            image=image_path,
            question=question,
            return_logits=True,
            generate_cot=False
        )

        soft_label['primary_answer'] = result.get('answer', '')
        soft_label['confidence'] = result.get('confidence', 0.0)
        soft_label['source'] = 'teacher_logits'

        if 'logits' in result:
            logits_data = result['logits']
            distribution = self._process_vqa_logits(logits_data, answer_candidates)
            soft_label['answer_distribution'] = distribution
        else:
            soft_label['answer_distribution'] = {
                result.get('answer', 'unknown').lower(): 1.0
            }

        return soft_label

        # Add primary answer for reference
        soft_label['primary_answer'] = result.get('answer', '')
        soft_label['confidence'] = result.get('confidence', 0.0)

        return soft_label

    def generate_captioning_soft_labels(
        self,
        image_path: str,
        num_captions: int = 3,
        image_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate soft labels for captioning.

        Args:
            image_path: Path to image
            num_captions: Number of caption variations
            image_id: Image identifier

        Returns:
            Soft label dictionary
        """
        self.logger.debug(f"Generating captioning soft labels for image {image_id}")

        # Get teacher inference with logits
        result = self.teacher.inference_captioning(
            image=image_path,
            return_logits=True,
            generate_cot=False,
            num_captions=num_captions
        )

        soft_label = {
            'image_id': image_id,
            'task': 'captioning',
            'temperature': self.temperature,
            'timestamp': datetime.now().isoformat(),
        }

        # Process caption logits
        if 'logits' in result:
            caption_distributions = []
            for i, logits_data in enumerate(result['logits']):
                distribution = self._process_caption_logits(logits_data)
                caption_distributions.append({
                    'caption_index': i,
                    'distribution': distribution,
                    'caption': result['captions'][i] if i < len(result['captions']) else '',
                })

            soft_label['caption_distributions'] = caption_distributions

        soft_label['captions'] = result.get('captions', [])
        soft_label['num_captions'] = len(result.get('captions', []))

        return soft_label

    def generate_detection_soft_labels(
        self,
        image_path: str,
        image_id: Optional[str] = None,
        hard_label_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate soft labels for object detection.

        改进：支持传入已有的hard_label结果，避免重复推理

        Args:
            image_path: Path to image
            image_id: Image identifier
            hard_label_result: 已有的hard_label结果（包含objects），避免重复推理

        Returns:
            Soft label dictionary
        """
        self.logger.debug(f"Generating detection soft labels for image {image_id}")

        # 🔧 优先使用已有的hard_label结果，避免重复推理导致结果不一致
        if hard_label_result and 'objects' in hard_label_result:
            objects = hard_label_result['objects']
            self.logger.debug(f"Using existing hard_label result with {len(objects)} objects")
        else:
            # 如果没有hard_label结果，才调用模型
            self.logger.warning(f"No hard_label result provided, calling inference_detection")
            result = self.teacher.inference_detection(
                image=image_path,
                return_logits=False,
                generate_cot=False
            )
            objects = result.get('objects', [])

        soft_label = {
            'image_id': image_id,
            'task': 'detection',
            'temperature': self.temperature,
            'timestamp': datetime.now().isoformat(),
        }

        # 基于检测结果生成分布
        if objects:
            # 为每个检测到的物体生成类别分布
            object_distributions = []

            for obj in objects:
                category = obj.get('category', 'unknown')
                confidence = obj.get('confidence', 0.5)
                bbox = obj.get('bbox', [])

                # 基于置信度生成分布
                # 高置信度物体：主要类别概率高
                # 低置信度物体：分布更均匀
                distribution = self._generate_object_category_distribution(
                    category=category,
                    confidence=confidence,
                    temperature=self.temperature
                )

                object_distributions.append({
                    'category': category,
                    'bbox': bbox,
                    'confidence': confidence,
                    'category_distribution': distribution
                })

            soft_label['object_distributions'] = object_distributions
            self.logger.debug(f"Generated distributions for {len(objects)} objects")

        # 🔧 不再重复保存 objects，避免数据冗余
        # objects 已在 hard_label 中保存，soft_label 只保存分布信息
        soft_label['num_objects'] = len(objects)

        return soft_label

    def _generate_object_category_distribution(
        self,
        category: str,
        confidence: float,
        temperature: float = 1.5
    ) -> Dict[str, float]:
        """
        为检测到的物体生成类别分布。

        改进：使用温度缩放和层次概率分配，避免过度自信

        Args:
            category: 检测到的类别
            confidence: 检测置信度
            temperature: 温度参数（控制分布的平滑程度）

        Returns:
            类别概率分布
        """
        import math

        distribution = {}

        # 🔧 改进1: 使用温度缩放，避免过度自信
        # 高置信度时，也要给其他类别合理的概率
        # temperature越高，分布越平滑

        # 应用温度缩放到置信度
        scaled_confidence = confidence / temperature

        # 🔧 改进2: 考虑类别层次关系
        # 主类别概率 = scaled_confidence * 父类别权重
        # 子类别共享剩余概率

        # 获取相似类别（子类别或兄弟类别）
        similar_categories = self._get_similar_categories(category)

        if similar_categories:
            # 主类别的实际概率（考虑温度缩放）
            main_prob = scaled_confidence * 0.4  # 降低主类别权重，避免过度自信

            # 剩余概率分配给子类别
            remaining_prob = 1.0 - main_prob

            # 🔧 改进3: 不均匀分配，给常见子类别更高概率
            # 例如：person的子类别中，man和woman更常见，child较少
            weights = self._get_subcategory_weights(category, len(similar_categories))

            for i, similar_cat in enumerate(similar_categories):
                distribution[similar_cat] = remaining_prob * weights[i]
        else:
            # 如果没有相似类别，主类别获得大部分概率
            main_prob = scaled_confidence * 0.7
            # 剩余30%分配给"unknown"或"other"
            distribution['other'] = 0.3

        # 添加主类别
        distribution[category] = main_prob

        # 🔧 改进4: 归一化，确保总和为1.0
        total = sum(distribution.values())
        if total > 0:
            distribution = {k: v / total for k, v in distribution.items()}

        return distribution

    def _get_subcategory_weights(self, category: str, num_subcategories: int) -> List[float]:
        """
        获取子类别的权重分配（不均匀）。

        根据类别的常见程度分配不同的权重。

        Args:
            category: 父类别
            num_subcategories: 子类别数量

        Returns:
            权重列表（总和为1.0）
        """
        # 预定义的权重分配（基于常见程度）
        weight_map = {
            'person': [0.35, 0.35, 0.30],  # man, woman, child - man/woman更常见
            'car': [0.4, 0.35, 0.25],      # truck, bus, vehicle
            'bicycle': [0.5, 0.5],          # motorcycle, bike - 两轮车
            'dog': [0.5, 0.5],              # cat, animal
            'chair': [0.35, 0.35, 0.30],    # sofa, couch, seat
        }

        weights = weight_map.get(category.lower())

        if weights and len(weights) == num_subcategories:
            return weights
        else:
            # 默认：均匀分配
            return [1.0 / num_subcategories] * num_subcategories

    def _get_similar_categories(self, category: str) -> List[str]:
        """
        获取相似类别（简化版本）。

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
            # 可以扩展更多
        }

        return similar_map.get(category.lower(), [])

    def _generate_answer_distribution_from_hard_label(
        self,
        answer: str,
        confidence: float,
        temperature: float = 1.5
    ) -> Dict[str, float]:
        """
        基于hard_label答案生成概率分布。

        Args:
            answer: 硬标签答案
            confidence: 置信度
            temperature: 温度参数

        Returns:
            答案概率分布
        """
        distribution = {}

        # 🔧 修复：直接使用confidence作为主答案概率，保持一致性
        # 不再人为放大，确保 confidence 和 distribution 一致
        main_prob = min(confidence, 0.98)  # 上限改为0.98，留2%给其他候选
        distribution[answer.lower()] = main_prob

        # 为相似答案分配剩余概率
        remaining_prob = 1.0 - main_prob

        # 根据答案类型生成相似的候选
        if answer.lower() in ['yes', 'no']:
            # 二元答案：给相反答案分配小概率
            opposite = 'no' if answer.lower() == 'yes' else 'yes'
            distribution[opposite] = remaining_prob * 0.7
            # 其他少量分配
            distribution['maybe'] = remaining_prob * 0.3

        elif answer.isdigit():
            # 数字答案：给相似数字分配小概率
            num = int(answer)
            for offset in [-1, 1]:
                neighbor_num = str(num + offset)
                if neighbor_num not in distribution:
                    distribution[neighbor_num] = remaining_prob * 0.3
            # 文字形式也可能
            word_forms = {'1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five'}
            if answer in word_forms:
                distribution[word_forms[answer]] = remaining_prob * 0.2

        elif answer.lower() in ['one', 'two', 'three', 'four', 'five']:
            # 文字数字：给数字形式分配小概率
            num_forms = {'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5'}
            if answer.lower() in num_forms:
                distribution[num_forms[answer.lower()]] = remaining_prob

        else:
            # 🔧 修复：普通答案（颜色、物体等），分配剩余概率给"other"
            # 这样可以反映概率分布的不确定性
            distribution['other'] = remaining_prob * 0.5
            distribution['unknown'] = remaining_prob * 0.5

        # 如果还有剩余概率，归一化到主答案
        total = sum(distribution.values())
        if total < 1.0:
            # 剩余概率归给主答案
            distribution[answer.lower()] += (1.0 - total)
        elif total > 1.0:
            # 归一化
            distribution = {k: v / total for k, v in distribution.items()}

        return distribution

    def generate_keypoints_soft_labels(
        self,
        image_path: str,
        image_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate soft labels for human pose estimation (keypoints).

        Args:
            image_path: Path to image
            image_id: Image identifier

        Returns:
            Soft label dictionary with keypoints distributions
        """
        self.logger.debug(f"Generating keypoints soft labels for image {image_id}")

        # Get teacher inference with logits
        result = self.teacher.inference_keypoints(
            image=image_path,
            return_logits=True,
            generate_cot=False
        )

        soft_label = {
            'image_id': image_id,
            'task': 'keypoints',
            'temperature': self.temperature,
            'timestamp': datetime.now().isoformat(),
        }

        # Process keypoints logits
        if 'logits' in result:
            keypoints_distributions = self._process_keypoints_logits(result['logits'])
            soft_label['keypoints_distributions'] = keypoints_distributions

        soft_label['persons'] = result.get('persons', [])
        soft_label['num_persons'] = len(result.get('persons', []))

        return soft_label

    def _process_keypoints_logits(
        self,
        logits_data: Dict[str, torch.Tensor]
    ) -> List[Dict[str, Any]]:
        """
        Process keypoints logits for each person.

        Args:
            logits_data: Logits dictionary

        Returns:
            List of keypoints distributions
        """
        probs = logits_data.get('probabilities')

        if probs is None:
            return []

        scaled_probs = self._apply_temperature(probs)
        top_k_probs = self._get_top_k_probabilities(scaled_probs)

        distributions = []

        # Keypoint logits typically involve coordinate predictions
        # For each keypoint, we have x, y coordinate distributions
        distributions.append({
            'coordinate_probabilities': {
                f'coord_{i}': float(prob_val)
                for i, prob_val in enumerate(top_k_probs['values'][:self.top_k])
                if prob_val >= self.min_probability
            },
        })

        return distributions

    def _process_vqa_logits(
        self,
        logits_data: Dict[str, torch.Tensor],
        answer_candidates: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Process VQA logits into answer probability distribution.

        Args:
            logits_data: Dictionary with logits (top_k_indices/top_k_values 或 probabilities)
            answer_candidates: Optional list of answer candidates

        Returns:
            Dictionary mapping answers to probabilities
        """
        distribution = {}

        # 方法1：如果有预定义的答案候选，使用它们
        if answer_candidates:
            for candidate in answer_candidates[:self.top_k]:
                distribution[candidate.lower()] = 1.0 / len(answer_candidates[:self.top_k])
            return distribution

        # 🔧 修复：兼容两种格式
        # 格式1：top_k_indices + top_k_values（teacher_model 返回）
        # 格式2：probabilities + indices（旧格式）
        if 'top_k_indices' in logits_data and 'top_k_values' in logits_data:
            # 新格式：直接使用 top-k
            token_indices = logits_data['top_k_indices']
            token_probs = logits_data['top_k_values']
        elif 'probabilities' in logits_data:
            # 旧格式：计算 top-k
            probs = logits_data['probabilities']
            scaled_probs = self._apply_temperature(probs)
            top_k_result = self._get_top_k_probabilities(scaled_probs)
            token_indices = top_k_result['indices']
            token_probs = top_k_result['values']
        else:
            return {}

        # 提取第一个 token 位置的概率分布
        # 取第一个生成位置
        if token_indices.dim() >= 1 and token_probs.dim() >= 1:
            # 取第一个位置的 top-k
            first_token_indices = token_indices[0] if token_indices.dim() == 1 else token_indices[0, 0]
            first_token_probs = token_probs[0] if token_probs.dim() == 1 else token_probs[0, 0]

            for idx, prob_val in zip(first_token_indices[:self.top_k], first_token_probs[:self.top_k]):
                if prob_val >= self.min_probability:
                    try:
                        word = self.teacher.tokenizer.decode([idx.item()])
                        word = word.strip().lower()
                        if word and word not in ['<s>', '</s>', '<pad>', '<|im', '|>', '<|', '|>']:
                            if len(word) > 1 or word.isdigit():
                                distribution[word] = float(prob_val)
                    except Exception:
                        pass

        # 归一化分布
        if distribution:
            total_prob = sum(distribution.values())
            if total_prob > 0:
                distribution = {k: v / total_prob for k, v in distribution.items()}

        return distribution

        return distribution

    def _process_caption_logits(
        self,
        logits_data: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Process captioning logits.

        Args:
            logits_data: Logits dictionary

        Returns:
            Processed distribution (without top-k data to reduce storage)
        """
        probs = logits_data.get('probabilities')

        if probs is None:
            return {}

        scaled_probs = self._apply_temperature(probs)
        # 🔧 不再保存 top_k_indices 和 top_k_values，减少 JSON 存储大小
        # 这些数据主要用于内部计算，不需要保存到最终输出中
        return {
            'distribution_shape': scaled_probs.shape,
        }

    def _process_detection_logits(
        self,
        logits_data: Dict[str, torch.Tensor]
    ) -> List[Dict[str, Any]]:
        """
        Process detection logits for each object.

        改进：对于Detection任务，不使用文本logits，而是基于检测到的物体置信度生成分布

        Args:
            logits_data: Logits dictionary

        Returns:
            List of object distributions
        """
        # Detection任务不应该使用文本生成的logits
        # 因为输出是结构化的JSON，而不是文本序列

        # 我们应该基于检测到的物体置信度生成分布
        # 但由于这个方法在软标签生成时调用，此时还没有检测结果
        # 所以我们返回空列表，实际分布应该在generate_detection_soft_labels中处理

        return []

    def _apply_temperature(
        self,
        probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply temperature scaling to probabilities.

        Args:
            probs: Probability tensor

        Returns:
            Temperature-scaled probabilities
        """
        # Convert back to logits (approximate)
        logits = torch.log(probs + 1e-10)

        # Apply temperature
        scaled_logits = logits / self.temperature

        # Convert back to probabilities
        scaled_probs = torch.softmax(scaled_logits, dim=-1)

        return scaled_probs

    def _get_top_k_probabilities(
        self,
        probs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Extract top-k probabilities.

        Args:
            probs: Probability tensor

        Returns:
            Dictionary with top-k indices and values
        """
        # Flatten if needed
        if probs.dim() > 1:
            probs_flat = probs.view(-1)
        else:
            probs_flat = probs

        # Get top-k
        top_k = min(self.top_k, probs_flat.size(0))
        top_values, top_indices = torch.topk(probs_flat, top_k)

        return {
            'indices': top_indices,
            'values': top_values,
        }

    def generate_batch_soft_labels(
        self,
        batch_data: Dict[str, Any],
        tasks: List[str]
    ) -> Dict[str, List[Dict]]:
        """
        Generate soft labels for batch of images.

        Args:
            batch_data: Batch data dictionary
            tasks: Tasks to process

        Returns:
            Dictionary with soft labels per task
        """
        results = {task: [] for task in tasks}

        for img_data in batch_data['images']:
            image_id = img_data['id']
            image_path = img_data['path']

            self.logger.info(f"Processing image {image_id} for soft labels")

            if 'vqa' in tasks:
                questions = batch_data['annotations']['vqa'].get(image_id, [])
                for q_data in questions:
                    question = q_data.get('question', '')
                    soft_label = self.generate_vqa_soft_labels(
                        image_path=image_path,
                        question=question,
                        image_id=image_id
                    )
                    results['vqa'].append(soft_label)

            if 'captioning' in tasks:
                soft_label = self.generate_captioning_soft_labels(
                    image_path=image_path,
                    num_captions=3,
                    image_id=image_id
                )
                results['captioning'].append(soft_label)

            if 'detection' in tasks:
                soft_label = self.generate_detection_soft_labels(
                    image_path=image_path,
                    image_id=image_id
                )
                results['detection'].append(soft_label)

            if 'keypoints' in tasks:
                soft_label = self.generate_keypoints_soft_labels(
                    image_path=image_path,
                    image_id=image_id
                )
                results['keypoints'].append(soft_label)

        return results

    def save_soft_labels(
        self,
        soft_labels: Dict[str, Any],
        output_path: str
    ) -> bool:
        """
        Save soft labels to file.

        Args:
            soft_labels: Soft label data
            output_path: Path to save

        Returns:
            True if successful
        """
        try:
            # Convert tensors to lists for JSON serialization
            serializable = self._make_serializable(soft_labels)

            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, 'w', encoding='utf-8') as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Soft labels saved to {path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save soft labels: {e}")
            return False

    def _make_serializable(
        self,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert tensors to serializable format.

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
        Compute statistics from soft labels.

        Args:
            soft_labels_list: List of soft labels

        Returns:
            Statistics dictionary
        """
        stats = {
            'total_count': len(soft_labels_list),
            'by_task': {},
            'average_temperature': self.temperature,
            'total_probabilities': 0,
        }

        for label in soft_labels_list:
            task = label.get('task', 'unknown')
            if task not in stats['by_task']:
                stats['by_task'][task] = 0
            stats['by_task'][task] += 1

            # Count probabilities
            if 'answer_distribution' in label:
                stats['total_probabilities'] += len(label['answer_distribution'])

        return stats

    def __repr__(self) -> str:
        """String representation."""
        return f"SoftLabelGenerator(teacher={self.teacher.model_name}, temp={self.temperature}, top_k={self.top_k})"