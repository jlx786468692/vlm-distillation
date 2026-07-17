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
from ..utils.vqa_token_filter import VQATokenFilter


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
        self.top_k = self.config.get("distillation.soft_labels.top_k_logits", 50)  # 🔧 修复：读取top_k_logits
        self.min_probability = self.config.get("distillation.soft_labels.min_probability", 0.01)

        # 🔧 初始化token过滤器（从主配置文件读取路径）
        try:
            self.token_filter = VQATokenFilter()
            self.logger.info("✓ VQA Token过滤器初始化成功")
        except Exception as e:
            self.logger.warning(f"VQA Token过滤器初始化失败: {e}，将不使用任务适配过滤")
            self.token_filter = None

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

        # 🔧 从 hard_label 获取 logits 和答案信息
        if hard_label_result and 'logits' in hard_label_result:
            logits_data = hard_label_result['logits']
            primary_answer = hard_label_result.get('answer', '')
            confidence = hard_label_result.get('confidence', 0.5)

            # 🔧 传入 primary_answer 和 confidence，确保分布合理
            # 🔧 新增：传入question用于上下文感知过滤
            distribution = self._process_vqa_logits(
                logits_data,
                answer_candidates,
                primary_answer=primary_answer,
                confidence=confidence,
                question=question
            )
            soft_label['answer_distribution'] = distribution
            soft_label['primary_answer'] = primary_answer
            soft_label['confidence'] = confidence
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
            # 🔧 新增：传入question用于上下文感知过滤
            distribution = self._process_vqa_logits(
                logits_data,
                answer_candidates,
                question=question
            )
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
        answer_candidates: Optional[List[str]] = None,
        primary_answer: Optional[str] = None,
        confidence: Optional[float] = None,
        question: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Process VQA logits into answer probability distribution.

        改进：
        1. 输入是原始logits（不是概率）
        2. 应用温度缩放后再计算softmax
        3. 标准公式：soft_probs = softmax(logits / temperature)
        4. 🔧 新增：在Logits层级应用Token白名单过滤（字符串判断 + Logits过滤）
        5. 🔧 新增：topk兜底策略（当白名单过滤后为空时）

        Args:
            logits_data: Dictionary with logits (top_k_indices/top_k_values)
            answer_candidates: Optional list of answer candidates
            primary_answer: 模型给出的主要答案（用于验证和保留）
            confidence: 答案的置信度（用于验证）
            question: 问题文本（用于上下文感知过滤）

        Returns:
            Dictionary mapping answers to probabilities
        """
        distribution = {}

        # 方法1：从原始logits提取top-k，应用温度缩放，然后计算概率
        if 'top_k_indices' in logits_data and 'top_k_values' in logits_data:
            token_indices = logits_data['top_k_indices']
            token_logits = logits_data['top_k_values']  # 🔧 现在是logits，不是概率

            # 🔧 修复：正确处理不同维度的tensor
            # 期望形状：[num_tokens, top_k] 或 [top_k]
            if token_indices.dim() >= 1 and token_logits.dim() >= 1:
                # 添加调试日志
                self.logger.debug(f"[VQA Logits] token_indices shape: {token_indices.shape}, token_logits shape: {token_logits.shape}")

                # 取第一个位置（答案的第一个 token）
                if token_indices.dim() == 1:
                    # 已经是 [top_k] 形状，直接使用
                    first_token_indices = token_indices
                    first_token_logits = token_logits
                    self.logger.debug(f"[VQA Logits] Using 1D tensor directly, shape: {first_token_indices.shape}")
                elif token_indices.dim() == 2:
                    # [num_tokens, top_k] 形状，取第一个token
                    first_token_indices = token_indices[0]
                    first_token_logits = token_logits[0]
                    self.logger.debug(f"[VQA Logits] Taking first token from 2D tensor, shape: {first_token_indices.shape}")
                else:
                    # [batch, num_tokens, top_k] 形状，取第一个batch的第一个token
                    first_token_indices = token_indices[0, 0]
                    first_token_logits = token_logits[0, 0]
                    self.logger.debug(f"[VQA Logits] Taking first batch, first token from 3D tensor, shape: {first_token_indices.shape}")

                # 🔧 Step 1: 应用温度缩放到logits
                # 标准公式：soft_probs = softmax(logits / temperature)
                scaled_logits = first_token_logits / self.temperature

                # ===== 🔧 三层防护策略（VQA过滤标准实践） =====
                # 第一层：黑名单（核心防线）- 拦截不可能作为单字答案的Token
                # 第二层：硬标签保护（安全网）- 确保正确答案永不丢失
                # 第三层：Top-K兜底（多样性保障）- 防止过滤后分布过于稀疏
                # ==========================================================

                valid_token_mask = torch.zeros_like(scaled_logits, dtype=torch.bool)
                primary_answer_lower = primary_answer.lower() if primary_answer else None

                # 🔧 新增：获取hard_label对应的token ID（用于第二层保护）
                hard_label_token_ids = set()
                if primary_answer_lower:
                    try:
                        # 将主答案编码为token ID
                        encoded_ids = self.teacher.tokenizer.encode(primary_answer_lower, add_special_tokens=False)
                        hard_label_token_ids = set(encoded_ids)
                        self.logger.debug(f"[Hard Label Protection] Primary answer '{primary_answer}' -> token IDs: {hard_label_token_ids}")
                    except Exception as e:
                        self.logger.warning(f"[Hard Label Protection] Failed to encode primary answer: {e}")

                self.logger.debug(f"[Blacklist Filter] Filtering {len(first_token_indices)} tokens...")

                for i, token_id in enumerate(first_token_indices):
                    try:
                        # 解码token ID到字符串
                        token_str = self.teacher.tokenizer.decode([token_id.item()]).strip()

                        # ===== 🔧 第二层：硬标签保护（安全网） =====
                        # 无论黑名单如何，强制将hard_label_id加入放行列表
                        # 原因：极少数情况下，正确答案的Token可能因为词表构造原因被黑名单误伤
                        if token_id.item() in hard_label_token_ids:
                            valid_token_mask[i] = True
                            self.logger.debug(f"[Hard Label Protection] ✓ Reserved hard label token: '{token_str}' (ID: {token_id.item()})")
                            continue

                        # ===== 🔧 第一层：黑名单（核心防线） =====
                        # 使用VQATokenFilter判断是否有效（已包含BPE碎片、特殊Token、标点等）
                        if self.token_filter and self.token_filter.is_valid_token(token_str, question):
                            valid_token_mask[i] = True
                            self.logger.debug(f"[Blacklist Filter] ✓ Valid token: '{token_str}'")
                        else:
                            self.logger.debug(f"[Blacklist Filter] ✗ Filtered out: '{token_str}'")
                    except Exception as e:
                        self.logger.warning(f"[Blacklist Filter] Failed to decode token {token_id}: {e}")

                # ===== 🔧 第三层：Top-K兜底（多样性保障） =====
                # 如果过滤后剩余Token少于N个（如10个），从原始Top-K中补充
                min_valid_tokens = 10  # 最少保留的有效token数量

                num_valid = valid_token_mask.sum().item()
                self.logger.info(f"[Blacklist Filter] {num_valid}/{len(first_token_indices)} tokens passed blacklist filter")

                # 🔧 第三层逻辑：如果过滤后少于min_valid_tokens个，从Top-K补充
                if num_valid < min_valid_tokens and num_valid > 0:
                    # 有有效token，但数量不足，需要补充
                    self.logger.info(f"[Top-K Fallback] Only {num_valid} tokens remaining, supplementing from Top-{self.top_k}")

                    # 从原始Top-K中补充未被黑名单拦截的token
                    # 计算原始概率分布
                    token_probs_raw = torch.softmax(scaled_logits, dim=-1)

                    # 按概率排序，取Top-50（或更多）
                    top_k_fallback = min(self.top_k * 2, len(first_token_indices))  # 取2倍的top_k作为候选集
                    top_k_indices = torch.topk(token_probs_raw, top_k_fallback).indices

                    # 补充逻辑：从Top-K中添加未被过滤的token
                    for idx in top_k_indices:
                        if not valid_token_mask[idx]:
                            # 检查这个token是否在黑名单中
                            try:
                                token_id = first_token_indices[idx]
                                token_str = self.teacher.tokenizer.decode([token_id.item()]).strip()

                                # 使用较宽松的过滤策略（只过滤绝对噪音）
                                # 注意：这里不使用上下文感知，避免过度过滤
                                if self.token_filter and self.token_filter.is_valid_token(token_str, None):
                                    valid_token_mask[idx] = True
                                    self.logger.debug(f"[Top-K Fallback] + Supplement token: '{token_str}'")

                                    # 检查是否达到最小数量
                                    if valid_token_mask.sum().item() >= min_valid_tokens:
                                        break
                            except Exception as e:
                                self.logger.warning(f"[Top-K Fallback] Failed to decode token: {e}")

                    self.logger.info(f"[Top-K Fallback] After supplementation: {valid_token_mask.sum().item()} tokens")

                # 🔧 应用mask到logits（将无效token的logits设为极小值）
                if num_valid > 0 or valid_token_mask.sum().item() > 0:
                    # 有有效token，应用过滤
                    # 将无效token的logits设为-1e9（softmax后会接近0）
                    scaled_logits_filtered = scaled_logits.clone()
                    scaled_logits_filtered[~valid_token_mask] = -1e9

                    # 计算softmax得到概率
                    token_probs = torch.softmax(scaled_logits_filtered, dim=-1)
                else:
                    # 极端情况：所有token都被过滤（不应该发生，因为有硬标签保护）
                    self.logger.warning(f"[Emergency Fallback] All tokens filtered! Using raw top-{self.top_k}")

                    # 回退到原始top-k策略：保留概率最高的k个token
                    token_probs = torch.softmax(scaled_logits, dim=-1)

                    # 取top-k（保留原始配置的top_k数量）
                    top_k = min(self.top_k, len(token_probs))
                    top_k_indices = torch.topk(token_probs, top_k).indices

                    # 只保留top-k的概率，其余置零
                    token_probs_filtered = torch.zeros_like(token_probs)
                    token_probs_filtered[top_k_indices] = token_probs[top_k_indices]
                    token_probs = token_probs_filtered

                # 🔧 Step 5: 提取并解码
                items = []
                for idx, prob_val in zip(first_token_indices, token_probs):
                    # 过滤掉概率太小的
                    if prob_val < 0.001:  # 过滤掉小于 0.1% 的
                        continue
                    try:
                        word = self.teacher.tokenizer.decode([idx.item()])
                        word = word.strip().lower()
                        # 过滤特殊 token
                        if word and word not in ['<s>', '</s>', '<pad>', '<|im', '|>', '<|', '|>', 'the', 'a', 'an']:
                            # 🔧 新增：过滤下标和上标字符（单字符且非数字）
                            # 这些字符通常是噪音，如：₀₁₂₃₄₅₆₇₈₉ 和 ⁰¹²³⁴⁵⁶⁷⁸⁹
                            if len(word) == 1 and not word.isdigit():
                                # 检查是否是下标或上标数字/字母
                                # Unicode范围：
                                # - 下标数字：U+2080-U+2089
                                # - 上标数字：U+2070, U+00B9, U+00B2, U+00B3, U+2074-U+2079
                                # - 下标字母：U+2090-U+209C
                                # - 上标字母：U+1D43-U+1DBF
                                char_code = ord(word)
                                is_subscript = (0x2080 <= char_code <= 0x2089 or  # 下标数字
                                                0x2090 <= char_code <= 0x209C)    # 下标字母
                                is_superscript = (0x2070 == char_code or           # 上标0
                                                  char_code == 0x00B9 or          # 上标1
                                                  0x00B2 <= char_code <= 0x00B3 or # 上标2-3
                                                  0x2074 <= char_code <= 0x2079 or # 上标4-9
                                                  0x1D43 <= char_code <= 0x1DBF)   # 上标字母
                                if is_subscript or is_superscript:
                                    self.logger.debug(f"[Token Filter] Filtered out subscript/superscript: '{word}' (U+{char_code:04X})")
                                    continue
                            if len(word) > 1 or word.isdigit():
                                items.append((word, float(prob_val)))
                    except Exception:
                        pass

                # 🔧 合并相同词的概率（如 'one' + 'One' = 'one'）
                word_probs = {}
                for word, prob in items:
                    if word in word_probs:
                        word_probs[word] += prob  # 合并
                    else:
                        word_probs[word] = prob

                # 🔧 按概率从大到小排序
                sorted_items = sorted(word_probs.items(), key=lambda x: x[1], reverse=True)

                # 构建 distribution
                for word, prob in sorted_items:
                    distribution[word] = prob

        # 方法2：如果没有 logits，使用 confidence 构建
        elif primary_answer and confidence:
            main_prob = min(confidence, 0.95)
            distribution[primary_answer.lower()] = main_prob
            remaining = 1.0 - main_prob
            if remaining > 0:
                distribution['other'] = remaining

        # 🔧 归一化分布
        if distribution:
            total_prob = sum(distribution.values())
            if total_prob > 0:
                distribution = {k: v / total_prob for k, v in distribution.items()}

        # ===== 🔧 新增：合并等价token的概率（如 '1' 和 'one'） =====
        if distribution and self.token_filter:
            distribution = self.token_filter.merge_equivalent_tokens(distribution)
            self.logger.debug(f"[Token Merge] After merging equivalent tokens: {len(distribution)} unique answers")

        # ===== 🔧 字符串层级二次过滤（可选，作为安全网） =====
        if distribution and self.token_filter:
            # 使用过滤器再次确认（确保万无一失）
            distribution = self.token_filter.filter_distribution(
                distribution=distribution,
                question=question,
                primary_answer=primary_answer,
                min_prob=0.001,
                max_answers=50
            )

            self.logger.debug(
                f"[Token Filter] After secondary filtering: {len(distribution)} tokens remaining, "
                f"primary_answer='{primary_answer}' with prob={distribution.get(primary_answer.lower(), 0):.4f}"
            )

        # ===== 🔧 第四层：任务适配过滤（白名单） =====
        # 根据问题类型应用白名单，过滤掉不属于该任务类型的token
        if distribution and primary_answer and question and self.token_filter:
            # 推断任务类型
            task_type = self.token_filter.infer_task_type(question, primary_answer)

            # 应用任务白名单过滤
            distribution = self.token_filter.filter_by_task_type(
                distribution=distribution,
                task_type=task_type,
                hard_label=primary_answer,
                preserve_hard_label=True
            )

            self.logger.info(
                f"[Task Filter] Task type: {task_type}, "
                f"tokens after filtering: {len(distribution)}, "
                f"primary_answer='{primary_answer}' with prob={distribution.get(primary_answer.lower(), 0):.4f}"
            )

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
        logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply temperature scaling to logits.

        改进：输入是原始logits，直接应用温度缩放

        标准公式：
            soft_logits = logits / temperature
            soft_probs = softmax(soft_logits, dim=-1)

        Args:
            logits: Logits tensor (NOT probabilities)

        Returns:
            Temperature-scaled probabilities
        """
        # 🔧 直接应用温度缩放到logits（标准公式）
        scaled_logits = logits / self.temperature

        # 计算softmax得到概率
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