"""
Hard Label Generator for VQA Tasks
==================================

专注于VQA任务的硬标签生成器。
Optimized for Qwen2.5-VL-32B

🔧 新增：集成候选集封闭，引导模型生成更精准的logits
"""

import json
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime

from ..models.teacher_model import TeacherModel
from ..utils.config import ConfigManager
from ..utils.logger import get_logger
from ..utils.answer_normalizer import normalize_answer

# 🔧 新增：导入候选集封闭模块
try:
    from tools.candidate.candidate_closure import CandidateClosure
    CANDIDATE_CLOSURE_AVAILABLE = True
except ImportError:
    CANDIDATE_CLOSURE_AVAILABLE = False
    CandidateClosure = None


class HardLabelGenerator:
    """
    Generates hard labels from teacher model predictions for VQA tasks.

    Hard labels are the final predictions with confidence scores.

    Optimized for 32B model:
    - Lower confidence threshold (32B model more reliable)
    - Longer max_new_tokens (32B can generate better reasoning)
    """

    def __init__(
        self,
        teacher_model: TeacherModel,
        config: Optional[ConfigManager] = None
    ):
        """
        Initialize Hard Label Generator.

        Args:
            teacher_model: Teacher model instance
            config: Configuration manager
        """
        self.teacher = teacher_model
        self.config = config or ConfigManager()
        self.logger = get_logger()

        # 置信度阈值（32B模型优化：降低阈值）
        self.confidence_threshold = self.config.get(
            "distillation.hard_labels.confidence_threshold", 0.4
        )

        # 🔧 新增：初始化候选集封闭模块
        try:
            candidate_config = {
                'enable_classifier': self.config.get("distillation.soft_labels.enable_candidate_classifier", False),
                'temperature': self.config.get("distillation.soft_labels.temperature", 2.0),
                'min_probability': self.config.get("distillation.soft_labels.min_probability", 0.01),
                'max_candidates': self.config.get("distillation.soft_labels.max_candidates", 100)
            }

            if CANDIDATE_CLOSURE_AVAILABLE and CandidateClosure:
                self.candidate_closure = CandidateClosure(candidate_config)
                self.logger.info("✓ HardLabelGenerator: 候选集封闭模块初始化成功")
                self.logger.info(f"  VQA词表大小: {len(self.candidate_closure.vqa_vocab)}个")
            else:
                self.candidate_closure = None
                self.logger.warning("HardLabelGenerator: 候选集封闭模块未加载")
        except Exception as e:
            self.logger.warning(f"HardLabelGenerator: 候选集封闭模块初始化失败: {e}")
            self.candidate_closure = None

    def generate_vqa_hard_labels(
        self,
        image_path: str,
        question: str,
        image_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate hard labels for VQA task.

        🔧 改进：集成候选集封闭，引导模型生成更精准的logits

        概念区分：
        - candidate_answers: 从VQA词表得到的预定义答案集，用于引导模型（硬标签阶段）
        - allowed_answers: 从软标签分布提取的可能答案，用于CoT生成（软标签阶段）

        Args:
            image_path: Path to image
            question: Question text
            image_id: Image identifier

        Returns:
            Hard label dictionary (包含 logits 供 soft_label 使用)
        """
        self.logger.debug(f"Generating VQA hard labels for image {image_id}")

        # 🔧 新增：生成候选答案集（用于引导模型）
        # 从VQA词表中选择候选答案，引导模型的logits更集中在有意义的答案上
        candidate_answers = None
        if self.candidate_closure:
            try:
                # 从VQA词表选择Top-K个候选答案
                max_candidates = self.config.get("distillation.hard_labels.max_candidates_for_prompt", 20)
                candidate_answers = self.candidate_closure.vqa_vocab[:max_candidates]
                self.logger.info(f"[Hard Label] 使用VQA词表作为候选答案集: {len(candidate_answers)}个")
                self.logger.debug(f"[Hard Label] 前5个候选: {candidate_answers[:5]}")
            except Exception as e:
                self.logger.warning(f"[Hard Label] 生成候选答案集失败: {e}")
                candidate_answers = None

        # 获取完整的logits用于软标签
        # 🔧 改进：传入候选答案集（candidate_answers），引导模型关注这些答案
        result = self.teacher.inference_vqa(
            image=image_path,
            question=question,
            return_logits=True,  # 获取 logits 供 soft_label 使用
            generate_cot=False,
            candidate_answers=candidate_answers  # 🔧 新增：传入候选答案集（用于硬标签阶段）
        )

        # Extract hard label
        answer_raw = result.get('answer', '')

        # 标准化答案格式：将数字转换为英文单词，确保与软标签一致
        answer = normalize_answer(answer_raw, target_format='word')

        if answer != answer_raw:
            self.logger.debug(f"[Answer Normalization] Hard label: '{answer_raw}' -> '{answer}'")

        hard_label = {
            'answer': answer,
            'confidence': result.get('confidence', 0.0),
        }

        # 🔧 移除置信度过滤：蒸馏阶段不做任何数据清洗和验证
        # 清洗逻辑由下游 cleaning 模块处理（RewardModelScorer）
        # 仅记录低置信度日志，不标记 filtered
        if hard_label['confidence'] < self.confidence_threshold:
            self.logger.debug(
                f"[Low Confidence] {hard_label['confidence']:.4f} < {self.confidence_threshold} (保留样本，由下游清洗模块处理)"
            )

        return hard_label

    def generate_batch_hard_labels(
        self,
        batch_data: Dict[str, Any],
        questions: Dict[int, List[Dict]]
    ) -> Dict[str, List[Dict]]:
        """
        Generate hard labels for batch of images (VQA only).

        Args:
            batch_data: Batch data dictionary
            questions: Dict of image_id -> list of question dicts

        Returns:
            Dictionary with VQA hard label results
        """
        results = {'vqa': []}

        for img_data in batch_data['images']:
            image_id = img_data['id']
            image_path = img_data['path']

            # 只处理VQA任务
            vqa_questions = questions.get(image_id, [])
            for q_data in vqa_questions:
                question = q_data.get('question', '')

                hard_label = self.generate_vqa_hard_labels(
                    image_path=image_path,
                    question=question,
                    image_id=str(image_id)
                )
                results['vqa'].append(hard_label)

        return results