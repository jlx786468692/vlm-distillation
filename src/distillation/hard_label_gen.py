"""
Hard Label Generator for VQA Tasks
==================================

专注于VQA任务的硬标签生成器。
Optimized for Qwen2.5-VL-32B
"""

import json
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime

from ..models.teacher_model import TeacherModel
from ..utils.config import ConfigManager
from ..utils.logger import get_logger
from ..utils.answer_normalizer import normalize_answer


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

    def generate_vqa_hard_labels(
        self,
        image_path: str,
        question: str,
        image_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate hard labels for VQA task.

        Args:
            image_path: Path to image
            question: Question text
            image_id: Image identifier

        Returns:
            Hard label dictionary (包含 logits 供 soft_label 使用)
        """
        self.logger.debug(f"Generating VQA hard labels for image {image_id}")

        # 获取完整的logits用于软标签
        result = self.teacher.inference_vqa(
            image=image_path,
            question=question,
            return_logits=True,  # 获取 logits 供 soft_label 使用
            generate_cot=False
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

        # 置信度过滤
        if hard_label['confidence'] < self.confidence_threshold:
            hard_label['filtered'] = True
            self.logger.debug(
                f"VQA result filtered: confidence {hard_label['confidence']:.4f} < threshold {self.confidence_threshold}"
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