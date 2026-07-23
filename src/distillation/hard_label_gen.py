"""
Hard Label Generator
====================

Generates hard labels (final predictions) from teacher model.
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
    Generates hard labels from teacher model predictions.

    Hard labels are the final predictions with confidence scores:
    - VQA: Final answer
    - Captioning: Generated caption
    - Detection: Detected objects with bounding boxes
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

        # Settings
        self.confidence_threshold = self.config.get("distillation.hard_labels.confidence_threshold", 0.7)

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

        # Get teacher model inference WITH logits to compute confidence
        result = self.teacher.inference_vqa(
            image=image_path,
            question=question,
            return_logits=True,  # 获取 logits 供 soft_label 使用
            generate_cot=False
        )

        # Extract hard label
        answer_raw = result.get('answer', '')

        # 🔧 标准化答案格式：将数字转换为英文单词，确保与软标签一致
        # 例如：'1' -> 'one', '2' -> 'two'
        answer = normalize_answer(answer_raw, target_format='word')

        if answer != answer_raw:
            self.logger.debug(f"[Answer Normalization] Hard label: '{answer_raw}' -> '{answer}'")

        hard_label = {
            'answer': answer,
            'confidence': result.get('confidence', 0.0),
        }

        # Filter by confidence if needed
        if hard_label['confidence'] < self.confidence_threshold:
            hard_label['filtered'] = True
            self.logger.debug(f"VQA result filtered: confidence {hard_label['confidence']} < threshold {self.confidence_threshold}")

        return hard_label

    def generate_captioning_hard_labels(
        self,
        image_path: str,
        num_captions: int = 1,
        image_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate hard labels for image captioning.

        Args:
            image_path: Path to image
            num_captions: Number of caption variations
            image_id: Image identifier

        Returns:
            Hard label dictionary
        """
        self.logger.debug(f"Generating captioning hard labels for image {image_id}")

        # Get teacher model inference WITH logits to compute confidence
        result = self.teacher.inference_captioning(
            image=image_path,
            return_logits=True,  # Changed from False to get confidence scores
            generate_cot=False,
            num_captions=num_captions
        )

        # Extract hard labels
        captions = result.get('captions', [])

        hard_label = {
            'captions': captions,
            'num_captions': len(captions),
            'primary_caption': captions[0] if captions else '',
            'confidence': result.get('confidence', 0.0),  # Add confidence
        }

        # Add caption scores/quality metrics (placeholder)
        hard_label['caption_scores'] = [1.0] * len(captions)  # Would be computed from model

        return hard_label

    def generate_detection_hard_labels(
        self,
        image_path: str,
        image_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate hard labels for object detection.

        Args:
            image_path: Path to image
            image_id: Image identifier

        Returns:
            Hard label dictionary with detected objects
        """
        self.logger.debug(f"Generating detection hard labels for image {image_id}")

        # Get teacher model inference WITH logits to compute overall confidence
        result = self.teacher.inference_detection(
            image=image_path,
            return_logits=True,  # Changed from False to get confidence scores
            generate_cot=False
        )

        # Extract hard labels
        objects = result.get('objects', [])

        # Filter objects by confidence
        filtered_objects = []
        for obj in objects:
            confidence = obj.get('confidence', 1.0)
            if confidence >= self.confidence_threshold:
                filtered_objects.append(obj)

        # 🔧 计算 confidence：使用 objects 的平均置信度
        if filtered_objects:
            avg_confidence = sum(obj.get('confidence', 0.9) for obj in filtered_objects) / len(filtered_objects)
        else:
            avg_confidence = 0.0

        hard_label = {
            'objects': filtered_objects,
            'num_objects': len(filtered_objects),
            'total_detected': len(objects),
            'confidence': avg_confidence,  # 使用计算的平均置信度
        }

        return hard_label

    def generate_keypoints_hard_labels(
        self,
        image_path: str,
        image_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate hard labels for human pose estimation (keypoints).

        Args:
            image_path: Path to image
            image_id: Image identifier

        Returns:
            Hard label dictionary with detected persons and their keypoints
        """
        self.logger.debug(f"Generating keypoints hard labels for image {image_id}")

        # Get teacher model inference
        result = self.teacher.inference_keypoints(
            image=image_path,
            return_logits=False,
            generate_cot=False
        )

        # Extract persons with keypoints
        persons = result.get('persons', [])

        # Filter persons by minimum visible keypoints
        filtered_persons = []
        for person in persons:
            keypoints = person.get('keypoints', [])
            visible_count = sum(1 for kp in keypoints if kp.get('visibility', 0) >= 2)
            if visible_count >= 5:  # Minimum 5 visible keypoints
                filtered_persons.append(person)

        hard_label = {
            'persons': filtered_persons,
            'num_persons': len(filtered_persons),
            'total_detected': len(persons),
        }

        return hard_label

    def generate_batch_hard_labels(
        self,
        batch_data: Dict[str, Any],
        tasks: List[str]
    ) -> Dict[str, List[Dict]]:
        """
        Generate hard labels for a batch of images across tasks.

        Args:
            batch_data: Batch data with images and annotations
            tasks: List of tasks to process

        Returns:
            Dictionary with hard labels for each task
        """
        results = {task: [] for task in tasks}

        for img_data in batch_data['images']:
            image_id = img_data['id']
            image_path = img_data['path']

            self.logger.info(f"Processing image {image_id} for hard labels")

            # Process each task
            if 'vqa' in tasks:
                questions = batch_data['annotations']['vqa'].get(image_id, [])
                for q_data in questions:
                    question = q_data.get('question', '')
                    hard_label = self.generate_vqa_hard_labels(
                        image_path=image_path,
                        question=question,
                        image_id=image_id
                    )
                    results['vqa'].append(hard_label)

            if 'captioning' in tasks:
                hard_label = self.generate_captioning_hard_labels(
                    image_path=image_path,
                    num_captions=3,
                    image_id=image_id
                )
                results['captioning'].append(hard_label)

            if 'detection' in tasks:
                hard_label = self.generate_detection_hard_labels(
                    image_path=image_path,
                    image_id=image_id
                )
                results['detection'].append(hard_label)

            if 'keypoints' in tasks:
                hard_label = self.generate_keypoints_hard_labels(
                    image_path=image_path,
                    image_id=image_id
                )
                results['keypoints'].append(hard_label)

        return results

    def format_hard_label_output(
        self,
        hard_labels: Dict[str, Any],
        format_type: str = "json"
    ) -> str:
        """
        Format hard labels for output.

        Args:
            hard_labels: Hard label dictionary
            format_type: Output format (json)

        Returns:
            Formatted string
        """
        if format_type == "json":
            return json.dumps(hard_labels, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def save_hard_labels(
        self,
        hard_labels: Dict[str, Any],
        output_path: str
    ) -> bool:
        """
        Save hard labels to file.

        Args:
            hard_labels: Hard label data
            output_path: Path to save

        Returns:
            True if successful
        """
        try:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            formatted = self.format_hard_label_output(hard_labels)

            with open(path, 'w', encoding='utf-8') as f:
                f.write(formatted)

            self.logger.info(f"Hard labels saved to {path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save hard labels: {e}")
            return False

    def validate_hard_labels(
        self,
        hard_labels: Dict[str, Any]
    ) -> bool:
        """
        Validate hard label structure and content.

        Args:
            hard_labels: Hard label dictionary

        Returns:
            True if valid
        """
        required_keys = ['image_id', 'task', 'timestamp']

        for key in required_keys:
            if key not in hard_labels:
                self.logger.warning(f"Missing required key in hard labels: {key}")
                return False

        # Task-specific validation
        task = hard_labels['task']

        if task == 'vqa':
            if 'answer' not in hard_labels:
                return False
            if not hard_labels['answer']:
                return False

        elif task == 'captioning':
            if 'captions' not in hard_labels or not hard_labels['captions']:
                return False

        elif task == 'detection':
            if 'objects' not in hard_labels:
                return False

        elif task == 'keypoints':
            if 'persons' not in hard_labels:
                return False

        return True

    def merge_hard_labels_with_metadata(
        self,
        hard_labels: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge hard labels with additional metadata.

        Args:
            hard_labels: Hard label data
            metadata: Additional metadata

        Returns:
            Merged dictionary
        """
        merged = hard_labels.copy()
        merged['metadata'] = {
            'teacher_model': self.teacher.model_name,
            'generator': 'HardLabelGenerator',
            'confidence_threshold': self.confidence_threshold,
            **metadata,
        }

        return merged

    def get_statistics(
        self,
        hard_labels_list: List[Dict]
    ) -> Dict[str, Any]:
        """
        Compute statistics from hard labels.

        Args:
            hard_labels_list: List of hard label dictionaries

        Returns:
            Statistics dictionary
        """
        stats = {
            'total_count': len(hard_labels_list),
            'by_task': {},
            'filtered_count': 0,
            'average_confidence': 0.0,
        }

        # Group by task
        for label in hard_labels_list:
            task = label.get('task', 'unknown')
            if task not in stats['by_task']:
                stats['by_task'][task] = 0
            stats['by_task'][task] += 1

            if label.get('filtered', False):
                stats['filtered_count'] += 1

        # Compute average confidence
        confidences = [l.get('confidence', 0) for l in hard_labels_list if 'confidence' in l]
        if confidences:
            stats['average_confidence'] = sum(confidences) / len(confidences)

        return stats

    def __repr__(self) -> str:
        """String representation."""
        return f"HardLabelGenerator(teacher={self.teacher.model_name}, threshold={self.confidence_threshold})"