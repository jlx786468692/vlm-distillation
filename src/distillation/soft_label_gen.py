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
        answer_candidates: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate soft labels for VQA.

        Args:
            image_path: Path to image
            question: Question text
            image_id: Image identifier
            answer_candidates: List of possible answer candidates (optional)

        Returns:
            Soft label dictionary
        """
        self.logger.debug(f"Generating VQA soft labels for image {image_id}")

        # Get teacher inference with logits
        result = self.teacher.inference_vqa(
            image=image_path,
            question=question,
            return_logits=True,
            generate_cot=False
        )

        # Process logits into soft labels
        soft_label = {
            'image_id': image_id,
            'task': 'vqa',
            'question': question,
            'temperature': self.temperature,
            'timestamp': datetime.now().isoformat(),
        }

        # Extract probability distribution
        if 'logits' in result:
            logits_data = result['logits']
            distribution = self._process_vqa_logits(logits_data, answer_candidates)
            soft_label['answer_distribution'] = distribution

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
        image_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate soft labels for object detection.

        Args:
            image_path: Path to image
            image_id: Image identifier

        Returns:
            Soft label dictionary
        """
        self.logger.debug(f"Generating detection soft labels for image {image_id}")

        # Get teacher inference with logits
        result = self.teacher.inference_detection(
            image=image_path,
            return_logits=True,
            generate_cot=False
        )

        soft_label = {
            'image_id': image_id,
            'task': 'detection',
            'temperature': self.temperature,
            'timestamp': datetime.now().isoformat(),
        }

        # Process detection logits
        if 'logits' in result:
            object_distributions = self._process_detection_logits(result['logits'])
            soft_label['object_distributions'] = object_distributions

        soft_label['objects'] = result.get('objects', [])
        soft_label['num_objects'] = len(result.get('objects', []))

        return soft_label

    def _process_vqa_logits(
        self,
        logits_data: Dict[str, torch.Tensor],
        answer_candidates: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Process VQA logits into answer probability distribution.

        Args:
            logits_data: Dictionary with logits and probabilities
            answer_candidates: Optional list of answer candidates

        Returns:
            Dictionary mapping answers to probabilities
        """
        probs = logits_data.get('probabilities')

        if probs is None:
            return {}

        # Apply temperature scaling
        scaled_probs = self._apply_temperature(probs)

        # Get top-k tokens/probabilities
        top_k_probs = self._get_top_k_probabilities(scaled_probs)

        # Map token IDs to answers (would need tokenizer)
        # For now, return as token probabilities
        distribution = {}

        if answer_candidates:
            # Use provided candidates
            # This would require computing probabilities for each candidate
            # Placeholder implementation
            for i, candidate in enumerate(answer_candidates[:self.top_k]):
                distribution[candidate] = 1.0 / len(answer_candidates[:self.top_k])
        else:
            # Use top token probabilities
            for i, prob_val in enumerate(top_k_probs['values'][:self.top_k]):
                if prob_val >= self.min_probability:
                    distribution[f'token_{i}'] = float(prob_val)

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
            Processed distribution
        """
        probs = logits_data.get('probabilities')

        if probs is None:
            return {}

        scaled_probs = self._apply_temperature(probs)
        top_k_probs = self._get_top_k_probabilities(scaled_probs)

        return {
            'top_k_indices': top_k_probs['indices'],
            'top_k_values': top_k_probs['values'],
            'distribution_shape': scaled_probs.shape,
        }

    def _process_detection_logits(
        self,
        logits_data: Dict[str, torch.Tensor]
    ) -> List[Dict[str, Any]]:
        """
        Process detection logits for each object.

        Args:
            logits_data: Logits dictionary

        Returns:
            List of object distributions
        """
        probs = logits_data.get('probabilities')

        if probs is None:
            return []

        scaled_probs = self._apply_temperature(probs)

        # Process logits for detection
        # Detection typically involves object class probabilities
        distributions = []

        # Placeholder - would need to extract per-object class distributions
        top_k_probs = self._get_top_k_probabilities(scaled_probs)

        distributions.append({
            'class_probabilities': {
                f'class_{i}': float(prob_val)
                for i, prob_val in enumerate(top_k_probs['values'][:self.top_k])
                if prob_val >= self.min_probability
            },
        })

        return distributions

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
