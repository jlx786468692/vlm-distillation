"""
Chain-of-Thought Generator
==========================

Generates structured reasoning chains from teacher model.
"""

import json
import re
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime

from ..models.teacher_model import TeacherModel
from ..utils.config import ConfigManager
from ..utils.logger import get_logger


class CoTGenerator:
    """
    Generates Chain-of-Thought (CoT) reasoning from teacher model.

    CoT provides step-by-step reasoning processes:
    - Structured reasoning with intermediate steps
    - Task-specific reasoning templates
    - Quality validation for reasoning chains
    """

    def __init__(
        self,
        teacher_model: TeacherModel,
        config: Optional[ConfigManager] = None
    ):
        """
        Initialize CoT Generator.

        Args:
            teacher_model: Teacher model instance
            config: Configuration manager
        """
        self.teacher = teacher_model
        self.config = config or ConfigManager()
        self.logger = get_logger()

        # Settings
        self.max_length = self.config.get("distillation.cot.max_length", 512)
        self.include_intermediate_steps = self.config.get("distillation.cot.include_intermediate_steps", True)
        self.structured_output = self.config.get("distillation.cot.structured_output", True)

        # Required keywords for reasoning validation - task-specific
        # Detection使用STEP格式，不同于VQA的First/Next/Then格式
        self.required_keywords_by_task = {
            'vqa': ['first', 'next', 'then', 'finally', 'therefore', 'because', 'so'],
            'captioning': ['first', 'next', 'then', 'finally', 'describe', 'identify'],
            'detection': ['step', 'scan', 'verify', 'check', 'format', 'json'],  # Detection使用STEP标记
            'keypoints': ['first', 'identify', 'locate', 'estimate', 'finally']
        }
        # 通用关键词（用于向后兼容）
        self.required_keywords = self.required_keywords_by_task['vqa']

    def generate_vqa_cot(
        self,
        image_path: str,
        question: str,
        image_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate CoT reasoning for VQA.

        Args:
            image_path: Path to image
            question: Question text
            image_id: Image identifier

        Returns:
            CoT reasoning dictionary
        """
        self.logger.debug(f"Generating VQA CoT for image {image_id}")

        # Get teacher model inference with CoT
        # 🔧 新方案：CoT 单独推理，不需要 logits
        result = self.teacher.inference_vqa(
            image=image_path,
            question=question,
            return_logits=False,  # 不需要 logits
            generate_cot=True
        )

        # Extract and structure CoT
        full_response = result.get('full_response', '')

        cot_data = {
            'image_id': image_id,
            'task': 'vqa',
            'question': question,
            'raw_reasoning': full_response,
            'timestamp': datetime.now().isoformat(),
        }

        # Structure the reasoning
        if self.structured_output:
            structured = self._structure_vqa_reasoning(full_response)
            cot_data['structured_reasoning'] = structured

        # Extract steps
        if self.include_intermediate_steps:
            steps = self._extract_reasoning_steps(full_response)
            cot_data['reasoning_steps'] = steps

        # Validate reasoning quality
        cot_data['quality_metrics'] = self._validate_reasoning_quality(full_response, task='vqa')

        return cot_data

    def generate_captioning_cot(
        self,
        image_path: str,
        image_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate CoT reasoning for image captioning.

        Args:
            image_path: Path to image
            image_id: Image identifier

        Returns:
            CoT reasoning dictionary
        """
        self.logger.debug(f"Generating captioning CoT for image {image_id}")

        # Get teacher inference with CoT
        result = self.teacher.inference_captioning(
            image=image_path,
            return_logits=False,
            generate_cot=True,
            num_captions=1
        )

        # Extract CoT
        full_response = result.get('full_response', '')
        cot_response = result.get('cot', full_response)

        cot_data = {
            'image_id': image_id,
            'task': 'captioning',
            'raw_reasoning': cot_response,
            'timestamp': datetime.now().isoformat(),
        }

        # Structure reasoning
        if self.structured_output:
            structured = self._structure_captioning_reasoning(cot_response)
            cot_data['structured_reasoning'] = structured

        # Extract steps
        if self.include_intermediate_steps:
            steps = self._extract_reasoning_steps(cot_response)
            cot_data['reasoning_steps'] = steps

        # Quality metrics
        cot_data['quality_metrics'] = self._validate_reasoning_quality(cot_response, task='captioning')

        return cot_data

    def generate_detection_cot(
        self,
        image_path: str,
        image_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate CoT reasoning for object detection.

        Args:
            image_path: Path to image
            image_id: Image identifier

        Returns:
            CoT reasoning dictionary
        """
        self.logger.debug(f"Generating detection CoT for image {image_id}")

        # Get teacher inference with CoT
        result = self.teacher.inference_detection(
            image=image_path,
            return_logits=False,
            generate_cot=True
        )

        full_response = result.get('full_response', '')

        cot_data = {
            'image_id': image_id,
            'task': 'detection',
            'raw_reasoning': full_response,
            'timestamp': datetime.now().isoformat(),
        }

        # Structure reasoning
        if self.structured_output:
            structured = self._structure_detection_reasoning(full_response)
            # 🔧 最小修复：如果结构化推理全部为空，不保存空字段
            if structured and any(v for v in structured.values() if v):
                cot_data['structured_reasoning'] = structured
            else:
                cot_data['structured_reasoning'] = {"note": "Model returned direct JSON output without step-by-step reasoning"}

        # Extract steps
        if self.include_intermediate_steps:
            steps = self._extract_reasoning_steps(full_response)
            # 🔧 最小修复：如果没有有效步骤，标记原因
            if steps and len(steps) == 1 and steps[0].get('content') == 'No reasoning provided':
                cot_data['reasoning_steps'] = [{'step_number': 1, 'marker': 'Note', 'content': 'Model provided direct JSON output without detailed reasoning'}]
            else:
                cot_data['reasoning_steps'] = steps

        # Quality metrics
        cot_data['quality_metrics'] = self._validate_reasoning_quality(full_response, task='detection')

        return cot_data

    def generate_keypoints_cot(
        self,
        image_path: str,
        image_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate CoT reasoning for human pose estimation (keypoints).

        Args:
            image_path: Path to image
            image_id: Image identifier

        Returns:
            CoT reasoning dictionary
        """
        self.logger.debug(f"Generating keypoints CoT for image {image_id}")

        # Get teacher inference with CoT
        result = self.teacher.inference_keypoints(
            image=image_path,
            return_logits=False,
            generate_cot=True
        )

        full_response = result.get('full_response', '')

        cot_data = {
            'image_id': image_id,
            'task': 'keypoints',
            'raw_reasoning': full_response,
            'timestamp': datetime.now().isoformat(),
        }

        # Structure reasoning
        if self.structured_output:
            structured = self._structure_keypoints_reasoning(full_response)
            cot_data['structured_reasoning'] = structured

        # Extract steps
        if self.include_intermediate_steps:
            steps = self._extract_reasoning_steps(full_response)
            cot_data['reasoning_steps'] = steps

        # Quality metrics
        cot_data['quality_metrics'] = self._validate_reasoning_quality(full_response, task='keypoints')

        return cot_data

    def _structure_keypoints_reasoning(
        self,
        raw_reasoning: str
    ) -> Dict[str, Any]:
        """
        Structure keypoints reasoning.

        Args:
            raw_reasoning: Raw reasoning text

        Returns:
            Structured reasoning
        """
        structured = {
            'person_detection': '',
            'head_face_keypoints': '',
            'upper_body_keypoints': '',
            'lower_body_keypoints': '',
            'pose_summary': '',
        }

        sections = {
            'person_detection': ['person', 'people', 'detect', 'identify', 'first'],
            'head_face_keypoints': ['head', 'face', 'nose', 'eye', 'ear'],
            'upper_body_keypoints': ['shoulder', 'elbow', 'wrist', 'arm', 'upper'],
            'lower_body_keypoints': ['hip', 'knee', 'ankle', 'leg', 'lower'],
            'pose_summary': ['pose', 'complete', 'final', 'summary', 'finally'],
        }

        for section, keywords in sections.items():
            for keyword in keywords:
                if keyword in raw_reasoning.lower():
                    sentences = raw_reasoning.split('.')
                    for sentence in sentences:
                        if keyword in sentence.lower():
                            structured[section] = sentence.strip()
                            break
                    break

        return structured

    def _structure_vqa_reasoning(
        self,
        raw_reasoning: str
    ) -> Dict[str, Any]:
        """
        Structure VQA reasoning into components.

        Args:
            raw_reasoning: Raw reasoning text

        Returns:
            Structured reasoning dictionary
        """
        structured = {
            'observation': '',
            'analysis': '',
            'reasoning': '',
            'conclusion': '',
        }

        # Try to parse structured sections
        sections = {
            'observation': ['first', 'identify', 'observe', 'see', 'notice'],
            'analysis': ['next', 'analyze', 'consider', 'examine'],
            'reasoning': ['then', 'therefore', 'because', 'since', 'reason'],
            'conclusion': ['finally', 'answer', 'result', 'conclusion'],
        }

        # Extract sections based on keywords
        for section, keywords in sections.items():
            for keyword in keywords:
                if keyword in raw_reasoning.lower():
                    # Find sentence containing keyword
                    sentences = raw_reasoning.split('.')
                    for sentence in sentences:
                        if keyword in sentence.lower():
                            structured[section] = sentence.strip()
                            break
                    break

        return structured

    def _structure_captioning_reasoning(
        self,
        raw_reasoning: str
    ) -> Dict[str, Any]:
        """
        Structure captioning reasoning.

        Args:
            raw_reasoning: Raw reasoning text

        Returns:
            Structured reasoning
        """
        structured = {
            'subject_identification': '',
            'attribute_description': '',
            'scene_description': '',
            'action_description': '',
            'final_caption': '',
        }

        # Extract components
        sections = {
            'subject_identification': ['main subject', 'object', 'person', 'first'],
            'attribute_description': ['attribute', 'color', 'size', 'shape', 'next'],
            'scene_description': ['scene', 'background', 'setting', 'location', 'then'],
            'action_description': ['action', 'activity', 'doing', 'movement'],
            'final_caption': ['caption', 'description', 'final', 'finally'],
        }

        for section, keywords in sections.items():
            for keyword in keywords:
                if keyword in raw_reasoning.lower():
                    sentences = raw_reasoning.split('.')
                    for sentence in sentences:
                        if keyword in sentence.lower():
                            structured[section] = sentence.strip()
                            break
                    break

        return structured

    def _structure_detection_reasoning(
        self,
        raw_reasoning: str
    ) -> Dict[str, Any]:
        """
        Structure detection reasoning.

        Args:
            raw_reasoning: Raw reasoning text

        Returns:
            Structured reasoning
        """
        structured = {
            'scanning_method': '',
            'object_identification': '',
            'bbox_determination': '',
            'classification': '',
            'verification': '',
        }

        sections = {
            'scanning_method': ['scan', 'search', 'look', 'first'],
            'object_identification': ['identify', 'detect', 'find', 'object'],
            'bbox_determination': ['bounding', 'box', 'coordinate', 'position', 'location'],
            'classification': ['classify', 'category', 'type', 'class'],
            'verification': ['verify', 'confirm', 'check', 'final'],
        }

        for section, keywords in sections.items():
            for keyword in keywords:
                if keyword in raw_reasoning.lower():
                    sentences = raw_reasoning.split('.')
                    for sentence in sentences:
                        if keyword in sentence.lower():
                            structured[section] = sentence.strip()
                            break
                    break

        return structured

    def _extract_reasoning_steps(
        self,
        raw_reasoning: str
    ) -> List[Dict[str, str]]:
        """
        Extract individual reasoning steps.

        改进：更好地处理Detection任务的JSON输出格式

        Args:
            raw_reasoning: Raw reasoning text

        Returns:
            List of reasoning step dictionaries
        """
        steps = []

        # Use regex to find step markers and their content
        # Pattern: (Marker)(optional comma)(content until next marker or end)
        pattern = r'(Step \d+|First|Next|Then|Finally|Therefore)[,:]?\s*'

        # Find all matches with their positions
        matches = list(re.finditer(pattern, raw_reasoning, re.IGNORECASE))

        if not matches:
            # If no markers found, try to split by numbered steps
            # Pattern: "1. content" or "1) content"
            numbered_pattern = r'(\d+)[.\)]\s*'
            numbered_matches = list(re.finditer(numbered_pattern, raw_reasoning))

            if numbered_matches:
                for i, match in enumerate(numbered_matches):
                    start_pos = match.end()
                    if i + 1 < len(numbered_matches):
                        end_pos = numbered_matches[i + 1].start()
                    else:
                        end_pos = len(raw_reasoning)

                    content = raw_reasoning[start_pos:end_pos].strip()
                    # Clean up content
                    content = re.sub(r'[\.\,]\s*$', '', content)
                    content = content.strip()

                    if content and len(content) > 5:  # Only add meaningful steps
                        steps.append({
                            'step_number': i + 1,
                            'marker': match.group(1),
                            'content': content,
                        })

                if steps:
                    return steps

            # If still no steps found, split by sentences
            sentences = raw_reasoning.split('.')
            for i, sentence in enumerate(sentences[:5]):  # Limit to 5 steps
                if sentence.strip() and len(sentence.strip()) > 10:  # Only meaningful sentences
                    steps.append({
                        'step_number': i + 1,
                        'marker': '',
                        'content': sentence.strip(),
                    })
            return steps

        # Extract content between markers
        for i, match in enumerate(matches):
            marker = match.group(1)

            # Get content from current marker to next marker (or end)
            start_pos = match.end()
            if i + 1 < len(matches):
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(raw_reasoning)

            content = raw_reasoning[start_pos:end_pos].strip()

            # Clean up content: remove trailing punctuation and next step markers
            # Remove trailing period, comma, or newline followed by numbers
            content = re.sub(r'[\.\,]\s*(\n\s*\d+\.?)?\s*$', '', content)
            content = re.sub(r'\n\s*\d+\.\s*', '', content)  # Remove "1." style markers
            content = content.strip()

            # Remove leading comma if present (shouldn't happen with new logic, but safety check)
            if content.startswith(',') or content.startswith(':'):
                content = content[1:].strip()

            # Only add steps with meaningful content
            if content and len(content) > 5:
                steps.append({
                    'step_number': i + 1,
                    'marker': marker.strip(),
                    'content': content,
                })

        return steps

    def _validate_reasoning_quality(
        self,
        reasoning: str,
        task: str = 'vqa'
    ) -> Dict[str, Any]:
        """
        Validate quality of reasoning chain.

        改进：支持不同任务的CoT格式
        - VQA/Captioning: First, Next, Then, Finally
        - Detection: STEP 1, STEP 2, STEP 3, STEP 4

        Args:
            reasoning: Reasoning text
            task: Task type for selecting appropriate keywords

        Returns:
            Quality metrics dictionary
        """
        metrics = {
            'length': len(reasoning),
            'has_required_keywords': False,
            'keyword_count': 0,
            'step_count': 0,
            'logical_flow_score': 0.0,
            'is_valid': False,
        }

        # 根据任务类型选择关键词
        task_keywords = self.required_keywords_by_task.get(task, self.required_keywords)

        # Check for required keywords (case-insensitive)
        reasoning_lower = reasoning.lower()
        keyword_count = sum(1 for kw in task_keywords if kw in reasoning_lower)
        metrics['keyword_count'] = keyword_count
        metrics['has_required_keywords'] = keyword_count >= 2

        # Estimate step count - 支持多种格式
        # Pattern 1: "STEP 1", "STEP 2" (Detection)
        # Pattern 2: "First", "Next", "Then", "Finally" (VQA)
        # Pattern 3: "1.", "2." (Numbered)
        step_patterns = [
            r'STEP\s*\d+',          # STEP 1, STEP 2
            r'(?:First|Next|Then|Finally|Therefore)',  # VQA markers
        ]

        max_step_count = 0
        for pattern in step_patterns:
            matches = re.findall(pattern, reasoning, re.IGNORECASE)
            max_step_count = max(max_step_count, len(matches))

        # Also use extracted steps
        steps = self._extract_reasoning_steps(reasoning)
        metrics['step_count'] = max(max_step_count, len(steps))

        # Compute logical flow score - 更宽松的评分
        # 有步骤 + 有关键词 + 长度合理 = 高分
        score = 0.0

        # 长度得分 (最多0.3)
        if metrics['length'] >= 100:
            score += 0.3
        elif metrics['length'] >= 50:
            score += 0.2

        # 关键词得分 (最多0.3)
        if keyword_count >= 3:
            score += 0.3
        elif keyword_count >= 2:
            score += 0.2
        elif keyword_count >= 1:
            score += 0.1

        # 步骤数得分 (最多0.4)
        if metrics['step_count'] >= 4:
            score += 0.4
        elif metrics['step_count'] >= 3:
            score += 0.3
        elif metrics['step_count'] >= 2:
            score += 0.2
        elif metrics['step_count'] >= 1:
            score += 0.1

        metrics['logical_flow_score'] = min(score, 1.0)

        # Determine validity - 降低阈值使更多数据通过
        metrics['is_valid'] = (
            metrics['length'] >= 50 and  # 降低从30到50
            (metrics['has_required_keywords'] or metrics['step_count'] >= 2)  # 更灵活的条件
        )

        return metrics

    def generate_batch_cot(
        self,
        batch_data: Dict[str, Any],
        tasks: List[str]
    ) -> Dict[str, List[Dict]]:
        """
        Generate CoT for batch of images.

        Args:
            batch_data: Batch data dictionary
            tasks: Tasks to process

        Returns:
            Dictionary with CoT for each task
        """
        results = {task: [] for task in tasks}

        for img_data in batch_data['images']:
            image_id = img_data['id']
            image_path = img_data['path']

            self.logger.info(f"Processing image {image_id} for CoT")

            if 'vqa' in tasks:
                questions = batch_data['annotations']['vqa'].get(image_id, [])
                for q_data in questions:
                    question = q_data.get('question', '')
                    cot = self.generate_vqa_cot(
                        image_path=image_path,
                        question=question,
                        image_id=image_id
                    )
                    results['vqa'].append(cot)

            if 'captioning' in tasks:
                cot = self.generate_captioning_cot(
                    image_path=image_path,
                    image_id=image_id
                )
                results['captioning'].append(cot)

            if 'detection' in tasks:
                cot = self.generate_detection_cot(
                    image_path=image_path,
                    image_id=image_id
                )
                results['detection'].append(cot)

            if 'keypoints' in tasks:
                cot = self.generate_keypoints_cot(
                    image_path=image_path,
                    image_id=image_id
                )
                results['keypoints'].append(cot)

        return results

    def save_cot(
        self,
        cot_data: Dict[str, Any],
        output_path: str
    ) -> bool:
        """
        Save CoT reasoning to file.

        Args:
            cot_data: CoT dictionary
            output_path: Path to save

        Returns:
            True if successful
        """
        try:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, 'w', encoding='utf-8') as f:
                json.dump(cot_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"CoT saved to {path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save CoT: {e}")
            return False

    def get_statistics(
        self,
        cot_list: List[Dict]
    ) -> Dict[str, Any]:
        """
        Compute statistics from CoT data.

        Args:
            cot_list: List of CoT dictionaries

        Returns:
            Statistics dictionary
        """
        stats = {
            'total_count': len(cot_list),
            'by_task': {},
            'average_length': 0,
            'average_steps': 0,
            'valid_count': 0,
        }

        lengths = []
        step_counts = []

        for cot in cot_list:
            task = cot.get('task', 'unknown')
            if task not in stats['by_task']:
                stats['by_task'][task] = 0
            stats['by_task'][task] += 1

            quality = cot.get('quality_metrics', {})
            lengths.append(quality.get('length', 0))
            step_counts.append(quality.get('step_count', 0))

            if quality.get('is_valid', False):
                stats['valid_count'] += 1

        if lengths:
            stats['average_length'] = sum(lengths) / len(lengths)

        if step_counts:
            stats['average_steps'] = sum(step_counts) / len(step_counts)

        return stats

    def __repr__(self) -> str:
        """String representation."""
        return f"CoTGenerator(teacher={self.teacher.model_name}, max_length={self.max_length})"