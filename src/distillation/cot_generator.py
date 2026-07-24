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
        # 🔧 新方案：匹配纯文本三段格式
        self.required_keywords_by_task = {
            'vqa': ['observation', 'analysis', 'conclusion', 'final answer'],
            'captioning': ['subject', 'attributes', 'scene'],
            'detection': ['scanning', 'objects', 'verification'],
            'keypoints': ['persons', 'keypoints', 'summary']
        }
        # 通用关键词（用于向后兼容）
        self.required_keywords = self.required_keywords_by_task['vqa']

    def generate_vqa_cot(
        self,
        image_path: str,
        question: str,
        image_id: Optional[str] = None,
        primary_answer: Optional[str] = None,
        allowed_answers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate CoT reasoning for VQA.

        Args:
            image_path: Path to image
            question: Question text
            image_id: Image identifier
            primary_answer: Reference answer from hard_label (for CoT prompt)
            allowed_answers: List of allowed answers from soft_label (for CoT prompt)

        Returns:
            CoT reasoning dictionary
        """
        self.logger.debug(f"Generating VQA CoT for image {image_id}")

        # Get teacher model inference with CoT
        # 🔧 第一层：传入 primary_answer 和 allowed_answers
        # 🔧 优化：使用缓存的视觉特征（第二次推理）
        result = self.teacher.inference_vqa(
            image=image_path,
            question=question,
            return_logits=False,  # 不需要 logits
            generate_cot=True,
            primary_answer=primary_answer,
            allowed_answers=allowed_answers,
            cache_visual=False,  # 不需要再次缓存
            use_cached_visual=True,  # 🔧 使用第一次缓存的视觉特征
            image_id=image_id  # 提供image_id用于缓存查找
        )

        # Extract and structure CoT
        full_response = result.get('full_response', '')

        cot_data = {}

        # Structure the reasoning
        if self.structured_output:
            structured = self._structure_vqa_reasoning(full_response)
            cot_data['structured_reasoning'] = structured

        # Validate reasoning quality
        quality_metrics = self._validate_reasoning_quality(full_response, task='vqa')
        # 🔧 修复：将is_valid放入quality_metrics对象中
        cot_data['quality_metrics'] = {
            'is_valid': quality_metrics.get('is_valid', False)
        }

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

        cot_data = {}

        # Structure reasoning
        if self.structured_output:
            structured = self._structure_captioning_reasoning(cot_response)
            cot_data['structured_reasoning'] = structured

        # Quality metrics
        quality_metrics = self._validate_reasoning_quality(cot_response, task='captioning')
        # 🔧 修复：将is_valid放入quality_metrics对象中
        cot_data['quality_metrics'] = {
            'is_valid': quality_metrics.get('is_valid', False)
        }

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

        cot_data = {}

        # Structure reasoning
        if self.structured_output:
            structured = self._structure_detection_reasoning(full_response)
            # 🔧 _structure_detection_reasoning 已经会处理空的情况，直接使用
            cot_data['structured_reasoning'] = structured

        # Quality metrics
        quality_metrics = self._validate_reasoning_quality(full_response, task='detection')
        # 🔧 修复：将is_valid放入quality_metrics对象中
        cot_data['quality_metrics'] = {
            'is_valid': quality_metrics.get('is_valid', False)
        }

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

        cot_data = {}

        # Structure reasoning
        if self.structured_output:
            structured = self._structure_keypoints_reasoning(full_response)
            cot_data['structured_reasoning'] = structured

        # Quality metrics
        quality_metrics = self._validate_reasoning_quality(full_response, task='keypoints')
        # 🔧 修复：将is_valid放入quality_metrics对象中
        cot_data['quality_metrics'] = {
            'is_valid': quality_metrics.get('is_valid', False)
        }

        return cot_data

    def _structure_keypoints_reasoning(
        self,
        raw_reasoning: str
    ) -> Dict[str, Any]:
        """
        Structure keypoints reasoning.

        🔧 新方案：解析纯三段自然文本格式
        格式：
        Persons:
        [段落内容]

        Keypoints:
        [段落内容]

        Summary:
        [段落内容]

        Args:
            raw_reasoning: Raw reasoning text

        Returns:
            Structured reasoning
        """
        structured = {
            'persons': '',
            'keypoints': '',
            'summary': '',
        }

        # 提取assistant回复
        if 'assistant' in raw_reasoning:
            response_part = raw_reasoning.split('assistant')[-1]
        else:
            response_part = raw_reasoning

        # 定义标签映射
        label_patterns = {
            'persons': [
                r'Persons\s*:',
                r'Step\s*1\s*:',
            ],
            'keypoints': [
                r'Keypoints\s*:',
                r'Step\s*2\s*:',
            ],
            'summary': [
                r'Summary\s*:',
                r'Step\s*3\s*:',
            ],
        }

        import re

        for section, patterns in label_patterns.items():
            for pattern in patterns:
                regex = rf'{pattern}\s*(.*?)(?=(?:Persons|Keypoints|Summary|Step)\s*:|$)'
                match = re.search(regex, response_part, re.DOTALL | re.IGNORECASE)
                if match:
                    content = match.group(1).strip()
                    if content:
                        content = self._clean_reasoning_text(content)
                        structured[section] = content
                        break

        # 如果正则没有匹配到，尝试按行解析
        if not any(structured.values()):
            lines = response_part.split('\n')
            current_section = None
            current_content = []

            for line in lines:
                line_lower = line.lower().strip()

                if 'persons' in line_lower or 'step 1' in line_lower:
                    if current_section and current_content:
                        structured[current_section] = ' '.join(current_content).strip()
                    current_section = 'persons'
                    current_content = []
                elif 'keypoints' in line_lower or 'step 2' in line_lower:
                    if current_section and current_content:
                        structured[current_section] = ' '.join(current_content).strip()
                    current_section = 'keypoints'
                    current_content = []
                elif 'summary' in line_lower or 'step 3' in line_lower:
                    if current_section and current_content:
                        structured[current_section] = ' '.join(current_content).strip()
                    current_section = 'summary'
                    current_content = []
                elif current_section:
                    cleaned_line = line.strip()
                    if cleaned_line and not cleaned_line.startswith('```') and not cleaned_line.startswith('{'):
                        current_content.append(cleaned_line)

            if current_section and current_content:
                structured[current_section] = ' '.join(current_content).strip()

        return structured

    def _structure_vqa_reasoning(
        self,
        raw_reasoning: str
    ) -> Dict[str, Any]:
        """
        Structure VQA reasoning into components.

        🔧 新方案：解析纯三段自然文本格式
        格式：
        Observation:
        [段落内容]

        Analysis:
        [段落内容]

        Conclusion:
        [段落内容]

        Args:
            raw_reasoning: Raw reasoning text

        Returns:
            Structured reasoning dictionary (observation, analysis, conclusion)
        """
        structured = {
            'observation': '',
            'analysis': '',
            'conclusion': '',
        }

        # 提取assistant回复
        if 'assistant' in raw_reasoning:
            response_part = raw_reasoning.split('assistant')[-1]
        else:
            response_part = raw_reasoning

        # 🔧 调试：显示原始回复内容（前500字符）
        self.logger.debug(f"[CoT解析] 原始回复长度: {len(raw_reasoning)}")
        self.logger.debug(f"[CoT解析] 处理后回复 (前500字符): {response_part[:500]}")

        # 🔧 核心逻辑：按标签分割三段文本
        # 匹配模式：Label: 后面跟着段落内容
        # 支持多种标签格式：Observation/Analysis/Conclusion 或 Segment 1/2/3

        # 定义标签映射
        label_patterns = {
            'observation': [
                r'Observation\s*:',
                r'Segment\s*1\s*:',
                r'Step\s*1\s*:',
                r'1\.\s*Observation',
            ],
            'analysis': [
                r'Analysis\s*:',
                r'Segment\s*2\s*:',
                r'Step\s*2\s*:',
                r'2\.\s*Analysis',
            ],
            'conclusion': [
                r'Conclusion\s*:',
                r'Segment\s*3\s*:',
                r'Step\s*3\s*:',
                r'3\.\s*Conclusion',
            ],
        }

        # 使用正则表达式提取每段内容
        import re

        for section, patterns in label_patterns.items():
            for pattern in patterns:
                # 匹配：标签: 后面的内容（直到下一个标签或文本结束）
                # 使用非贪婪匹配，避免跨段落
                regex = rf'{pattern}\s*(.*?)(?=(?:Observation|Analysis|Conclusion|Segment|Step)\s*:|$)'
                match = re.search(regex, response_part, re.DOTALL | re.IGNORECASE)
                if match:
                    content = match.group(1).strip()
                    if content:
                        # 清理内容：移除可能的JSON格式符号
                        content = self._clean_reasoning_text(content)
                        if content:  # 清理后再次检查
                            structured[section] = content
                            self.logger.debug(f"[CoT解析] {section}: 匹配成功，长度={len(content)}")
                            break

        # 如果正则没有匹配到，尝试按行解析
        if not any(structured.values()):
            self.logger.debug(f"[CoT解析] 正则匹配失败，尝试按行解析")
            structured = self._parse_reasoning_by_lines(response_part)

        # 🔧 如果仍然为空，记录警告并显示实际内容
        if not any(structured.values()):
            self.logger.warning(f"[CoT解析] 所有字段为空，模型可能没有按格式输出")
            self.logger.warning(f"[CoT解析] 实际回复内容 (前200字符): {response_part[:200]}")

        return structured

    def _clean_reasoning_text(self, text: str) -> str:
        """
        清理推理文本，移除JSON格式符号。

        Args:
            text: 原始文本

        Returns:
            清理后的纯文本
        """
        import re

        original_text = text  # 保存原始文本用于调试

        # 移除开头的标签示例（如 "Observation:" 本身）
        text = re.sub(r'^(Observation|Analysis|Conclusion|Segment\s*\d+)\s*:\s*', '', text, flags=re.IGNORECASE)

        # 移除可能的JSON格式符号
        # 但保留 "Final Answer: xxx" 这样的自然文本

        # 移除单独的 {} 符号
        text = re.sub(r'\{[^{}]*\}', '', text)

        # 移除 ```json ... ``` 代码块
        text = re.sub(r'```json\s*.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'```\s*.*?```', '', text, flags=re.DOTALL)

        # 移除多余的空白行
        text = re.sub(r'\n\s*\n', '\n', text)

        # 清理开头和结尾的空白
        text = text.strip()

        # 🔧 如果清理后为空，返回原始文本
        if not text:
            self.logger.debug(f"[CoT清理] 清理后文本为空，返回原始文本")
            return original_text.strip()

        return text

    def _parse_reasoning_by_lines(self, text: str) -> Dict[str, str]:
        """
        按行解析推理文本（备用方法）。

        Args:
            text: 原始文本

        Returns:
            结构化的推理字典
        """
        structured = {
            'observation': '',
            'analysis': '',
            'conclusion': '',
        }

        lines = text.split('\n')

        current_section = None
        current_content = []

        for line in lines:
            line_lower = line.lower().strip()

            # 检测段落标签
            if 'observation' in line_lower or 'segment 1' in line_lower or 'step 1' in line_lower:
                if current_section and current_content:
                    structured[current_section] = ' '.join(current_content).strip()
                current_section = 'observation'
                current_content = []
            elif 'analysis' in line_lower or 'segment 2' in line_lower or 'step 2' in line_lower:
                if current_section and current_content:
                    structured[current_section] = ' '.join(current_content).strip()
                current_section = 'analysis'
                current_content = []
            elif 'conclusion' in line_lower or 'segment 3' in line_lower or 'step 3' in line_lower:
                if current_section and current_content:
                    structured[current_section] = ' '.join(current_content).strip()
                current_section = 'conclusion'
                current_content = []
            elif current_section:
                # 收集当前段落的内容
                cleaned_line = line.strip()
                if cleaned_line and not cleaned_line.startswith('```') and not cleaned_line.startswith('{'):
                    current_content.append(cleaned_line)

        # 保存最后一个段落
        if current_section and current_content:
            structured[current_section] = ' '.join(current_content).strip()

        return structured

    def _structure_captioning_reasoning(
        self,
        raw_reasoning: str
    ) -> Dict[str, Any]:
        """
        Structure captioning reasoning.

        🔧 新方案：解析纯三段自然文本格式
        格式：
        Subject:
        [段落内容]

        Attributes:
        [段落内容]

        Scene:
        [段落内容]

        Args:
            raw_reasoning: Raw reasoning text

        Returns:
            Structured reasoning
        """
        structured = {
            'subject': '',
            'attributes': '',
            'scene': '',
        }

        # 提取assistant回复
        if 'assistant' in raw_reasoning:
            response_part = raw_reasoning.split('assistant')[-1]
        else:
            response_part = raw_reasoning

        # 定义标签映射
        label_patterns = {
            'subject': [
                r'Subject\s*:',
                r'Step\s*1\s*:',
            ],
            'attributes': [
                r'Attributes\s*:',
                r'Step\s*2\s*:',
            ],
            'scene': [
                r'Scene\s*:',
                r'Step\s*3\s*:',
            ],
        }

        import re

        for section, patterns in label_patterns.items():
            for pattern in patterns:
                regex = rf'{pattern}\s*(.*?)(?=(?:Subject|Attributes|Scene|Step)\s*:|$)'
                match = re.search(regex, response_part, re.DOTALL | re.IGNORECASE)
                if match:
                    content = match.group(1).strip()
                    if content:
                        content = self._clean_reasoning_text(content)
                        structured[section] = content
                        break

        # 如果正则没有匹配到，尝试按行解析
        if not any(structured.values()):
            lines = response_part.split('\n')
            current_section = None
            current_content = []

            for line in lines:
                line_lower = line.lower().strip()

                if 'subject' in line_lower or 'step 1' in line_lower:
                    if current_section and current_content:
                        structured[current_section] = ' '.join(current_content).strip()
                    current_section = 'subject'
                    current_content = []
                elif 'attributes' in line_lower or 'step 2' in line_lower:
                    if current_section and current_content:
                        structured[current_section] = ' '.join(current_content).strip()
                    current_section = 'attributes'
                    current_content = []
                elif 'scene' in line_lower or 'step 3' in line_lower:
                    if current_section and current_content:
                        structured[current_section] = ' '.join(current_content).strip()
                    current_section = 'scene'
                    current_content = []
                elif current_section:
                    cleaned_line = line.strip()
                    if cleaned_line and not cleaned_line.startswith('```') and not cleaned_line.startswith('{'):
                        current_content.append(cleaned_line)

            if current_section and current_content:
                structured[current_section] = ' '.join(current_content).strip()

        return structured

    def _structure_detection_reasoning(
        self,
        raw_reasoning: str
    ) -> Dict[str, Any]:
        """
        Structure detection reasoning.

        🔧 新方案：解析纯三段自然文本格式
        格式：
        Scanning:
        [段落内容]

        Objects:
        [段落内容]

        Verification:
        [段落内容]

        Args:
            raw_reasoning: Raw reasoning text

        Returns:
            Structured reasoning
        """
        structured = {
            'scanning': '',
            'objects': '',
            'verification': '',
        }

        # 提取assistant回复
        if 'assistant' in raw_reasoning:
            response_part = raw_reasoning.split('assistant')[-1]
        else:
            response_part = raw_reasoning

        # 🔧 如果回答部分主要是JSON，返回说明
        if '```json' in response_part or response_part.strip().startswith('[') or response_part.strip().startswith('{'):
            return {"note": "Model returned direct JSON output without step-by-step reasoning"}

        # 定义标签映射
        label_patterns = {
            'scanning': [
                r'Scanning\s*:',
                r'Step\s*1\s*:',
            ],
            'objects': [
                r'Objects\s*:',
                r'Step\s*2\s*:',
            ],
            'verification': [
                r'Verification\s*:',
                r'Step\s*3\s*:',
            ],
        }

        import re

        for section, patterns in label_patterns.items():
            for pattern in patterns:
                regex = rf'{pattern}\s*(.*?)(?=(?:Scanning|Objects|Verification|Step)\s*:|$)'
                match = re.search(regex, response_part, re.DOTALL | re.IGNORECASE)
                if match:
                    content = match.group(1).strip()
                    if content:
                        content = self._clean_reasoning_text(content)
                        structured[section] = content
                        break

        # 如果正则没有匹配到，尝试按行解析
        if not any(structured.values()):
            lines = response_part.split('\n')
            current_section = None
            current_content = []

            for line in lines:
                line_lower = line.lower().strip()

                if 'scanning' in line_lower or 'step 1' in line_lower:
                    if current_section and current_content:
                        structured[current_section] = ' '.join(current_content).strip()
                    current_section = 'scanning'
                    current_content = []
                elif 'objects' in line_lower or 'step 2' in line_lower:
                    if current_section and current_content:
                        structured[current_section] = ' '.join(current_content).strip()
                    current_section = 'objects'
                    current_content = []
                elif 'verification' in line_lower or 'step 3' in line_lower:
                    if current_section and current_content:
                        structured[current_section] = ' '.join(current_content).strip()
                    current_section = 'verification'
                    current_content = []
                elif current_section:
                    cleaned_line = line.strip()
                    if cleaned_line and not cleaned_line.startswith('```') and not cleaned_line.startswith('{'):
                        current_content.append(cleaned_line)

            if current_section and current_content:
                structured[current_section] = ' '.join(current_content).strip()

        return structured

    def _validate_reasoning_quality(
        self,
        reasoning: str,
        task: str = 'vqa'
    ) -> Dict[str, Any]:
        """
        Validate quality of reasoning chain.

        🔧 新方案：验证纯文本三段格式
        - VQA: Observation, Analysis, Conclusion
        - Captioning: Subject, Attributes, Scene
        - Detection: Scanning, Objects, Verification
        - Keypoints: Persons, Keypoints, Summary

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

        # Estimate step count - 检测段落标签数量
        # VQA: Observation, Analysis, Conclusion
        # Detection: Scanning, Objects, Verification
        step_patterns = [
            r'(?:Observation|Analysis|Conclusion)',
            r'(?:Scanning|Objects|Verification)',
            r'(?:Subject|Attributes|Scene)',
            r'(?:Persons|Keypoints|Summary)',
        ]

        max_step_count = 0
        for pattern in step_patterns:
            matches = re.findall(pattern, reasoning, re.IGNORECASE)
            max_step_count = max(max_step_count, len(matches))

        metrics['step_count'] = max_step_count

        # Compute logical flow score
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
        if metrics['step_count'] >= 3:
            score += 0.4
        elif metrics['step_count'] >= 2:
            score += 0.3
        elif metrics['step_count'] >= 1:
            score += 0.2

        metrics['logical_flow_score'] = min(score, 1.0)

        # Determine validity - 有关键词且有合理长度
        metrics['is_valid'] = (
            metrics['length'] >= 50 and
            metrics['has_required_keywords']
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