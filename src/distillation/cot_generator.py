"""
CoT Generator for VQA Tasks
============================

专注于VQA任务的思维链（Chain-of-Thought）生成器。
"""

from datetime import datetime
from typing import Dict, Any, Optional, List

from ..models.teacher_model import TeacherModel
from ..utils.config import ConfigManager
from ..utils.logger import get_logger

# 🔧 新增：导入后置文本清洗模块
try:
    from ..cleaning.text_cleaner import clean_text, TextCleaner
    TEXT_CLEANER_AVAILABLE = True
    text_cleaner = TextCleaner()
except ImportError:
    TEXT_CLEANER_AVAILABLE = False
    text_cleaner = None


class CoTGenerator:
    """
    VQA任务的CoT生成器

    功能：
    - 为VQA问题生成结构化推理过程
    - 验证推理质量
    - 输出Observation/Analysis/Conclusion结构
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

        # CoT parameters
        self.structured_output = self.config.get("distillation.cot.structured_output", True)

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
        result = self.teacher.inference_vqa(
            image=image_path,
            question=question,
            return_logits=False,  # 不需要 logits
            generate_cot=True,
            primary_answer=primary_answer,
            allowed_answers=allowed_answers,
            cache_visual=False,  # 不需要再次缓存
            use_cached_visual=True,  # 使用缓存的视觉特征
            image_id=image_id
        )

        # Extract and structure CoT
        full_response = result.get('full_response', '')

        # 🔧 调试日志：查看原始 CoT 响应
        self.logger.debug(f"[CoT] Raw full_response length: {len(full_response)}")
        self.logger.debug(f"[CoT] full_response (first 500 chars): {full_response[:500]}")

        cot_data = {}

        # Structure the reasoning
        if self.structured_output:
            structured = self._structure_vqa_reasoning(full_response)
            cot_data['structured_reasoning'] = structured

        # 🔧 移除验证操作：蒸馏阶段不做任何数据清洗和验证
        # 清洗逻辑由下游 cleaning 模块处理（RewardModelScorer）

        return cot_data

    def _structure_vqa_reasoning(self, raw_reasoning: str) -> Dict[str, str]:
        """
        Structure VQA reasoning.

        Args:
            raw_reasoning: Raw reasoning text

        Returns:
            Structured reasoning with observation/analysis/conclusion
        """
        structured = {
            'observation': '',
            'analysis': '',
            'conclusion': ''
        }

        # 🔧 调试日志：查看原始推理文本
        self.logger.debug(f"[Structure] Raw reasoning length: {len(raw_reasoning)}")
        if not raw_reasoning or len(raw_reasoning.strip()) < 10:
            self.logger.warning(f"[Structure] Empty or too short reasoning: '{raw_reasoning[:100]}'")
            return structured

        # 提取assistant回复
        if 'assistant' in raw_reasoning:
            response_part = raw_reasoning.split('assistant')[-1]
            self.logger.debug(f"[Structure] Extracted assistant response (first 200 chars): {response_part[:200]}")
        else:
            response_part = raw_reasoning
            self.logger.debug(f"[Structure] No 'assistant' found, using full text")

        # 定义标签映射
        label_patterns = {
            'observation': [r'Observation\s*:', r'Step\s*1\s*:'],
            'analysis': [r'Analysis\s*:', r'Step\s*2\s*:'],
            'conclusion': [r'Conclusion\s*:', r'Step\s*3\s*:', r'Final Answer\s*:']
        }

        import re

        # 提取各部分内容
        for key, patterns in label_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, response_part, re.IGNORECASE)
                if match:
                    start_idx = match.end()
                    # 找到下一个标签或文本结束
                    next_labels = ['Observation:', 'Analysis:', 'Conclusion:', 'Final Answer:']
                    end_idx = len(response_part)
                    for label in next_labels:
                        next_match = re.search(label, response_part[start_idx:], re.IGNORECASE)
                        if next_match:
                            end_idx = start_idx + next_match.start()
                            break

                    content = response_part[start_idx:end_idx].strip()

                    # 🔧 新增：后置文本清洗（移除 Markdown 符号和多余引号）
                    if TEXT_CLEANER_AVAILABLE and content:
                        # 先清洗闭合问题 CoT 中的引号
                        content = text_cleaner.clean_cot_quotes(content)
                        # 再清洗其他 Markdown 符号
                        content = clean_text(content)

                    if content:
                        structured[key] = content
                        self.logger.debug(f"[Structure] Extracted {key}: {content[:100]}")
                        break

        # 🔧 调试日志：显示最终解析结果
        self.logger.debug(f"[Structure] Final structured result: obs={bool(structured['observation'])}, ana={bool(structured['analysis'])}, con={bool(structured['conclusion'])}")

        return structured

    def _validate_reasoning_quality(
        self,
        reasoning: str,
        task: str = 'vqa'
    ) -> Dict[str, bool]:
        """
        Validate reasoning quality.

        Args:
            reasoning: Raw reasoning text
            task: Task type (only 'vqa' supported)

        Returns:
            Quality metrics
        """
        metrics = {
            'is_valid': False
        }

        if not reasoning or len(reasoning.strip()) < 10:
            return metrics

        # VQA质量检查
        if task == 'vqa':
            required_keywords = ['observe', 'analy', 'conclude']
            has_keywords = sum(1 for kw in required_keywords if kw.lower() in reasoning.lower())

            # 至少包含2个关键词，且长度>50字符
            if has_keywords >= 2 and len(reasoning) > 50:
                metrics['is_valid'] = True

        return metrics

    def generate_batch_cot(
        self,
        batch_data: Dict[str, Any],
        questions: Dict[int, List[Dict]]
    ) -> Dict[str, List[Dict]]:
        """
        Generate CoT for batch of images (VQA only).

        Args:
            batch_data: Batch data dictionary
            questions: Dict of image_id -> list of question dicts

        Returns:
            Dictionary with VQA CoT results
        """
        results = {'vqa': []}

        for img_data in batch_data['images']:
            image_id = img_data['id']
            image_path = img_data['path']

            # 只处理VQA任务
            vqa_questions = questions.get(image_id, [])
            for q_data in vqa_questions:
                question = q_data.get('question', '')

                cot = self.generate_vqa_cot(
                    image_path=image_path,
                    question=question,
                    image_id=str(image_id)
                )
                results['vqa'].append(cot)

        return results