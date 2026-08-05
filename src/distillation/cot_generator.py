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
        allowed_answers: Optional[List[str]] = None,
        question_type: Optional[str] = None  # 🔧 新增：问题类型
    ) -> Dict[str, Any]:
        """
        Generate CoT reasoning for VQA.

        🔧 统一方案：开放和闭合问题都生成两段式CoT
        - 内在逻辑：观察→分析→结论
        - 外在形式：[Reasoning]...[Answer]...

        Args:
            image_path: Path to image
            question: Question text
            image_id: Image identifier
            primary_answer: Reference answer from hard_label (for CoT prompt)
            allowed_answers: List of allowed answers from soft_label (for CoT prompt)
            question_type: Question type (open_descriptive, closed_choice, etc.)

        Returns:
            CoT reasoning dictionary with 'reasoning_paragraph' and 'answer'
        """
        self.logger.debug(f"Generating VQA CoT for image {image_id}, question_type={question_type}")

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
            image_id=image_id,
            question_type=question_type  # 🔧 新增：传递问题类型
        )

        # Extract and structure CoT
        full_response = result.get('full_response', '')

        # 🔧 调试日志：查看原始 CoT 响应
        self.logger.debug(f"[CoT] Raw full_response length: {len(full_response)}")
        self.logger.debug(f"[CoT] full_response (first 500 chars): {full_response[:500]}")

        # 🔧 新格式：直接返回 reasoning_paragraph 和 answer
        if self.structured_output:
            cot_data = self._structure_vqa_reasoning(full_response)
        else:
            # 如果不使用结构化输出，返回空结构
            cot_data = {
                'reasoning_paragraph': '',
                'answer': ''
            }

        # 🔧 移除验证操作：蒸馏阶段不做任何数据清洗和验证
        # 清洗逻辑由下游 cleaning 模块处理（RewardModelScorer）

        return cot_data

    def _structure_vqa_reasoning(self, raw_reasoning: str) -> Dict[str, str]:
        """
        Structure VQA reasoning into two-part format.

        🔧 统一格式（开放和闭合问题）：
        - 内在逻辑：观察→分析→结论
        - 外在形式：两段式
          - 【推理】：连贯自然的推理段落
          - 【答案】：简短的结论

        Args:
            raw_reasoning: Raw reasoning text

        Returns:
            Structured reasoning with reasoning_paragraph and answer
        """
        structured = {
            'reasoning_paragraph': '',  # 推理段落
            'answer': ''                # 最终答案
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

        import re

        # ───────────────────────────────────────────────────────
        # 尝试匹配两段式格式：[Reasoning]...[Answer]...
        # ───────────────────────────────────────────────────────
        reasoning_match = re.search(r'\[Reasoning\]\s*(.*?)(?=\[Answer\]|$)', response_part, re.DOTALL)
        answer_match = re.search(r'\[Answer\]\s*(.*?)(?=\n\n|$)', response_part, re.DOTALL)

        if reasoning_match and answer_match:
            # 提取两段式格式的内容
            reasoning_content = reasoning_match.group(1).strip()
            answer_content = answer_match.group(1).strip()

            self.logger.debug(f"[Structure] Found two-part format: reasoning={len(reasoning_content)}, answer={len(answer_content)}")

            # 清洗内容
            if TEXT_CLEANER_AVAILABLE:
                reasoning_content = clean_text(reasoning_content)
                answer_content = clean_text(answer_content)

            structured['reasoning_paragraph'] = reasoning_content
            structured['answer'] = answer_content

        else:
            # ───────────────────────────────────────────────────────
            # 回退：尝试提取旧的Observation/Analysis/Conclusion格式
            # 并转换为两段式
            # ───────────────────────────────────────────────────────
            self.logger.debug(f"[Structure] No two-part format found, trying old format")

            label_patterns = {
                'observation': [r'Observation\s*:', r'Step\s*1\s*:'],
                'analysis': [r'Analysis\s*:', r'Step\s*2\s*:'],
                'conclusion': [r'Conclusion\s*:', r'Step\s*3\s*:', r'Final Answer\s*:']
            }

            old_structured = {}

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

                        # 🔧 清洗内容
                        if TEXT_CLEANER_AVAILABLE and content:
                            content = text_cleaner.clean_cot_quotes(content)
                            content = clean_text(content)

                        if content:
                            old_structured[key] = content
                            self.logger.debug(f"[Structure] Extracted {key}: {content[:100]}")
                            break

            # ───────────────────────────────────────────────────────
            # 转换为两段式
            # ───────────────────────────────────────────────────────
            if old_structured:
                # 合并observation和analysis为推理段落
                reasoning_parts = []
                if old_structured.get('observation'):
                    reasoning_parts.append(old_structured['observation'])
                if old_structured.get('analysis'):
                    reasoning_parts.append(old_structured['analysis'])

                if reasoning_parts:
                    structured['reasoning_paragraph'] = ' '.join(reasoning_parts)

                # 结论作为答案
                if old_structured.get('conclusion'):
                    structured['answer'] = old_structured['conclusion']

                self.logger.debug(f"[Structure] Converted old format to two-part: reasoning={len(structured['reasoning_paragraph'])}, answer={len(structured['answer'])}")

            else:
                # ───────────────────────────────────────────────────────
                # 最终回退：直接使用整个响应
                # ───────────────────────────────────────────────────────
                self.logger.warning(f"[Structure] No structured format found, using full response")
                structured['reasoning_paragraph'] = response_part.strip()

        # 🔧 调试日志：显示最终结果
        self.logger.debug(
            f"[Structure] Final result: "
            f"reasoning={bool(structured['reasoning_paragraph'])}, "
            f"answer={bool(structured['answer'])}"
        )

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