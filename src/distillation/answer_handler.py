"""
答案标准化辅助模块（集成到蒸馏流程）
====================================

【三条红线 - 绝对禁止】
❌ 红线1：不要把 color 弱候选池映射逻辑用在 GT、教师数据集；映射只上线推理。
❌ 红线2：counting 不要把过滤阈值写进 Prompt 强制模型只能输出该区间数字。
❌ 红线3：软标签（教师文本）不能做语义归一，只允许格式清洗。

【使用方式】
集成到蒸馏流程中，根据问题类型和标签类型（GT硬标签/教师软标签）使用不同的标准化策略。
"""

from typing import Tuple, Optional
from ..classification.question_classifier import QuestionType
from ..normalization import AnswerNormalizer


class DistillationAnswerHandler:
    """
    蒸馏流程答案处理辅助类

    三条红线：
    ❌ 不要把 color 弱候选池映射逻辑用在 GT、教师数据集；映射只上线推理。
    ❌ counting 不要把过滤阈值写进 Prompt 强制模型只能输出该区间数字。
    ❌ 软标签（教师文本）不能做语义归一，只允许格式清洗。
    """

    def __init__(self):
        """初始化答案处理器"""
        self.normalizer = AnswerNormalizer()

    def process_hard_label(
        self,
        answer: str,
        question_type: QuestionType
    ) -> Tuple[str, float]:
        """
        处理硬标签（GT标准化）

        处理规则：
        - counting: 统一阿拉伯数字（one/two → 1/2）
        - closed_yesno: 统一小写yes/no
        - closed_choice: 和候选池对齐（小写）
        - color: 剥离修饰词，保留核心颜色词（dark blue → blue）
        - location: 提取核心位置词
        - open: 仅格式清洗

        Args:
            answer: 原始答案
            question_type: 问题类型（细分类型）

        Returns:
            (标准化后的答案, 置信度)
        """
        # 获取4大类问题类型
        major_category = question_type.to_major_category()

        # 映射到normalizer的参数
        type_mapping = {
            QuestionType.CLOSED_ENUMERATE: {
                'counting': 'counting',
                'color': 'color',
                'location': 'location'
            }
        }

        # 特殊处理closed_enumerate的细分类型
        if major_category == QuestionType.CLOSED_ENUMERATE:
            # 根据细分类型选择处理方法
            if question_type == QuestionType.COUNT:
                return self.normalizer.normalize_gt(answer, 'counting')
            elif question_type == QuestionType.COLOR:
                return self.normalizer.normalize_gt(answer, 'color')
            elif question_type == QuestionType.LOCATION:
                return self.normalizer.normalize_gt(answer, 'location')

        # 其他类型直接使用4大类
        return self.normalizer.normalize_gt(answer, major_category.value)

    def process_soft_label(
        self,
        answer: str,
        question_type: QuestionType
    ) -> Tuple[str, float]:
        """
        处理软标签（教师输出清洗）

        处理规则：
        - 仅做格式清洗（大小写、空格、标点）
        - ❌ 禁止语义归一（红线3）
        - counting: 验证数字格式+阈值过滤
        - color/location: 保留原始语义（dark blue不转为blue）

        Args:
            answer: 教师输出
            question_type: 问题类型（细分类型）

        Returns:
            (清洗后的答案, 置信度)
        """
        # 获取4大类问题类型
        major_category = question_type.to_major_category()

        # 特殊处理closed_enumerate的细分类型
        if major_category == QuestionType.CLOSED_ENUMERATE:
            # 根据细分类型选择处理方法
            if question_type == QuestionType.COUNT:
                return self.normalizer.clean_teacher_output(answer, 'counting')
            elif question_type == QuestionType.COLOR:
                return self.normalizer.clean_teacher_output(answer, 'color')
            elif question_type == QuestionType.LOCATION:
                return self.normalizer.clean_teacher_output(answer, 'location')

        # 其他类型直接使用4大类
        return self.normalizer.clean_teacher_output(answer, major_category.value)

    def process_inference_output(
        self,
        answer: str,
        question_type: QuestionType
    ) -> Tuple[str, float]:
        """
        处理学生推理输出（语义归一）

        处理规则：
        - 仅用于学生推理后处理
        - ❌ 禁止用于GT、教师数据集（红线1）
        - color/location: 语义归一（dark blue → blue）

        Args:
            answer: 原始答案
            question_type: 问题类型（细分类型）

        Returns:
            (归一化后的答案, 置信度)
        """
        # 获取4大类问题类型
        major_category = question_type.to_major_category()

        # 特殊处理closed_enumerate的细分类型
        if major_category == QuestionType.CLOSED_ENUMERATE:
            if question_type == QuestionType.COLOR:
                return self.normalizer.validate_for_inference(answer, 'color')
            elif question_type == QuestionType.LOCATION:
                return self.normalizer.validate_for_inference(answer, 'location')
            else:
                # counting不需要语义归一
                return self.normalizer.format_cleaner.clean_text(answer)

        return self.normalizer.validate_for_inference(answer, major_category.value)


# ===== 便捷函数 =====
_global_handler = None


def get_answer_handler() -> DistillationAnswerHandler:
    """获取全局答案处理器"""
    global _global_handler
    if _global_handler is None:
        _global_handler = DistillationAnswerHandler()
    return _global_handler


def normalize_hard_label(answer: str, question_type: QuestionType) -> Tuple[str, float]:
    """
    标准化硬标签（便捷函数）

    Args:
        answer: 原始答案
        question_type: 问题类型

    Returns:
        (标准化后的答案, 置信度)
    """
    return get_answer_handler().process_hard_label(answer, question_type)


def clean_soft_label(answer: str, question_type: QuestionType) -> Tuple[str, float]:
    """
    清洗软标签（便捷函数）

    Args:
        answer: 教师输出
        question_type: 问题类型

    Returns:
        (清洗后的答案, 置信度)
    """
    return get_answer_handler().process_soft_label(answer, question_type)


def normalize_inference_output(answer: str, question_type: QuestionType) -> Tuple[str, float]:
    """
    归一化推理输出（便捷函数）

    Args:
        answer: 原始答案
        question_type: 问题类型

    Returns:
        (归一化后的答案, 置信度)
    """
    return get_answer_handler().process_inference_output(answer, question_type)