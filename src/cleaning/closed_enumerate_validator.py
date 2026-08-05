"""
closed_enumerate 样本验证器
===========================

专门处理 closed_enumerate 问题（counting/color/location）的答案验证。

【三条红线 - 绝对禁止】
❌ 红线1：不要把 color 弱候选池映射逻辑用在 GT、教师数据集；映射只上线推理。
❌ 红线2：counting 不要把过滤阈值写进 Prompt 强制模型只能输出该区间数字。
❌ 红线3：软标签（教师文本）不能做语义归一，只允许格式清洗。
"""

import logging
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

from ..normalization import AnswerNormalizer
from ..classification.question_classifier import QuestionType


class ClosedEnumerateValidator:
    """
    closed_enumerate 样本验证器

    三条红线：
    ❌ 不要把 color 弱候选池映射逻辑用在 GT、教师数据集；映射只上线推理。
    ❌ counting 不要把过滤阈值写进 Prompt 强制模型只能输出该区间数字。
    ❌ 软标签（教师文本）不能做语义归一，只允许格式清洗。
    """

    def __init__(self, config: Optional[Any] = None, logger: Optional[logging.Logger] = None):
        """
        初始化验证器

        Args:
            config: 配置管理器
            logger: 日志记录器
        """
        self.logger = logger if logger else logging.getLogger(__name__)
        self.config = config
        self.normalizer = AnswerNormalizer()

        self.logger.info("✓ closed_enumerate 样本验证器初始化完成")
        self.logger.info("  ⚠️ 严格遵守三条红线：")
        self.logger.info("    ❌ 红线1：color弱候选池映射仅用于学生推理")
        self.logger.info("    ❌ 红线2：counting阈值仅用于数据清洗")
        self.logger.info("    ❌ 红线3：软标签仅允许格式清洗")

    def validate_sample(
        self,
        sample: Dict[str, Any],
        question_type: QuestionType
    ) -> Tuple[bool, float, str]:
        """
        验证样本

        Args:
            sample: 样本数据
            question_type: 问题类型（细分类型）

        Returns:
            (是否有效, 置信度, 验证信息)
        """
        # 获取答案
        hard_label = sample.get('hard_label', {})
        soft_label = sample.get('soft_label', {})

        hard_answer = hard_label.get('answer', '')
        soft_distribution = soft_label.get('answer_distribution', {})

        # 根据问题类型选择验证方法
        major_category = question_type.to_major_category()

        if major_category != QuestionType.CLOSED_ENUMERATE:
            # 不是closed_enumerate类型，直接返回有效
            return True, 1.0, "非closed_enumerate类型，无需特殊验证"

        # 根据细分类型选择验证方法
        if question_type == QuestionType.COUNT:
            return self._validate_counting(hard_answer, soft_distribution)
        elif question_type == QuestionType.COLOR:
            return self._validate_color(hard_answer, soft_distribution)
        elif question_type == QuestionType.LOCATION:
            return self._validate_location(hard_answer, soft_distribution)
        else:
            return True, 1.0, "未知细分类型"

    def _validate_counting(
        self,
        hard_answer: str,
        soft_distribution: Dict[str, float]
    ) -> Tuple[bool, float, str]:
        """
        验证 counting 问题

        规则：
        - 硬标签：统一阿拉伯数字
        - 阈值验证：0-50（仅用于过滤脏样本，不写入Prompt）

        Args:
            hard_answer: 硬标签答案
            soft_distribution: 软标签分布

        Returns:
            (是否有效, 置信度, 验证信息)
        """
        # 标准化硬标签
        normalized_answer, conf = self.normalizer.normalize_gt(hard_answer, 'counting')

        # 验证是否为有效数字
        if conf < 0.5:
            return False, conf, f"无效的数字格式: {hard_answer}"

        # 验证是否在阈值内（0-50，仅用于数据清洗）
        try:
            num = int(normalized_answer)
            if num < 0 or num > 50:
                return False, 0.3, f"数字超出阈值范围(0-50): {num}"
            else:
                return True, conf, f"有效的计数答案: {normalized_answer}"
        except ValueError:
            return False, 0.3, f"无法解析数字: {hard_answer}"

    def _validate_color(
        self,
        hard_answer: str,
        soft_distribution: Dict[str, float]
    ) -> Tuple[bool, float, str]:
        """
        验证 color 问题

        规则：
        - 硬标签：剥离修饰词，保留核心颜色词
        - ❌ 不使用弱候选池映射（红线1）

        Args:
            hard_answer: 硬标签答案
            soft_distribution: 软标签分布

        Returns:
            (是否有效, 置信度, 验证信息)
        """
        # 标准化硬标签
        normalized_answer, conf = self.normalizer.normalize_gt(hard_answer, 'color')

        if conf < 0.5:
            return False, conf, f"无效的颜色答案: {hard_answer}"

        return True, conf, f"有效的颜色答案: {hard_answer} → {normalized_answer}"

    def _validate_location(
        self,
        hard_answer: str,
        soft_distribution: Dict[str, float]
    ) -> Tuple[bool, float, str]:
        """
        验证 location 问题

        规则：
        - 硬标签：提取核心位置词
        - ❌ 不使用弱候选池映射（红线1）

        Args:
            hard_answer: 硬标签答案
            soft_distribution: 软标签分布

        Returns:
            (是否有效, 置信度, 验证信息)
        """
        # 标准化硬标签
        normalized_answer, conf = self.normalizer.normalize_gt(hard_answer, 'location')

        if conf < 0.5:
            return False, conf, f"无效的位置答案: {hard_answer}"

        return True, conf, f"有效的位置答案: {hard_answer} → {normalized_answer}"


# ===== 便捷函数 =====

def validate_closed_enumerate_sample(
    sample: Dict[str, Any],
    question_type: QuestionType
) -> Tuple[bool, float, str]:
    """
    验证 closed_enumerate 样本（便捷函数）

    Args:
        sample: 样本数据
        question_type: 问题类型

    Returns:
        (是否有效, 置信度, 验证信息)
    """
    validator = ClosedEnumerateValidator()
    return validator.validate_sample(sample, question_type)