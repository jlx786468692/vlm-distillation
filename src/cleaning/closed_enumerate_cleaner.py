"""
closed_enumerate 样本清洗器
===========================

对closed_enumerate类型（counting/color/location）的样本进行格式清洗和过滤。

【三条红线 - 绝对禁止】
❌ 红线1：不要把 color 弱候选池映射逻辑用在 GT、教师数据集；映射只上线推理。
❌ 红线2：counting 不要把过滤阈值写进 Prompt 强制模型只能输出该区间数字。
❌ 红线3：软标签（教师文本）不能做语义归一，只允许格式清洗。
"""

import re
import logging
from typing import Dict, Any, Tuple, Optional


class ClosedEnumerateCleaner:
    """
    closed_enumerate 样本清洗器

    步骤：
    1. 格式清洗（全部子类型统一）
    2. 按enum_subtype执行样本过滤（不合格直接丢弃）
    """

    def __init__(self, config: Optional[Any] = None):
        """
        初始化清洗器

        Args:
            config: 配置管理器
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 数字过滤阈值（仅用于过滤，不写入Prompt）
        self.max_number_threshold = 50

        # 颜色词汇列表（用于验证）
        self.color_keywords = [
            'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink',
            'black', 'white', 'gray', 'grey', 'brown', 'beige', 'tan',
            'gold', 'silver', 'cyan', 'magenta', 'turquoise', 'cream',
            'dark', 'light', 'bright', 'pale'  # 修饰词
        ]

        # 位置词汇列表（用于验证）
        self.location_keywords = [
            'left', 'right', 'top', 'bottom', 'center', 'middle',
            'front', 'back', 'side', 'corner', 'inside', 'outside',
            'background', 'foreground', 'on', 'in', 'at', 'under', 'above'
        ]

        self.logger.info("✓ closed_enumerate清洗器初始化完成")

    def clean_and_validate(
        self,
        sample: Dict[str, Any],
        enum_subtype: str
    ) -> Tuple[bool, Dict[str, Any], str]:
        """
        清洗和验证closed_enumerate样本

        Args:
            sample: 样本数据
            enum_subtype: 子类型（counting/color/location）

        Returns:
            (是否有效, 清洗后的样本, 验证信息)
        """
        # 获取教师输出（硬标签答案）
        hard_label = sample.get('hard_label', {})
        teacher_output = hard_label.get('answer', '')

        if not teacher_output:
            return False, sample, "教师输出为空"

        # ───────────────────────────────────────────────────────
        # 步骤A：格式清洗（全部子类型统一）
        # ───────────────────────────────────────────────────────
        cleaned_output = self._format_clean(teacher_output)

        # 更新样本
        cleaned_sample = sample.copy()
        cleaned_sample['hard_label']['answer'] = cleaned_output

        # ───────────────────────────────────────────────────────
        # 步骤B：按enum_subtype执行样本过滤
        # ───────────────────────────────────────────────────────
        if enum_subtype == 'counting':
            is_valid, msg = self._validate_counting(cleaned_output)
        elif enum_subtype == 'color':
            is_valid, msg = self._validate_color(cleaned_output)
        elif enum_subtype == 'location':
            is_valid, msg = self._validate_location(cleaned_output)
        else:
            is_valid = True
            msg = "未知子类型，默认保留"

        return is_valid, cleaned_sample, msg

    def _format_clean(self, text: str) -> str:
        """
        格式清洗（只做格式清洗，禁止语义归一）

        步骤：
        1. strip前后空白
        2. 移除末尾句号、逗号
        3. 保留内部空格
        4. 大小写归一：Dark Blue → dark blue
        5. ❌ 不做：dark blue → blue
        6. ❌ 不做：改写数字文本

        Args:
            text: 原始文本

        Returns:
            清洗后的文本
        """
        if not text:
            return ""

        # Step 1: strip前后空白
        cleaned = text.strip()

        # Step 2: 统一小写
        cleaned = cleaned.lower()

        # Step 3: 去除多余空格
        cleaned = re.sub(r'\s+', ' ', cleaned)

        # Step 4: 移除末尾标点（保留内部标点）
        cleaned = re.sub(r'[.!?]+$', '', cleaned)

        # ❌ 禁止语义归一（红线3）
        # 不做：dark blue → blue
        # 不做：改写数字文本

        return cleaned

    def _validate_counting(self, answer: str) -> Tuple[bool, str]:
        """
        验证counting答案

        规则：
        - 从Conclusion解析内容，必须能提取阿拉伯数字
        - 如果出现英文数词one/two、长句描述、解析不出数字 → 丢弃
        - 数字超过max_threshold → 丢弃

        Args:
            answer: 答案文本

        Returns:
            (是否有效, 验证信息)
        """
        # 尝试提取阿拉伯数字
        numbers = re.findall(r'\b\d+\b', answer)

        if numbers:
            # 找到阿拉伯数字
            num = int(numbers[0])

            # 检查是否在阈值内（红线2：阈值仅用于过滤，不写入Prompt）
            if 0 <= num <= self.max_number_threshold:
                return True, f"有效的计数答案（阿拉伯数字）: {num}"
            else:
                return False, f"数字超出阈值(0-{self.max_number_threshold}): {num}"

        # 检查是否是英文数词
        number_words = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
        answer_lower = answer.lower()

        for word in number_words:
            if word in answer_lower:
                return False, f"使用英文数词而非阿拉伯数字: {answer}"

        # 无法解析数字
        return False, f"无法解析出数字: {answer}"

    def _validate_color(self, answer: str) -> Tuple[bool, str]:
        """
        验证color答案

        规则：
        - 输出必须包含至少一个颜色词汇
        - 如果输出是大段描述、完全无颜色词 → 丢弃
        - 保留修饰词：dark blue, light red（不做语义归一）

        Args:
            answer: 答案文本

        Returns:
            (是否有效, 验证信息)
        """
        answer_lower = answer.lower()

        # 检查是否包含颜色词汇
        has_color = any(color in answer_lower for color in self.color_keywords)

        if has_color:
            return True, f"有效的颜色答案: {answer}"
        else:
            return False, f"输出不包含颜色词汇: {answer}"

    def _validate_location(self, answer: str) -> Tuple[bool, str]:
        """
        验证location答案

        规则：
        - 输出为空、大段无关描述 → 丢弃
        - 保留原始位置短语

        Args:
            answer: 答案文本

        Returns:
            (是否有效, 验证信息)
        """
        if not answer or len(answer.strip()) == 0:
            return False, "位置答案为空"

        answer_lower = answer.lower()

        # 检查是否包含位置词汇
        has_location = any(loc in answer_lower for loc in self.location_keywords)

        if has_location:
            return True, f"有效的位置答案: {answer}"
        else:
            # 即使不包含已知位置词，如果是短答案也保留
            if len(answer.split()) <= 5:
                return True, f"位置答案（短答案）: {answer}"
            else:
                return False, f"输出过长或无位置信息: {answer}"


# ===== 使用示例 =====
if __name__ == "__main__":
    cleaner = ClosedEnumerateCleaner()

    # 测试counting
    test_cases = [
        ("3", 'counting', True),
        ("three", 'counting', False),  # 英文数词 → 丢弃
        ("100", 'counting', False),  # 超出阈值 → 丢弃
        ("dark blue", 'color', True),  # 保留修饰词
        ("the car", 'color', False),  # 无颜色词 → 丢弃
        ("on the table", 'location', True),
        ("", 'location', False),
    ]

    print("="*70)
    print("closed_enumerate清洗测试")
    print("="*70)

    for answer, subtype, expected_valid in test_cases:
        sample = {'hard_label': {'answer': answer}}
        is_valid, cleaned, msg = cleaner.clean_and_validate(sample, subtype)

        status = "✓" if is_valid == expected_valid else "✗"
        print(f"\n{status} 输入: '{answer}' ({subtype})")
        print(f"  有效: {is_valid} (预期: {expected_valid})")
        print(f"  清洗后: '{cleaned['hard_label']['answer']}'")
        print(f"  信息: {msg}")