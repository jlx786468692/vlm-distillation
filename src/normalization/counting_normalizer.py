"""
counting 专用处理器
==================

【三条红线 - 绝对禁止】
❌ 红线1：不要把 color 弱候选池映射逻辑用在 GT、教师数据集；映射只上线推理。
❌ 红线2：counting 不要把过滤阈值写进 Prompt 强制模型只能输出该区间数字。
❌ 红线3：软标签（教师文本）不能做语义归一，只允许格式清洗。

【处理规则】
GT 硬标签：
  - 统一阿拉伯数字：one/two 全部映射为数字
  - 数字阈值：统计 GT 分布取 99.9%*1.5；无统计临时用 0‑50，仅用于过滤脏样本

教师输出：
  - 能解析出阿拉伯数字且在阈值内则保留
  - 英文数词、无数字、超阈值样本丢弃

注意：
  - 不把数字范围作为 Prompt 生成约束，仅做数据集清洗
"""

import re
from typing import Tuple, Optional


class CountingNormalizer:
    """
    counting 专用处理器

    三条红线：
    ❌ 不要把 color 弱候选池映射逻辑用在 GT、教师数据集；映射只上线推理。
    ❌ counting 不要把过滤阈值写进 Prompt 强制模型只能输出该区间数字。
    ❌ 软标签（教师文本）不能做语义归一，只允许格式清洗。
    """

    # 英文数词 → 阿拉伯数字映射（1-20）
    NUMBER_WORD_MAP = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
        "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20
    }

    # 临时阈值（0-50），用于过滤脏样本
    # ❌ 红线2：不要把此阈值写进 Prompt
    DEFAULT_MAX_THRESHOLD = 50
    DEFAULT_MIN_THRESHOLD = 0

    def __init__(self, max_threshold: Optional[int] = None, min_threshold: int = 0):
        """
        初始化 counting 处理器

        Args:
            max_threshold: 数字上限阈值（默认50，仅用于数据清洗，不写入Prompt）
            min_threshold: 数字下限阈值（默认0）
        """
        self.max_threshold = max_threshold or self.DEFAULT_MAX_THRESHOLD
        self.min_threshold = min_threshold

        print(f"✓ CountingNormalizer 初始化完成")
        print(f"  数字范围阈值: {self.min_threshold}-{self.max_threshold}")
        print(f"  ⚠️ 注意：此阈值仅用于数据清洗，不写入Prompt（红线2）")

    def normalize_gt(self, answer: str) -> Tuple[str, float]:
        """
        GT 硬标签标准化

        规则：
        - 统一阿拉伯数字：one/two 全部映射为数字
        - 英文数词 → 阿拉伯数字
        - 阿拉伯数字 → 保持不变

        Args:
            answer: 原始答案

        Returns:
            (标准化后的答案, 置信度)
        """
        answer_clean = answer.strip()

        # 规则1：已经是阿拉伯数字
        if answer_clean.isdigit():
            num = int(answer_clean)
            # 检查是否在阈值内（用于标记异常值，但不丢弃）
            if self.min_threshold <= num <= self.max_threshold:
                return str(num), 1.0
            else:
                # 超出阈值，降低置信度（但保留原始值）
                return str(num), 0.5

        # 规则2：英文数词 → 阿拉伯数字
        answer_lower = answer_clean.lower()
        if answer_lower in self.NUMBER_WORD_MAP:
            num = self.NUMBER_WORD_MAP[answer_lower]
            return str(num), 1.0

        # 规则3：无法解析 → 标记为低置信度
        return answer_clean, 0.3

    def validate_teacher_output(self, answer: str) -> Tuple[bool, Optional[str], float]:
        """
        教师输出验证（用于数据清洗，不用于Prompt生成）

        规则：
        - 能解析出阿拉伯数字且在阈值内则保留
        - 英文数词、无数字、超阈值样本丢弃

        Args:
            answer: 教师输出

        Returns:
            (是否有效, 解析后的数字, 置信度)
        """
        answer_clean = answer.strip()

        # 规则1：阿拉伯数字
        if answer_clean.isdigit():
            num = int(answer_clean)
            # 检查是否在阈值内
            if self.min_threshold <= num <= self.max_threshold:
                return True, str(num), 1.0
            else:
                # 超出阈值 → 丢弃（红线2：阈值仅用于数据清洗）
                return False, str(num), 0.0

        # 规则2：英文数词
        answer_lower = answer_clean.lower()
        if answer_lower in self.NUMBER_WORD_MAP:
            num = self.NUMBER_WORD_MAP[answer_lower]
            # 英文数词 → 丢弃（统一要求阿拉伯数字）
            return False, str(num), 0.0

        # 规则3：无法解析 → 丢弃
        return False, None, 0.0

    def parse_number(self, text: str) -> Optional[int]:
        """
        从文本中解析数字

        支持格式：
        - 阿拉伯数字：1, 2, 3
        - 英文数词：one, two, three

        Args:
            text: 输入文本

        Returns:
            解析出的数字，失败返回 None
        """
        text_clean = text.strip()

        # 尝试阿拉伯数字
        if text_clean.isdigit():
            return int(text_clean)

        # 尝试英文数词
        text_lower = text_clean.lower()
        if text_lower in self.NUMBER_WORD_MAP:
            return self.NUMBER_WORD_MAP[text_lower]

        # 尝试从文本中提取数字（如 "There are 5 people"）
        match = re.search(r'\b(\d+)\b', text_clean)
        if match:
            return int(match.group(1))

        return None


# ===== 使用示例 =====
if __name__ == "__main__":
    normalizer = CountingNormalizer()

    print("="*70)
    print("GT 硬标签标准化测试")
    print("="*70)

    # 测试GT硬标签标准化
    test_cases = [
        ("one", "1"),
        ("two", "2"),
        ("15", "15"),
        ("twenty", "20"),
        ("fifty", "fifty"),  # 超出映射范围
    ]

    for input_text, expected in test_cases:
        result, conf = normalizer.normalize_gt(input_text)
        status = "✓" if result == expected else "✗"
        print(f"\n{status} 输入: '{input_text}'")
        print(f"  输出: '{result}'")
        print(f"  预期: '{expected}'")
        print(f"  置信度: {conf}")

    print("\n" + "="*70)
    print("教师输出验证测试（数据清洗，不用于Prompt）")
    print("="*70)

    # 测试教师输出验证
    teacher_outputs = [
        ("2", True, "2"),      # 阿拉伯数字，在阈值内
        ("two", False, "2"),   # 英文数词，丢弃
        ("60", False, "60"),   # 超出阈值，丢弃
        ("unknown", False, None),  # 无法解析，丢弃
    ]

    for input_text, expected_valid, expected_num in teacher_outputs:
        is_valid, parsed_num, conf = normalizer.validate_teacher_output(input_text)
        status = "✓" if is_valid == expected_valid and parsed_num == expected_num else "✗"
        print(f"\n{status} 输入: '{input_text}'")
        print(f"  是否有效: {is_valid} (预期: {expected_valid})")
        print(f"  解析结果: {parsed_num} (预期: {expected_num})")
        print(f"  置信度: {conf}")