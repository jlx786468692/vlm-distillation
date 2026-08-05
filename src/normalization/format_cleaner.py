"""
格式清洗器（软标签专用）
======================

【三条红线 - 绝对禁止】
❌ 红线1：不要把 color 弱候选池映射逻辑用在 GT、教师数据集；映射只上线推理。
❌ 红线2：counting 不要把过滤阈值写进 Prompt 强制模型只能输出该区间数字。
❌ 红线3：软标签（教师文本）不能做语义归一，只允许格式清洗。

【使用范围】
- 教师输出软标签的格式清洗
- 仅处理：大小写、空格、末尾标点
- 禁止：语义层面的合并、改写、映射
"""

import re
from typing import Tuple


class FormatCleaner:
    """
    格式清洗器（仅做格式清洗，不做语义归一）

    三条红线：
    ❌ 不要把 color 弱候选池映射逻辑用在 GT、教师数据集；映射只上线推理。
    ❌ counting 不要把过滤阈值写进 Prompt 强制模型只能输出该区间数字。
    ❌ 软标签（教师文本）不能做语义归一，只允许格式清洗。
    """

    @staticmethod
    def clean_text(text: str) -> Tuple[str, float]:
        """
        格式清洗（通用）

        仅处理：
        - 统一小写
        - 去除多余空格
        - 去除末尾标点（保留内部标点）
        - 去除首尾空白

        禁止：
        - 语义层面的合并、改写、映射
        - 修改文本内容（如 "dark blue" → "blue"）

        Args:
            text: 原始文本

        Returns:
            (清洗后的文本, 置信度)
        """
        if not text:
            return "", 0.0

        # Step 1: 去除首尾空白
        cleaned = text.strip()

        # Step 2: 统一小写
        cleaned = cleaned.lower()

        # Step 3: 去除多余空格（多个空格 → 单个空格）
        cleaned = re.sub(r'\s+', ' ', cleaned)

        # Step 4: 去除末尾标点（保留内部标点）
        # 例如："yes." → "yes"，但 "top-left" 保留
        cleaned = re.sub(r'[.!?]+$', '', cleaned)

        return cleaned, 1.0

    @staticmethod
    def clean_yesno(text: str) -> Tuple[str, float]:
        """
        格式清洗（closed_yesno专用）

        仅处理：
        - 提取 yes/no 关键词
        - 统一小写
        - 去除标点和空格

        禁止：
        - 语义层面的修改

        Args:
            text: 原始文本

        Returns:
            (清洗后的文本, 置信度)
        """
        if not text:
            return "", 0.0

        # 格式清洗
        cleaned, conf = FormatCleaner.clean_text(text)

        # 提取 yes/no 关键词
        if 'yes' in cleaned:
            return 'yes', 1.0
        elif 'no' in cleaned:
            return 'no', 1.0
        else:
            # 无效样本（既不包含yes也不包含no）
            return cleaned, 0.0

    @staticmethod
    def clean_choice(text: str, candidate_pool: list) -> Tuple[str, float]:
        """
        格式清洗（closed_choice专用）

        仅处理：
        - 统一小写
        - 去除空格和标点
        - 检查是否在候选池中

        禁止：
        - 语义层面的修改

        Args:
            text: 原始文本
            candidate_pool: 候选答案池 ["red", "blue"]

        Returns:
            (清洗后的文本, 置信度)
        """
        if not text:
            return "", 0.0

        # 格式清洗
        cleaned, _ = FormatCleaner.clean_text(text)

        # 检查是否在候选池中
        if cleaned in candidate_pool:
            return cleaned, 1.0
        else:
            # 无效样本（不在候选池中）
            return cleaned, 0.0


# ===== 使用示例 =====
if __name__ == "__main__":
    cleaner = FormatCleaner()

    # 测试通用格式清洗
    test_cases = [
        ("Yes, there is a dog.", 'yes, there is a dog'),  # 去除末尾标点
        ("  DARK BLUE  ", 'dark blue'),  # 统一小写、去空格
        ("top-left corner.", 'top-left corner'),  # 保留内部标点
        ("light red", 'light red'),  # ❌ 不做语义归一
    ]

    print("="*70)
    print("格式清洗测试（仅做格式清洗，不做语义归一）")
    print("="*70)

    for input_text, expected in test_cases:
        cleaned, conf = cleaner.clean_text(input_text)
        status = "✓" if cleaned == expected else "✗"
        print(f"\n{status} 输入: '{input_text}'")
        print(f"  输出: '{cleaned}'")
        print(f"  预期: '{expected}'")
        print(f"  置信度: {conf}")

    # 测试yesno格式清洗
    print("\n" + "="*70)
    print("closed_yesno格式清洗测试")
    print("="*70)

    yesno_cases = [
        ("Yes, there is.", 'yes'),
        ("No.", 'no'),
        ("Maybe", 'maybe'),  # 无效样本
    ]

    for input_text, expected in yesno_cases:
        cleaned, conf = cleaner.clean_yesno(input_text)
        status = "✓" if cleaned == expected else "✗"
        print(f"\n{status} 输入: '{input_text}'")
        print(f"  输出: '{cleaned}'")
        print(f"  置信度: {conf}")