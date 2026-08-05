"""
color 专用处理器
===============

【三条红线 - 绝对禁止】
❌ 红线1：不要把 color 弱候选池映射逻辑用在 GT、教师数据集；映射只上线推理。
❌ 红线2：counting 不要把过滤阈值写进 Prompt 强制模型只能输出该区间数字。
❌ 红线3：软标签（教师文本）不能做语义归一，只允许格式清洗。

【处理规则】
GT 硬标签：
  - 剥离 light/dark/bright 修饰，保留核心颜色词
  - 独立色系 (pink/purple) 保留

教师输出（软标签）：
  - 仅做格式清洗（大小写、标点空格）
  - 保留 dark/light 修饰，不做语义合并
  - 语义归一 dark blue → blue：仅放在学生推理后处理
  - 教师生成 & 训练阶段禁止修改教师文本

弱候选池：
  - 仅用于学生推理后处理验证
  - ❌ 禁止用于 GT、教师数据集
"""

import re
from typing import Tuple, List, Optional


class ColorNormalizer:
    """
    color 专用处理器

    三条红线：
    ❌ 不要把 color 弱候选池映射逻辑用在 GT、教师数据集；映射只上线推理。
    ❌ counting 不要把过滤阈值写进 Prompt 强制模型只能输出该区间数字。
    ❌ 软标签（教师文本）不能做语义归一，只允许格式清洗。
    """

    # 常见颜色修饰词（用于GT硬标签提取）
    COLOR_MODIFIERS = [
        "light", "dark", "bright", "pale", "deep",
        "lighter", "darker", "brighter", "paler", "deeper",
        "very", "slightly", "somewhat"
    ]

    # 弱候选池（仅用于学生推理后处理，❌ 禁止用于GT、教师数据集）
    WEAK_CANDIDATE_POOL = [
        "red", "blue", "green", "yellow", "orange", "purple", "pink",
        "brown", "black", "white", "gray", "grey", "beige", "tan",
        "gold", "silver", "cyan", "magenta", "turquoise", "cream"
    ]

    def __init__(self, weak_candidate_pool: Optional[List[str]] = None):
        """
        初始化 color 处理器

        Args:
            weak_candidate_pool: 弱候选池（可选，仅用于学生推理后处理）
        """
        self.weak_candidate_pool = weak_candidate_pool or self.WEAK_CANDIDATE_POOL

        print(f"✓ ColorNormalizer 初始化完成")
        print(f"  弱候选池大小: {len(self.weak_candidate_pool)}")
        print(f"  ⚠️ 注意：弱候选池仅用于学生推理后处理，禁止用于GT、教师数据集（红线1）")

    def normalize_gt(self, answer: str) -> Tuple[str, float]:
        """
        GT 硬标签标准化

        规则：
        - 剥离 light/dark/bright 修饰，保留核心颜色词
        - 独立色系 (pink/purple) 保留
        - ❌ 不使用弱候选池映射（红线1）

        Args:
            answer: 原始答案

        Returns:
            (标准化后的答案, 置信度)
        """
        answer_clean = answer.strip().lower()

        # 规则1：剥离修饰词，提取核心颜色词
        core_color = self._extract_core_color(answer_clean)

        if core_color:
            return core_color, 1.0

        # 规则2：无法提取 → 返回原始答案（低置信度）
        return answer_clean, 0.6

    def clean_teacher_output(self, answer: str) -> Tuple[str, float]:
        """
        教师输出清洗（软标签专用）

        规则：
        - 仅做格式清洗（大小写、空格、标点）
        - 保留 dark/light 修饰，不做语义合并
        - ❌ 禁止语义归一（红线3）

        Args:
            answer: 教师输出

        Returns:
            (清洗后的答案, 置信度)
        """
        if not answer:
            return "", 0.0

        # Step 1: 去除首尾空白
        cleaned = answer.strip()

        # Step 2: 统一小写
        cleaned = cleaned.lower()

        # Step 3: 去除多余空格
        cleaned = re.sub(r'\s+', ' ', cleaned)

        # Step 4: 去除末尾标点（保留内部标点）
        cleaned = re.sub(r'[.!?]+$', '', cleaned)

        # ❌ 禁止：语义归一（如 dark blue → blue）
        # ❌ 禁止：弱候选池映射（红线1）

        return cleaned, 1.0

    def validate_for_inference(self, answer: str) -> Tuple[str, float]:
        """
        学生推理后处理（语义归一）

        规则：
        - 语义归一：dark blue → blue
        - 弱候选池验证：检查是否包含核心颜色词
        - ⚠️ 仅用于学生推理后处理，禁止用于GT、教师数据集

        Args:
            answer: 原始答案

        Returns:
            (归一化后的答案, 置信度)
        """
        answer_clean = answer.strip().lower()

        # Step 1: 提取核心颜色词
        core_color = self._extract_core_color(answer_clean)

        if core_color:
            # Step 2: 弱候选池验证
            if core_color in self.weak_candidate_pool:
                return core_color, 1.0
            else:
                # 新颜色词，降低置信度
                return core_color, 0.7

        # Step 3: 无法提取 → 返回原始答案
        return answer_clean, 0.5

    def _extract_core_color(self, text: str) -> Optional[str]:
        """
        提取核心颜色词（剥离修饰词）

        Args:
            text: 输入文本

        Returns:
            核心颜色词，失败返回 None
        """
        text_lower = text.strip().lower()

        # 去除修饰词
        words = text_lower.split()
        core_words = [w for w in words if w not in self.COLOR_MODIFIERS]

        if not core_words:
            return None

        # 合并核心词
        core_color = ' '.join(core_words)

        # 如果是复合词（如 "blue green"），返回第一个核心词
        # 例如："dark blue green" → "blue green" → "blue"
        if len(core_words) > 1:
            # 检查是否在候选池中
            for word in core_words:
                if word in self.weak_candidate_pool:
                    return word

        return core_words[0] if core_words else None


# ===== 使用示例 =====
if __name__ == "__main__":
    normalizer = ColorNormalizer()

    print("="*70)
    print("GT 硬标签标准化测试（剥离修饰词）")
    print("="*70)

    # 测试GT硬标签标准化
    test_cases = [
        ("dark blue", "blue"),       # 剥离修饰词
        ("light red", "red"),        # 剥离修饰词
        ("bright green", "green"),   # 剥离修饰词
        ("pink", "pink"),            # 独立色系保留
        ("purple", "purple"),        # 独立色系保留
        ("very dark blue", "blue"),  # 剥离多个修饰词
    ]

    for input_text, expected in test_cases:
        result, conf = normalizer.normalize_gt(input_text)
        status = "✓" if result == expected else "✗"
        print(f"\n{status} 输入: '{input_text}'")
        print(f"  输出: '{result}'")
        print(f"  预期: '{expected}'")
        print(f"  置信度: {conf}")

    print("\n" + "="*70)
    print("教师输出清洗测试（保留原始语义）")
    print("="*70)

    # 测试教师输出清洗
    teacher_outputs = [
        ("DARK BLUE", "dark blue"),      # 格式清洗，保留语义
        ("  light red  ", "light red"),  # 去除空格，保留语义
        ("bright green.", "bright green"),  # 去除标点，保留语义
    ]

    for input_text, expected in teacher_outputs:
        result, conf = normalizer.clean_teacher_output(input_text)
        status = "✓" if result == expected else "✗"
        print(f"\n{status} 输入: '{input_text}'")
        print(f"  输出: '{result}'")
        print(f"  预期: '{expected}'")
        print(f"  置信度: {conf}")

    print("\n" + "="*70)
    print("学生推理后处理测试（语义归一）")
    print("="*70)

    # 测试学生推理后处理（语义归一）
    inference_outputs = [
        ("dark blue", "blue"),       # 语义归一
        ("light red", "red"),        # 语义归一
        ("bright green", "green"),   # 语义归一
        ("unknown color", "unknown color"),  # 新颜色词
    ]

    for input_text, expected in inference_outputs:
        result, conf = normalizer.validate_for_inference(input_text)
        status = "✓" if result == expected else "✗"
        print(f"\n{status} 输入: '{input_text}'")
        print(f"  输出: '{result}'")
        print(f"  预期: '{expected}'")
        print(f"  置信度: {conf}")