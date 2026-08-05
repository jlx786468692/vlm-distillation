"""
答案标准化主模块
===============

【三条红线 - 绝对禁止】
❌ 红线1：不要把 color 弱候选池映射逻辑用在 GT、教师数据集；映射只上线推理。
❌ 红线2：counting 不要把过滤阈值写进 Prompt 强制模型只能输出该区间数字。
❌ 红线3：软标签（教师文本）不能做语义归一，只允许格式清洗。

【使用方式】
GT 硬标签标准化：
    normalizer.normalize_gt(answer="dark blue", question_type="color")
    → ("blue", 1.0)

教师软标签清洗：
    normalizer.clean_teacher_output(answer="dark blue", question_type="color")
    → ("dark blue", 1.0)  # 保留原始语义

学生推理后处理：
    normalizer.validate_for_inference(answer="dark blue", question_type="color")
    → ("blue", 1.0)  # 语义归一
"""

from typing import Tuple, List, Optional
from .counting_normalizer import CountingNormalizer
from .color_normalizer import ColorNormalizer
from .location_normalizer import LocationNormalizer
from .format_cleaner import FormatCleaner


class AnswerNormalizer:
    """
    答案标准化主模块

    三条红线：
    ❌ 不要把 color 弱候选池映射逻辑用在 GT、教师数据集；映射只上线推理。
    ❌ counting 不要把过滤阈值写进 Prompt 强制模型只能输出该区间数字。
    ❌ 软标签（教师文本）不能做语义归一，只允许格式清洗。
    """

    def __init__(
        self,
        counting_max_threshold: Optional[int] = None,
        color_weak_pool: Optional[List[str]] = None,
        location_weak_pool: Optional[List[str]] = None
    ):
        """
        初始化答案标准化器

        Args:
            counting_max_threshold: counting数字上限阈值（仅用于数据清洗）
            color_weak_pool: color弱候选池（仅用于学生推理后处理）
            location_weak_pool: location弱候选池（仅用于学生推理后处理）
        """
        # 初始化各类型处理器
        self.counting_normalizer = CountingNormalizer(max_threshold=counting_max_threshold)
        self.color_normalizer = ColorNormalizer(weak_candidate_pool=color_weak_pool)
        self.location_normalizer = LocationNormalizer(weak_candidate_pool=location_weak_pool)
        self.format_cleaner = FormatCleaner()

        print("✓ AnswerNormalizer 初始化完成")
        print("  ⚠️ 严格遵守三条红线：")
        print("    ❌ 红线1：color弱候选池映射仅用于学生推理，禁止用于GT、教师数据集")
        print("    ❌ 红线2：counting阈值仅用于数据清洗，禁止写入Prompt")
        print("    ❌ 红线3：软标签仅允许格式清洗，禁止语义归一")

    def normalize_gt(self, answer: str, question_type: str) -> Tuple[str, float]:
        """
        GT 硬标签标准化

        处理规则：
        - counting: 统一阿拉伯数字（one/two → 1/2）
        - closed_yesno: 统一小写yes/no
        - closed_choice: 和候选池对齐（小写）
        - color: 剥离修饰词，保留核心颜色词（dark blue → blue）
        - location: 提取核心位置词
        - open: 仅格式清洗

        Args:
            answer: 原始答案
            question_type: 问题类型

        Returns:
            (标准化后的答案, 置信度)
        """
        if question_type == 'counting':
            return self.counting_normalizer.normalize_gt(answer)

        elif question_type == 'closed_yesno':
            return self.format_cleaner.clean_yesno(answer)

        elif question_type == 'closed_choice':
            # closed_choice需要候选池，由外部调用处理
            # 这里只做格式清洗
            return self.format_cleaner.clean_text(answer)

        elif question_type == 'color':
            return self.color_normalizer.normalize_gt(answer)

        elif question_type == 'location':
            return self.location_normalizer.normalize_gt(answer)

        else:  # open
            return self.format_cleaner.clean_text(answer)

    def clean_teacher_output(self, answer: str, question_type: str) -> Tuple[str, float]:
        """
        教师软标签清洗

        处理规则：
        - 仅做格式清洗（大小写、空格、标点）
        - ❌ 禁止语义归一（红线3）
        - counting: 验证数字格式+阈值过滤
        - color/location: 保留原始语义（dark blue不转为blue）

        Args:
            answer: 教师输出
            question_type: 问题类型

        Returns:
            (清洗后的答案, 置信度)
        """
        if question_type == 'counting':
            # counting特殊处理：验证数字格式
            is_valid, parsed_num, conf = self.counting_normalizer.validate_teacher_output(answer)
            if is_valid:
                return parsed_num, conf
            else:
                # 无效样本（英文数词、无数字、超阈值）
                return "", 0.0

        elif question_type == 'closed_yesno':
            return self.format_cleaner.clean_yesno(answer)

        elif question_type == 'closed_choice':
            # closed_choice需要候选池，由外部调用处理
            return self.format_cleaner.clean_text(answer)

        elif question_type == 'color':
            # ❌ 红线3：保留原始语义，不做语义归一
            return self.color_normalizer.clean_teacher_output(answer)

        elif question_type == 'location':
            # ❌ 红线3：保留原始语义，不做语义归一
            return self.location_normalizer.clean_teacher_output(answer)

        else:  # open
            return self.format_cleaner.clean_text(answer)

    def validate_for_inference(self, answer: str, question_type: str) -> Tuple[str, float]:
        """
        学生推理后处理（语义归一）

        处理规则：
        - 仅用于学生推理后处理
        - ❌ 禁止用于GT、教师数据集（红线1）
        - color/location: 语义归一（dark blue → blue）

        Args:
            answer: 原始答案
            question_type: 问题类型

        Returns:
            (归一化后的答案, 置信度)
        """
        if question_type == 'color':
            # ⚠️ 仅用于学生推理后处理
            return self.color_normalizer.validate_for_inference(answer)

        elif question_type == 'location':
            # ⚠️ 仅用于学生推理后处理
            return self.location_normalizer.validate_for_inference(answer)

        else:
            # 其他类型不需要语义归一
            return self.format_cleaner.clean_text(answer)


# ===== 使用示例 =====
if __name__ == "__main__":
    normalizer = AnswerNormalizer()

    print("="*70)
    print("GT 硬标签标准化测试")
    print("="*70)

    # 测试GT硬标签标准化
    test_cases = [
        ("one", "counting", "1"),
        ("dark blue", "color", "blue"),
        ("top-left corner", "location", "top-left"),
        ("Yes, there is.", "closed_yesno", "yes"),
    ]

    for answer, qtype, expected in test_cases:
        result, conf = normalizer.normalize_gt(answer, qtype)
        status = "✓" if result == expected else "✗"
        print(f"\n{status} 输入: '{answer}' (类型: {qtype})")
        print(f"  输出: '{result}'")
        print(f"  预期: '{expected}'")
        print(f"  置信度: {conf}")

    print("\n" + "="*70)
    print("教师软标签清洗测试（保留原始语义）")
    print("="*70)

    # 测试教师软标签清洗
    teacher_outputs = [
        ("two", "counting", "2"),
        ("dark blue", "color", "dark blue"),  # 保留原始语义
        ("top-left corner", "location", "top-left corner"),  # 保留原始语义
        ("Yes.", "closed_yesno", "yes"),
    ]

    for answer, qtype, expected in teacher_outputs:
        result, conf = normalizer.clean_teacher_output(answer, qtype)
        status = "✓" if result == expected else "✗"
        print(f"\n{status} 输入: '{answer}' (类型: {qtype})")
        print(f"  输出: '{result}'")
        print(f"  预期: '{expected}'")
        print(f"  置信度: {conf}")

    print("\n" + "="*70)
    print("学生推理后处理测试（语义归一）")
    print("="*70)

    # 测试学生推理后处理
    inference_outputs = [
        ("dark blue", "color", "blue"),
        ("top-left corner", "location", "top-left"),
    ]

    for answer, qtype, expected in inference_outputs:
        result, conf = normalizer.validate_for_inference(answer, qtype)
        status = "✓" if result == expected else "✗"
        print(f"\n{status} 输入: '{answer}' (类型: {qtype})")
        print(f"  输出: '{result}'")
        print(f"  预期: '{expected}'")
        print(f"  置信度: {conf}")