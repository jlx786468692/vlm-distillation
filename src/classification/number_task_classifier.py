"""
数字任务分类器（计数 vs 精确读数）
================================

区分两种数字任务：
1. COUNTING（计数任务）：
   - 问题模式："How many X?"
   - 答案语义：可数实例的数量
   - 数值范围：0-20（离散整数）
   - 视觉能力：目标检测 + 计数

2. READING_NUMBER（精确读数任务）：
   - 问题模式："What is the number on X?"
   - 答案语义：图像中印刷/显示的数字
   - 数值范围：0-9999+（连续数值）
   - 视觉能力：OCR + 文本识别

作者：Claude
日期：2026-08-10
"""

import re
from enum import Enum
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


class NumberTaskType(str, Enum):
    """数字任务类型枚举"""
    COUNTING = "counting"              # 计数任务（0-20）
    READING_NUMBER = "reading_number"  # 精确读数任务（OCR）
    UNKNOWN = "unknown"                # 未知类型


@dataclass
class NumberTaskClassificationResult:
    """数字任务分类结果"""
    task_type: NumberTaskType
    confidence: float
    method: str
    reasoning: str

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "task_type": self.task_type.value,
            "confidence": self.confidence,
            "method": self.method,
            "reasoning": self.reasoning
        }


class NumberTaskClassifier:
    """
    数字任务分类器

    区分计数任务和精确读数任务

    使用方法：
    >>> classifier = NumberTaskClassifier()
    >>> result = classifier.classify("How many people are there?", "3")
    >>> result.task_type
    NumberTaskType.COUNTING
    """

    # 计数任务特征模式
    COUNTING_PATTERNS = [
        r"how many",
        r"count the",
        r"number of",
        r"total number",
        r"quantity of",
    ]

    # 精确读数任务特征模式
    READING_NUMBER_PATTERNS = [
        r"what is the (number|bus number|train number)",
        r"bus number",
        r"train number",
        r"license plate",
        r"plate number",
        r"phone number",
        r"what number",
        r"number written",
        r"printed number",
        r"what (year|date|time)",
        r"which (year|date|time)",
    ]

    def __init__(self, config: Optional[Dict] = None):
        """初始化数字任务分类器"""
        self.config = config or {}
        self.counting_range = self.config.get("counting_range", (0, 20))

    def classify(
        self,
        question: str,
        hard_label: Optional[str] = None,
        method: str = "hybrid"
    ) -> NumberTaskClassificationResult:
        """
        分类数字任务类型

        Args:
            question: 问题文本
            hard_label: 真值答案（可选）
            method: 分类方法 ("rule", "range_based", "hybrid")

        Returns:
            NumberTaskClassificationResult对象
        """
        if method == "rule":
            return self._classify_by_rule(question)
        elif method == "range_based":
            return self._classify_by_range(hard_label)
        elif method == "hybrid":
            return self._classify_hybrid(question, hard_label)
        else:
            raise ValueError(f"未知的分类方法: {method}")

    def _classify_by_rule(self, question: str) -> NumberTaskClassificationResult:
        """基于规则分类"""
        question_lower = question.lower().strip()

        # 优先级1：计数任务检测
        for pattern in self.COUNTING_PATTERNS:
            if re.search(pattern, question_lower):
                return NumberTaskClassificationResult(
                    task_type=NumberTaskType.COUNTING,
                    confidence=0.95,
                    method="rule",
                    reasoning=f"匹配计数模式: {pattern}"
                )

        # 优先级2：精确读数任务检测
        for pattern in self.READING_NUMBER_PATTERNS:
            if re.search(pattern, question_lower):
                return NumberTaskClassificationResult(
                    task_type=NumberTaskType.READING_NUMBER,
                    confidence=0.90,
                    method="rule",
                    reasoning=f"匹配精确读数模式: {pattern}"
                )

        # 优先级3：未知类型
        return NumberTaskClassificationResult(
            task_type=NumberTaskType.UNKNOWN,
            confidence=0.0,
            method="rule",
            reasoning="规则未匹配"
        )

    def _classify_by_range(self, hard_label: Optional[str]) -> NumberTaskClassificationResult:
        """基于答案范围分类"""
        if hard_label is None:
            return NumberTaskClassificationResult(
                task_type=NumberTaskType.UNKNOWN,
                confidence=0.0,
                method="range_based",
                reasoning="缺少真值，无法判断"
            )

        try:
            clean_label = self._extract_number(hard_label)

            if clean_label is None:
                return NumberTaskClassificationResult(
                    task_type=NumberTaskType.UNKNOWN,
                    confidence=0.0,
                    method="range_based",
                    reasoning=f"无法从 '{hard_label}' 提取数字"
                )

            value = int(clean_label)

            if self.counting_range[0] <= value <= self.counting_range[1]:
                return NumberTaskClassificationResult(
                    task_type=NumberTaskType.COUNTING,
                    confidence=0.7,
                    method="range_based",
                    reasoning=f"答案范围 [{self.counting_range[0]}, {self.counting_range[1]}]: {value}"
                )
            else:
                return NumberTaskClassificationResult(
                    task_type=NumberTaskType.READING_NUMBER,
                    confidence=0.7,
                    method="range_based",
                    reasoning=f"答案超出计数范围: {value} > {self.counting_range[1]}"
                )

        except (ValueError, TypeError):
            return NumberTaskClassificationResult(
                task_type=NumberTaskType.UNKNOWN,
                confidence=0.0,
                method="range_based",
                reasoning=f"无法解析数字: '{hard_label}'"
            )

    def _classify_hybrid(
        self,
        question: str,
        hard_label: Optional[str]
    ) -> NumberTaskClassificationResult:
        """混合策略分类（推荐）"""
        # Step 1: 规则匹配
        rule_result = self._classify_by_rule(question)

        if rule_result.confidence >= 0.9:
            return rule_result

        # Step 2: 答案范围判断（后备）
        if hard_label is not None:
            range_result = self._classify_by_range(hard_label)

            if range_result.task_type != NumberTaskType.UNKNOWN:
                if rule_result.task_type == range_result.task_type:
                    return NumberTaskClassificationResult(
                        task_type=rule_result.task_type,
                        confidence=max(rule_result.confidence, range_result.confidence) + 0.1,
                        method="hybrid",
                        reasoning=f"规则和范围一致: {rule_result.reasoning}"
                    )
                else:
                    return NumberTaskClassificationResult(
                        task_type=rule_result.task_type,
                        confidence=rule_result.confidence,
                        method="hybrid",
                        reasoning=f"规则与范围不一致，采用规则"
                    )

        return rule_result

    def _extract_number(self, text: str) -> Optional[str]:
        """从文本中提取数字"""
        text = text.lower().strip()

        match = re.search(r'(\d+)', text)
        if match:
            return match.group(1)

        word_to_num = {
            "zero": "0", "one": "1", "two": "2", "three": "3",
            "four": "4", "five": "5", "six": "6", "seven": "7",
            "eight": "8", "nine": "9", "ten": "10"
        }

        for word, num in word_to_num.items():
            if word in text:
                return num

        return None

    def validate_sample(
        self,
        question: str,
        hard_label: str
    ) -> Dict[str, any]:
        """
        验证样本的任务类型一致性

        用于检测标注错误或异常样本

        Args:
            question: 问题文本
            hard_label: 真值答案

        Returns:
            验证结果字典
        """
        result = self.classify(question, hard_label, method="hybrid")

        # 检查异常情况
        warnings = []

        # 异常1：计数问题但答案 > 20
        if result.task_type == NumberTaskType.COUNTING:
            try:
                number = self._extract_number(hard_label)
                if number:
                    value = int(number)
                    if value > 30:  # 宽松阈值
                        warnings.append(f"计数任务答案过大: {value} (> 30)")
            except:
                pass

        # 异常2：精确读数但答案很小（< 10）
        if result.task_type == NumberTaskType.READING_NUMBER:
            try:
                number = self._extract_number(hard_label)
                if number:
                    value = int(number)
                    if value < 10:
                        warnings.append(f"精确读数任务答案较小: {value} (< 10)，可能是计数任务")
            except:
                pass

        return {
            "task_type": result.task_type.value,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "warnings": warnings,
            "is_valid": len(warnings) == 0
        }


if __name__ == "__main__":
    print("数字任务分类器测试")
    classifier = NumberTaskClassifier()

    test_cases = [
        ("How many people are in the image?", "3", "counting"),
        ("What is the bus number?", "413", "reading_number"),
    ]

    for question, hard_label, expected in test_cases:
        result = classifier.classify(question, hard_label)
        print(f"问题: {question}")
        print(f"任务类型: {result.task_type.value}")
        print(f"置信度: {result.confidence:.2f}")
        print()