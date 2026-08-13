"""
精确读数候选生成器
==================

为精确读数任务（OCR/识别）生成候选池

核心功能：
1. 基于真值生成邻域候选（方案A，推荐）
2. 生成OCR识别的常见错误变体
3. 数字位数感知的候选生成
4. 答案格式归一化

作者：Claude
日期：2026-08-10
"""

import re
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from itertools import permutations


@dataclass
class CandidatePoolConfig:
    """候选池配置"""
    # 精确邻域
    precise_deltas: List[int] = None
    # 十位邻域
    decade_deltas: List[int] = None
    # 百位邻域
    century_deltas: List[int] = None
    # 数字重排
    include_permutations: bool = True
    # OCR变体（⚠️ 已禁用：包含字母的候选不符合数字任务）
    # 原因：如 '4l3', '4I3' 等包含字母，不是有效数字
    include_ocr_variants: bool = False

    def __post_init__(self):
        if self.precise_deltas is None:
            self.precise_deltas = [-2, -1, 0, 1, 2]
        if self.decade_deltas is None:
            self.decade_deltas = [-10, 10]
        if self.century_deltas is None:
            self.century_deltas = [-100, 100]


class ReadingNumberCandidateGenerator:
    """
    精确读数候选生成器

    适用于：
    - 车牌号识别："What is the bus number?" → "413"
    - 时间识别："What year?" → "2024"
    - 编号识别："What number is on the train?" → "42"

    使用方法：
    >>> generator = ReadingNumberCandidateGenerator()
    >>> candidates = generator.generate("413")
    >>> print(candidates)
    ['413', '412', '414', '411', '415', '403', '423', '313', '513', '143', '341', '431', '4l3', '4I3', '4 13', '41 3']
    """

    # OCR常见字符混淆映射
    CHAR_CONFUSION_MAP = {
        '0': ['O', 'o'],
        '1': ['l', 'I', 'i'],
        '2': ['Z', 'z'],
        '5': ['S', 's'],
        '8': ['B'],
    }

    def __init__(self, config: Optional[CandidatePoolConfig] = None):
        """
        初始化候选生成器

        Args:
            config: 候选池配置
        """
        self.config = config or CandidatePoolConfig()

    def generate(
        self,
        hard_label: str,
        max_candidates: int = 50
    ) -> List[str]:
        """
        生成候选池

        Args:
            hard_label: 真值答案
            max_candidates: 最大候选数

        Returns:
            候选列表
        """
        # 提取数字
        number = self._extract_number(hard_label)

        if number is None:
            # 无法提取数字，返回原值
            return [hard_label]

        try:
            value = int(number)
        except ValueError:
            return [hard_label]

        candidates = set()

        # 1. 基础候选（原值）
        candidates.add(str(value))

        # 2. 邻域候选
        candidates.update(self._generate_neighbors(value))

        # 3. 数字重排候选
        if self.config.include_permutations:
            candidates.update(self._generate_permutations(number))

        # 4. OCR变体候选
        if self.config.include_ocr_variants:
            candidates.update(self._generate_ocr_variants(number))

        # 过滤和限制
        candidates = self._filter_candidates(candidates, max_candidates)

        return sorted(candidates, key=lambda x: (len(x), x))

    def _extract_number(self, text: str) -> Optional[str]:
        """从文本中提取数字"""
        match = re.search(r'(\d+)', str(text))
        return match.group(1) if match else None

    def _generate_neighbors(self, value: int) -> List[str]:
        """生成邻域候选"""
        neighbors = set()

        # 精确邻域
        for delta in self.config.precise_deltas:
            neighbor = value + delta
            if neighbor >= 0:
                neighbors.add(str(neighbor))

        # 十位邻域
        for delta in self.config.decade_deltas:
            neighbor = value + delta
            if neighbor >= 0:
                neighbors.add(str(neighbor))

        # 百位邻域
        for delta in self.config.century_deltas:
            neighbor = value + delta
            if neighbor >= 0:
                neighbors.add(str(neighbor))

        return list(neighbors)

    def _generate_permutations(self, number: str) -> List[str]:
        """生成数字重排候选"""
        if len(number) > 4:
            # 超过4位数字，不进行重排（避免候选爆炸）
            return []

        perms = set()
        for perm in permutations(number):
            perm_str = ''.join(perm)
            if perm_str != number:  # 排除原值
                perms.add(perm_str)

        # 限制重排数量
        return list(perms)[:min(len(perms), 10)]

    def _generate_ocr_variants(self, number: str) -> List[str]:
        """生成OCR识别的常见错误变体"""
        variants = set()

        # 1. 字符混淆替换
        for i, char in enumerate(number):
            if char in self.CHAR_CONFUSION_MAP:
                for replacement in self.CHAR_CONFUSION_MAP[char]:
                    variant = number[:i] + replacement + number[i+1:]
                    variants.add(variant)

        # 2. 空格插入（模拟分割错误）
        for i in range(1, len(number)):
            variant = number[:i] + ' ' + number[i:]
            variants.add(variant)

        return list(variants)

    def _filter_candidates(
        self,
        candidates: set,
        max_candidates: int
    ) -> List[str]:
        """过滤和限制候选数量"""
        # 移除负数
        candidates = {c for c in candidates if self._is_valid_candidate(c)}

        # 按优先级排序（数字优先，然后是字母+数字，最后是空格）
        def sort_key(x):
            has_letter = any(c.isalpha() for c in x)
            has_space = ' ' in x
            return (has_space, has_letter, len(x), x)

        sorted_candidates = sorted(candidates, key=sort_key)

        # 限制数量
        return sorted_candidates[:max_candidates]

    def _is_valid_candidate(self, candidate: str) -> bool:
        """检查候选是否有效"""
        # 空候选
        if not candidate or candidate.strip() == '':
            return False

        # 纯空格
        if candidate.strip() == '':
            return False

        return True

    def generate_with_variants(
        self,
        hard_label: str,
        include_word_variants: bool = False
    ) -> Dict[str, List[str]]:
        """
        生成候选池及其变体映射

        Args:
            hard_label: 真值答案
            include_word_variants: 是否包含英文单词变体

        Returns:
            字典：{
                "candidates": 候选列表,
                "variant_map": {候选: 归一化形式}
            }
        """
        candidates = self.generate(hard_label)

        variant_map = {}
        for candidate in candidates:
            # 归一化（移除空格、字母等）
            normalized = self._normalize_candidate(candidate)
            variant_map[candidate] = normalized

        # 添加英文单词变体（如果需要）
        if include_word_variants:
            number = self._extract_number(hard_label)
            if number:
                word_variants = self._generate_word_variants(number)
                for word in word_variants:
                    variant_map[word] = number
                    if word not in candidates:
                        candidates.append(word)

        return {
            "candidates": candidates,
            "variant_map": variant_map
        }

    def _normalize_candidate(self, candidate: str) -> str:
        """归一化候选（提取纯数字）"""
        # 移除空格
        candidate = candidate.replace(' ', '')

        # 提取数字部分
        match = re.search(r'(\d+)', candidate)
        if match:
            return match.group(1)

        return candidate

    def _generate_word_variants(self, number: str) -> List[str]:
        """生成英文单词变体（简化版）"""
        try:
            value = int(number)
        except ValueError:
            return []

        # 仅处理0-20的数字
        if value > 20:
            return []

        number_to_word = {
            0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
            5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
            10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
            14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
            18: "eighteen", 19: "nineteen", 20: "twenty"
        }

        word = number_to_word.get(value)
        return [word] if word else []


class ReadingNumberExtractor:
    """
    精确读数任务提取器（集成候选生成和概率计算）

    使用方法：
    >>> generator = ReadingNumberCandidateGenerator()
    >>> candidates = generator.generate("413")
    >>> # 然后使用Teacher模型评估每个候选的概率
    """

    def __init__(
        self,
        candidate_generator: Optional[ReadingNumberCandidateGenerator] = None
    ):
        """
        初始化提取器

        Args:
            candidate_generator: 候选生成器实例
        """
        self.generator = candidate_generator or ReadingNumberCandidateGenerator()

    def extract(
        self,
        hard_label: str,
        teacher_model=None,
        tokenizer=None,
        question: str = "",
        batch_size: int = 8
    ) -> Dict[str, float]:
        """
        提取精确读数的分布

        Args:
            hard_label: 真值答案
            teacher_model: Teacher模型（可选，用于演示）
            tokenizer: Tokenizer（可选）
            question: 问题文本
            batch_size: 批次大小

        Returns:
            候选分布字典
        """
        # 生成候选池
        candidate_data = self.generator.generate_with_variants(
            hard_label,
            include_word_variants=False
        )

        candidates = candidate_data["candidates"]
        variant_map = candidate_data["variant_map"]

        print(f"生成候选池: {len(candidates)} 个候选")
        print(f"示例: {candidates[:10]}")

        # 实际应用中，这里应该调用Teacher模型评估每个候选
        # 这里仅演示接口
        if teacher_model is None:
            print("⚠️ Teacher模型未提供，返回模拟分布")
            return self._mock_distribution(candidates, hard_label)

        # 真实流程：
        # 1. 构建prompts: [question + "\nAnswer: " + candidate for candidate in candidates]
        # 2. 批量forward
        # 3. 计算序列对数概率
        # 4. Softmax归一化

        # 这里简化处理
        return self._mock_distribution(candidates, hard_label)

    def _mock_distribution(
        self,
        candidates: List[str],
        hard_label: str
    ) -> Dict[str, float]:
        """生成模拟分布（仅用于演示）"""
        distribution = {}

        # 真值最高概率
        for candidate in candidates:
            if candidate == hard_label:
                distribution[candidate] = 0.85
            elif abs(int(candidate) - int(hard_label)) <= 2:
                distribution[candidate] = 0.05
            else:
                distribution[candidate] = 0.01

        # 归一化
        total = sum(distribution.values())
        distribution = {k: v/total for k, v in distribution.items()}

        return distribution


# ===== 测试用例 =====
if __name__ == "__main__":
    print("\n" + "="*70)
    print("精确读数候选生成器测试")
    print("="*70)

    generator = ReadingNumberCandidateGenerator()

    # 测试1：生成候选池
    test_cases = ["413", "42", "2024", "7"]

    for hard_label in test_cases:
        print(f"\n--- 真值: {hard_label} ---")

        candidates = generator.generate(hard_label, max_candidates=30)
        print(f"候选数: {len(candidates)}")
        print(f"候选池: {candidates[:15]}")

        # 带变体映射
        data = generator.generate_with_variants(hard_label, include_word_variants=True)
        print(f"\n变体映射示例:")
        for i, (candidate, normalized) in list(data["variant_map"].items())[:5]:
            print(f"  {candidate} → {normalized}")

    # 测试2：提取器演示
    print("\n" + "="*70)
    print("精确读数提取器演示")
    print("="*70)

    extractor = ReadingNumberExtractor()

    result = extractor.extract("413", teacher_model=None)
    print(f"\n真值: 413")
    print(f"分布 (Top 5):")
    sorted_dist = sorted(result.items(), key=lambda x: x[1], reverse=True)[:5]
    for candidate, prob in sorted_dist:
        print(f"  {candidate}: {prob:.4f}")

    print("\n" + "="*70)
    print("测试完成")
    print("="*70)