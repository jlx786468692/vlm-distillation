"""
多Token实体归一化工具
======================

统一多Token实体的写法，避免同一实体多种文本形式增加训练难度。

核心功能：
1. 复合词统一写法（如 "hot dog" -> "hotdog"）
2. 实体完整性校验（检测中途截断）
3. 幻觉过滤（检测未见过的复合实体）

使用方法：
    from src.utils.multi_token_normalizer import MultiTokenNormalizer

    normalizer = MultiTokenNormalizer()

    # 归一化
    text = normalizer.normalize("A hot dog with ketchup")
    # 输出: "A hotdog with ketchup"

    # 实体完整性校验
    is_valid, issue = normalizer.validate_entity("hot dog")
    # 输出: (True, None)

    # 幻觉检测
    is_hallucination = normalizer.is_hallucination("spacehotdog")
    # 输出: True
"""

from typing import Dict, List, Tuple, Optional, Set
import re
from pathlib import Path
import json


class MultiTokenNormalizer:
    """
    多Token实体归一化器

    功能：
    1. 复合词统一写法
    2. 实体完整性校验
    3. 幻觉过滤
    """

    # ==================
    # 多Token实体映射表
    # ==================

    # 常见的多Token实体（拆分写法 -> 合并写法）
    MULTI_TOKEN_ENTITIES = {
        # 食物类
        "hot dog": "hotdog",
        "hot dogs": "hotdogs",
        "ice cream": "icecream",
        "french fries": "frenchfries",
        "banana split": "bananasplit",
        "soft drink": "softdrink",

        # 交通工具类
        "fire truck": "firetruck",
        "fire trucks": "firetrucks",
        "motor cycle": "motorcycle",
        "motor cycles": "motorcycles",
        "motor bike": "motorbike",
        "motor bikes": "motorbikes",

        # 其他常见实体
        "baseball bat": "baseballbat",
        "tennis racket": "tennisracket",
        "parking meter": "parkingmeter",
        "traffic light": "trafficlight",
    }

    # 实体白名单（用于幻觉检测）
    ENTITY_WHITELIST = set(MULTI_TOKEN_ENTITIES.values())

    # 常见的前缀和后缀（用于检测截断）
    COMMON_PREFIXES = {"hot", "fire", "motor", "ice", "soft", "tennis", "baseball", "parking", "traffic"}
    COMMON_SUFFIXES = {"dog", "truck", "cycle", "bike", "cream", "drink", "racket", "bat", "meter", "light"}

    def __init__(self, custom_entities_file: Optional[str] = None):
        """
        初始化多Token实体归一化器

        Args:
            custom_entities_file: 自定义实体映射文件路径（JSON格式）
        """
        self.entities = self.MULTI_TOKEN_ENTITIES.copy()
        self.whitelist = self.ENTITY_WHITELIST.copy()

        # 加载自定义实体映射
        if custom_entities_file:
            self._load_custom_entities(custom_entities_file)

        # 构建反向映射（用于快速查找）
        self._build_reverse_mapping()

    def _load_custom_entities(self, file_path: str):
        """加载自定义实体映射"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                custom_entities = json.load(f)

            self.entities.update(custom_entities)
            self.whitelist.update(custom_entities.values())

        except Exception as e:
            print(f"⚠ 加载自定义实体映射失败: {e}")

    def _build_reverse_mapping(self):
        """构建反向映射（合并写法 -> 拆分写法列表）"""
        self.reverse_mapping = {}
        for split_form, merged_form in self.entities.items():
            if merged_form not in self.reverse_mapping:
                self.reverse_mapping[merged_form] = []
            self.reverse_mapping[merged_form].append(split_form)

    def normalize(self, text: str) -> str:
        """
        归一化文本中的多Token实体

        Args:
            text: 原始文本

        Returns:
            归一化后的文本

        Examples:
            >>> normalizer.normalize("A hot dog with ketchup")
            "A hotdog with ketchup"
        """
        if not text:
            return text

        normalized_text = text

        # 按照实体长度排序（优先匹配更长的实体）
        sorted_entities = sorted(
            self.entities.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )

        # 替换所有多Token实体
        for split_form, merged_form in sorted_entities:
            # 使用正则表达式替换（忽略大小写）
            pattern = re.compile(re.escape(split_form), re.IGNORECASE)
            normalized_text = pattern.sub(merged_form, normalized_text)

        return normalized_text

    def normalize_answer(self, answer: str) -> str:
        """
        归一化单个答案（针对闭合样本）

        Args:
            answer: 原始答案

        Returns:
            归一化后的答案
        """
        # 先进行基本归一化（小写、去除空格）
        answer = answer.strip().lower()

        # 再进行多Token归一化
        return self.normalize(answer)

    def validate_entity(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        校验实体完整性

        Args:
            text: 待校验的文本

        Returns:
            (is_valid, issue)
            - is_valid: 是否有效
            - issue: 问题描述（如果无效）
        """
        if not text:
            return False, "空文本"

        text = text.strip().lower()

        # 检查是否是已知的完整实体
        if text in self.whitelist or text in self.entities:
            return True, None

        # 检查是否是截断的实体（如只有 "hot"，缺失 "dog"）
        if text in self.COMMON_PREFIXES:
            return False, f"实体截断：'{text}' 可能是多Token实体的前缀，但缺失后缀"

        if text in self.COMMON_SUFFIXES:
            return False, f"实体截断：'{text}' 可能是多Token实体的后缀，但缺失前缀"

        # 检查是否包含空格（可能是拆分写法）
        if ' ' in text:
            # 尝试归一化
            normalized = self.normalize(text)
            if normalized != text:
                return True, f"需要归一化：'{text}' -> '{normalized}'"

        return True, None

    def is_hallucination(self, entity: str, threshold: float = 0.8) -> bool:
        """
        检测是否是幻觉（未见过的复合实体）

        Args:
            entity: 待检测的实体
            threshold: 相似度阈值

        Returns:
            True 如果是幻觉
        """
        if not entity:
            return False

        entity = entity.strip().lower()

        # 如果在白名单中，不是幻觉
        if entity in self.whitelist:
            return False

        # 检查是否包含常见实体成分（可能是未见过的复合实体）
        for prefix in self.COMMON_PREFIXES:
            if entity.startswith(prefix) and len(entity) > len(prefix):
                # 检查后缀是否在白名单中
                suffix = entity[len(prefix):]
                if suffix not in self.COMMON_SUFFIXES and suffix not in self.whitelist:
                    # 可能是幻觉（如 "spacehotdog"）
                    return True

        for suffix in self.COMMON_SUFFIXES:
            if entity.endswith(suffix) and len(entity) > len(suffix):
                # 检查前缀是否在白名单中
                prefix = entity[:-len(suffix)]
                if prefix not in self.COMMON_PREFIXES and prefix not in self.whitelist:
                    # 可能是幻觉
                    return True

        return False

    def detect_truncated_entities(self, text: str) -> List[Dict]:
        """
        检测文本中的截断实体

        Args:
            text: 待检测的文本

        Returns:
            截断实体列表
        """
        truncated_entities = []

        words = text.split()

        for i, word in enumerate(words):
            word_lower = word.lower()

            # 检查是否是截断的前缀
            if word_lower in self.COMMON_PREFIXES:
                # 检查下一个词是否是预期的后缀
                if i + 1 < len(words):
                    next_word = words[i + 1].lower()
                    expected_entity = f"{word_lower} {next_word}"

                    if expected_entity in self.entities:
                        # 完整实体，正常
                        pass
                    else:
                        # 可能是截断或幻觉
                        truncated_entities.append({
                            "position": i,
                            "text": word,
                            "type": "prefix",
                            "suggestion": f"可能是多Token实体的前缀，建议检查下一个token"
                        })
                else:
                    # 文本结尾，确实是截断
                    truncated_entities.append({
                        "position": i,
                        "text": word,
                        "type": "truncated_prefix",
                        "suggestion": f"多Token实体截断：'{word}' 缺失后缀"
                    })

        return truncated_entities

    def get_normalized_forms(self, entity: str) -> Dict[str, str]:
        """
        获取实体的所有可能形式

        Args:
            entity: 实体名称

        Returns:
            {
                "original": 原始形式,
                "merged": 合并形式,
                "split": 拆分形式（如果有）
            }
        """
        entity_lower = entity.strip().lower()

        result = {
            "original": entity,
            "merged": None,
            "split": None
        }

        # 如果已经是合并形式
        if entity_lower in self.reverse_mapping:
            result["merged"] = entity_lower
            result["split"] = self.reverse_mapping[entity_lower][0]  # 取第一个拆分形式

        # 如果是拆分形式
        elif entity_lower in self.entities:
            result["merged"] = self.entities[entity_lower]
            result["split"] = entity_lower

        return result


# ==================
# 便捷函数
# ==================

def normalize_multi_token_text(text: str) -> str:
    """
    归一化文本中的多Token实体（便捷函数）

    Args:
        text: 原始文本

    Returns:
        归一化后的文本
    """
    normalizer = MultiTokenNormalizer()
    return normalizer.normalize(text)


def validate_multi_token_entity(entity: str) -> Tuple[bool, Optional[str]]:
    """
    校验多Token实体完整性（便捷函数）

    Args:
        entity: 实体名称

    Returns:
        (is_valid, issue)
    """
    normalizer = MultiTokenNormalizer()
    return normalizer.validate_entity(entity)


def is_multi_token_hallucination(entity: str) -> bool:
    """
    检测多Token实体是否是幻觉（便捷函数）

    Args:
        entity: 实体名称

    Returns:
        True 如果是幻觉
    """
    normalizer = MultiTokenNormalizer()
    return normalizer.is_hallucination(entity)


# ==================
# 使用示例
# ==================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("多Token实体归一化工具测试")
    print("="*70)

    normalizer = MultiTokenNormalizer()

    # 测试1: 归一化
    test_texts = [
        "A hot dog with ketchup on top.",
        "The fire truck is parked nearby.",
        "He is riding a motor cycle.",
        "There are ice cream and french fries."
    ]

    print("\n【测试1: 归一化】")
    for text in test_texts:
        normalized = normalizer.normalize(text)
        print(f"  原文: {text}")
        print(f"  归一化: {normalized}")
        print()

    # 测试2: 实体完整性校验
    test_entities = [
        "hot dog",
        "hot",
        "dog",
        "hotdog"
    ]

    print("\n【测试2: 实体完整性校验】")
    for entity in test_entities:
        is_valid, issue = normalizer.validate_entity(entity)
        status = "✓" if is_valid else "✗"
        print(f"  {status} 实体: '{entity}'")
        if issue:
            print(f"      问题: {issue}")

    # 测试3: 幻觉检测
    test_hallucinations = [
        "hotdog",
        "spacehotdog",
        "firetruck",
        "watertruck"
    ]

    print("\n【测试3: 幻觉检测】")
    for entity in test_hallucinations:
        is_hallucination = normalizer.is_hallucination(entity)
        status = "幻觉" if is_hallucination else "正常"
        print(f"  实体: '{entity}' -> {status}")

    print("\n" + "="*70)