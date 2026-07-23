"""
答案格式标准化工具
==================

统一硬标签和软标签的答案格式，确保一致性。

支持格式：
- 'word': 英文单词格式（one, two, three...）
- 'number': 数字格式（1, 2, 3...）

使用方法：
    from src.utils.answer_normalizer import normalize_answer, normalize_distribution_keys

    # 转换单个答案
    normalize_answer('1', 'word')  # 返回 'one'

    # 转换概率分布的键
    normalize_distribution_keys({'1': 0.8, '2': 0.2}, 'word')  # 返回 {'one': 0.8, 'two': 0.2}
"""

from typing import Dict


# ==================
# 数字到英文的映射
# ==================

NUMBER_TO_WORD = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
    '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen',
    '14': 'fourteen', '15': 'fifteen', '16': 'sixteen', '17': 'seventeen',
    '18': 'eighteen', '19': 'nineteen', '20': 'twenty'
}

# 英文到数字的映射
WORD_TO_NUMBER = {v: k for k, v in NUMBER_TO_WORD.items()}


def normalize_answer(answer: str, target_format: str = 'word') -> str:
    """
    标准化答案格式

    Args:
        answer: 原始答案（可能是 "1" 或 "one"）
        target_format: 目标格式，'word'（英文）或 'number'（阿拉伯数字）

    Returns:
        标准化后的答案

    Examples:
        >>> normalize_answer('1', 'word')
        'one'
        >>> normalize_answer('one', 'number')
        '1'
        >>> normalize_answer('yes', 'word')
        'yes'  # 非数字答案保持不变
    """
    if not answer:
        return answer

    # 统一转小写
    answer = answer.strip().lower()

    # 转换为英文格式（默认格式，与软标签一致）
    if target_format == 'word':
        # 如果已经是英文单词，直接返回
        if answer in WORD_TO_NUMBER:
            return answer
        # 如果是数字，转换为英文
        if answer in NUMBER_TO_WORD:
            return NUMBER_TO_WORD[answer]
        # 其他情况（如 "yes", "no", "kitchen" 等）直接返回
        return answer

    # 转换为数字格式
    elif target_format == 'number':
        # 如果已经是数字，直接返回
        if answer in NUMBER_TO_WORD:
            return answer
        # 如果是英文单词，转换为数字
        if answer in WORD_TO_NUMBER:
            return WORD_TO_NUMBER[answer]
        # 其他情况直接返回
        return answer

    return answer


def normalize_distribution_keys(distribution: Dict[str, float], target_format: str = 'word') -> Dict[str, float]:
    """
    标准化概率分布的键

    Args:
        distribution: 原始概率分布，如 {'one': 0.25, 'two': 0.17} 或 {'1': 0.25, '2': 0.17}
        target_format: 目标格式，'word' 或 'number'

    Returns:
        标准化后的概率分布

    Examples:
        >>> normalize_distribution_keys({'1': 0.8, '2': 0.2}, 'word')
        {'one': 0.8, 'two': 0.2}
        >>> normalize_distribution_keys({'one': 0.8, 'two': 0.2}, 'number')
        {'1': 0.8, '2': 0.2}
    """
    if not distribution:
        return distribution

    return {
        normalize_answer(key, target_format): value
        for key, value in distribution.items()
    }


def is_numeric_answer(answer: str) -> bool:
    """
    判断答案是否是数字类型（包括数字字符串或英文数字单词）

    Args:
        answer: 待判断的答案

    Returns:
        True 如果是数字类型

    Examples:
        >>> is_numeric_answer('1')
        True
        >>> is_numeric_answer('one')
        True
        >>> is_numeric_answer('kitchen')
        False
    """
    if not answer:
        return False

    answer = answer.strip().lower()

    return answer in NUMBER_TO_WORD or answer in WORD_TO_NUMBER