"""
后置文本清洗模块
================

对生成的文本（answer、CoT等）进行轻量级后置清洗，移除 Markdown 符号。

清洗规则：
1. 移除所有 ### ## # 标题标记
2. 移除加粗标记 **、斜体 *
3. 移除 - 、1. 、2. 有序/无序列表前缀
4. 将连续换行、多个空格压缩为单个空格
5. 剔除空行

使用方式：
    from src.cleaning.text_cleaner import TextCleaner

    cleaner = TextCleaner()
    clean_text = cleaner.clean(raw_text)
"""

import re
from typing import Optional
import logging


class TextCleaner:
    """
    后置文本清洗器（轻量级，毫秒级）

    移除 Markdown 符号，统一空白符
    """

    def __init__(self):
        """初始化清洗器"""
        self.logger = logging.getLogger(__name__)

        # 预编译正则表达式（性能优化）
        # 1. 标题标记：### ## #
        self.heading_pattern = re.compile(r'^#{1,6}\s*', re.MULTILINE)

        # 2. 加粗/斜体：**text** *text*
        self.bold_pattern = re.compile(r'\*\*([^*]+)\*\*')
        self.italic_pattern = re.compile(r'\*([^*]+)\*')

        # 3. 有序/无序列表：- item, 1. item, 2. item
        self.list_pattern = re.compile(r'^\s*[-\d]+\.\s*', re.MULTILINE)

        # 4. 多个空格/换行
        self.whitespace_pattern = re.compile(r'\s+')

        # 5. 空行
        self.empty_line_pattern = re.compile(r'\n\s*\n')

    def clean(self, text: str) -> str:
        """
        清洗文本

        Args:
            text: 原始文本

        Returns:
            清洗后的文本
        """
        if not text:
            return text

        original_text = text

        # 1. 移除标题标记（### ## #）
        text = self.heading_pattern.sub('', text)

        # 2. 移除加粗标记（**text** -> text）
        text = self.bold_pattern.sub(r'\1', text)

        # 3. 移除斜体标记（*text* -> text）
        text = self.italic_pattern.sub(r'\1', text)

        # 4. 移除列表前缀（- item, 1. item -> item）
        text = self.list_pattern.sub('', text)

        # 5. 压缩连续空白符为单个空格
        text = self.whitespace_pattern.sub(' ', text)

        # 6. 移除首尾空白
        text = text.strip()

        # 记录清洗效果（仅在有变化时）
        if text != original_text:
            self.logger.debug(f"[TextCleaner] 清洗完成: {len(original_text)} -> {len(text)} 字符")

        return text

    def clean_cot_quotes(self, text: str) -> str:
        """
        清洗 CoT 文本中多余的引号

        针对闭合问题 CoT 中的特定模式：
        - "yes" -> yes
        - "no" -> no
        - "yes," -> yes,
        - "no." -> no.

        Args:
            text: 原始文本

        Returns:
            清洗后的文本
        """
        if not text:
            return text

        # 清洗模式：引号包围的答案选项（闭合问题常见）
        # 模式1: "yes" 或 "yes," 或 "yes." 等
        # 模式2: "no" 或 "no," 或 "no." 等
        patterns = [
            # "yes" 相关
            (r'"yes([,\.])?"', r'yes\1'),
            (r'"yes"', 'yes'),
            # "no" 相关
            (r'"no([,\.])?"', r'no\1'),
            (r'"no"', 'no'),
            # 其他常见答案选项（数字、颜色等）
            (r'"(\d+)"', r'\1'),  # "1" -> 1
            (r'"(red|blue|green|yellow|orange|purple|pink|black|white|brown|gray)"', r'\1'),  # 颜色
        ]

        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def clean_cot(self, cot_dict: dict) -> dict:
        """
        清洗 CoT 结构中的文本

        🔧 支持两种格式：
        - 新格式（两段式）：reasoning_paragraph, answer
        - 旧格式（三段式）：structured_reasoning: {observation, analysis, conclusion}

        Args:
            cot_dict: CoT 字典

        Returns:
            清洗后的 CoT 字典
        """
        if not cot_dict:
            return cot_dict

        # ───────────────────────────────────────────────────────
        # 新格式：两段式（reasoning_paragraph + answer）
        # ───────────────────────────────────────────────────────
        if 'reasoning_paragraph' in cot_dict or 'answer' in cot_dict:
            # 清洗推理段落
            if 'reasoning_paragraph' in cot_dict and cot_dict['reasoning_paragraph']:
                text = self.clean_cot_quotes(cot_dict['reasoning_paragraph'])
                cot_dict['reasoning_paragraph'] = self.clean(text)

            # 清洗答案
            if 'answer' in cot_dict and cot_dict['answer']:
                text = self.clean_cot_quotes(cot_dict['answer'])
                cot_dict['answer'] = self.clean(text)

            return cot_dict

        # ───────────────────────────────────────────────────────
        # 旧格式：三段式（structured_reasoning 或直接字段）
        # ───────────────────────────────────────────────────────
        # 处理嵌套的 structured_reasoning
        if 'structured_reasoning' in cot_dict:
            structured = cot_dict['structured_reasoning']
            for key in ['observation', 'analysis', 'conclusion']:
                if key in structured and structured[key]:
                    # 先清洗引号，再清洗其他内容
                    text = self.clean_cot_quotes(structured[key])
                    structured[key] = self.clean(text)
        else:
            # 直接子字段
            for key in ['observation', 'analysis', 'conclusion']:
                if key in cot_dict and cot_dict[key]:
                    # 先清洗引号，再清洗其他内容
                    text = self.clean_cot_quotes(cot_dict[key])
                    cot_dict[key] = self.clean(text)

        return cot_dict


# 全局单例（性能优化）
_cleaner_instance = None

def get_cleaner() -> TextCleaner:
    """获取全局清洗器实例"""
    global _cleaner_instance
    if _cleaner_instance is None:
        _cleaner_instance = TextCleaner()
    return _cleaner_instance


def clean_text(text: str) -> str:
    """
    便捷函数：清洗文本

    Args:
        text: 原始文本

    Returns:
        清洗后的文本
    """
    return get_cleaner().clean(text)


def clean_cot(cot_dict: dict) -> dict:
    """
    便捷函数：清洗 CoT 结构

    Args:
        cot_dict: CoT 字典

    Returns:
        清洗后的 CoT 字典
    """
    return get_cleaner().clean_cot(cot_dict)


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("后置文本清洗模块测试")
    print("="*70)

    cleaner = TextCleaner()

    # 测试用例
    test_cases = [
        # 标题
        ("### Analysis\nThis is a test.", "This is a test."),

        # 加粗
        ("This is **important** text.", "This is important text."),

        # 斜体
        ("This is *emphasized* text.", "This is emphasized text."),

        # 列表
        ("- First item\n- Second item", "First item Second item"),

        # 有序列表
        ("1. First\n2. Second\n3. Third", "First Second Third"),

        # 混合
        ("### Heading\n\n**Bold** and *italic* text.\n\n- List item", "Heading Bold and italic text. List item"),

        # 实际例子（来自用户）
        (
            "The image shows a hot dog sandwich. Here's a detailed explanation:\n\n1. **Bun**: The sandwich is served in a long, soft bun.\n\n2. **Sausage**: Inside the bun, there is a sausage.\n\nIn conclusion, this is a hot dog.",
            "The image shows a hot dog sandwich. Here's a detailed explanation: Bun: The sandwich is served in a long, soft bun. Sausage: Inside the bun, there is a sausage. In conclusion, this is a hot dog."
        )
    ]

    print("\n测试用例：")
    print("-" * 70)

    for i, (input_text, expected) in enumerate(test_cases, 1):
        result = cleaner.clean(input_text)
        status = "✓" if result == expected else "✗"
        print(f"\n{i}. {status}")
        print(f"   输入: {input_text[:50]}...")
        print(f"   输出: {result[:50]}...")
        if result != expected:
            print(f"   期望: {expected[:50]}...")

    # CoT 引号清洗测试
    print("\n" + "="*70)
    print("CoT 引号清洗测试")
    print("="*70)

    cot_test_cases = [
        # yes/no 引号
        ('The primary answer is "yes," as the fire hydrant is red.', 'The primary answer is yes, as the fire hydrant is red.'),
        ('The answer is "no".', 'The answer is no.'),
        ('"yes" or "no"', 'yes or no'),
        # 数字引号
        ('There are "3" dogs.', 'There are 3 dogs.'),
        # 颜色引号
        ('The color is "red".', 'The color is red.'),
        # 混合情况
        ('The answer might be "yes", "no", or "1".', 'The answer might be yes, no, or 1.'),
    ]

    print("\nCoT 引号清洗测试用例：")
    print("-" * 70)

    for i, (input_text, expected) in enumerate(cot_test_cases, 1):
        result = cleaner.clean_cot_quotes(input_text)
        status = "✓" if result == expected else "✗"
        print(f"\n{i}. {status}")
        print(f"   输入: {input_text}")
        print(f"   输出: {result}")
        if result != expected:
            print(f"   期望: {expected}")

    # CoT 结构清洗测试
    print("\n" + "="*70)
    print("CoT 结构清洗测试")
    print("="*70)

    cot_dict = {
        'structured_reasoning': {
            'observation': 'The fire hydrant in the image is predominantly red in color.',
            'analysis': 'The primary answer is "yes," as the fire hydrant\'s main color is red, which directly matches the question. There is no visual evidence suggesting the hydrant is any other color, making "no" an unlikely answer.',
            'conclusion': 'yes'
        }
    }

    print("\n原始 CoT:")
    print(f"  observation: {cot_dict['structured_reasoning']['observation']}")
    print(f"  analysis: {cot_dict['structured_reasoning']['analysis']}")
    print(f"  conclusion: {cot_dict['structured_reasoning']['conclusion']}")

    cleaned_cot = cleaner.clean_cot(cot_dict)

    print("\n清洗后 CoT:")
    print(f"  observation: {cleaned_cot['structured_reasoning']['observation']}")
    print(f"  analysis: {cleaned_cot['structured_reasoning']['analysis']}")
    print(f"  conclusion: {cleaned_cot['structured_reasoning']['conclusion']}")

    print("\n" + "="*70)