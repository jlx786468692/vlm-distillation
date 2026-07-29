"""
开放样本正则清洗器（官方标准）
================================

严格实现官方蒸馏脚本的正则清洗流程：
1. Token解码基础清洗（去除特殊符号）
2. 正则规则清洗（Markdown、空白字符、乱码）
3. 格式隔离校验（检测闭合任务CoT结构）
4. 长度阈值筛选（answer长度<80字符丢弃）

使用方式：
    from src.cleaning.open_answer_cleaner import OpenAnswerCleaner

    cleaner = OpenAnswerCleaner()
    cleaned_answer = cleaner.clean(raw_answer)
"""

import re
from typing import Dict, Any, Tuple
import logging


class OpenAnswerCleaner:
    """
    开放样本答案正则清洗器（官方标准）

    核心清洗流程：
    1. Token解码基础清洗
    2. 正则规则清洗（Markdown、空白字符、乱码）
    3. 格式隔离校验（CoT结构检测）
    4. 长度阈值筛选
    """

    # 官方标准：长度阈值（字符数）
    MIN_ANSWER_CHARS = 80    # 最少有效字符数
    MAX_ANSWER_CHARS = 2000  # 最大字符数（防止冗余重复）

    def __init__(self):
        """初始化清洗器"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("✓ 开放样本答案正则清洗器初始化完成")
        self.logger.info(f"  - 最小字符数: {self.MIN_ANSWER_CHARS}")
        self.logger.info(f"  - 最大字符数: {self.MAX_ANSWER_CHARS}")

    def clean(self, answer: str) -> Tuple[str, bool, Dict[str, Any]]:
        """
        清洗开放问题答案（完整流程）

        Args:
            answer: 原始答案字符串

        Returns:
            (cleaned_answer, is_valid, metadata)
            - cleaned_answer: 清洗后的答案
            - is_valid: 是否有效（True=保留，False=丢弃）
            - metadata: 清洗元数据（包含清洗动作、问题列表）
        """
        metadata = {
            'original_length': len(answer),
            'cleaning_actions': [],
            'issues': []
        }

        # ───────────────────────────────────────────────────────
        # Step 1: Token解码基础清洗
        # ───────────────────────────────────────────────────────
        answer = self._token_decode_cleaning(answer)
        metadata['cleaning_actions'].append('token_decode_cleaning')

        # ───────────────────────────────────────────────────────
        # Step 2: 正则规则清洗
        # ───────────────────────────────────────────────────────
        answer, regex_actions = self._regex_cleaning(answer)
        metadata['cleaning_actions'].extend(regex_actions)

        # ───────────────────────────────────────────────────────
        # Step 3: 格式隔离校验
        # ───────────────────────────────────────────────────────
        is_isolated, format_issues = self._format_isolation_check(answer)
        metadata['issues'].extend(format_issues)

        if not is_isolated:
            # 检测到闭合任务格式，直接丢弃
            metadata['final_length'] = len(answer)
            return answer, False, metadata

        # ───────────────────────────────────────────────────────
        # Step 4: 长度阈值筛选
        # ───────────────────────────────────────────────────────
        answer_length = len(answer.strip())

        if answer_length < self.MIN_ANSWER_CHARS:
            metadata['issues'].append(f"答案过短：{answer_length}字符 < {self.MIN_ANSWER_CHARS}")
            metadata['final_length'] = answer_length
            return answer, False, metadata

        if answer_length > self.MAX_ANSWER_CHARS:
            metadata['issues'].append(f"答案过长：{answer_length}字符 > {self.MAX_ANSWER_CHARS}")
            metadata['final_length'] = answer_length
            return answer, False, metadata

        # ───────────────────────────────────────────────────────
        # 清洗完成
        # ───────────────────────────────────────────────────────
        metadata['final_length'] = len(answer)
        metadata['cleaning_actions'].append('length_check_passed')

        return answer, True, metadata

    def _token_decode_cleaning(self, text: str) -> str:
        """
        Token解码基础清洗

        作用：
        - 去除模型生成的系统/对话模板标记（<|im_start|><|im_end|>等）
        - 去除分词残留空白字符
        - 去除特殊符号

        Args:
            text: 原始文本

        Returns:
            清洗后的文本
        """
        # 去除常见的特殊token标记
        special_tokens = [
            '<|im_start|>',
            '<|im_end|>',
            '<|endoftext|>',
            '<|im_sep|>',
            '<start_of_turn>',
            '<end_of_turn>',
            '<|begin_of_text|>',
            '<|eot_id|>',
        ]

        for token in special_tokens:
            text = text.replace(token, '')

        # 去除控制字符（保留换行符和制表符）
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

        # 去除零宽字符
        text = re.sub(r'[​-‏ -  -⁯﻿]', '', text)

        return text.strip()

    def _regex_cleaning(self, text: str) -> Tuple[str, list]:
        """
        正则规则清洗（官方标准）

        清洗内容：
        1. Markdown语法清除（#标题、列表、加粗、分割线）
        2. 空白字符归一化（连续换行、多余空格）
        3. 乱码、特殊符号过滤
        4. 无意义文本清除

        Args:
            text: 原始文本

        Returns:
            (cleaned_text, cleaning_actions)
        """
        actions = []
        original_text = text

        # ───────────────────────────────────────────────────────
        # 1. Markdown语法清除
        # ───────────────────────────────────────────────────────

        # 清除标题标记（# ## ###）
        if re.search(r'^#{1,6}\s+', text, re.MULTILINE):
            text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
            actions.append('remove_markdown_headers')

        # 清除有序列表（1. 2. 3.）
        if re.search(r'^\d+\.\s+', text, re.MULTILINE):
            text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
            actions.append('remove_ordered_lists')

        # 清除无序列表（- *）
        if re.search(r'^[-*]\s+', text, re.MULTILINE):
            text = re.sub(r'^[-*]\s+', '', text, flags=re.MULTILINE)
            actions.append('remove_unordered_lists')

        # 清除加粗标记（**text**）
        if re.search(r'\*\*[^*]+\*\*', text):
            text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
            actions.append('remove_bold_markers')

        # 清除斜体标记（*text*）
        if re.search(r'\*[^*]+\*', text):
            text = re.sub(r'\*([^*]+)\*', r'\1', text)
            actions.append('remove_italic_markers')

        # 清除分割线（---）
        if re.search(r'^---+$', text, re.MULTILINE):
            text = re.sub(r'^---+$', '', text, flags=re.MULTILINE)
            actions.append('remove_horizontal_rules')

        # 清除代码块标记（```）
        if '```' in text:
            text = re.sub(r'```[\s\S]*?```', '', text)
            actions.append('remove_code_blocks')

        # ───────────────────────────────────────────────────────
        # 2. 空白字符归一化
        # ───────────────────────────────────────────────────────

        # 压缩连续换行（保留段落结构）
        if re.search(r'\n{3,}', text):
            text = re.sub(r'\n{3,}', '\n\n', text)
            actions.append('normalize_line_breaks')

        # 去除行尾空格
        if re.search(r' +\n', text):
            text = re.sub(r' +\n', '\n', text)
            actions.append('remove_trailing_spaces')

        # 去除行首空格（保留缩进）
        if re.search(r'\n +', text):
            text = re.sub(r'\n +', '\n', text)
            actions.append('remove_leading_spaces')

        # 压缩多个连续空格为单个空格
        if re.search(r' {2,}', text):
            text = re.sub(r' {2,}', ' ', text)
            actions.append('normalize_spaces')

        # ───────────────────────────────────────────────────────
        # 3. 乱码、特殊符号过滤
        # ───────────────────────────────────────────────────────

        # 去除无法显示的Unicode乱码
        text_before = text
        text = re.sub(r'[�]', '', text)
        if text != text_before:
            actions.append('remove_unicode_garbage')

        # 去除多余emoji（保留常见emoji）
        text_before = text
        text = re.sub(r'[^\x00-\x7F一-鿿︀-️‍\U0001F600-\U0001F64F\U0001F300-\U0001F5FF]', '', text)
        if text != text_before:
            actions.append('filter_emoji')

        # ───────────────────────────────────────────────────────
        # 4. 无意义文本清除
        # ───────────────────────────────────────────────────────

        # 去除重复的空段落标记
        if re.search(r'\n\s*\n\s*\n', text):
            text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
            actions.append('remove_empty_paragraphs')

        # 去除开头的assistant标记
        prefixes = ['assistant\n', 'Assistant\n', 'ASSISTANT\n', 'Assistant: ', 'assistant: ']
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                actions.append(f'remove_prefix_{prefix.strip()}')
                break

        # 最终清理
        text = text.strip()

        return text, actions

    def _format_isolation_check(self, text: str) -> Tuple[bool, list]:
        """
        格式隔离校验（官方标准）

        检测闭合任务相关文本：
        1. 三段式CoT结构（Observation/Analysis/Conclusion）
        2. 候选词列表、概率数值
        3. hard_label/soft_label相关描述

        Args:
            text: 答案文本

        Returns:
            (is_isolated, issues)
            - is_isolated: True=格式干净，False=包含闭合任务格式
            - issues: 问题列表
        """
        issues = []
        text_lower = text.lower()

        # ───────────────────────────────────────────────────────
        # 检测1：三段式CoT结构
        # ───────────────────────────────────────────────────────

        # 检测 "Observation:", "Analysis:", "Conclusion:" 标记
        cot_markers = [
            'observation:',
            'analysis:',
            'conclusion:',
            'final answer:',
            'allowed answers:',
            'primary answer:',
        ]

        detected_markers = []
        for marker in cot_markers:
            if marker in text_lower:
                detected_markers.append(marker)

        if len(detected_markers) >= 2:
            issues.append(f"检测到CoT结构标记：{detected_markers}")
            return False, issues

        # ───────────────────────────────────────────────────────
        # 检测2：候选词列表、概率分布
        # ───────────────────────────────────────────────────────

        # 检测 "answer distribution:", "probability:" 等关键词
        distribution_markers = [
            'answer distribution',
            'probability distribution',
            'candidate answers',
            'allowed answers',
            'primary answer',
        ]

        for marker in distribution_markers:
            if marker in text_lower:
                issues.append(f"检测到候选集/概率分布标记：{marker}")
                return False, issues

        # ───────────────────────────────────────────────────────
        # 检测3：hard_label/soft_label相关描述
        # ───────────────────────────────────────────────────────

        # 检测JSON格式的标签数据
        if re.search(r'"hard_label":\s*{', text) or re.search(r'"soft_label":\s*{', text):
            issues.append("检测到hard_label/soft_label JSON结构")
            return False, issues

        # ───────────────────────────────────────────────────────
        # 检测4：数字列表（可能是候选答案）
        # ───────────────────────────────────────────────────────

        # 检测类似 "1. yes 2. no 3. maybe" 的格式
        if re.search(r'\d+\.\s+\w+\s+\d+\.\s+\w+\s+\d+\.\s+\w+', text):
            issues.append("检测到候选答案列表格式")
            return False, issues

        # ───────────────────────────────────────────────────────
        # 检测5：概率数值（如 "0.85, 0.08, 0.05"）
        # ───────────────────────────────────────────────────────

        # 检测连续的浮点数（可能是概率分布）
        if re.search(r'0\.\d+,\s*0\.\d+,\s*0\.\d+', text):
            issues.append("检测到概率分布数值")
            return False, issues

        # ───────────────────────────────────────────────────────
        # 格式干净
        # ───────────────────────────────────────────────────────

        return True, issues

    def is_valid_length(self, text: str) -> Tuple[bool, str]:
        """
        检查答案长度是否在有效范围内

        Args:
            text: 答案文本

        Returns:
            (is_valid, reason)
        """
        length = len(text.strip())

        if length < self.MIN_ANSWER_CHARS:
            return False, f"答案过短：{length}字符 < {self.MIN_ANSWER_CHARS}"

        if length > self.MAX_ANSWER_CHARS:
            return False, f"答案过长：{length}字符 > {self.MAX_ANSWER_CHARS}"

        return True, "长度有效"


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    cleaner = OpenAnswerCleaner()

    print("\n" + "="*70)
    print("开放样本答案正则清洗器测试")
    print("="*70)

    # 测试样本
    test_samples = [
        {
            "name": "正常样本（PETA问题）",
            "answer": "PETA is an animal rights organization that opposes animal exploitation for entertainment or tourism. The image shows an elephant being ridden by tourists. Elephants used for rides often endure harsh training, confinement and physical strain, which violates animal welfare standards. This exploitative use of elephants would make PETA advocates upset."
        },
        {
            "name": "包含Markdown的样本",
            "answer": "### Analysis\n\n1. The elephant is being used for rides.\n2. This involves animal exploitation.\n\n**Conclusion**: PETA would be upset about this."
        },
        {
            "name": "包含CoT结构的样本",
            "answer": "Observation: The image shows an elephant being ridden by tourists.\nAnalysis: Animal rights organizations oppose this practice.\nConclusion: PETA advocates would be upset."
        },
        {
            "name": "过短样本",
            "answer": "Yes"
        },
        {
            "name": "包含候选集的样本",
            "answer": "The answer could be one of: yes, no, maybe. The probability distribution is 0.7, 0.2, 0.1."
        }
    ]

    print("\n测试结果：")
    print("-" * 70)

    for test in test_samples:
        cleaned, is_valid, metadata = cleaner.clean(test["answer"])

        status = "✓" if is_valid else "✗"
        print(f"\n{status} {test['name']}:")
        print(f"  原始长度: {metadata['original_length']}")
        print(f"  最终长度: {metadata['final_length']}")
        print(f"  有效: {is_valid}")

        if metadata['cleaning_actions']:
            print(f"  清洗动作: {metadata['cleaning_actions']}")

        if metadata['issues']:
            print(f"  问题: {metadata['issues']}")

        if is_valid:
            print(f"  清洗后答案: {cleaned[:100]}...")

    print("\n" + "="*70)
    print("官方标准清洗流程：")
    print("  1. Token解码基础清洗")
    print("  2. 正则规则清洗（Markdown、空白字符、乱码）")
    print("  3. 格式隔离校验（CoT结构检测）")
    print("  4. 长度阈值筛选（<80字符丢弃）")
    print("="*70)