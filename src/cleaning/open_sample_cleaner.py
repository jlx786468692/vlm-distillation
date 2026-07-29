"""
开放样本清洗器（官方标准）
==========================

严格按照官方 filter_tool.py 标准实现开放样本清洗。

官方标准：开放样本仅 4 条基础规则
1. 文本长度过滤：回答 token < 8（过短无有效信息）或 > 512（冗余重复描述）直接丢弃
2. 重复文本过滤：回答存在大面积重复句式、词语循环，标记脏样本剔除
3. 图像-回答一致性粗校验：回答完全脱离画面、描述不存在物体（重度幻觉）丢弃
4. 空输出兜底过滤：模型生成空字符串、全空白文本直接丢弃

不执行闭合集相关校验：
- 硬标签是否在候选集
- CoT 结论匹配
- 推测词数量过滤
"""

import re
from typing import Dict, Any, List
import logging


class OpenSampleCleaner:
    """
    开放样本清洗器（官方标准）

    仅应用 4 条基础规则，不做闭合集校验
    """

    # 官方标准：长度阈值
    MIN_ANSWER_TOKENS = 8    # 官方标准：最少 token 数
    MAX_ANSWER_TOKENS = 512  # 官方标准：最多 token 数

    def __init__(self):
        """初始化清洗器"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("✓ 开放样本清洗器初始化完成（官方标准）")
        self.logger.info(f"  - 最小 token 数: {self.MIN_ANSWER_TOKENS}")
        self.logger.info(f"  - 最大 token 数: {self.MAX_ANSWER_TOKENS}")

    def clean(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        清洗开放样本（官方标准 4 条规则）

        Args:
            sample: 样本数据（包含 answer 字段）

        Returns:
            {
                "is_valid": bool,
                "issues": List[str],
                "actions": List[str]
            }
        """
        issues = []
        actions = []
        is_valid = True

        # 获取回答文本
        answer = sample.get('answer', '')

        # ───────────────────────────────────────────────────────
        # 规则1：空输出兜底过滤
        # ───────────────────────────────────────────────────────
        if not answer or not answer.strip():
            issues.append("空输出：模型生成空字符串")
            actions.append("直接丢弃")
            return {
                "is_valid": False,
                "issues": issues,
                "actions": actions
            }

        answer = answer.strip()

        # ───────────────────────────────────────────────────────
        # 规则2：文本长度过滤
        # 官方标准：token < 8 或 > 512 丢弃
        # ───────────────────────────────────────────────────────
        # 简单分词（空格分隔）
        tokens = answer.split()
        token_count = len(tokens)

        if token_count < self.MIN_ANSWER_TOKENS:
            issues.append(f"文本过短：{token_count} tokens < {self.MIN_ANSWER_TOKENS}")
            actions.append("直接丢弃")
            is_valid = False

        elif token_count > self.MAX_ANSWER_TOKENS:
            issues.append(f"文本过长：{token_count} tokens > {self.MAX_ANSWER_TOKENS}")
            actions.append("直接丢弃")
            is_valid = False

        # ───────────────────────────────────────────────────────
        # 规则3：重复文本过滤
        # 大面积重复句式、词语循环
        # ───────────────────────────────────────────────────────
        if is_valid and self._has_repetition(answer):
            issues.append("重复文本：存在大面积重复句式或词语循环")
            actions.append("标记为脏样本并剔除")
            is_valid = False

        # ───────────────────────────────────────────────────────
        # 规则4：图像-回答一致性粗校验
        # 检测重度幻觉：回答完全脱离画面
        # ───────────────────────────────────────────────────────
        # 注意：这是粗校验，真正的图像-回答一致性需要结合图像内容
        # 这里仅检测明显的幻觉标记

        if is_valid and self._has_heavy_hallucination(answer):
            issues.append("重度幻觉：回答完全脱离画面")
            actions.append("直接丢弃")
            is_valid = False

        # ───────────────────────────────────────────────────────
        # 返回清洗结果
        # ───────────────────────────────────────────────────────
        return {
            "is_valid": is_valid,
            "issues": issues,
            "actions": actions
        }

    def _has_repetition(self, text: str) -> bool:
        """
        检测重复文本（官方标准）

        检测方法：
        1. 重复短语检测（连续重复 ≥ 3 次）
        2. 重复句式检测（相同句式循环）

        Args:
            text: 文本内容

        Returns:
            是否存在重复
        """
        # 简单检测：连续重复短语
        # 例如："The cat is The cat is The cat is"

        # 分词
        words = text.lower().split()

        # 检测连续重复短语（3 个词一组）
        if len(words) >= 9:  # 至少 3 组
            for i in range(0, len(words) - 6, 3):
                phrase1 = ' '.join(words[i:i+3])
                phrase2 = ' '.join(words[i+3:i+6])
                phrase3 = ' '.join(words[i+6:i+9])

                if phrase1 == phrase2 == phrase3:
                    return True

        # 检测重复单词（连续出现 ≥ 5 次）
        # 例如："cat cat cat cat cat"
        word_counts = {}
        prev_word = None
        consecutive_count = 0

        for word in words:
            if word == prev_word:
                consecutive_count += 1
                if consecutive_count >= 5:
                    return True
            else:
                consecutive_count = 1
                prev_word = word

        return False

    def _has_heavy_hallucination(self, text: str) -> bool:
        """
        检测重度幻觉（官方标准）

        检测方法：
        1. 检测明显的幻觉标记（"I cannot see", "not visible in the image"）
        2. 检测完全脱离画面的描述

        Args:
            text: 文本内容

        Returns:
            是否存在重度幻觉
        """
        text_lower = text.lower()

        # 官方标准：检测明显的幻觉标记
        hallucination_markers = [
            "i cannot see",          # 无法看到
            "not visible in the",    # 图像中不可见
            "not present in the",    # 不存在于图像
            "i don't see",           # 我没看到
            "cannot determine",      # 无法确定
            "unable to identify",    # 无法识别
            "no clear answer",       # 无明确答案
        ]

        for marker in hallucination_markers:
            if marker in text_lower:
                return True

        return False


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    cleaner = OpenSampleCleaner()

    print("\n" + "="*70)
    print("开放样本清洗器测试（官方标准）")
    print("="*70)

    # 测试样本
    test_samples = [
        {
            "name": "正常样本",
            "data": {
                "answer": "PETA advocates animal welfare. The photo shows an elephant being used for tourist rides. Animal rights organizations oppose this practice because elephants are often exploited and mistreated for entertainment."
            },
            "expected": True
        },
        {
            "name": "过短样本",
            "data": {
                "answer": "Yes"  # 仅 1 个 token
            },
            "expected": False
        },
        {
            "name": "空输出样本",
            "data": {
                "answer": ""
            },
            "expected": False
        },
        {
            "name": "重复文本样本",
            "data": {
                "answer": "The cat is The cat is The cat is The cat is on the mat."
            },
            "expected": False
        },
        {
            "name": "重度幻觉样本",
            "data": {
                "answer": "I cannot see the elephant clearly in this image. The animal is not visible in the picture."
            },
            "expected": False
        }
    ]

    print("\n测试结果：")
    print("-" * 70)

    for test in test_samples:
        result = cleaner.clean(test["data"])
        status = "✓" if result["is_valid"] == test["expected"] else "✗"

        print(f"\n{status} {test['name']}:")
        print(f"  预期: {'有效' if test['expected'] else '无效'}")
        print(f"  实际: {'有效' if result['is_valid'] else '无效'}")

        if result["issues"]:
            print(f"  问题: {result['issues']}")

    print("\n" + "="*70)
    print("官方标准 4 条基础规则：")
    print("  1. 文本长度过滤：token < 8 或 > 512")
    print("  2. 重复文本过滤：大面积重复句式、词语循环")
    print("  3. 图像-回答一致性：重度幻觉检测")
    print("  4. 空输出兜底：空字符串直接丢弃")
    print("="*70)