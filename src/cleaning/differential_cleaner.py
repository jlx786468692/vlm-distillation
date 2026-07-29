"""
差异化清洗过滤逻辑
==================

根据问题类型应用不同的清洗策略：
- 闭合样本（count/color/binary/location）：候选集校验、置信度过滤、CoT 推测词过滤
- 开放样本（open）：仅做文本基础过滤

使用方式：
    from src.cleaning.differential_cleaner import DifferentialCleaner

    cleaner = DifferentialCleaner(config)
    result = cleaner.clean_sample(sample)
"""

import re
from typing import Dict, Any, List, Optional, Set
import logging

# 导入问题类型枚举
try:
    from ..classification.question_classifier import QuestionType
except ImportError:
    # 兼容性：定义问题类型（官方标准）
    class QuestionType:
        COUNT = 'counting'
        COLOR = 'color'
        BINARY = 'yes_no'
        LOCATION = 'location'
        OPEN = 'open_descriptive'  # 官方标准命名


class DifferentialCleaner:
    """
    差异化清洗器

    根据问题类型应用不同的清洗策略：
    - 闭合样本：严格过滤（候选集校验、置信度过滤、CoT推测词过滤）
    - 开放样本：宽松过滤（仅文本基础过滤）
    """

    # 推测词黑名单（用于闭合样本）
    SPECULATIVE_WORDS = [
        'appear', 'seem', 'probably', 'likely', 'possibly', 'perhaps',
        'maybe', 'might', 'could be', 'may be', 'would be',
        'suggest', 'unclear', 'uncertain', 'ambiguous'
    ]

    # 闭合问题类型的白名单
    CLOSED_WHITELISTS = {
        'count': {
            'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'
        },
        'color': {
            'red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'brown', 'black', 'white',
            'gray', 'grey', 'cyan', 'magenta'
        },
        'binary': {'yes', 'no'},
        'location': {
            'left', 'right', 'top', 'bottom', 'center', 'middle',
            'front', 'back', 'side', 'corner', 'edge'
        }
    }

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化差异化清洗器

        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # 读取配置参数
        self.min_confidence = self.config.get('cleaning.min_confidence', 0.3)
        self.max_confidence = self.config.get('cleaning.max_confidence', 0.95)
        self.min_cot_quality = self.config.get('cleaning.min_cot_quality', 0.3)

        self.logger.info("✓ 差异化清洗器初始化完成")
        self.logger.info(f"  - 最小置信度: {self.min_confidence}")
        self.logger.info(f"  - 最大置信度: {self.max_confidence}")
        self.logger.info(f"  - 最小CoT质量: {self.min_cot_quality}")

    def clean_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        清洗单个样本（差异化策略）

        Args:
            sample: 样本数据（包含 question_type 字段）

        Returns:
            清洗结果，包含：
            - is_valid: 是否有效
            - question_type: 问题类型
            - cleaning_mode: 清洗模式（'closed' 或 'open'）
            - issues: 发现的问题列表
            - actions: 执行的操作列表
        """
        # 获取问题类型
        question_type = sample.get('question_type', 'open')

        # 根据问题类型选择清洗策略
        if question_type in ['count', 'color', 'binary', 'location']:
            # 闭合样本：严格清洗
            return self._clean_closed_sample(sample, question_type)
        else:
            # 开放样本：宽松清洗
            return self._clean_open_sample(sample)

    def _clean_closed_sample(self, sample: Dict[str, Any], question_type: str) -> Dict[str, Any]:
        """
        闭合样本清洗（严格）

        清洗步骤：
        1. 候选集校验：检查答案是否在白名单中
        2. 置信度过滤：检查置信度是否在合理范围
        3. CoT 推测词过滤：检查 CoT 是否包含推测词

        Args:
            sample: 样本数据
            question_type: 问题类型

        Returns:
            清洗结果
        """
        issues = []
        actions = []
        is_valid = True

        # 获取数据
        hard_label = sample.get('tasks', {}).get('vqa', {}).get('hard_label', {})
        soft_label = sample.get('tasks', {}).get('vqa', {}).get('soft_label', {})
        cot = sample.get('tasks', {}).get('vqa', {}).get('cot_reasoning', {})

        # ───────────────────────────────────────────────────────
        # 检查1：候选集校验
        # ───────────────────────────────────────────────────────
        whitelist = self.CLOSED_WHITELISTS.get(question_type, set())

        # 检查 hard_label
        if hard_label and 'answer' in hard_label:
            answer = hard_label['answer'].lower().strip()
            if whitelist and answer not in whitelist:
                issues.append(f"Hard label '{answer}' not in {question_type} whitelist")
                actions.append(f"Flag hard_label: {answer}")
                # 不直接标记为无效，仅记录问题

        # 检查 soft_label 分布
        if soft_label and 'answer_distribution' in soft_label:
            distribution = soft_label['answer_distribution']
            invalid_tokens = []

            for token in distribution.keys():
                token_lower = token.lower().strip()
                if whitelist and token_lower not in whitelist:
                    invalid_tokens.append(token)

            if invalid_tokens:
                issues.append(f"Soft label contains {len(invalid_tokens)} out-of-whitelist tokens")
                actions.append(f"Flag soft_label tokens: {invalid_tokens[:5]}")

        # ───────────────────────────────────────────────────────
        # 检查2：置信度过滤
        # ───────────────────────────────────────────────────────
        if hard_label and 'confidence' in hard_label:
            confidence = hard_label['confidence']

            # 过低置信度
            if confidence < self.min_confidence:
                issues.append(f"Low confidence: {confidence:.4f} < {self.min_confidence}")
                actions.append("Flag low confidence")
                is_valid = False

            # 过高置信度（可能过拟合）
            elif confidence > self.max_confidence:
                issues.append(f"Over-confidence: {confidence:.4f} > {self.max_confidence}")
                actions.append("Flag over-confidence")

        # ───────────────────────────────────────────────────────
        # 检查3：CoT 推测词过滤
        # ───────────────────────────────────────────────────────
        if cot and 'analysis' in cot:
            analysis_text = cot['analysis'].lower()
            found_speculative = []

            for word in self.SPECULATIVE_WORDS:
                if word in analysis_text:
                    found_speculative.append(word)

            if found_speculative:
                issues.append(f"CoT contains speculative words: {found_speculative}")
                actions.append("Flag speculative CoT")
                # 标记为低质量，但不直接无效
                # is_valid = False  # 可选：是否直接无效

        # ───────────────────────────────────────────────────────
        # 返回清洗结果
        # ───────────────────────────────────────────────────────
        return {
            'is_valid': is_valid,
            'question_type': question_type,
            'cleaning_mode': 'closed',
            'issues': issues,
            'actions': actions,
            'quality_score': self._compute_quality_score(issues)
        }

    def _clean_open_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        开放样本清洗（宽松）

        清洗步骤：
        1. 文本基础过滤：检查是否为空、过短、包含非法字符等
        2. 基本置信度检查（可选）

        Args:
            sample: 样本数据

        Returns:
            清洗结果
        """
        issues = []
        actions = []
        is_valid = True

        # 获取数据
        hard_label = sample.get('tasks', {}).get('vqa', {}).get('hard_label', {})
        cot = sample.get('tasks', {}).get('vqa', {}).get('cot_reasoning', {})

        # ───────────────────────────────────────────────────────
        # 检查1：文本基础过滤
        # ───────────────────────────────────────────────────────

        # 检查 hard_label
        if hard_label and 'answer' in hard_label:
            answer = hard_label['answer'].strip()

            # 空答案
            if not answer:
                issues.append("Empty hard_label answer")
                actions.append("Mark as invalid")
                is_valid = False

            # 过短答案
            elif len(answer) < 2:
                issues.append(f"Too short answer: '{answer}'")
                actions.append("Flag short answer")

        # 检查 CoT
        if cot:
            observation = cot.get('observation', '').strip()
            conclusion = cot.get('conclusion', '').strip()

            # 空观察
            if not observation:
                issues.append("Empty CoT observation")
                actions.append("Flag empty observation")

            # 空结论
            if not conclusion:
                issues.append("Empty CoT conclusion")
                actions.append("Flag empty conclusion")

        # ───────────────────────────────────────────────────────
        # 检查2：基本置信度检查
        # ───────────────────────────────────────────────────────
        if hard_label and 'confidence' in hard_label:
            confidence = hard_label['confidence']

            # 极低置信度才标记（更宽松）
            if confidence < 0.1:
                issues.append(f"Very low confidence: {confidence:.4f}")
                actions.append("Flag very low confidence")
                is_valid = False

        # ───────────────────────────────────────────────────────
        # 返回清洗结果
        # ───────────────────────────────────────────────────────
        return {
            'is_valid': is_valid,
            'question_type': 'open',
            'cleaning_mode': 'open',
            'issues': issues,
            'actions': actions,
            'quality_score': self._compute_quality_score(issues)
        }

    def _compute_quality_score(self, issues: List[str]) -> float:
        """
        根据问题数量计算质量分数

        Args:
            issues: 问题列表

        Returns:
            质量分数（0-100）
        """
        if not issues:
            return 100.0

        # 每个问题扣10分，最低0分
        score = max(0, 100 - len(issues) * 10)
        return float(score)


# ============================================================
# 集成到 DataCleaner
# ============================================================

def integrate_to_data_cleaner():
    """
    将差异化清洗器集成到 DataCleaner

    修改文件：src/cleaning/data_cleaner.py

    在 DataCleaner.__init__ 中添加：
        from .differential_cleaner import DifferentialCleaner
        self.differential_cleaner = DifferentialCleaner(self.config)

    在 DataCleaner._clean_single_file 中修改：
        # 使用差异化清洗
        cleaning_result = self.differential_cleaner.clean_sample(data)

        if not cleaning_result['is_valid']:
            # 标记为无效，移除
            data['cleaning_result'] = cleaning_result
            return None

        # 保留样本，附加清洗结果
        data['cleaning_result'] = cleaning_result
        return data
    """
    pass


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    # 示例数据
    sample_closed = {
        "image_id": 123,
        "question_type": "count",
        "tasks": {
            "vqa": {
                "question": "How many people?",
                "hard_label": {
                    "answer": "two",
                    "confidence": 0.89
                },
                "soft_label": {
                    "answer_distribution": {"one": 0.05, "two": 0.89, "three": 0.06},
                    "primary_answer": "two",
                    "allowed_answers": ["one", "two", "three"]
                },
                "cot_reasoning": {
                    "observation": "Two people visible",
                    "analysis": "Probably two people",  # 包含推测词
                    "conclusion": "two"
                }
            }
        }
    }

    sample_open = {
        "image_id": 456,
        "question_type": "open",
        "tasks": {
            "vqa": {
                "question": "What kind of sandwich is this?",
                "hard_label": {
                    "answer": "turkey sandwich",
                    "confidence": 0.75
                },
                "cot_reasoning": {
                    "observation": "Turkey and lettuce",
                    "analysis": "Appears to be a turkey sandwich",
                    "conclusion": "turkey sandwich"
                }
            }
        }
    }

    # 初始化清洗器
    cleaner = DifferentialCleaner()

    print("\n" + "="*70)
    print("差异化清洗示例")
    print("="*70)

    # 清洗闭合样本
    print("\n【闭合样本】")
    result_closed = cleaner.clean_sample(sample_closed)
    print(f"问题类型: {result_closed['question_type']}")
    print(f"清洗模式: {result_closed['cleaning_mode']}")
    print(f"是否有效: {result_closed['is_valid']}")
    print(f"问题列表: {result_closed['issues']}")
    print(f"质量分数: {result_closed['quality_score']}")

    # 清洗开放样本
    print("\n【开放样本】")
    result_open = cleaner.clean_sample(sample_open)
    print(f"问题类型: {result_open['question_type']}")
    print(f"清洗模式: {result_open['cleaning_mode']}")
    print(f"是否有效: {result_open['is_valid']}")
    print(f"问题列表: {result_open['issues']}")
    print(f"质量分数: {result_open['quality_score']}")

    print("\n" + "="*70)