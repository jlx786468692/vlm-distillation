"""
候选集查询器（兼容层）
=====================

保留此类以保持向后兼容性。
内部实现已迁移到 scene_candidate_loader.py。

使用方式：
    from tools.candidate.candidate_closure import CandidateClosure

    closure = CandidateClosure(config)
    candidates = closure.get_candidates_for_question(question, primary_answer)
"""

from typing import Dict, Any, List, Optional
import logging

# 导入新的查询器
from .scene_candidate_loader import SceneCandidateLoader


class CandidateClosure:
    """
    候选集查询器（兼容层）

    内部使用 SceneCandidateLoader 实现。
    只负责查询，不负责生成。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化候选集查询器

        Args:
            config: 配置字典（可选）
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # 配置参数
        self.max_candidates = self.config.get('max_candidates', 100)

        # 使用新的查询器
        self._loader = SceneCandidateLoader(max_candidates=self.max_candidates)

        # 为了保持向后兼容，提供 vqa_vocab 属性
        self.vqa_vocab = self._loader.vqa_vocab

        self.logger.info("✓ CandidateClosure 初始化完成（兼容模式）")
        self.logger.info(f"  VQA词表大小: {len(self.vqa_vocab)}")
        self.logger.info(f"  最大候选数: {self.max_candidates}")

    def get_candidates_for_question(
        self,
        question: str,
        primary_answer: Optional[str] = None
    ) -> List[str]:
        """
        根据问题类型获取候选答案集

        Args:
            question: 问题文本
            primary_answer: 主答案（用于保底）

        Returns:
            候选答案列表
        """
        return self._loader.get_candidates_for_question(question, primary_answer)


# 保持向后兼容的别名
CandidateClosureV2 = CandidateClosure