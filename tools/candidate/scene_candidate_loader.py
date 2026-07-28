"""
场景候选集查询器
================

从阶段3生成的场景候选集文件中查询候选答案。

核心功能：
- 加载 data/scene_candidates.json
- 根据问题类型查询对应的候选集
- 保底策略：添加 primary_answer 到候选集

使用方式：
    from tools.candidate.scene_candidate_loader import SceneCandidateLoader

    loader = SceneCandidateLoader()
    candidates = loader.get_candidates_for_question(question, primary_answer)
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging


class SceneCandidateLoader:
    """
    场景候选集查询器（轻量级）

    只负责查询，不负责生成。
    数据来源：data/scene_candidates.json（由三阶段流水线生成）
    """

    def __init__(self, max_candidates: int = 100):
        """
        初始化场景候选集查询器

        Args:
            max_candidates: 每个场景最大候选数
        """
        self.logger = logging.getLogger(__name__)
        self.max_candidates = max_candidates

        # 加载场景候选集
        self.scene_candidates = None
        self.vqa_vocab = []
        self._load_scene_candidates()

    def _load_scene_candidates(self):
        """加载分场景候选集文件"""
        candidates_file = Path(__file__).parent.parent.parent / "data" / "scene_candidates.json"

        if not candidates_file.exists():
            self.logger.warning(
                f"场景候选集文件不存在: {candidates_file}\n"
                "请先运行: python -m tools candidate_closure"
            )
            return

        try:
            with open(candidates_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.scene_candidates = data.get('scenes', {})

            # 合并所有场景的候选集作为通用词表
            for scene_type, scene_data in self.scene_candidates.items():
                candidates = scene_data.get('candidates', [])
                for ans in candidates:
                    if ans not in self.vqa_vocab:
                        self.vqa_vocab.append(ans)

            self.logger.info(f"✓ 加载场景候选集: {candidates_file}")
            self.logger.info(f"  总场景数: {len(self.scene_candidates)}")
            self.logger.info(f"  总候选数: {len(self.vqa_vocab)}")

        except Exception as e:
            self.logger.error(f"加载场景候选集失败: {e}")
            self.scene_candidates = None
            self.vqa_vocab = []

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
        if not self.scene_candidates:
            self.logger.warning("场景候选集未加载，返回空列表")
            return []

        # Step 1: 简单的问题类型分类（基于关键词）
        question_type = self._classify_question_type(question)

        # Step 2: 根据类型获取候选集
        if question_type in self.scene_candidates:
            scene_data = self.scene_candidates[question_type]
            candidates = scene_data.get('candidates', []).copy()
            self.logger.info(
                f"✓ 使用 {question_type} 场景候选集: {len(candidates)}个答案"
            )
        else:
            # 如果场景不存在，使用 'other' 场景
            if 'other' in self.scene_candidates:
                candidates = self.scene_candidates['other'].get('candidates', []).copy()
                self.logger.info(
                    f"✓ 使用 other 场景候选集: {len(candidates)}个答案"
                )
            else:
                candidates = self.vqa_vocab[:self.max_candidates]
                self.logger.info(
                    f"✓ 使用默认候选集: {len(candidates)}个答案"
                )

        # Step 3: 保底策略 - 添加 primary_answer 到候选集
        if primary_answer:
            primary_lower = primary_answer.lower().strip()
            if primary_lower not in candidates:
                candidates.insert(0, primary_lower)
                self.logger.debug(f"✓ 保底策略: 添加 primary_answer '{primary_lower}' 到候选集")

        # Step 4: 限制候选数量
        if len(candidates) > self.max_candidates:
            candidates = candidates[:self.max_candidates]

        return candidates

    def _classify_question_type(self, question: str) -> str:
        """
        基于关键词的简单分类

        Args:
            question: 问题文本

        Returns:
            问题类型
        """
        question_lower = question.lower()

        # 计数问题
        if any(kw in question_lower for kw in ['how many', 'how much', 'count', 'number']):
            return 'count'

        # 颜色问题
        elif any(kw in question_lower for kw in ['what color', 'color is', 'what colour']):
            return 'color'

        # 二元问题
        elif any(kw in question_lower for kw in ['is there', 'are there', 'is it', 'are they', 'does', 'do you', 'can you']):
            return 'binary'

        # 其他
        else:
            return 'other'


def main():
    """测试入口"""
    import argparse

    parser = argparse.ArgumentParser(description="场景候选集查询器测试")
    parser.add_argument('--question', type=str, default="How many people are in the image?")
    parser.add_argument('--primary-answer', type=str, default='two')

    args = parser.parse_args()

    # 初始化
    loader = SceneCandidateLoader()

    if not loader.scene_candidates:
        print("❌ 请先运行: python -m tools candidate_closure")
        return

    # 测试
    print(f"\n问题: {args.question}")
    candidates = loader.get_candidates_for_question(args.question, args.primary_answer)
    print(f"候选答案: {candidates}")
    print(f"候选数: {len(candidates)}")


if __name__ == "__main__":
    main()