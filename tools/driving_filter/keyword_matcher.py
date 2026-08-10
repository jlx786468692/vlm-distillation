"""
智能驾驶关键词匹配器
====================

功能：
- 加载智能驾驶相关关键词库
- 支持多种匹配模式（精确、模糊、短语）
- 高效文本匹配算法
"""

import re
from pathlib import Path
from typing import List, Set, Dict, Tuple
from collections import defaultdict


class KeywordMatcher:
    """
    智能驾驶关键词匹配器

    支持：
    - 精确匹配：完整单词匹配
    - 模糊匹配：支持复数形式
    - 短语匹配：支持多词短语
    - 负向关键词：排除非驾驶场景
    """

    def __init__(self, keywords: Set[str] = None, keywords_file: str = None):
        """
        初始化关键词匹配器

        Args:
            keywords: 关键词集合（直接传入）
            keywords_file: 关键词文件路径（从文件加载）
        """
        self.keywords = set()
        self.phrase_keywords = set()  # 多词短语
        self.single_keywords = set()  # 单词关键词
        self.negative_keywords = set()  # 负向关键词（排除场景）

        # 加载关键词
        if keywords_file and Path(keywords_file).exists():
            self._load_keywords_from_file(keywords_file)
        elif keywords:
            self._load_keywords_from_set(keywords)

        # 构建匹配模式
        self._build_match_patterns()

        # 统计信息
        self.stats = {
            'total_keywords': len(self.keywords),
            'single_keywords': len(self.single_keywords),
            'phrase_keywords': len(self.phrase_keywords),
            'negative_keywords': len(self.negative_keywords),
            'match_calls': 0,
            'match_hits': 0
        }

    def _load_keywords_from_file(self, keywords_file: str):
        """
        从文件加载关键词

        文件格式：
        - 每行一个关键词
        - 支持#开头的注释
        - 支持空行
        - 支持"负向关键词"部分（以"负向关键词"开头的段落）
        """
        keywords_path = Path(keywords_file)

        if not keywords_path.exists():
            raise FileNotFoundError(f"关键词文件不存在: {keywords_file}")

        in_negative_section = False

        with open(keywords_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                # 跳过注释和空行
                if not line or line.startswith('#'):
                    # 检测是否进入负向关键词段落
                    if '负向关键词' in line or '排除' in line:
                        in_negative_section = True
                    continue

                # 检测是否进入新的段落（非负向关键词）
                if line.startswith('===') or line.startswith('【'):
                    if '负向' not in line and '排除' not in line:
                        in_negative_section = False
                    continue

                # 添加到对应的关键词集合
                if in_negative_section:
                    self.negative_keywords.add(line.lower())
                else:
                    self.keywords.add(line.lower())

        print(f"✓ 从文件加载 {len(self.keywords)} 个正向关键词")
        print(f"✓ 从文件加载 {len(self.negative_keywords)} 个负向关键词")
        print(f"  文件: {keywords_file}")

    def _load_keywords_from_set(self, keywords: Set[str]):
        """从关键词集合加载"""
        self.keywords = {kw.lower() for kw in keywords}

    def _build_match_patterns(self):
        """
        构建匹配模式

        分类：
        - 单词关键词：用于快速匹配
        - 短语关键词：用于精确匹配
        """
        for keyword in self.keywords:
            if ' ' in keyword:
                # 多词短语
                self.phrase_keywords.add(keyword)
            else:
                # 单词
                self.single_keywords.add(keyword)

        # 构建正则表达式（用于单词匹配）
        # \b表示单词边界，支持复数形式
        if self.single_keywords:
            # 转义特殊字符
            escaped_keywords = [re.escape(kw) for kw in self.single_keywords]
            # 构建正则：支持复数形式（加s）
            pattern = r'\b(?:' + '|'.join(escaped_keywords) + r')(?:s)?\b'
            self.word_pattern = re.compile(pattern, re.IGNORECASE)
        else:
            self.word_pattern = None

    def match(self, text: str) -> Tuple[bool, List[str]]:
        """
        检测文本中是否包含关键词

        Args:
            text: 待检测文本

        Returns:
            (是否匹配, 匹配到的关键词列表)
        """
        self.stats['match_calls'] += 1

        if not text:
            return False, []

        text_lower = text.lower()
        matched_keywords = set()

        # 1. 检测负向关键词（排除场景）
        negative_matches = set()
        if self.negative_keywords:
            for neg_kw in self.negative_keywords:
                if ' ' in neg_kw:
                    # 短语匹配
                    if neg_kw in text_lower:
                        negative_matches.add(neg_kw)
                else:
                    # 单词匹配
                    if re.search(r'\b' + re.escape(neg_kw) + r'\b', text_lower):
                        negative_matches.add(neg_kw)

        # 2. 单词匹配（使用正则表达式）
        if self.word_pattern:
            matches = self.word_pattern.findall(text_lower)
            matched_keywords.update(matches)

        # 3. 短语匹配（精确匹配）
        for phrase in self.phrase_keywords:
            if phrase in text_lower:
                matched_keywords.add(phrase)

        # 4. 如果有负向关键词匹配，返回False（排除该样本）
        if negative_matches:
            # 记录被负向关键词过滤
            self.stats['match_hits'] += 1
            return False, []

        # 更新统计信息
        if matched_keywords:
            self.stats['match_hits'] += 1

        return len(matched_keywords) > 0, list(matched_keywords)

    def match_count(self, text: str) -> int:
        """
        返回匹配到的关键词数量

        Args:
            text: 待检测文本

        Returns:
            匹配到的关键词数量
        """
        matched, keywords = self.match(text)
        return len(keywords) if matched else 0

    def get_stats(self) -> Dict:
        """获取匹配统计信息"""
        hit_rate = (
            self.stats['match_hits'] / self.stats['match_calls']
            if self.stats['match_calls'] > 0 else 0
        )

        return {
            **self.stats,
            'hit_rate': f"{hit_rate:.2%}"
        }

    @staticmethod
    def get_default_keywords() -> Set[str]:
        """
        获取默认关键词库（内置关键词）

        Returns:
            默认关键词集合
        """
        return {
            # ============================================
            # 交通工具（高频）
            # ============================================
            'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'vehicle',
            'taxi', 'ambulance', 'police car', 'van', 'sedan',
            'suv', 'pickup', 'trailer', 'minivan',

            # ============================================
            # 道路设施（高频）
            # ============================================
            'traffic light', 'stop sign', 'road', 'street', 'highway',
            'intersection', 'crosswalk', 'parking', 'bridge', 'lane',
            'freeway', 'tunnel', 'overpass', 'underpass', 'roundabout',
            'sidewalk', 'curb', 'median', 'guardrail',

            # ============================================
            # 驾驶行为（中频）
            # ============================================
            'driving', 'parked', 'moving', 'stopped', 'waiting',
            'turning', 'passing', 'overtaking', 'reversing',
            'accelerating', 'braking', 'yielding', 'merging',

            # ============================================
            # 安全相关（中频）
            # ============================================
            'pedestrian', 'crossing', 'safety', 'helmet', 'seatbelt',
            'accident', 'collision', 'crash', 'hazard',

            # ============================================
            # 交通参与者（中频）
            # ============================================
            'driver', 'passenger', 'rider', 'walker', 'jogger',

            # ============================================
            # 天气与时间（低频，但对自动驾驶重要）
            # ============================================
            'rainy', 'sunny', 'foggy', 'snowy', 'night',
            'dusk', 'dawn', 'daylight', 'dark',

            # ============================================
            # 其他相关词汇（低频）
            # ============================================
            'traffic', 'transportation', 'commute', 'route',
            'signal', 'indicator', 'headlight', 'taillight',
            'mirror', 'windshield', 'wheel', 'tire'
        }


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    # 使用默认关键词库测试
    matcher = KeywordMatcher(keywords=KeywordMatcher.get_default_keywords())

    print("\n" + "="*60)
    print("智能驾驶关键词匹配器测试")
    print("="*60)

    # 测试文本
    test_texts = [
        "A red car is parked on the street.",
        "There are several people walking on the sidewalk.",
        "The traffic light is green for the crossing pedestrians.",
        "A truck is driving on the highway at night.",
        "This is a beautiful sunset over the mountains.",  # 无匹配
        "Multiple cars waiting at the intersection for the stop sign."
    ]

    print("\n测试结果：")
    for text in test_texts:
        matched, keywords = matcher.match(text)
        status = "✓" if matched else "✗"
        print(f"{status} '{text}'")
        if matched:
            print(f"  匹配关键词: {keywords}")

    # 显示统计信息
    print("\n" + "="*60)
    print("统计信息：")
    print("="*60)
    stats = matcher.get_stats()
    for key, value in stats.items():
        print(f"{key:20s}: {value}")