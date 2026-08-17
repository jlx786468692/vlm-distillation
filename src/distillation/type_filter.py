"""
类型匹配器（Type Matcher）
==========================

核心功能：
1. 分层匹配逻辑（Level 1 → Level 2 → Level 3 → Level 4）
2. 语义簇归并（处理 BPE 变体）
3. GT 兜底策略

作者: Claude
日期: 2026-08-13
"""

from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import yaml
from pathlib import Path
import re


@dataclass
class TypeMatchResult:
    """类型匹配结果"""
    token: str
    token_types: List[str]
    matched: bool
    matched_types: List[str]
    mismatched_types: List[str]
    safety_level: int  # 1-4
    kl_weight: float
    is_safety_critical: bool
    semantic_cluster: Optional[str] = None


@dataclass
class FilterResult:
    """过滤结果"""
    filtered_logits: Dict[str, float]
    kl_weight: float
    level_1_mismatches: int
    level_2_mismatches: int
    level_3_mismatches: int
    level_4_mismatches: int
    gt_retained: bool
    gt_fallback_applied: bool


class TypeMatcher:
    """
    类型匹配器

    核心逻辑：
    1. 分层匹配（Level 1 → Level 4）
    2. 语义簇归并（BPE 变体处理）
    3. GT 兜底策略
    """

    def __init__(self, schema_path: str = "configs/vqa_type_schema.yaml"):
        """
        初始化类型匹配器

        Args:
            schema_path: 类型分类体系配置文件路径
        """
        self.schema_path = Path(schema_path)
        self.schema = self._load_schema()

        # 构建反向索引
        self.token_to_types = self._build_token_to_types()
        self.type_to_level = self._build_type_to_level()
        self.semantic_clusters = self._build_semantic_clusters()

        # 统计信息
        self.stats = {
            'total_matches': 0,
            'level_1_matches': 0,
            'level_2_matches': 0,
            'level_3_matches': 0,
            'level_4_matches': 0,
            'semantic_cluster_merges': 0,
            'gt_fallbacks': 0
        }

    def _load_schema(self) -> dict:
        """加载 Schema 配置"""
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")

        with open(self.schema_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _build_token_to_types(self) -> Dict[str, List[str]]:
        """
        构建 token -> types 反向索引

        Returns:
            {token: [type1, type2, ...]}
        """
        token_to_types = {}
        taxonomy = self.schema.get('type_taxonomy', {})

        for domain, types in taxonomy.items():
            for type_name, keywords in types.items():
                for keyword in keywords:
                    keyword_lower = keyword.lower().strip()
                    if keyword_lower not in token_to_types:
                        token_to_types[keyword_lower] = []
                    token_to_types[keyword_lower].append(type_name)

        return token_to_types

    def _build_type_to_level(self) -> Dict[str, int]:
        """
        构建 type -> level 映射

        Returns:
            {type_name: level}
        """
        type_to_level = {}
        safety_levels = self.schema.get('safety_levels', {})

        for level_name, level_config in safety_levels.items():
            level_num = int(level_name.split('_')[1])  # level_1_critical -> 1
            for pattern in level_config['types']:
                if pattern.endswith('_*'):
                    # 通配符匹配，需要遍历所有类型
                    prefix = pattern[:-2]
                    for domain_types in self.schema['type_taxonomy'].values():
                        for type_name in domain_types.keys():
                            if type_name.startswith(prefix):
                                type_to_level[type_name] = level_num
                else:
                    type_to_level[pattern] = level_num

        return type_to_level

    def _build_semantic_clusters(self) -> Dict[str, dict]:
        """
        构建语义簇映射

        Returns:
            {canonical: {'variants': [...], 'gt_types': [...]}}
        """
        clusters = {}
        semantic_clusters = self.schema.get('semantic_clusters', {})

        for cluster_name, cluster_config in semantic_clusters.items():
            canonical = cluster_config['canonical']
            clusters[canonical] = {
                'variants': [v.lower().strip() for v in cluster_config['variants']],
                'gt_types': cluster_config['gt_types']
            }

        return clusters

    def get_token_types(self, token: str) -> List[str]:
        """
        获取 token 的类型标签

        Args:
            token: "pedestrian"

        Returns:
            ["driving_pedestrian"]
        """
        token_lower = token.lower().strip()

        # 1. 直接匹配
        if token_lower in self.token_to_types:
            return self.token_to_types[token_lower]

        # 2. 语义簇匹配
        for canonical, cluster_config in self.semantic_clusters.items():
            if token_lower in cluster_config['variants']:
                return cluster_config['gt_types']

        # 3. 未找到类型
        return []

    def get_type_level(self, type_name: str) -> int:
        """
        获取类型的安全等级

        Args:
            type_name: "driving_pedestrian"

        Returns:
            1 (Level 1 - Critical)
        """
        return self.type_to_level.get(type_name, 4)  # 默认 Level 4

    def get_kl_weight(self, level: int) -> float:
        """
        根据 Level 获取 KL 权重

        Args:
            level: 1-4

        Returns:
            kl_weight: 0.0 / 0.1 / 0.3 / 0.0
        """
        kl_weights = {
            1: 0.0,  # Level 1: 直接丢弃
            2: 0.1,  # Level 2: 降权到 10%
            3: 0.3,  # Level 3: 轻降权到 30%
            4: 0.0   # Level 4: 直接丢弃
        }
        return kl_weights.get(level, 1.0)

    def match_token(
        self,
        token: str,
        gt_types: List[str],
        enable_semantic_cluster: bool = True
    ) -> TypeMatchResult:
        """
        匹配单个 token 的类型

        Args:
            token: "pedestrian"
            gt_types: ["driving_pedestrian"]
            enable_semantic_cluster: 是否启用语义簇归并

        Returns:
            TypeMatchResult
        """
        # 1. 获取 token 类型
        token_types = self.get_token_types(token)

        # 2. 检查匹配
        matched_types = set(token_types) & set(gt_types)
        mismatched_types = set(token_types) - set(gt_types)

        # 3. 检查语义簇匹配
        semantic_cluster = None
        if enable_semantic_cluster and not matched_types:
            for canonical, cluster_config in self.semantic_clusters.items():
                if token.lower().strip() in cluster_config['variants']:
                    cluster_gt_types = cluster_config['gt_types']
                    matched_types = set(cluster_gt_types) & set(gt_types)
                    if matched_types:
                        semantic_cluster = canonical
                        self.stats['semantic_cluster_merges'] += 1
                        break

        # 4. 确定 Level
        level = 4  # 默认 Level 4
        for type_name in mismatched_types:
            type_level = self.get_type_level(type_name)
            level = min(level, type_level)  # 最危险的 Level

        # 5. 获取 KL 权重
        kl_weight = self.get_kl_weight(level)

        # 6. 是否安全 Critical
        is_safety_critical = (level == 1 and mismatched_types)

        # 7. 更新统计
        self.stats['total_matches'] += 1
        if matched_types:
            if level == 1:
                self.stats['level_1_matches'] += 1
            elif level == 2:
                self.stats['level_2_matches'] += 1
            elif level == 3:
                self.stats['level_3_matches'] += 1
            elif level == 4:
                self.stats['level_4_matches'] += 1

        return TypeMatchResult(
            token=token,
            token_types=token_types,
            matched=bool(matched_types),
            matched_types=list(matched_types),
            mismatched_types=list(mismatched_types),
            safety_level=level,
            kl_weight=kl_weight,
            is_safety_critical=is_safety_critical,
            semantic_cluster=semantic_cluster
        )

    def filter_top_k_logits(
        self,
        top_k_logits: Dict[str, float],
        gt_answer: str,
        gt_fallback_prob: float = 0.01,
        enable_semantic_cluster: bool = True
    ) -> FilterResult:
        """
        过滤 Top-K logits（分层匹配）

        Args:
            top_k_logits: {"pedestrian": 2.8, "left": 2.5, ...}
            gt_answer: "left front pedestrian" or "letters"
            gt_fallback_prob: GT 不在 Top-K 时给予的兜底概率
            enable_semantic_cluster: 是否启用语义簇归并

        Returns:
            FilterResult
        """
        # 1. 提取 GT 类型
        gt_types = self.get_token_types(gt_answer)

        # 2. 分层过滤
        filtered_logits = {}
        level_mismatches = {1: 0, 2: 0, 3: 0, 4: 0}

        # ───────────────────────────────────────────────────────
        # 🔧 新增：GT 无类型时的默认过滤策略
        # ───────────────────────────────────────────────────────
        # 当 GT 不在类型体系中时（如 "letters"），应用默认逻辑：
        # - 过滤明显不相关的类型（数字、颜色、空间位置）
        # - 保留其他候选（开放性）
        # ───────────────────────────────────────────────────────

        if not gt_types:
            # GT 无类型：使用默认过滤策略
            for token, logit in top_k_logits.items():
                # GT 答案：强制保留
                if token.lower().strip() == gt_answer.lower().strip():
                    filtered_logits[token] = logit
                    continue

                # 获取 token 类型
                token_types = self.get_token_types(token)

                if not token_types:
                    # token 也无类型：保留（可能是新词或专有名词）
                    filtered_logits[token] = logit
                else:
                    # 检查是否属于禁止类型
                    forbidden = self._is_forbidden_for_unknown_gt(token_types)

                    if forbidden:
                        # 过滤禁止类型
                        level_mismatches[3] += 1  # Level 3（中等）
                    else:
                        # 保留其他候选
                        filtered_logits[token] = logit
        else:
            # GT 有类型：使用原有分层匹配逻辑
            for token, logit in top_k_logits.items():
                # 匹配类型
                match_result = self.match_token(token, gt_types, enable_semantic_cluster)

                if match_result.matched:
                    # 保留匹配的 token
                    filtered_logits[token] = logit
                else:
                    # 统计不匹配的 Level
                    level_mismatches[match_result.safety_level] += 1

        # 3. GT 兜底策略
        gt_retained = self._check_gt_retention(gt_answer, filtered_logits)
        gt_fallback_applied = False

        if not gt_retained and gt_fallback_prob > 0:
            # GT 不在 Top-K，给予兜底概率
            filtered_logits[gt_answer] = gt_fallback_prob
            gt_retained = True
            gt_fallback_applied = True
            self.stats['gt_fallbacks'] += 1

        # 4. 计算 KL 权重（取最高 Level 的惩罚）
        kl_weight = 1.0
        if level_mismatches[1] > 0:
            kl_weight = 0.0  # Level 1 不匹配，直接丢弃
        elif level_mismatches[2] > 0:
            kl_weight = 0.1  # Level 2 不匹配，降权
        elif level_mismatches[3] > 0:
            kl_weight = 0.3  # Level 3 不匹配，轻降权
        elif level_mismatches[4] > 0:
            kl_weight = 0.0  # Level 4 不匹配，丢弃

        return FilterResult(
            filtered_logits=filtered_logits,
            kl_weight=kl_weight,
            level_1_mismatches=level_mismatches[1],
            level_2_mismatches=level_mismatches[2],
            level_3_mismatches=level_mismatches[3],
            level_4_mismatches=level_mismatches[4],
            gt_retained=gt_retained,
            gt_fallback_applied=gt_fallback_applied
        )

    def _is_forbidden_for_unknown_gt(self, token_types: List[str]) -> bool:
        """
        检查 token 类型是否属于禁止类型（针对无类型GT）

        当 GT 不在类型体系中时，过滤以下类型：
        - 数字类型（number_digit）
        - 颜色类型（color_*）
        - 空间位置（spatial_*）
        - 否定词（negation_*）

        Args:
            token_types: token 的类型列表

        Returns:
            是否应该被过滤
        """
        forbidden_prefixes = [
            'number_',  # 数字（包括 number_digit, number_many等）
            'color_',   # 颜色
            'spatial_',  # 空间位置
            'negation_',  # 否定词
        ]

        for token_type in token_types:
            for prefix in forbidden_prefixes:
                if token_type.startswith(prefix):
                    return True

        return False

    def _check_gt_retention(self, gt_answer: str, filtered_logits: Dict[str, float]) -> bool:
        """
        检查 GT 是否保留在过滤后的 logits 中

        Args:
            gt_answer: "pedestrian"
            filtered_logits: {"pedestrian": 2.8, "left": 2.5, ...}

        Returns:
            True/False
        """
        gt_lower = gt_answer.lower().strip()

        # 1. 直接匹配
        if gt_lower in filtered_logits:
            return True

        # 2. 语义簇匹配
        for canonical, cluster_config in self.semantic_clusters.items():
            if gt_lower in cluster_config['variants']:
                # GT 在语义簇中，检查是否有簇内 token 在 filtered_logits 中
                for variant in cluster_config['variants']:
                    if variant in filtered_logits:
                        return True

        return False

    def _is_gt_semantic_valid(self, gt_answer: str, gt_types: List[str]) -> bool:
        """
        检查 GT 是否语义合法（类型不为空）

        Args:
            gt_answer: "pedestrian"
            gt_types: ["driving_pedestrian"]

        Returns:
            True/False
        """
        return len(gt_types) > 0

    def get_stats(self) -> dict:
        """获取统计信息"""
        return self.stats.copy()

    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_matches': 0,
            'level_1_matches': 0,
            'level_2_matches': 0,
            'level_3_matches': 0,
            'level_4_matches': 0,
            'semantic_cluster_merges': 0,
            'gt_fallbacks': 0
        }


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    import json

    # 1. 初始化类型匹配器
    matcher = TypeMatcher()

    # 2. 示例：Top-K logits
    top_k_logits = {
        "pedestrian": 2.8,
        "left": 2.5,
        "red": 2.3,
        "sitting": 2.1,
        "no sign": 1.9,
        "front": 1.8
    }

    # 3. 示例：GT 答案
    gt_answer = "left front pedestrian"

    # 4. 过滤 Top-K logits
    filter_result = matcher.filter_top_k_logits(top_k_logits, gt_answer)

    # 5. 输出结果
    print("=" * 60)
    print("类型过滤结果")
    print("=" * 60)
    print(f"GT 答案: {gt_answer}")
    print(f"过滤前: {len(top_k_logits)} 个 token")
    print(f"过滤后: {len(filter_result.filtered_logits)} 个 token")
    print(f"过滤率: {(1 - len(filter_result.filtered_logits) / len(top_k_logits)) * 100:.1f}%")
    print()
    print(f"Level 1 不匹配: {filter_result.level_1_mismatches}")
    print(f"Level 2 不匹配: {filter_result.level_2_mismatches}")
    print(f"Level 3 不匹配: {filter_result.level_3_mismatches}")
    print(f"Level 4 不匹配: {filter_result.level_4_mismatches}")
    print()
    print(f"KL 权重: {filter_result.kl_weight}")
    print(f"GT 留存: {filter_result.gt_retained}")
    print(f"GT 兜底: {filter_result.gt_fallback_applied}")
    print()
    print("过滤后的 logits:")
    for token, logit in filter_result.filtered_logits.items():
        print(f"  {token}: {logit:.4f}")

    # 6. 输出统计信息
    print()
    print("=" * 60)
    print("统计信息")
    print("=" * 60)
    stats = matcher.get_stats()
    print(json.dumps(stats, indent=2, ensure_ascii=False))