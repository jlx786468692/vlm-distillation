"""
智能驾驶数据筛选引擎
==================

核心筛选流程：
1. 加载COCO数据
2. 多维度评分
3. 阈值筛选
4. 数据导出
"""

import yaml
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

from .scorer import DrivingDataScorer
from .keyword_matcher import KeywordMatcher
from .data_exporter import DataExporter


class DrivingDataFilter:
    """
    智能驾驶数据筛选器

    主要功能：
    - 多维度综合打分
    - 阈值筛选
    - 数据导出
    """

    def __init__(self, config_path: str = None, config: Dict = None):
        """
        初始化筛选器

        Args:
            config_path: 配置文件路径
            config: 配置字典（直接传入）
        """
        # 加载配置
        if config_path:
            self.config = self._load_config(config_path)
        elif config:
            self.config = config
        else:
            self.config = self._get_default_config()

        # 初始化组件
        self._init_components()

        # 统计信息
        self.stats = {
            'total_images': 0,
            'filtered_images': 0,
            'scores': {}
        }

    def _load_config(self, config_path: str) -> Dict:
        """加载YAML配置文件"""
        path = Path(config_path)
        if not path.exists():
            print(f"⚠ 配置文件不存在: {config_path}，使用默认配置")
            return self._get_default_config()

        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        print(f"✓ 加载配置文件: {config_path}")
        return config

    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'driving_data_filter': {
                'categories': {
                    'driving_categories': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15],
                    'weights': {
                        3: 2.0, 6: 2.0, 8: 2.0, 10: 2.0, 13: 2.0,
                        'default': 1.0
                    }
                },
                'scoring': {
                    'category_weight': 0.4,
                    'text_weight': 0.3,
                    'scene_weight': 0.3,
                    'score_threshold': 0.5
                },
                'output': {
                    'root': './data/filter_coco',
                    'copy_images': False
                }
            }
        }

    def _init_components(self):
        """初始化各个组件"""
        filter_config = self.config.get('driving_data_filter', {})

        # 1. 初始化关键词匹配器
        keywords_file = filter_config.get('semantics', {}).get('keywords_file')
        if keywords_file and Path(keywords_file).exists():
            self.keyword_matcher = KeywordMatcher(keywords_file=keywords_file)
        else:
            # 使用内置关键词库
            self.keyword_matcher = KeywordMatcher(
                keywords=KeywordMatcher.get_default_keywords()
            )

        # 2. 初始化打分器
        categories_config = filter_config.get('categories', {})
        driving_categories = set(categories_config.get('driving_categories', []))
        category_weights = categories_config.get('weights', {})

        self.scorer = DrivingDataScorer(
            driving_categories=driving_categories,
            category_weights=category_weights,
            keywords_matcher=self.keyword_matcher,
            config=filter_config
        )

        # 3. 初始化导出器
        self.exporter = DataExporter(config=filter_config)

        print(f"✓ 筛选器组件初始化完成")

    def run(self, coco_loader, output_dir: str = None) -> Tuple[List[int], Dict[int, float]]:
        """
        执行筛选流程

        Args:
            coco_loader: COCO数据加载器实例
            output_dir: 输出目录（可选，默认从配置读取）

        Returns:
            (筛选后的图像ID列表, 图像得分字典)
        """
        print("\n" + "="*60)
        print("智能驾驶数据筛选流程")
        print("="*60)

        # 1. 获取所有图像ID
        all_img_ids = coco_loader.get_image_ids()
        self.stats['total_images'] = len(all_img_ids)

        print(f"\n总图像数: {len(all_img_ids)}")

        # 2. 评分阶段
        print(f"\n正在评分...")
        scores = {}

        # 使用进度条
        show_progress = self.config.get('driving_data_filter', {}).get('processing', {}).get('show_progress', True)

        if show_progress:
            pbar = tqdm(all_img_ids, desc="评分进度", unit="img")
            for img_id in pbar:
                score = self.scorer.score_image(img_id, coco_loader)
                scores[img_id] = score
        else:
            for img_id in all_img_ids:
                score = self.scorer.score_image(img_id, coco_loader)
                scores[img_id] = score

        self.stats['scores'] = scores

        # 3. 阈值筛选
        threshold = self.config.get('driving_data_filter', {}).get('scoring', {}).get('score_threshold', 0.5)

        filtered_img_ids = [
            img_id for img_id, score in scores.items()
            if score >= threshold
        ]

        self.stats['filtered_images'] = len(filtered_img_ids)

        print(f"\n✓ 评分完成")
        print(f"  - 筛选阈值: {threshold}")
        print(f"  - 筛选图像数: {len(filtered_img_ids)}/{len(all_img_ids)}")

        # 🔧 修复：防止除零错误
        if len(all_img_ids) > 0:
            print(f"  - 保留率: {len(filtered_img_ids)/len(all_img_ids):.2%}")
        else:
            print(f"  - 保留率: N/A (无图像)")

        # 🔧 新增：检查是否有图像
        if len(all_img_ids) == 0:
            print("\n⚠️ 警告：数据集为空，请检查：")
            print("  1. 数据集路径配置是否正确（configs/driving_filter.yaml 或 configs/default.yaml）")
            print("  2. 标注文件是否存在（data/coco/annotations/）")
            print("  3. 使用正确的数据集 split（val2014 或 val2017）")
            print("\n运行诊断脚本：python diagnose_coco_loading.py")
            return [], {}

        # 4. 导出数据
        if output_dir is None:
            output_dir = self.config.get('driving_data_filter', {}).get('output', {}).get('root', './data/filter_coco')

        self.exporter.export(filtered_img_ids, coco_loader, output_dir, scores)

        # 5. 打印统计信息
        self._print_statistics()

        return filtered_img_ids, scores

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = {
            'total_images': self.stats['total_images'],
            'filtered_images': self.stats['filtered_images'],
            'retention_rate': self.stats['filtered_images'] / self.stats['total_images'] if self.stats['total_images'] > 0 else 0,
            'scorer_stats': self.scorer.get_stats_summary()
        }

        return stats

    def _print_statistics(self):
        """打印统计信息"""
        print("\n" + "="*60)
        print("筛选统计报告")
        print("="*60)

        # 基本统计
        print(f"\n基本统计:")
        print(f"  总图像数: {self.stats['total_images']}")
        print(f"  筛选图像数: {self.stats['filtered_images']}")
        print(f"  保留率: {self.stats['filtered_images']/self.stats['total_images']:.2%}")

        # 得分分布
        if self.stats['scores']:
            import numpy as np
            score_values = np.array(list(self.stats['scores'].values()))

            print(f"\n得分分布:")
            print(f"  平均分: {np.mean(score_values):.3f}")
            print(f"  中位数: {np.median(score_values):.3f}")
            print(f"  标准差: {np.std(score_values):.3f}")
            print(f"  最小值: {np.min(score_values):.3f}")
            print(f"  最大值: {np.max(score_values):.3f}")

            # 分数段统计
            threshold = self.config.get('driving_data_filter', {}).get('scoring', {}).get('score_threshold', 0.5)
            high_quality_threshold = self.config.get('driving_data_filter', {}).get('scoring', {}).get('high_quality_threshold', 0.8)

            high_quality = np.sum(score_values >= high_quality_threshold)
            medium_quality = np.sum((score_values >= threshold) & (score_values < high_quality_threshold))
            low_quality = np.sum(score_values < threshold)

            print(f"\n质量分布:")
            print(f"  高质量 (>={high_quality_threshold}): {high_quality} ({high_quality/len(score_values):.2%})")
            print(f"  中等质量 ([{threshold}, {high_quality_threshold})): {medium_quality} ({medium_quality/len(score_values):.2%})")
            print(f"  低质量 (<{threshold}): {low_quality} ({low_quality/len(score_values):.2%})")


# ============================================================
# 便捷函数
# ============================================================
def filter_driving_data(
    coco_loader,
    config_path: str = None,
    output_dir: str = None
) -> Tuple[List[int], Dict[int, float]]:
    """
    便捷函数：筛选智能驾驶数据

    Args:
        coco_loader: COCO数据加载器
        config_path: 配置文件路径
        output_dir: 输出目录

    Returns:
        (筛选后的图像ID列表, 图像得分字典)
    """
    filter_engine = DrivingDataFilter(config_path=config_path)
    return filter_engine.run(coco_loader, output_dir)


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("智能驾驶数据筛选器测试")
    print("="*60)

    # 模拟测试
    filter_engine = DrivingDataFilter()

    print(f"\n✓ 筛选器初始化成功")
    print(f"  - 智能驾驶类别数: {len(filter_engine.scorer.driving_categories)}")
    print(f"  - 关键词数: {len(filter_engine.keyword_matcher.keywords)}")