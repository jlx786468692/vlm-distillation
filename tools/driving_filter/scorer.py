"""
智能驾驶数据综合打分器
======================

功能：
- 多维度评分：类别（40%）+ 文本（30%）+ 场景（30%）
- 类别权重：核心类别权重更高
- 场景特征：宽高比、对象数量等
"""

from typing import Dict, List, Set, Tuple
from collections import Counter


class DrivingDataScorer:
    """
    智能驾驶数据综合打分器

    评分维度：
    1. 类别得分（40%）：检测对象类别
    2. 文本语义得分（30%）：Captions和VQA问题
    3. 场景特征得分（30%）：图像元数据
    """

    def __init__(
        self,
        driving_categories: Set[int] = None,
        category_weights: Dict[int, float] = None,
        keywords_matcher=None,
        config: Dict = None
    ):
        """
        初始化打分器

        Args:
            driving_categories: 智能驾驶相关类别ID集合
            category_weights: 类别权重字典
            keywords_matcher: 关键词匹配器实例
            config: 配置字典
        """
        # 智能驾驶核心类别（COCO ID）
        self.driving_categories = driving_categories or self._get_default_driving_categories()

        # 类别权重（核心类别权重更高）
        self.category_weights = category_weights or self._get_default_category_weights()

        # 关键词匹配器
        self.keywords_matcher = keywords_matcher

        # 配置
        self.config = config or {}

        # 统计信息
        self.stats = {
            'total_images': 0,
            'scored_images': 0,
            'category_scores': [],
            'text_scores': [],
            'scene_scores': [],
            'final_scores': []
        }

    @staticmethod
    def _get_default_driving_categories() -> Set[int]:
        """获取默认智能驾驶类别集合"""
        return {
            # ========================================
            # 核心车辆类别（必须有这些才算驾驶场景）
            # ========================================
            2,   # bicycle ⭐⭐⭐
            3,   # car ⭐⭐⭐
            4,   # motorcycle ⭐⭐⭐
            6,   # bus ⭐⭐⭐
            8,   # truck ⭐⭐⭐

            # ========================================
            # 交通设施类别（强信号）
            # ========================================
            10,  # traffic light ⭐⭐⭐
            13,  # stop sign ⭐⭐⭐
            14,  # parking meter ⭐⭐

            # ========================================
            # 辅助类别（不单独计数，需配合核心类别）
            # ========================================
            1,   # person (行人) - 辅助
            12,  # fire hydrant - 辅助
            15,  # bench - 辅助

            # ========================================
            # 场景多样性类别（权重低）
            # ========================================
            5,   # airplane - 场景多样性
            7,   # train - 场景多样性
            9,   # boat - 场景多样性
        }

    @staticmethod
    def _get_default_category_weights() -> Dict[int, float]:
        """获取默认类别权重"""
        return {
            # ========================================
            # 核心车辆（权重最高）
            # ========================================
            3: 4.0,   # car ⭐⭐⭐
            6: 4.0,   # bus ⭐⭐⭐
            8: 4.0,   # truck ⭐⭐⭐
            2: 3.0,   # bicycle ⭐⭐⭐
            4: 3.0,   # motorcycle ⭐⭐⭐

            # ========================================
            # 交通设施（权重高）
            # ========================================
            10: 4.0,  # traffic light ⭐⭐⭐
            13: 4.0,  # stop sign ⭐⭐⭐
            14: 3.0,  # parking meter ⭐⭐

            # ========================================
            # 辅助类别（权重低）
            # ========================================
            1: 0.5,   # person - 辅助，权重低
            12: 1.0,  # fire hydrant
            15: 0.5,  # bench

            # ========================================
            # 场景多样性（权重最低）
            # ========================================
            5: 0.3,   # airplane
            7: 0.3,   # train
            9: 0.3,   # boat

            'default': 1.0
        }

    @staticmethod
    def _get_core_vehicle_categories() -> Set[int]:
        """
        获取核心车辆类别（必须有这些才算驾驶场景）

        Returns:
            核心车辆类别ID集合
        """
        return {
            2,   # bicycle
            3,   # car
            4,   # motorcycle
            6,   # bus
            8,   # truck
        }

    def score_image(self, img_id: int, coco_loader) -> float:
        """
        对图像进行综合评分

        🔧 改进：道路场景优先，车辆非必须
        - 有道路文本关键词（road/street/highway）→ 基础分
        - 有交通设施（traffic light/stop sign）→ 基础分
        - 有车辆（car/truck/bus）→ 加分项，非必须
        - 有负向关键词（室内/运动等）→ 排除

        Args:
            img_id: 图像ID
            coco_loader: COCO数据加载器实例

        Returns:
            综合得分（0-1）
        """
        self.stats['total_images'] += 1

        # 1. 类别得分（40%）
        category_score = self._score_categories(img_id, coco_loader)
        self.stats['category_scores'].append(category_score)

        # 2. 文本语义得分（30%）
        text_score = self._score_text_semantics(img_id, coco_loader)
        self.stats['text_scores'].append(text_score)

        # 3. 场景特征得分（30%）
        scene_score = self._score_scene_features(img_id, coco_loader)
        self.stats['scene_scores'].append(scene_score)

        # 4. 加权求和
        weights = self.config.get('scoring', {})
        category_weight = weights.get('category_weight', 0.4)
        text_weight = weights.get('text_weight', 0.35)  # 提高文本权重
        scene_weight = weights.get('scene_weight', 0.25)

        # 🔧 新逻辑：道路场景优先
        # 只要有道路文本证据，就是驾驶场景（车辆非必须）
        final_score = (
            category_score * category_weight +
            text_score * text_weight +
            scene_score * scene_weight
        )

        # 记录统计信息
        self.stats['scored_images'] += 1
        self.stats['final_scores'].append(final_score)

        return min(1.0, final_score)

    def _score_categories(self, img_id: int, coco_loader) -> float:
        """
        基于检测对象评分

        🔧 改进：交通设施和车辆都是有效信号，车辆非必须

        评分策略：
        - 交通设施（traffic light/stop sign）→ 基础分（强信号）
        - 车辆（car/truck/bus）→ 加分项（但非必须）
        - 辅助类别（person等）→ 补充分数
        """
        instances = coco_loader.instances_data.get(img_id, [])
        if not instances:
            return 0.0

        # 交通设施类别（强信号，独立证据）
        traffic_facility_cats = {10, 13, 14}  # traffic light, stop sign, parking meter

        # 核心车辆类别
        vehicle_cats = {2, 3, 4, 6, 8}  # bicycle, car, motorcycle, bus, truck

        # 辅助类别
        auxiliary_cats = {1, 12, 15}  # person, fire hydrant, bench

        has_traffic_facility = False
        has_vehicle = False
        total_weight = 0.0

        for inst in instances:
            cat_id = inst['category_id']
            if cat_id not in self.driving_categories:
                continue

            weight = self.category_weights.get(cat_id, self.category_weights.get('default', 1.0))

            # 交通设施（独立强信号）
            if cat_id in traffic_facility_cats:
                has_traffic_facility = True
                total_weight += weight
            # 车辆（加分项）
            elif cat_id in vehicle_cats:
                has_vehicle = True
                total_weight += weight
            # 辅助类别（补充）
            elif cat_id in auxiliary_cats:
                total_weight += weight * 0.3

        # 🔧 计算得分（不再强制要求车辆）
        # 有交通设施或车辆，给基础分
        if has_traffic_facility or has_vehicle:
            base_score = 0.5
            bonus_score = min(0.5, total_weight / 15.0)
            return base_score + bonus_score

        # 只有辅助类别，给少量分（可能通过文本关键词补偿）
        elif total_weight > 0:
            return min(0.3, total_weight / 20.0)

        return 0.0

    def _score_text_semantics(self, img_id: int, coco_loader) -> float:
        """
        基于文本描述评分

        评分策略：
        - Caption匹配：+0.5分
        - VQA问题匹配：+0.5分
        - 上限1.0分
        """
        if not self.keywords_matcher:
            return 0.0

        score = 0.0

        # 检查Captions
        captions = coco_loader.captions_data.get(img_id, [])
        if captions:
            for cap in captions:
                text = cap.get('caption', '').lower()
                matched, _ = self.keywords_matcher.match(text)
                if matched:
                    score += 0.5
                    break

        # 检查VQA问题
        questions = coco_loader.vqa_data_by_image.get(img_id, [])
        if questions:
            for q in questions:
                text = q.get('question', '').lower()
                matched, _ = self.keywords_matcher.match(text)
                if matched:
                    score += 0.5
                    break

        return min(1.0, score)

    def _score_scene_features(self, img_id: int, coco_loader) -> float:
        """
        基于场景特征评分

        评分策略：
        - 宽高比（道路场景多为宽屏）：+0.3分
        - 对象数量（复杂场景）：+0.2分
        - 对象密度（适中密度）：+0.1分
        """
        img_info = coco_loader.images_data.get(img_id, {})
        if not img_info:
            return 0.0

        score = 0.5  # 基础分

        # 1. 宽高比特征（道路场景多为宽屏）
        width = img_info.get('width', 0)
        height = img_info.get('height', 0)
        if width > 0 and height > 0:
            aspect_ratio = width / height
            # 宽屏场景（1.5-2.5）加分
            if 1.5 <= aspect_ratio <= 2.5:
                score += 0.3
            # 超宽屏场景（2.5-3.0）适度加分
            elif 2.5 < aspect_ratio <= 3.0:
                score += 0.2

        # 2. 对象数量特征（复杂场景）
        instances = coco_loader.instances_data.get(img_id, [])
        num_objects = len(instances)
        if 5 <= num_objects <= 20:  # 适中对象数量（复杂但不拥挤）
            score += 0.2
        elif 20 < num_objects <= 50:  # 较多对象
            score += 0.1

        return min(1.0, score)

    def get_category_name(self, cat_id: int, coco_loader) -> str:
        """获取类别名称"""
        return coco_loader.categories.get(cat_id, f"unknown_{cat_id}")

    def get_stats_summary(self) -> Dict:
        """获取统计摘要"""
        import numpy as np

        if not self.stats['final_scores']:
            return {}

        scores = np.array(self.stats['final_scores'])

        return {
            'total_images': self.stats['total_images'],
            'scored_images': self.stats['scored_images'],
            'score_distribution': {
                'mean': float(np.mean(scores)),
                'median': float(np.median(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'q25': float(np.percentile(scores, 25)),
                'q75': float(np.percentile(scores, 75)),
            },
            'category_score_mean': float(np.mean(self.stats['category_scores'])) if self.stats['category_scores'] else 0,
            'text_score_mean': float(np.mean(self.stats['text_scores'])) if self.stats['text_scores'] else 0,
            'scene_score_mean': float(np.mean(self.stats['scene_scores'])) if self.stats['scene_scores'] else 0,
        }


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    from .keyword_matcher import KeywordMatcher

    print("\n" + "="*60)
    print("智能驾驶数据打分器测试")
    print("="*60)

    # 初始化关键词匹配器和打分器
    matcher = KeywordMatcher(keywords=KeywordMatcher.get_default_keywords())
    scorer = DrivingDataScorer(keywords_matcher=matcher)

    print(f"\n✓ 打分器已初始化")
    print(f"  - 智能驾驶类别数: {len(scorer.driving_categories)}")
    print(f"  - 类别权重配置: {len(scorer.category_weights)} 个类别")

    # 测试场景特征评分（独立函数）
    print("\n" + "-"*60)
    print("场景特征评分测试：")
    print("-"*60)

    # 模拟图像信息
    test_cases = [
        {'width': 640, 'height': 480, 'name': '普通宽高比（4:3）'},
        {'width': 1920, 'height': 1080, 'name': '宽屏（16:9）'},
        {'width': 1280, 'height': 720, 'name': 'HD宽屏（16:9）'},
    ]

    for case in test_cases:
        # 模拟coco_loader
        class MockCOCOLoader:
            def __init__(self, img_info):
                self.images_data = {1: img_info}
                self.instances_data = {1: []}

        mock_loader = MockCOCOLoader({'width': case['width'], 'height': case['height']})
        score = scorer._score_scene_features(1, mock_loader)
        print(f"  {case['name']}: {score:.2f}")

    print("\n✓ 测试完成")