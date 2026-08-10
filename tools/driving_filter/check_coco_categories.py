#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查COCO数据集类别
==================

列出所有COCO类别，查看是否有车道线相关标注
"""

import sys
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.coco_loader import COCODataLoader


def main():
    print("\n" + "="*70)
    print("COCO数据集类别检查")
    print("="*70)

    # 加载COCO数据
    print("\n正在加载COCO数据...")
    try:
        coco_loader = COCODataLoader()
        coco_loader.initialize(split="val2014")

        print(f"✓ 加载成功")
        print(f"  总图像数: {len(coco_loader.images_data)}")
        print(f"  总类别数: {len(coco_loader.categories)}")

    except Exception as e:
        print(f"✗ 加载失败: {e}")
        return

    # 打印所有类别
    print("\n" + "="*70)
    print("COCO 80个类别完整列表")
    print("="*70)

    # 按类别ID排序
    sorted_categories = sorted(coco_loader.categories.items(), key=lambda x: x[0])

    for cat_id, cat_name in sorted_categories:
        # 标记智能驾驶相关类别
        marker = ""
        if cat_id in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15}:
            marker = " ⭐ 智能驾驶相关"
        elif 'road' in cat_name.lower() or 'lane' in cat_name.lower() or 'street' in cat_name.lower():
            marker = " ⭐⭐⭐ 道路相关（找到！）"

        print(f"  {cat_id:3d}: {cat_name:20s} {marker}")

    print("\n" + "="*70)
    print("检查结果")
    print("="*70)

    # 检查是否有道路相关类别
    road_related = ['road', 'street', 'lane', 'highway', 'sidewalk', 'crosswalk']

    found = []
    for cat_id, cat_name in sorted_categories:
        if any(keyword in cat_name.lower() for keyword in road_related):
            found.append((cat_id, cat_name))

    if found:
        print("\n✓ 找到道路相关类别：")
        for cat_id, cat_name in found:
            print(f"  {cat_id}: {cat_name}")
    else:
        print("\n✗ 没有找到道路相关类别")
        print("  COCO数据集不包含：车道线、道路、人行道等标注")

    print("\n结论：")
    print("  - COCO 80个类别中，没有车道线（lane line）类别")
    print("  - 也没有道路（road）、人行道（sidewalk）等类别")
    print("  - 最接近的是：traffic light（交通灯）、stop sign（停止标志）")

    print("\n建议筛选策略：")
    print("  1. 必须有核心车辆类别（car/truck/bus）")
    print("  2. 结合交通设施类别（traffic light/stop sign）")
    print("  3. 使用文本语义匹配道路关键词（road/street/highway）")


if __name__ == "__main__":
    main()