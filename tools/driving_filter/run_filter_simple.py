#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版智能驾驶数据筛选（仅导出标注，不复制图像）
=====================================================

使用场景：
- 图像复制失败时
- 只需要标注文件时
- 验证筛选逻辑时
"""

import sys
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tools.driving_filter.filter_engine import DrivingDataFilter
from src.data.coco_loader import COCODataLoader


def main():
    """简化版筛选流程：只导出标注，跳过图像处理"""
    print("\n" + "="*70)
    print("   简化版智能驾驶数据筛选（仅标注）")
    print("="*70)

    # 1. 加载COCO数据
    print("\n步骤1：加载COCO数据...")
    try:
        coco_loader = COCODataLoader()
        coco_loader.initialize(split="val2014")
        print(f"✓ 加载成功，共 {len(coco_loader.images_data)} 张图像")
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        return

    # 2. 执行筛选
    print("\n步骤2：执行筛选...")
    filter_engine = DrivingDataFilter()

    # 只评分，不导出
    all_img_ids = coco_loader.get_image_ids()
    scores = {}

    from tqdm import tqdm
    for img_id in tqdm(all_img_ids, desc="评分"):
        score = filter_engine.scorer.score_image(img_id, coco_loader)
        scores[img_id] = score

    # 3. 筛选
    threshold = 0.5
    filtered_img_ids = [img_id for img_id, score in scores.items() if score >= threshold]

    print(f"\n✓ 筛选完成")
    print(f"  筛选图像数: {len(filtered_img_ids)}/{len(all_img_ids)}")
    print(f"  保留率: {len(filtered_img_ids)/len(all_img_ids):.2%}")

    # 4. 仅导出标注（跳过图像处理）
    print("\n步骤3：导出标注文件（跳过图像）...")
    output_dir = Path("data/filter_coco")

    from tools.driving_filter.data_exporter import DataExporter
    exporter = DataExporter(config={'output': {'copy_images': True}})

    # 导出标注文件（调用内部方法，跳过图像处理）
    annotations_dir = output_dir / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    # 导出instances
    exporter._export_instances(filtered_img_ids, coco_loader, annotations_dir)

    # 导出captions
    exporter._export_captions(filtered_img_ids, coco_loader, annotations_dir)

    # 导出VQA
    exporter._export_vqa(filtered_img_ids, coco_loader, annotations_dir)

    # 导出统计信息
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    exporter._export_statistics(filtered_img_ids, coco_loader, scores, metadata_dir)
    exporter._export_scores(scores, metadata_dir)

    print(f"\n✓ 标注文件导出完成: {annotations_dir}")

    # 5. 提示用户
    print("\n" + "="*70)
    print("后续操作建议")
    print("="*70)
    print("\n1. 标注文件已导出，可直接用于训练（修改配置文件路径）")
    print("\n2. 如需复制图像，请手动执行：")
    print("   mkdir -p data/filter_coco/images/val2014")
    print("   # 根据筛选结果复制对应图像")

    print(f"\n3. 查看筛选结果：")
    print(f"   cat {output_dir}/metadata/filter_statistics.json")


if __name__ == "__main__":
    main()