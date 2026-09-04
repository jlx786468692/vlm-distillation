#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按 val2014 方式过滤 train2014，输出到 ./data/filter_coco

与 val2014 完全相同的筛选配置（configs/driving_filter.yaml，阈值 0.9），
仅数据源切换为 train2014，导出文件以 train2014 命名：
  annotations/instances_train2014.json, captions_train2014.json,
  v2_mscoco_train2014_questions.json, v2_mscoco_train2014_annotations.json,
  person_keypoints_train2014.json
  images/train2014/
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import ConfigManager
from src.data.coco_loader import COCODataLoader
from tools.driving_filter.filter_engine import DrivingDataFilter


def main():
    print("\n" + "=" * 70)
    print("   COCO 智能驾驶数据筛选 - train2014 (同 val2014 配置)")
    print("=" * 70)

    # 1. 构建指向【源 COCO】的配置
    #    default.yaml 当前指向 ./data/filter_coco（蒸馏训练用），这里覆盖回源数据
    cm = ConfigManager()
    cm.config['data']['coco_root'] = './data/coco'
    cm.config['data']['annotations_root'] = './data/coco/annotations'
    cm.config['data']['images_root'] = './data/coco'
    # 让 DataExporter 以 train2014 命名输出文件与 images 子目录
    cm.config['data']['val_split'] = 'train2014'
    cm.config['data']['train_split'] = 'train2014'
    # GT 映射缓存与过滤无关，禁用避免误用 val2014 旧缓存
    cm.config.setdefault('cleaning', {}).setdefault('gt_mapping', {})['cache_mode'] = 'disabled'

    # 2. 加载 train2014 数据
    print("\n步骤1：加载 COCO train2014 数据集...")
    coco_loader = COCODataLoader(config=cm)
    coco_loader.initialize(split='train2014')
    print(f"✓ 加载成功，共 {len(coco_loader.images_data)} 张图像")

    # 3. 初始化筛选引擎（与 val2014 同配置）
    filter_engine = DrivingDataFilter(config_path='configs/driving_filter.yaml')
    # 复制图像（与 val2014 一致，非符号链接）
    filter_engine.config.setdefault('driving_data_filter', {}).setdefault('output', {})['copy_images'] = True

    # 4. 执行筛选，输出到 ./data/filter_coco
    output_dir = './data/filter_coco'
    print(f"\n步骤2：执行筛选，输出到 {output_dir}")
    filtered_img_ids, scores = filter_engine.run(coco_loader, output_dir)

    print("\n" + "=" * 70)
    print(f"✓ train2014 筛选完成：{len(filtered_img_ids)} 张图像")
    print("=" * 70)


if __name__ == '__main__':
    main()
