#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
智能驾驶数据筛选运行脚本
======================

使用方式：
    python tools/driving_filter/run_filter.py
    python tools/driving_filter/run_filter.py --config configs/driving_filter.yaml
    python tools/driving_filter/run_filter.py --split val2014 --threshold 0.6
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tools.driving_filter.filter_engine import DrivingDataFilter
from src.data.coco_loader import COCODataLoader
from src.utils.logger import get_logger


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='COCO智能驾驶数据筛选工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python tools/driving_filter/run_filter.py                          # 使用默认配置（自动复制）
  python tools/driving_filter/run_filter.py --config configs/driving_filter.yaml  # 使用自定义配置
  python tools/driving_filter/run_filter.py --split train2014        # 处理训练集
  python tools/driving_filter/run_filter.py --threshold 0.6          # 调整筛选阈值
  python tools/driving_filter/run_filter.py --output ./data/driving_only  # 指定输出目录
  python tools/driving_filter/run_filter.py --copy-images            # 明确指定复制模式
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/driving_filter.yaml',
        help='配置文件路径（默认：configs/driving_filter.yaml）'
    )

    parser.add_argument(
        '--split',
        type=str,
        default='val2014',
        help='数据集分割（默认：val2014）'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='筛选阈值（覆盖配置文件中的设置）'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出目录（覆盖配置文件中的设置）'
    )

    parser.add_argument(
        '--copy-images',
        action='store_true',
        default=True,  # 🔧 默认使用复制模式（避免符号链接问题）
        help='复制图像文件（默认开启，避免符号链接问题）'
    )

    parser.add_argument(
        '--use-symlink',
        action='store_true',
        help='使用符号链接（仅Linux/Mac，不推荐）'
    )

    args = parser.parse_args()

    # 🔧 默认使用复制模式（除非明确指定使用符号链接）
    if not args.use_symlink:
        args.copy_images = True

    # 打印欢迎信息
    print("\n" + "="*70)
    print("   COCO智能驾驶数据筛选工具")
    print("="*70)
    print(f"\n配置:")
    print(f"  配置文件: {args.config}")
    print(f"  数据集: {args.split}")
    print(f"  输出阈值: {args.threshold if args.threshold else '使用配置文件设置'}")
    print(f"  输出目录: {args.output if args.output else '使用配置文件设置'}")
    print(f"  图像处理: {'复制' if args.copy_images else '符号链接'}")

    # 1. 初始化COCO数据加载器
    print("\n" + "="*70)
    print("步骤1：加载COCO数据集")
    print("="*70)

    try:
        coco_loader = COCODataLoader()
        coco_loader.initialize(split=args.split)
        print(f"✓ COCO数据集加载成功")
        print(f"  - 图像数: {len(coco_loader.images_data)}")
        print(f"  - 类别数: {len(coco_loader.categories)}")
    except Exception as e:
        print(f"✗ COCO数据集加载失败: {e}")
        print("\n请检查数据集路径配置:")
        print("  - configs/default.yaml 中的 data.coco_root")
        print("  - 确保 data/coco/annotations/ 目录下有相应的标注文件")
        return

    # 2. 初始化筛选器
    print("\n" + "="*70)
    print("步骤2：初始化筛选引擎")
    print("="*70)

    # 检查配置文件是否存在
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"⚠ 配置文件不存在: {args.config}")
        print("  使用默认配置")
        config_path = None

    filter_engine = DrivingDataFilter(config_path=config_path)

    # 应用命令行参数覆盖
    if args.threshold is not None:
        filter_engine.config.setdefault('driving_data_filter', {}).setdefault('scoring', {})
        filter_engine.config['driving_data_filter']['scoring']['score_threshold'] = args.threshold
        print(f"✓ 覆盖筛选阈值: {args.threshold}")

    if args.copy_images:
        filter_engine.config.setdefault('driving_data_filter', {}).setdefault('output', {})
        filter_engine.config['driving_data_filter']['output']['copy_images'] = True
        print(f"✓ 设置图像处理方式: 复制")

    # 3. 执行筛选
    print("\n" + "="*70)
    print("步骤3：执行智能驾驶数据筛选")
    print("="*70)

    output_dir = args.output
    if output_dir is None:
        output_dir = filter_engine.config.get('driving_data_filter', {}).get('output', {}).get('root', './data/filter_coco')

    try:
        filtered_img_ids, scores = filter_engine.run(coco_loader, output_dir)
    except Exception as e:
        print(f"\n✗ 筛选过程失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. 显示最终结果
    print("\n" + "="*70)
    print("筛选完成")
    print("="*70)

    print(f"\n✓ 成功筛选 {len(filtered_img_ids)} 张智能驾驶相关图像")
    print(f"✓ 数据已导出到: {output_dir}")

    print("\n输出文件结构:")
    print(f"  {output_dir}/")
    print(f"    ├── annotations/")
    print(f"    │   ├── instances_{args.split}.json")
    print(f"    │   ├── captions_{args.split}.json")
    print(f"    │   ├── v2_mscoco_{args.split}_questions.json")
    print(f"    │   ├── v2_mscoco_{args.split}_annotations.json")
    print(f"    │   └── person_keypoints_{args.split}.json")
    print(f"    ├── images/{args.split}/")
    print(f"    └── metadata/")
    print(f"        ├── filter_statistics.json")
    print(f"        └── filter_scores.json")

    # 5. 显示使用建议
    print("\n" + "="*70)
    print("使用建议")
    print("="*70)
    print("\n1. 查看筛选结果:")
    print(f"   cat {output_dir}/metadata/filter_statistics.json")

    print("\n2. 使用筛选后的数据训练模型:")
    print(f"   修改配置文件中的数据路径:")
    print(f"     data:")
    print(f"       coco_root: \"{output_dir}\"")
    print(f"       annotations_root: \"{output_dir}/annotations\"")
    print(f"       images_root: \"{output_dir}/images/{args.split}\"")

    print("\n3. 调整筛选阈值:")
    print(f"   python tools/driving_filter/run_filter.py --threshold 0.6  # 更严格")
    print(f"   python tools/driving_filter/run_filter.py --threshold 0.4  # 更宽松")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()