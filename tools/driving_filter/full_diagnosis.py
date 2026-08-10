#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整诊断：COCO数据集和筛选数据
================================

诊断所有可能的问题
"""

import sys
from pathlib import Path
import json

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    print("\n" + "="*80)
    print("完整诊断：COCO数据集和筛选数据")
    print("="*80)

    # 1. 检查当前工作目录
    print("\n步骤1：检查当前工作目录")
    print("-" * 80)
    print(f"当前目录: {Path.cwd()}")
    print(f"项目根目录: {project_root.absolute()}")

    # 2. 检查原始COCO数据
    print("\n步骤2：检查原始COCO数据")
    print("-" * 80)

    possible_coco_paths = [
        Path("data/coco"),
        Path("/data/workspace2/jlx/workspace/vlm-distillation/data/coco"),
        Path("D:/data/01-project/vlm/data/coco"),  # 可能的其他位置
    ]

    coco_found = False
    for path in possible_coco_paths:
        print(f"\n检查路径: {path}")
        if path.exists():
            print(f"  ✓ 路径存在")

            # 检查子目录
            annotations = path / "annotations"
            images = path / "val2014"

            if annotations.exists():
                annotation_files = list(annotations.glob("*.json"))
                print(f"  ✓ 标注目录存在: {len(annotation_files)} 个文件")

                # 显示几个文件
                for f in annotation_files[:3]:
                    size_kb = f.stat().st_size / 1024
                    print(f"    - {f.name}: {size_kb:.2f} KB")
            else:
                print(f"  ✗ 标注目录不存在")

            if images.exists():
                image_files = list(images.glob("*.jpg"))
                print(f"  ✓ 图像目录存在: {len(image_files)} 张图像")

                # 检查前几张图像
                for img in image_files[:3]:
                    try:
                        size_kb = img.stat().st_size / 1024
                        print(f"    - {img.name}: {size_kb:.2f} KB ✓")
                    except Exception as e:
                        print(f"    - {img.name}: ✗ {e}")

                coco_found = True
            else:
                print(f"  ✗ 图像目录不存在")
        else:
            print(f"  ✗ 路径不存在")

    if not coco_found:
        print("\n" + "!"*80)
        print("警告：未找到COCO数据集")
        print("!"*80)
        print("\n请下载COCO数据集：")
        print("1. 访问：https://cocodataset.org/#download")
        print("2. 下载：2014 Train/Val images")
        print("3. 下载：2014 Train/Val annotations")
        print("4. 解压到：data/coco/")
        print("\n或者使用已有的数据集，修改配置文件：")
        print("  configs/default.yaml:")
        print("    data:")
        print("      coco_root: \"实际路径\"")
        print("      images_root: \"实际图像路径\"")

    # 3. 检查筛选后的数据
    print("\n步骤3：检查筛选后的数据")
    print("-" * 80)

    filter_paths = [
        Path("data/filter_coco"),
        Path("/data/workspace2/jlx/workspace/vlm-distillation/data/filter_coco"),
    ]

    filter_found = False
    for path in filter_paths:
        print(f"\n检查路径: {path}")

        if path.exists():
            print(f"  ✓ 筛选数据目录存在")

            # 检查标注
            annotations = path / "annotations"
            if annotations.exists():
                annotation_files = list(annotations.glob("*.json"))
                print(f"  ✓ 标注文件: {len(annotation_files)} 个")

                # 读取统计信息
                instances_file = annotations / "instances_val2014.json"
                if instances_file.exists():
                    try:
                        with open(instances_file, 'r') as f:
                            data = json.load(f)
                            print(f"    - 图像数: {len(data.get('images', []))}")
                            print(f"    - 标注数: {len(data.get('annotations', []))}")
                    except Exception as e:
                        print(f"    - 读取失败: {e}")

            # 检查图像
            images = path / "images" / "val2014"
            if images.exists():
                image_files = list(images.glob("*.jpg"))
                print(f"  ✓ 图像文件: {len(image_files)} 张")

                # 检查符号链接 vs 真实文件
                if image_files:
                    sample = image_files[0]
                    if sample.is_symlink():
                        target = sample.resolve()
                        print(f"    类型: 符号链接")
                        print(f"    目标: {target}")
                        print(f"    目标存在: {target.exists()}")

                        if not target.exists():
                            print(f"    ✗ 符号链接目标不存在！")
                            print(f"    这可能是问题所在！")
                    else:
                        size_mb = sample.stat().st_size / (1024*1024)
                        print(f"    类型: 真实文件")
                        print(f"    样本大小: {size_mb:.2f} MB")

                        # 尝试打开图像
                        try:
                            from PIL import Image
                            img = Image.open(sample)
                            print(f"    ✓ 可以打开: {img.size}")
                            img.close()
                        except Exception as e:
                            print(f"    ✗ 无法打开: {e}")

                filter_found = True
            else:
                print(f"  ✗ 图像目录不存在: {images}")
                print(f"  ⚠ 标注存在但图像不存在！")
                print(f"  这就是报错的原因！")
        else:
            print(f"  ✗ 筛选数据目录不存在")

    # 4. 诊断总结
    print("\n" + "="*80)
    print("诊断总结")
    print("="*80)

    if not coco_found:
        print("\n❌ 问题1：原始COCO数据集未找到")
        print("   解决：下载COCO数据集或修改配置文件路径")

    if not filter_found:
        print("\n❌ 问题2：筛选数据不存在")
        print("   解决：运行筛选工具")
        print("   命令：python tools/driving_filter/run_filter.py --copy-images")

    # 5. 提供修复方案
    if not coco_found or not filter_found:
        print("\n" + "="*80)
        print("推荐修复方案")
        print("="*80)

        print("\n方案1：重新运行筛选（推荐）")
        print("```bash")
        print("# 1. 确保原始COCO数据存在")
        print("ls data/coco/val2014/*.jpg | head -5")
        print("")
        print("# 2. 运行筛选工具（强制复制图像）")
        print("python tools/driving_filter/run_filter.py --copy-images")
        print("")
        print("# 3. 验证结果")
        print("ls data/filter_coco/images/val2014/*.jpg | head -5")
        print("```")

        print("\n方案2：只导出标注，不复制图像")
        print("```bash")
        print("python tools/driving_filter/run_filter_simple.py")
        print("```")

        print("\n方案3：检查配置文件")
        print("```bash")
        print("# 修改 configs/default.yaml")
        print("data:")
        print("  coco_root: \"实际的COCO数据路径\"")
        print("  images_root: \"实际的图像路径\"")
        print("```")

    else:
        print("\n✓ 所有数据都存在，应该可以正常加载")
        print("  如果仍有问题，请检查图像是否损坏")


if __name__ == "__main__":
    main()