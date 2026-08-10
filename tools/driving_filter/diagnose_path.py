#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图像路径诊断脚本
================

用于诊断COCO图像路径问题
"""

import sys
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.coco_loader import COCODataLoader


def main():
    print("\n" + "="*70)
    print("COCO图像路径诊断")
    print("="*70)

    # 初始化COCO加载器
    print("\n步骤1：加载COCO数据...")
    try:
        coco_loader = COCODataLoader()
        coco_loader.initialize(split="val2014")
        print(f"✓ 加载成功")
        print(f"  总图像数: {len(coco_loader.images_data)}")
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        return

    # 测试几个图像路径
    print("\n步骤2：测试图像路径...")
    test_img_ids = list(coco_loader.images_data.keys())[:10]

    success_count = 0
    for img_id in test_img_ids:
        img_info = coco_loader.images_data[img_id]

        # 获取图像路径
        img_path = coco_loader.get_image_path(img_id, "val2014")

        print(f"\n图像ID {img_id}:")
        print(f"  文件名: {img_info.get('file_name', 'N/A')}")

        if img_path:
            img_path = Path(img_path)
            print(f"  计算路径: {img_path}")
            print(f"  绝对路径: {img_path.absolute()}")
            print(f"  文件存在: {img_path.exists()}")

            if img_path.exists():
                success_count += 1
                # 尝试获取文件大小
                try:
                    size_mb = img_path.stat().st_size / (1024 * 1024)
                    print(f"  文件大小: {size_mb:.2f} MB")
                except:
                    pass
        else:
            print(f"  ✗ 无法获取路径")

    print("\n" + "="*70)
    print(f"测试结果：{success_count}/{len(test_img_ids)} 张图像路径正确")
    print("="*70)

    # 检查配置
    print("\n步骤3：检查配置路径...")
    config = coco_loader.config
    print(f"  coco_root: {config.get('data.coco_root', 'N/A')}")
    print(f"  annotations_root: {config.get('data.annotations_root', 'N/A')}")
    print(f"  images_root: {config.get('data.images_root', 'N/A')}")

    # 检查这些目录是否存在
    for key in ['data.coco_root', 'data.annotations_root', 'data.images_root']:
        path_str = config.get(key)
        if path_str:
            path = Path(path_str)
            print(f"  {key} 存在: {path.exists()}")

    # 检查目标目录
    print("\n步骤4：检查目标目录...")
    target_dir = Path("data/filter_coco/images/val2014")
    print(f"  目标目录: {target_dir.absolute()}")
    print(f"  目录存在: {target_dir.exists()}")

    if not target_dir.exists():
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ 目录创建成功")

            # 测试写入权限
            test_file = target_dir / "test_write.txt"
            test_file.write_text("test")
            print(f"  ✓ 写入权限正常")
            test_file.unlink()
        except Exception as e:
            print(f"  ✗ 目录创建失败: {e}")


if __name__ == "__main__":
    main()