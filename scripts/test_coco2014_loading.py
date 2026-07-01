"""
测试COCO 2014数据集加载
=======================

验证COCO 2014数据集是否正确配置和加载。

Usage:
    python scripts/test_coco2014_loading.py
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from data import COCODataLoader
from utils import ConfigManager

def test_coco2014_loading():
    """测试COCO 2014数据集加载"""

    print("="*70)
    print("COCO 2014 数据集加载测试")
    print("="*70)

    # 加载配置
    config = ConfigManager('configs/default.yaml')

    # 检查配置
    print("\n配置检查:")
    print(f"  Annotations root: {config.get('data.annotations_root')}")
    print(f"  Images root: {config.get('data.images_root')}")
    print(f"  Val split: {config.get('data.val_split')}")

    # 检查路径是否存在
    annotations_root = Path(config.get('data.annotations_root'))
    images_root = Path(config.get('data.images_root'))
    val_split = config.get('data.val_split')

    print("\n路径检查:")
    print(f"  Annotations目录: {annotations_root.exists()} - {annotations_root}")
    print(f"  Images目录: {images_root.exists()} - {images_root}")

    # 检查标注文件
    print("\n标注文件检查:")
    required_annotations = [
        f"captions_{val_split}.json",
        f"instances_{val_split}.json",
        f"v2_mscoco_{val_split}_questions.json"
    ]

    for ann_file in required_annotations:
        ann_path = annotations_root / ann_file
        exists = ann_path.exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {ann_file}: {exists}")

        if not exists:
            print(f"      ⚠️  文件缺失！请下载: {ann_file}")

    # 检查图像目录
    image_dir = images_root / val_split
    print(f"\n图像目录检查:")
    print(f"  目录: {image_dir}")
    print(f"  存在: {image_dir.exists()}")

    if image_dir.exists():
        image_files = list(image_dir.glob("*.jpg"))
        print(f"  图像数量: {len(image_files)}")
        if len(image_files) > 0:
            print(f"  示例图像: {image_files[0].name}")
    else:
        print(f"  ⚠️  图像目录不存在！请下载并解压: {val_split}.zip")

    # 初始化数据加载器
    print("\n" + "="*70)
    print("开始加载COCO数据集...")
    print("="*70)

    try:
        loader = COCODataLoader(config)

        # 测试加载
        print(f"\nInitializing split: {val_split}")
        loader.initialize(val_split)

        # 获取摘要
        summary = loader.get_annotation_summary()

        print("\n数据集摘要:")
        print(f"  总图像数: {summary['total_images']}")
        print(f"  有Caption的图像: {summary['images_with_captions']}")
        print(f"  有Instance的图像: {summary['images_with_instances']}")
        print(f"  有VQA的图像: {summary['images_with_vqa']}")
        print(f"  Caption总数: {summary['total_captions']}")
        print(f"  Instance总数: {summary['total_instances']}")
        print(f"  VQA问题总数: {summary['total_vqa_questions']}")
        print(f"  目标类别数: {summary['num_categories']}")

        if summary['num_categories'] > 0:
            print(f"  示例类别: {summary['categories'][:5]}")

        # 测试图像加载
        if summary['total_images'] > 0:
            test_image_id = list(loader.images_data.keys())[0]
            print(f"\n测试图像加载:")
            print(f"  测试Image ID: {test_image_id}")

            image_path = loader.get_image_path(test_image_id, val_split)

            if image_path:
                print(f"  ✅ 图像路径: {image_path}")
                print(f"  ✅ 文件存在: {image_path.exists()}")

                # 尝试加载图像
                try:
                    image = loader.load_image(test_image_id, val_split)
                    if image:
                        print(f"  ✅ 图像加载成功: {image.size}")
                    else:
                        print(f"  ❌ 图像加载失败")
                except Exception as e:
                    print(f"  ❌ 图像加载错误: {e}")
            else:
                print(f"  ❌ 未找到图像路径")

            # 测试标注获取
            print(f"\n测试标注获取:")

            # Caption标注
            captions = loader.get_captions(test_image_id)
            if captions:
                print(f"  ✅ Caption数量: {len(captions)}")
                print(f"  示例Caption: '{captions[0]['caption'][:50]}...'")
            else:
                print(f"  ❌ 无Caption标注")

            # Instance标注
            instances = loader.get_instances(test_image_id)
            if instances:
                print(f"  ✅ Instance数量: {len(instances)}")
                print(f"  示例类别: {instances[0].get('category_name', 'unknown')}")
            else:
                print(f"  ❌ 无Instance标注")

            # VQA问题
            vqa_questions = loader.get_vqa_questions(test_image_id)
            if vqa_questions:
                print(f"  ✅ VQA问题数量: {len(vqa_questions)}")
                print(f"  示例问题: '{vqa_questions[0]['question'][:50]}...'")
            else:
                print(f"  ❌ 无VQA问题")

        # 最终状态
        print("\n" + "="*70)
        if summary['total_images'] > 0:
            print("✅ COCO 2014数据集加载成功！")
            print("="*70)
            print("\n建议:")
            print(f"  - 使用 {summary['total_images']} 张图像进行蒸馏")
            print(f"  - 运行命令: python scripts/run_full_pipeline.py --samples 10")
            return True
        else:
            print("❌ COCO 2014数据集加载失败：0张图像")
            print("="*70)
            print("\n解决方案:")
            print("  1. 检查标注文件是否下载并解压")
            print("  2. 检查图像文件是否下载并解压")
            print("  3. 检查配置文件路径是否正确")
            return False

    except Exception as e:
        print(f"\n❌ 加载失败: {e}")
        print(f"\n错误类型: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_coco2014_loading()
    sys.exit(0 if success else 1)