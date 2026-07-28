"""
阶段2：COCO图像分场景
======================

基于COCO instances检测标注，使用12大supercategory作为场景划分标准。

使用方式：
    python tools/candidate/stage2_scene_mapping.py
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List
import sys

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# COCO 12大supercategory
COCO_SUPER_CATEGORIES = [
    'person', 'vehicle', 'outdoor', 'animal', 'accessory',
    'sports', 'kitchen', 'food', 'furniture', 'electronic',
    'appliance', 'indoor'
]


def load_coco_instances(instances_file: Path) -> Dict:
    """加载COCO instances标注"""
    print(f"📖 加载COCO instances: {instances_file}")

    with open(instances_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"  ✓ 加载 {len(data.get('images', []))} 张图像")
    print(f"  ✓ 加载 {len(data.get('annotations', []))} 个实例")

    return data


def build_category_mapping(data: Dict) -> tuple:
    """
    构建类别映射：category_id → supercategory
    """
    categories = {}

    for cat in data.get('categories', []):
        cat_id = cat['id']
        cat_name = cat['name']
        supercategory = cat.get('supercategory', cat_name)

        categories[cat_id] = {
            'name': cat_name,
            'supercategory': supercategory
        }

    print(f"✓ 构建类别映射: {len(categories)} 个类别")

    return categories


def determine_image_scene(
    image_id: int,
    annotations: List[Dict],
    categories: Dict
) -> tuple:
    """
    单图像场景判定逻辑

    统计各类场景出现频次，出现最多的大类作为图像主场景

    Returns:
        primary_scene: 主场景
        scene_counter: 场景频次统计
    """
    scene_counter = Counter()

    # 统计该图像所有实例的场景
    for ann in annotations:
        if ann['image_id'] != image_id:
            continue

        cat_id = ann['category_id']
        if cat_id in categories:
            supercategory = categories[cat_id]['supercategory']
            scene_counter[supercategory] += 1

    if not scene_counter:
        return 'mixed', scene_counter

    # 找出出现最多的场景
    primary_scene = scene_counter.most_common(1)[0][0]

    # 检查是否为混合场景（多场景频次持平）
    top_freq = scene_counter[primary_scene]
    tied_scenes = [s for s, f in scene_counter.items() if f == top_freq]

    if len(tied_scenes) > 1:
        return 'mixed', scene_counter

    return primary_scene, scene_counter


def build_scene_mapping(data: Dict) -> Dict[int, Dict]:
    """
    构建图像到场景的映射

    Returns:
        imgid2scene: {image_id: {'primary_scene': str, 'scenes': list}}
    """
    print("\n" + "="*60)
    print("构建图像-场景映射")
    print("="*60)

    # 构建类别映射
    categories = build_category_mapping(data)

    # 按图像ID分组标注
    image_annotations = defaultdict(list)
    for ann in data.get('annotations', []):
        image_annotations[ann['image_id']].append(ann)

    # 统计每张图像的场景
    imgid2scene = {}
    scene_counter = Counter()

    total_images = len(data.get('images', []))

    for i, image in enumerate(data.get('images', [])):
        if i % 1000 == 0:
            print(f"  处理进度: {i}/{total_images} ({i/total_images*100:.1f}%)")

        image_id = image['id']
        annotations = image_annotations.get(image_id, [])

        # 判定主场景
        primary_scene, scene_stats = determine_image_scene(
            image_id,
            annotations,
            categories
        )

        imgid2scene[image_id] = {
            'primary_scene': primary_scene,
            'scenes': list(scene_stats.keys()),
            'scene_stats': dict(scene_stats)
        }

        scene_counter[primary_scene] += 1

    print(f"\n✓ 映射构建完成：")
    print(f"  总图像数: {len(imgid2scene)}")

    print(f"\n场景分布：")
    for scene, count in scene_counter.most_common():
        print(f"  {scene:15s}: {count:5d} 张图像")

    return imgid2scene


def main():
    parser = argparse.ArgumentParser(description="阶段2：COCO图像分场景")

    parser.add_argument(
        '--instances',
        default='data/coco/annotations/instances_train2014.json',
        help='COCO instances标注文件'
    )

    parser.add_argument(
        '--output',
        default='data/imgid2scene.json',
        help='输出文件路径'
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("阶段2：COCO图像分场景")
    print("="*60)
    print("\n使用COCO原生12大supercategory作为场景划分：")
    print(f"  {', '.join(COCO_SUPER_CATEGORIES)}")

    instances_file = Path(args.instances)

    if not instances_file.exists():
        print(f"\n❌ 文件不存在: {instances_file}")
        print("\n请下载COCO数据集：")
        print("  访问: https://cocodataset.org/#download")
        print("  下载: 2014 Train/Val annotations")
        return

    # 加载instances标注
    data = load_coco_instances(instances_file)

    # 构建场景映射
    imgid2scene = build_scene_mapping(data)

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'metadata': {
            'source': 'Qwen Official Pipeline Stage 2',
            'instances_file': str(instances_file),
            'total_images': len(imgid2scene),
            'super_categories': COCO_SUPER_CATEGORIES
        },
        'imgid2scene': imgid2scene
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print("✓ 阶段2完成")
    print("="*60)
    print(f"\n输出文件: {output_path}")
    print(f"图像-场景映射: {len(imgid2scene)} 张图像")


if __name__ == "__main__":
    main()