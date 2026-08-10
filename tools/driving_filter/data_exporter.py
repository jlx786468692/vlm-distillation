"""
COCO数据导出器
==============

功能：
- 导出筛选后的COCO标注文件
- 处理图像文件（符号链接或复制）
- 导出统计信息和元数据
"""

import json
import shutil
from pathlib import Path
from typing import List, Dict, Set
from datetime import datetime


class DataExporter:
    """
    COCO数据导出器

    导出内容：
    - 标注文件：instances, captions, vqa, keypoints
    - 图像文件：符号链接或复制
    - 元数据：统计信息、得分详情
    """

    def __init__(self, config: Dict = None):
        """
        初始化导出器

        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.stats = {
            'exported_images': 0,
            'exported_instances': 0,
            'exported_captions': 0,
            'exported_vqa_questions': 0,
            'exported_vqa_annotations': 0,
            'exported_keypoints': 0
        }

    def export(
        self,
        filtered_img_ids: List[int],
        coco_loader,
        output_dir: str,
        scores: Dict[int, float] = None
    ):
        """
        导出筛选后的数据

        Args:
            filtered_img_ids: 筛选后的图像ID列表
            coco_loader: COCO数据加载器实例
            output_dir: 输出目录
            scores: 图像得分字典（可选）
        """
        output_root = Path(output_dir)
        annotations_dir = output_root / "annotations"
        images_dir = output_root / "images"
        metadata_dir = output_root / "metadata"

        # 创建目录
        annotations_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n正在导出筛选数据到: {output_root}")
        print(f"  - 筛选图像数: {len(filtered_img_ids)}")

        # 1. 导出instances标注
        self._export_instances(filtered_img_ids, coco_loader, annotations_dir)

        # 2. 导出captions标注
        self._export_captions(filtered_img_ids, coco_loader, annotations_dir)

        # 3. 导出VQA问题和答案
        self._export_vqa(filtered_img_ids, coco_loader, annotations_dir)

        # 4. 导出keypoints标注
        self._export_keypoints(filtered_img_ids, coco_loader, annotations_dir)

        # 5. 处理图像文件（符号链接或复制）
        self._handle_images(filtered_img_ids, coco_loader, images_dir)

        # 6. 导出统计信息
        self._export_statistics(filtered_img_ids, coco_loader, scores, metadata_dir)

        # 7. 导出得分详情
        if scores:
            self._export_scores(scores, metadata_dir)

        print(f"\n✓ 数据导出完成")
        self._print_export_summary()

    def _export_instances(self, img_ids: List[int], coco_loader, output_dir: Path):
        """导出instances标注"""
        output_file = output_dir / f"instances_{coco_loader.config.get('data.val_split', 'val2014')}.json"

        print(f"\n导出instances标注: {output_file}")

        # 构建COCO格式数据
        coco_data = {
            'info': {
                'description': 'COCO 2014 Dataset - Filtered for Autonomous Driving',
                'url': 'http://cocodataset.org',
                'version': '1.0',
                'year': datetime.now().year,
                'contributor': 'Filtered by Driving Data Filter Tool',
                'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'licenses': [
                {
                    'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/',
                    'id': 1,
                    'name': 'Attribution-NonCommercial-ShareAlike License'
                }
            ],
            'images': [],
            'annotations': [],
            'categories': []
        }

        # 添加图像信息
        for img_id in img_ids:
            if img_id in coco_loader.images_data:
                img_info = coco_loader.images_data[img_id].copy()
                coco_data['images'].append(img_info)

        # 添加标注信息
        annotation_id = 1
        for img_id in img_ids:
            instances = coco_loader.instances_data.get(img_id, [])
            for inst in instances:
                inst_copy = inst.copy()
                inst_copy['id'] = annotation_id
                coco_data['annotations'].append(inst_copy)
                annotation_id += 1
                self.stats['exported_instances'] += 1

        # 添加类别信息（保留所有80个类别）
        if coco_loader.coco_instance:
            for cat in coco_loader.coco_instance.cats.values():
                coco_data['categories'].append(cat)
        else:
            # 使用内置类别列表
            coco_data['categories'] = self._get_coco_categories()

        # 保存文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f)

        self.stats['exported_images'] = len(img_ids)
        print(f"  ✓ 导出 {len(coco_data['images'])} 张图像")
        print(f"  ✓ 导出 {len(coco_data['annotations'])} 个实例标注")

    def _export_captions(self, img_ids: List[int], coco_loader, output_dir: Path):
        """导出captions标注"""
        output_file = output_dir / f"captions_{coco_loader.config.get('data.val_split', 'val2014')}.json"

        print(f"\n导出captions标注: {output_file}")

        # 构建COCO格式数据
        coco_data = {
            'info': {
                'description': 'COCO 2014 Captions - Filtered for Autonomous Driving',
                'version': '1.0',
                'year': datetime.now().year
            },
            'images': [],
            'annotations': []
        }

        # 添加图像信息
        for img_id in img_ids:
            if img_id in coco_loader.images_data:
                img_info = coco_loader.images_data[img_id].copy()
                coco_data['images'].append(img_info)

        # 添加标注信息
        annotation_id = 1
        for img_id in img_ids:
            captions = coco_loader.captions_data.get(img_id, [])
            for cap in captions:
                cap_copy = cap.copy()
                cap_copy['id'] = annotation_id
                coco_data['annotations'].append(cap_copy)
                annotation_id += 1
                self.stats['exported_captions'] += 1

        # 保存文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f)

        print(f"  ✓ 导出 {len(coco_data['annotations'])} 个描述标注")

    def _export_vqa(self, img_ids: List[int], coco_loader, output_dir: Path):
        """导出VQA问题和答案"""
        # 导出问题
        questions_file = output_dir / f"v2_mscoco_{coco_loader.config.get('data.val_split', 'val2014')}_questions.json"

        print(f"\n导出VQA问题: {questions_file}")

        vqa_questions_data = {
            'info': {
                'description': 'VQA v2 Questions - Filtered for Autonomous Driving',
                'version': '1.0',
                'year': datetime.now().year
            },
            'questions': []
        }

        # 添加问题
        for img_id in img_ids:
            questions = coco_loader.vqa_data_by_image.get(img_id, [])
            for q in questions:
                vqa_questions_data['questions'].append(q)
                self.stats['exported_vqa_questions'] += 1

        # 保存问题文件
        with open(questions_file, 'w', encoding='utf-8') as f:
            json.dump(vqa_questions_data, f)

        print(f"  ✓ 导出 {len(vqa_questions_data['questions'])} 个问题")

        # 导出答案（如果存在）
        if coco_loader.vqa_answers_by_question:
            annotations_file = output_dir / f"v2_mscoco_{coco_loader.config.get('data.val_split', 'val2014')}_annotations.json"

            print(f"\n导出VQA答案: {annotations_file}")

            vqa_annotations_data = {
                'info': {
                    'description': 'VQA v2 Annotations - Filtered for Autonomous Driving',
                    'version': '1.0',
                    'year': datetime.now().year
                },
                'annotations': []
            }

            # 添加答案
            for img_id in img_ids:
                questions = coco_loader.vqa_data_by_image.get(img_id, [])
                for q in questions:
                    question_id = q.get('question_id')
                    if question_id and question_id in coco_loader.vqa_answers_by_question:
                        ann = coco_loader.vqa_answers_by_question[question_id]
                        vqa_annotations_data['annotations'].append(ann)
                        self.stats['exported_vqa_annotations'] += 1

            # 保存答案文件
            with open(annotations_file, 'w', encoding='utf-8') as f:
                json.dump(vqa_annotations_data, f)

            print(f"  ✓ 导出 {len(vqa_annotations_data['annotations'])} 个答案标注")

    def _export_keypoints(self, img_ids: List[int], coco_loader, output_dir: Path):
        """导出keypoints标注"""
        output_file = output_dir / f"person_keypoints_{coco_loader.config.get('data.val_split', 'val2014')}.json"

        print(f"\n导出keypoints标注: {output_file}")

        # 构建COCO格式数据
        coco_data = {
            'info': {
                'description': 'COCO 2014 Person Keypoints - Filtered for Autonomous Driving',
                'version': '1.0',
                'year': datetime.now().year
            },
            'images': [],
            'annotations': [],
            'categories': []
        }

        # 添加图像信息
        for img_id in img_ids:
            if img_id in coco_loader.images_data:
                img_info = coco_loader.images_data[img_id].copy()
                coco_data['images'].append(img_info)

        # 添加标注信息
        annotation_id = 1
        for img_id in img_ids:
            keypoints = coco_loader.keypoints_data.get(img_id, [])
            for kp in keypoints:
                kp_copy = kp.copy()
                kp_copy['id'] = annotation_id
                coco_data['annotations'].append(kp_copy)
                annotation_id += 1
                self.stats['exported_keypoints'] += 1

        # 添加类别信息（Person Keypoints）
        coco_data['categories'] = self._get_keypoint_categories()

        # 保存文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f)

        print(f"  ✓ 导出 {len(coco_data['annotations'])} 个关键点标注")

    def _handle_images(self, img_ids: List[int], coco_loader, images_dir: Path):
        """
        处理图像文件（符号链接或复制）

        Args:
            img_ids: 图像ID列表
            coco_loader: COCO数据加载器
            images_dir: 输出目录
        """
        import platform
        import os

        copy_images = self.config.get('output', {}).get('copy_images', False)

        # Windows系统强制使用复制（符号链接需要管理员权限）
        if platform.system() == 'Windows' and not copy_images:
            print("⚠ Windows系统检测到，自动切换为复制模式（符号链接需要管理员权限）")
            copy_images = True

        print(f"\n处理图像文件...")
        print(f"  方式: {'复制' if copy_images else '符号链接'}")

        # 获取split名称
        split = coco_loader.config.get('data.val_split', 'val2014')
        target_images_dir = images_dir / split

        # 创建目标目录
        try:
            target_images_dir.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ 目标目录创建成功: {target_images_dir}")
        except Exception as e:
            print(f"  ✗ 目标目录创建失败: {e}")
            return

        success_count = 0
        failed_count = 0
        skipped_count = 0

        for img_id in img_ids:
            # 获取图像路径
            src_path = coco_loader.get_image_path(img_id, split)

            if not src_path:
                skipped_count += 1
                continue

            src_path = Path(src_path)  # 确保是Path对象

            # 检查源文件是否存在
            if not src_path.exists():
                skipped_count += 1
                if skipped_count <= 3:  # 只显示前3个跳过的文件
                    print(f"  ⚠ 源文件不存在，跳过: {src_path}")
                continue

            # 目标路径
            file_name = src_path.name
            dst_path = target_images_dir / file_name

            try:
                # 确保目标目录存在（再次确认）
                if not target_images_dir.exists():
                    target_images_dir.mkdir(parents=True, exist_ok=True)

                # 如果目标文件已存在，跳过
                if dst_path.exists():
                    success_count += 1
                    continue

                if copy_images:
                    # 复制文件（使用更稳健的方法）
                    # 方法1: 使用shutil.copy（不保留元数据，更兼容）
                    import shutil
                    shutil.copy(str(src_path), str(dst_path))
                    success_count += 1
                else:
                    # 创建符号链接
                    try:
                        dst_path.symlink_to(src_path)
                        success_count += 1
                    except OSError as e:
                        # 符号链接失败，回退到复制
                        print(f"    ⚠ 符号链接失败，回退到复制: {file_name}")
                        print(f"       错误: {e}")

                        try:
                            # 🔧 确保目标目录存在
                            if not target_images_dir.exists():
                                target_images_dir.mkdir(parents=True, exist_ok=True)

                            # 🔧 使用绝对路径复制
                            shutil.copy(str(src_path.absolute()), str(dst_path.absolute()))
                            print(f"       ✓ 复制成功: {file_name}")
                            success_count += 1
                        except Exception as copy_error:
                            # 复制也失败了
                            print(f"       ✗ 复制也失败: {copy_error}")
                            print(f"          源路径: {src_path.absolute()}")
                            print(f"          目标路径: {dst_path.absolute()}")
                            print(f"          源存在: {src_path.exists()}")
                            print(f"          目标目录存在: {target_images_dir.exists()}")
                            failed_count += 1

            except PermissionError as e:
                failed_count += 1
                if failed_count <= 5:
                    print(f"  ✗ 权限错误 {file_name}: {e}")
                    print(f"     源路径: {src_path.absolute()}")
                    print(f"     目标路径: {dst_path.absolute()}")
            except FileNotFoundError as e:
                failed_count += 1
                if failed_count <= 5:
                    print(f"  ✗ 文件不存在 {file_name}: {e}")
                    print(f"     源路径: {src_path.absolute()}")
                    print(f"     源文件存在: {src_path.exists()}")
                    print(f"     目标目录: {target_images_dir.absolute()}")
                    print(f"     目标目录存在: {target_images_dir.exists()}")
            except Exception as e:
                failed_count += 1
                if failed_count <= 5:  # 只打印前5个错误
                    print(f"  ✗ 处理图像失败 {file_name}: {e}")
                    print(f"     错误类型: {type(e).__name__}")
                    print(f"     源路径: {src_path.absolute()}")
                    print(f"     目标路径: {dst_path.absolute()}")
                    print(f"     源文件存在: {src_path.exists()}")
                    print(f"     目标目录存在: {target_images_dir.exists()}")

        print(f"\n  ✓ 图像处理完成:")
        print(f"    成功: {success_count}/{len(img_ids)}")
        if skipped_count > 0:
            print(f"    跳过（源文件不存在）: {skipped_count}")
        if failed_count > 0:
            print(f"    失败: {failed_count}")

    def _export_statistics(self, img_ids: List[int], coco_loader, scores: Dict[int, float], output_dir: Path):
        """导出统计信息"""
        output_file = output_dir / "filter_statistics.json"

        print(f"\n导出统计信息: {output_file}")

        # 统计类别分布
        category_counter = {}
        for img_id in img_ids:
            instances = coco_loader.instances_data.get(img_id, [])
            for inst in instances:
                cat_id = inst['category_id']
                cat_name = coco_loader.categories.get(cat_id, f"unknown_{cat_id}")
                category_counter[cat_name] = category_counter.get(cat_name, 0) + 1

        # 统计得分分布
        score_stats = {}
        if scores:
            import numpy as np
            score_values = np.array(list(scores.values()))
            score_stats = {
                'mean': float(np.mean(score_values)),
                'median': float(np.median(score_values)),
                'std': float(np.std(score_values)),
                'min': float(np.min(score_values)),
                'max': float(np.max(score_values)),
                'q25': float(np.percentile(score_values, 25)),
                'q75': float(np.percentile(score_values, 75)),
            }

        # 构建统计信息
        statistics = {
            'filter_time': datetime.now().isoformat(),
            'total_images': len(coco_loader.images_data),
            'filtered_images': len(img_ids),
            'retention_rate': len(img_ids) / len(coco_loader.images_data) if coco_loader.images_data else 0,
            'category_distribution': category_counter,
            'score_statistics': score_stats,
            'export_stats': self.stats
        }

        # 保存文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)

        print(f"  ✓ 统计信息已保存")

    def _export_scores(self, scores: Dict[int, float], output_dir: Path):
        """导出得分详情"""
        output_file = output_dir / "filter_scores.json"

        print(f"\n导出得分详情: {output_file}")

        # 转换为可序列化格式
        scores_data = {
            'description': 'Image scores from driving data filter',
            'total_images': len(scores),
            'scores': {str(img_id): score for img_id, score in scores.items()}
        }

        # 保存文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(scores_data, f, indent=2)

        print(f"  ✓ 得分详情已保存")

    def _print_export_summary(self):
        """打印导出摘要"""
        print("\n" + "="*60)
        print("导出摘要")
        print("="*60)
        print(f"导出图像数: {self.stats['exported_images']}")
        print(f"导出实例标注数: {self.stats['exported_instances']}")
        print(f"导出描述标注数: {self.stats['exported_captions']}")
        print(f"导出VQA问题数: {self.stats['exported_vqa_questions']}")
        print(f"导出VQA答案数: {self.stats['exported_vqa_annotations']}")
        print(f"导出关键点标注数: {self.stats['exported_keypoints']}")

    @staticmethod
    def _get_coco_categories():
        """获取COCO 80类类别列表"""
        categories = [
            {'id': 1, 'name': 'person', 'supercategory': 'person'},
            {'id': 2, 'name': 'bicycle', 'supercategory': 'vehicle'},
            {'id': 3, 'name': 'car', 'supercategory': 'vehicle'},
            {'id': 4, 'name': 'motorcycle', 'supercategory': 'vehicle'},
            {'id': 5, 'name': 'airplane', 'supercategory': 'vehicle'},
            {'id': 6, 'name': 'bus', 'supercategory': 'vehicle'},
            {'id': 7, 'name': 'train', 'supercategory': 'vehicle'},
            {'id': 8, 'name': 'truck', 'supercategory': 'vehicle'},
            {'id': 9, 'name': 'boat', 'supercategory': 'vehicle'},
            {'id': 10, 'name': 'traffic light', 'supercategory': 'outdoor'},
            {'id': 11, 'name': 'fire hydrant', 'supercategory': 'outdoor'},
            {'id': 13, 'name': 'stop sign', 'supercategory': 'outdoor'},
            {'id': 14, 'name': 'parking meter', 'supercategory': 'outdoor'},
            {'id': 15, 'name': 'bench', 'supercategory': 'outdoor'},
            {'id': 16, 'name': 'bird', 'supercategory': 'animal'},
            {'id': 17, 'name': 'cat', 'supercategory': 'animal'},
            {'id': 18, 'name': 'dog', 'supercategory': 'animal'},
            {'id': 19, 'name': 'horse', 'supercategory': 'animal'},
            {'id': 20, 'name': 'sheep', 'supercategory': 'animal'},
            {'id': 21, 'name': 'cow', 'supercategory': 'animal'},
            {'id': 22, 'name': 'elephant', 'supercategory': 'animal'},
            {'id': 23, 'name': 'bear', 'supercategory': 'animal'},
            {'id': 24, 'name': 'zebra', 'supercategory': 'animal'},
            {'id': 25, 'name': 'giraffe', 'supercategory': 'animal'},
            {'id': 27, 'name': 'backpack', 'supercategory': 'accessory'},
            {'id': 28, 'name': 'umbrella', 'supercategory': 'accessory'},
            {'id': 31, 'name': 'handbag', 'supercategory': 'accessory'},
            {'id': 32, 'name': 'tie', 'supercategory': 'accessory'},
            {'id': 33, 'name': 'suitcase', 'supercategory': 'accessory'},
            {'id': 34, 'name': 'frisbee', 'supercategory': 'sports'},
            {'id': 35, 'name': 'skis', 'supercategory': 'sports'},
            {'id': 36, 'name': 'snowboard', 'supercategory': 'sports'},
            {'id': 37, 'name': 'sports ball', 'supercategory': 'sports'},
            {'id': 38, 'name': 'kite', 'supercategory': 'sports'},
            {'id': 39, 'name': 'baseball bat', 'supercategory': 'sports'},
            {'id': 40, 'name': 'baseball glove', 'supercategory': 'sports'},
            {'id': 41, 'name': 'skateboard', 'supercategory': 'sports'},
            {'id': 42, 'name': 'surfboard', 'supercategory': 'sports'},
            {'id': 43, 'name': 'tennis racket', 'supercategory': 'sports'},
            {'id': 44, 'name': 'bottle', 'supercategory': 'kitchen'},
            {'id': 46, 'name': 'wine glass', 'supercategory': 'kitchen'},
            {'id': 47, 'name': 'cup', 'supercategory': 'kitchen'},
            {'id': 48, 'name': 'fork', 'supercategory': 'kitchen'},
            {'id': 49, 'name': 'knife', 'supercategory': 'kitchen'},
            {'id': 50, 'name': 'spoon', 'supercategory': 'kitchen'},
            {'id': 51, 'name': 'bowl', 'supercategory': 'kitchen'},
            {'id': 52, 'name': 'banana', 'supercategory': 'food'},
            {'id': 53, 'name': 'apple', 'supercategory': 'food'},
            {'id': 54, 'name': 'sandwich', 'supercategory': 'food'},
            {'id': 55, 'name': 'orange', 'supercategory': 'food'},
            {'id': 56, 'name': 'broccoli', 'supercategory': 'food'},
            {'id': 57, 'name': 'carrot', 'supercategory': 'food'},
            {'id': 58, 'name': 'hot dog', 'supercategory': 'food'},
            {'id': 59, 'name': 'pizza', 'supercategory': 'food'},
            {'id': 60, 'name': 'donut', 'supercategory': 'food'},
            {'id': 61, 'name': 'cake', 'supercategory': 'food'},
            {'id': 62, 'name': 'chair', 'supercategory': 'furniture'},
            {'id': 63, 'name': 'couch', 'supercategory': 'furniture'},
            {'id': 64, 'name': 'potted plant', 'supercategory': 'furniture'},
            {'id': 65, 'name': 'bed', 'supercategory': 'furniture'},
            {'id': 67, 'name': 'dining table', 'supercategory': 'furniture'},
            {'id': 70, 'name': 'toilet', 'supercategory': 'furniture'},
            {'id': 72, 'name': 'tv', 'supercategory': 'electronic'},
            {'id': 73, 'name': 'laptop', 'supercategory': 'electronic'},
            {'id': 74, 'name': 'mouse', 'supercategory': 'electronic'},
            {'id': 75, 'name': 'remote', 'supercategory': 'electronic'},
            {'id': 76, 'name': 'keyboard', 'supercategory': 'electronic'},
            {'id': 77, 'name': 'cell phone', 'supercategory': 'electronic'},
            {'id': 78, 'name': 'microwave', 'supercategory': 'appliance'},
            {'id': 79, 'name': 'oven', 'supercategory': 'appliance'},
            {'id': 80, 'name': 'toaster', 'supercategory': 'appliance'},
            {'id': 81, 'name': 'sink', 'supercategory': 'appliance'},
            {'id': 82, 'name': 'refrigerator', 'supercategory': 'appliance'},
            {'id': 84, 'name': 'book', 'supercategory': 'indoor'},
            {'id': 85, 'name': 'clock', 'supercategory': 'indoor'},
            {'id': 86, 'name': 'vase', 'supercategory': 'indoor'},
            {'id': 87, 'name': 'scissors', 'supercategory': 'indoor'},
            {'id': 88, 'name': 'teddy bear', 'supercategory': 'indoor'},
            {'id': 89, 'name': 'hair drier', 'supercategory': 'indoor'},
            {'id': 90, 'name': 'toothbrush', 'supercategory': 'indoor'}
        ]
        return categories

    @staticmethod
    def _get_keypoint_categories():
        """获取Person Keypoints类别"""
        return [{
            'id': 1,
            'name': 'person',
            'supercategory': 'person',
            'keypoints': [
                'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
            ],
            'skeleton': [
                [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
                [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10],
                [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
            ]
        }]