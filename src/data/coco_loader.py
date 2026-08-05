"""
COCO Dataset Loader
===================

Handles loading and accessing COCO dataset annotations and images.
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime

try:
    from pycocotools.coco import COCO
except ImportError:
    COCO = None

from PIL import Image
import numpy as np

from ..utils.config import ConfigManager
from ..utils.logger import get_logger


class COCODataLoader:
    """
    Loads and manages COCO dataset for multi-task distillation.

    Supports:
    - Image Captioning (COCO Captions)
    - Object Detection (COCO Instances)
    - VQA (COCO VQA annotations)
    """

    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize COCODataLoader.

        Args:
            config: Configuration manager instance
        """
        self.config = config or ConfigManager()
        self.logger = get_logger()

        # Data paths
        self.coco_root = Path(self.config.get("data.coco_root", "./data/coco"))
        self.annotations_root = Path(self.config.get("data.annotations_root", "./data/coco/annotations"))
        self.images_root = Path(self.config.get("data.images_root", "./data/coco/images"))

        # COCO API instances
        self.coco_caption = None
        self.coco_instance = None
        self.coco_keypoints = None  # Person keypoints API
        self.vqa_data = None

        # Loaded data
        self.images_data: Dict[int, Dict] = {}
        self.captions_data: Dict[int, List[Dict]] = defaultdict(list)
        self.instances_data: Dict[int, List[Dict]] = defaultdict(list)
        self.keypoints_data: Dict[int, List[Dict]] = defaultdict(list)  # Person keypoints
        self.vqa_data_by_image: Dict[int, List[Dict]] = defaultdict(list)

        # 🔧 新增：VQA标注答案（ground truth）及置信度
        self.vqa_answers_by_question: Dict[int, Dict] = {}  # question_id -> annotation

        # 🔧 新增：VQA标注置信度信息
        # 存储格式：{
        #     'answer': str,              # 最终答案（过滤no后最常见）
        #     'gt_consistency': float,    # 置信度（最常见答案频率）
        #     'gt_dist': Dict[str, float] # 答案分布
        # }
        self.vqa_gt_with_confidence: Dict[int, Dict] = {}  # question_id -> gt_info

        # Categories
        self.categories: Dict[int, str] = {}

        # Keypoint metadata
        self.keypoint_names: List[str] = []  # 17 keypoint names
        self.skeleton: List[List[int]] = []  # Skeleton connections

        self._initialized = False

    def initialize(self, split: str = "val2017") -> None:
        """
        Initialize COCO APIs and load annotations.

        Args:
            split: Dataset split (train2017, val2017)
        """
        if self._initialized:
            self.logger.info("COCO data already initialized")
            return

        self.logger.info(f"Initializing COCO dataset for split: {split}")

        # Load caption annotations
        caption_ann_file = self.annotations_root / f"captions_{split}.json"
        if caption_ann_file.exists():
            if COCO:
                self.coco_caption = COCO(str(caption_ann_file))
                self.logger.info(f"Loaded caption annotations from {caption_ann_file}")

                # Load image data
                self.images_data = self.coco_caption.imgs
                # Load captions per image
                for ann in self.coco_caption.anns.values():
                    img_id = ann['image_id']
                    self.captions_data[img_id].append(ann)

                self.logger.info(f"Loaded {len(self.images_data)} images with captions")
            else:
                self.logger.warning("pycocotools not installed, loading from JSON directly")
                self._load_caption_json(caption_ann_file)

        # Load instance annotations (for detection)
        instance_ann_file = self.annotations_root / f"instances_{split}.json"
        if instance_ann_file.exists():
            if COCO:
                self.coco_instance = COCO(str(instance_ann_file))
                self.logger.info(f"Loaded instance annotations from {instance_ann_file}")

                # Load categories
                self.categories = {cat['id']: cat['name'] for cat in self.coco_instance.cats.values()}

                # Load instances per image
                for ann in self.coco_instance.anns.values():
                    img_id = ann['image_id']
                    self.instances_data[img_id].append(ann)

                self.logger.info(f"Loaded {len(self.categories)} object categories")
            else:
                self._load_instance_json(instance_ann_file)

        # Load VQA annotations if available
        vqa_ann_file = self.annotations_root / f"v2_mscoco_{split}_questions.json"
        if vqa_ann_file.exists():
            self._load_vqa_json(vqa_ann_file)
            self.logger.info(f"Loaded VQA questions from {vqa_ann_file}")

        # 🔧 新增：加载VQA标注答案（ground truth）
        # 尝试多种文件名格式
        possible_annotation_files = [
            self.annotations_root / f"v2_mscoco_{split}_annotations.json",
            self.annotations_root / f"v2_Annotations_{split.capitalize()}_mscoco" / f"v2_mscoco_{split}_annotations.json",
        ]

        # 对于train2014/val2014格式
        if 'train' in split:
            possible_annotation_files.append(
                self.annotations_root / "v2_mscoco_train2014_annotations.json"
            )
        elif 'val' in split:
            possible_annotation_files.append(
                self.annotations_root / "v2_mscoco_val2014_annotations.json"
            )

        for ann_file in possible_annotation_files:
            if ann_file.exists():
                self._load_vqa_annotations(ann_file)
                self.logger.info(f"Loaded VQA annotations from {ann_file}")
                break

        # Load person keypoints annotations
        keypoints_ann_file = self.annotations_root / f"person_keypoints_{split}.json"
        if keypoints_ann_file.exists():
            if COCO:
                self.coco_keypoints = COCO(str(keypoints_ann_file))
                self.logger.info(f"Loaded keypoints annotations from {keypoints_ann_file}")

                # Load keypoint names and skeleton from categories
                for cat in self.coco_keypoints.cats.values():
                    self.keypoint_names = cat.get('keypoints', [])
                    self.skeleton = cat.get('skeleton', [])

                # Load keypoints per image
                for ann in self.coco_keypoints.anns.values():
                    img_id = ann['image_id']
                    self.keypoints_data[img_id].append(ann)

                self.logger.info(f"Loaded {len(self.keypoint_names)} keypoint definitions")
                self.logger.info(f"Loaded keypoints for {len(self.keypoints_data)} images")
            else:
                self.logger.warning("pycocotools not installed, loading keypoints from JSON directly")
                self._load_keypoints_json(keypoints_ann_file)

        self._initialized = True
        self.logger.info("COCO dataset initialization complete")

        # 🔧 新增：构建GT映射缓存（包含置信度和分布）
        # 在加载VQA标注后立即构建，供硬标签生成使用
        if self.vqa_gt_with_confidence:
            try:
                cache_mode = self.config.get("cleaning.gt_mapping.cache_mode", "auto")
                cache_file = self.config.get("cleaning.gt_mapping.cache_file", "./data/gt_mapping_cache.json")

                self.logger.info("Building GT mapping cache with confidence...")
                gt_mapping = self.build_gt_mapping(
                    cache_mode=cache_mode,
                    cache_file=cache_file
                )
                self.logger.info(f"✓ GT mapping cache built: {len(gt_mapping)} entries")
            except Exception as e:
                self.logger.warning(f"Failed to build GT mapping cache: {e}")
                self.logger.warning("GT data will be used from memory")

    def _load_caption_json(self, ann_file: Path) -> None:
        """Load caption annotations from JSON directly (fallback)."""
        with open(ann_file, 'r') as f:
            data = json.load(f)

        self.images_data = {img['id']: img for img in data['images']}
        for ann in data['annotations']:
            self.captions_data[ann['image_id']].append(ann)

        self.logger.info(f"Loaded {len(self.images_data)} images with {len(data['annotations'])} captions")

    def _load_instance_json(self, ann_file: Path) -> None:
        """Load instance annotations from JSON directly (fallback)."""
        with open(ann_file, 'r') as f:
            data = json.load(f)

        if not self.images_data:
            self.images_data = {img['id']: img for img in data['images']}

        self.categories = {cat['id']: cat['name'] for cat in data['categories']}
        for ann in data['annotations']:
            self.instances_data[ann['image_id']].append(ann)

        self.logger.info(f"Loaded {len(self.categories)} categories with {len(data['annotations'])} instances")

    def _load_vqa_json(self, ann_file: Path) -> None:
        """Load VQA question annotations."""
        with open(ann_file, 'r') as f:
            data = json.load(f)

        for question in data['questions']:
            img_id = question['image_id']
            self.vqa_data_by_image[img_id].append(question)

        self.logger.info(f"Loaded {len(data['questions'])} VQA questions")

    def _load_vqa_annotations(self, ann_file: Path) -> None:
        """
        🔧 新增：加载VQA标注答案（ground truth）并计算置信度

        VQA标注文件格式：
        {
            "annotations": [
                {
                    "question_type": "what is this",
                    "multiple_choice_answer": "net",  # 投票最多的答案
                    "answers": [  # 10个标注员的答案
                        {"answer": "net", "answer_confidence": "maybe", "answer_id": 1},
                        ...
                    ],
                    "image_id": 458752,
                    "answer_type": "other",
                    "question_id": 458752000
                },
                ...
            ]
        }

        🔧 置信度计算方法：
        1. 过滤掉 answer_confidence="no" 的标注
        2. 统计剩余有效标注的答案分布
        3. gt_consistency = 最常见答案的频率
        4. gt_dist = 答案分布字典

        示例：
        - 原始10条标注，过滤掉2条"no"，剩余8条
        - 答案"3"出现6次 → 6/8 = 0.75
        - 答案"2"出现2次 → 2/8 = 0.25
        - 结果：{'answer': '3', 'gt_consistency': 0.75, 'gt_dist': {'3': 0.75, '2': 0.25}}
        """
        with open(ann_file, 'r') as f:
            data = json.load(f)

        count = 0
        for annotation in data.get('annotations', []):
            question_id = annotation.get('question_id')
            if question_id:
                # 存储原始标注
                self.vqa_answers_by_question[question_id] = annotation

                # 🔧 新增：计算置信度
                gt_info = self._compute_gt_confidence(annotation)
                self.vqa_gt_with_confidence[question_id] = gt_info

                count += 1

        self.logger.info(f"Loaded {count} VQA annotations with confidence scores")

    def _compute_gt_confidence(self, annotation: Dict) -> Dict:
        """
        🔧 新增：计算VQA标注的置信度

        Args:
            annotation: VQA标注数据（包含10个answers）

        Returns:
            {
                'answer': str,              # 最终答案
                'gt_consistency': float,    # 置信度
                'gt_dist': Dict[str, float] # 答案分布
            }
        """
        answers_list = annotation.get('answers', [])

        # ───────────────────────────────────────────────────────
        # 步骤1：过滤掉 answer_confidence="no" 的标注
        # ───────────────────────────────────────────────────────
        valid_answers = []
        for ans_data in answers_list:
            confidence = ans_data.get('answer_confidence', 'yes')
            answer = ans_data.get('answer', '').strip().lower()

            # 过滤掉 "no" 置信度的标注
            if confidence != 'no' and answer:
                valid_answers.append(answer)

        # 如果没有有效答案，使用原始的 multiple_choice_answer
        if not valid_answers:
            mc_answer = annotation.get('multiple_choice_answer', '').strip().lower()
            return {
                'answer': mc_answer,
                'gt_consistency': 1.0,  # 默认置信度
                'gt_dist': {mc_answer: 1.0}
            }

        # ───────────────────────────────────────────────────────
        # 步骤2：统计答案分布
        # ───────────────────────────────────────────────────────
        from collections import Counter
        answer_counts = Counter(valid_answers)
        total_count = len(valid_answers)

        # 计算频率分布
        gt_dist = {
            answer: count / total_count
            for answer, count in answer_counts.items()
        }

        # ───────────────────────────────────────────────────────
        # 步骤3：获取最常见答案（gt_consistency）
        # ───────────────────────────────────────────────────────
        most_common = answer_counts.most_common(1)[0]
        most_common_answer = most_common[0]
        gt_consistency = most_common[1] / total_count

        return {
            'answer': most_common_answer,
            'gt_consistency': round(gt_consistency, 4),  # 保留4位小数
            'gt_dist': {k: round(v, 4) for k, v in gt_dist.items()}  # 保留4位小数
        }

    def _load_keypoints_json(self, ann_file: Path) -> None:
        """Load keypoints annotations from JSON directly (fallback)."""
        with open(ann_file, 'r') as f:
            data = json.load(f)

        if not self.images_data:
            self.images_data = {img['id']: img for img in data['images']}

        # Load keypoint metadata from categories
        for cat in data['categories']:
            self.keypoint_names = cat.get('keypoints', [])
            self.skeleton = cat.get('skeleton', [])

        # Load keypoints annotations
        for ann in data['annotations']:
            self.keypoints_data[ann['image_id']].append(ann)

        self.logger.info(f"Loaded {len(self.keypoint_names)} keypoint definitions with {len(data['annotations'])} person keypoints")

    def get_image_path(self, image_id: int, split: str = "val2017") -> Optional[Path]:
        """
        Get image file path for given image ID.

        Args:
            image_id: COCO image ID
            split: Dataset split

        Returns:
            Path to image file or None if not found
        """
        if image_id not in self.images_data:
            self.logger.warning(f"Image ID {image_id} not found in dataset")
            return None

        img_info = self.images_data[image_id]
        file_name = img_info.get('file_name', f"COCO_{split}_{image_id:012d}.jpg")

        # Try multiple possible locations
        possible_paths = [
            self.images_root / split / file_name,
            self.images_root / file_name,
            self.coco_root / split / file_name,
        ]

        for path in possible_paths:
            if path.exists():
                return path

        self.logger.warning(f"Image file not found: {file_name}")
        return None

    def load_image(self, image_id: int, split: str = "val2017") -> Optional[Image.Image]:
        """
        Load PIL Image for given image ID.

        Args:
            image_id: COCO image ID
            split: Dataset split

        Returns:
            PIL Image or None if not found
        """
        image_path = self.get_image_path(image_id, split)
        if image_path is None:
            return None

        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            self.logger.error(f"Error loading image {image_id}: {e}")
            return None

    def get_vqa_questions(self, image_id: int) -> List[Dict]:
        """
        Get VQA questions for an image.

        Args:
            image_id: COCO image ID

        Returns:
            List of question dictionaries
        """
        return self.vqa_data_by_image.get(image_id, [])

    def get_vqa_ground_truth(self, question_id: int) -> Optional[Dict]:
        """
        🔧 新增：获取VQA问题的标注答案及置信度

        Args:
            question_id: VQA问题ID

        Returns:
            {
                'answer': str,              # 最终答案
                'gt_consistency': float,    # 置信度（过滤no后最常见答案的频率）
                'gt_dist': Dict[str, float] # 答案分布
            }
            如果不存在则返回None
        """
        return self.vqa_gt_with_confidence.get(question_id)

    def get_vqa_ground_truth_answer(self, question_id: int) -> Optional[str]:
        """
        🔧 兼容性方法：仅获取答案文本

        Args:
            question_id: VQA问题ID

        Returns:
            标注答案文本，如果不存在则返回None
        """
        gt_info = self.vqa_gt_with_confidence.get(question_id)
        if gt_info:
            return gt_info.get('answer')
        return None

    def get_vqa_annotation(self, question_id: int) -> Optional[Dict]:
        """
        🔧 新增：获取VQA问题的完整标注信息

        Args:
            question_id: VQA问题ID

        Returns:
            完整的标注信息（包含multiple_choice_answer和10个answers）
        """
        return self.vqa_answers_by_question.get(question_id)

    def _save_gt_mapping_to_cache(
        self,
        gt_mapping: Dict[Tuple[str, str], Dict],
        cache_path: str
    ) -> bool:
        """
        保存GT映射到缓存文件

        🔧 新格式：保存完整的GT信息（answer、gt_consistency、gt_dist）

        Args:
            gt_mapping: GT映射数据 {
                (image_id, question): {
                    'answer': str,
                    'gt_consistency': float,
                    'gt_dist': Dict[str, float]
                }
            }
            cache_path: 缓存文件路径

        Returns:
            是否保存成功
        """
        try:
            # 确保输出目录存在
            cache_file = Path(cache_path)
            cache_file.parent.mkdir(parents=True, exist_ok=True)

            # 构建保存数据（包含元数据）
            save_data = {
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'total_questions': len(gt_mapping),
                    'cache_version': '2.0',  # 新版本（包含置信度）
                    'description': 'GT mapping with confidence and distribution'
                },
                'mapping': {}
            }

            # 将tuple key转换为字符串key（JSON不支持tuple作为key）
            for (image_id, question), gt_info in gt_mapping.items():
                # 使用 "image_id||question" 作为key（||是分隔符，不太可能出现在问题中）
                key_str = f"{image_id}||{question}"
                save_data['mapping'][key_str] = gt_info

            # 保存为JSON格式
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)

            self.logger.info(f"✓ GT映射已保存到缓存文件: {cache_file}")
            self.logger.info(f"  - 映射条目数: {len(gt_mapping)}")
            self.logger.info(f"  - 缓存版本: 2.0 (包含置信度和分布)")
            return True

        except Exception as e:
            self.logger.warning(f"保存GT映射缓存失败: {e}")
            return False

    def _load_gt_mapping_from_cache(
        self,
        cache_path: str
    ) -> Optional[Dict[Tuple[str, str], Dict]]:
        """
        从缓存文件加载GT映射

        🔧 支持新旧两种格式：
        - 新格式（v2.0）：包含metadata和mapping，mapping包含完整GT信息
        - 旧格式（v1.0）：仅包含answer字符串

        Args:
            cache_path: 缓存文件路径

        Returns:
            GT映射数据，如果加载失败返回None
        """
        try:
            cache_file = Path(cache_path)

            # 检查缓存文件是否存在
            if not cache_file.exists():
                self.logger.info(f"GT映射缓存文件不存在: {cache_file}")
                return None

            # 加载JSON文件
            with open(cache_file, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)

            # 判断格式类型
            if isinstance(loaded_data, dict) and 'metadata' in loaded_data and 'mapping' in loaded_data:
                # 新格式（v2.0）：包含metadata和mapping
                metadata = loaded_data['metadata']
                serializable_mapping = loaded_data['mapping']

                self.logger.info(f"加载GT映射缓存（v{metadata.get('cache_version', '2.0')}）")
                self.logger.info(f"  - 创建时间: {metadata.get('created_at', 'N/A')}")
                self.logger.info(f"  - 条目数: {metadata.get('total_questions', 'N/A')}")

                # 将字符串key转换回tuple key
                gt_mapping = {}
                for key_str, gt_info in serializable_mapping.items():
                    # 解析 "image_id||question" 格式
                    if '||' in key_str:
                        parts = key_str.split('||', 1)  # 只分割第一个||
                        if len(parts) == 2:
                            image_id, question = parts
                            gt_mapping[(image_id, question)] = gt_info

                self.logger.info(f"✓ GT映射缓存加载成功: {len(gt_mapping)} 条")
                return gt_mapping

            else:
                # 旧格式（v1.0）：仅包含answer字符串
                self.logger.warning("检测到旧格式缓存文件（仅包含answer），将重新构建")
                return None

        except Exception as e:
            self.logger.warning(f"加载GT映射缓存失败: {e}")
            return None

    def build_gt_mapping(self, cache_mode: str = "auto", cache_file: str = None) -> Dict[Tuple[str, str], Dict]:
        """
        🔧 改进：构建GT真值映射（包含置信度信息），支持缓存机制

        通过遍历已加载的VQA问题和答案，构建：
        {
            (image_id, question): {
                'answer': str,              # 最终答案
                'gt_consistency': float,    # 置信度
                'gt_dist': Dict[str, float] # 答案分布
            }
        }

        Args:
            cache_mode: 缓存模式（枚举）
                - "auto": 优先使用缓存，缓存不存在才构建并保存（推荐）
                - "rebuild": 强制重新构建，更新缓存文件
                - "disabled": 禁用缓存，每次从COCO标注构建
            cache_file: 缓存文件路径（从配置文件读取）

        Returns:
            GT真值映射（包含置信度信息）
        """
        # ───────────────────────────────────────────────────────
        # 从配置读取缓存文件路径
        # ───────────────────────────────────────────────────────
        if not cache_file:
            cache_file = self.config.get('cleaning.gt_mapping.cache_file', './data/gt_mapping_cache.json')

        self.logger.info(f"【GT真值】构建GT真值映射...")
        self.logger.info(f"  - cache_mode: {cache_mode}")
        self.logger.info(f"  - cache_file: {cache_file}")

        # ───────────────────────────────────────────────────────
        # 根据缓存模式处理
        # ───────────────────────────────────────────────────────
        if cache_mode == "disabled":
            # 禁用缓存：直接从COCO构建
            self.logger.info("缓存已禁用，从COCO标注构建...")
            return self._build_gt_mapping_from_coco()

        elif cache_mode == "rebuild":
            # 强制重建：从COCO构建并保存到缓存
            self.logger.info("强制重建模式，从COCO标注构建并更新缓存...")
            gt_mapping = self._build_gt_mapping_from_coco()
            if gt_mapping:
                self._save_gt_mapping_to_cache(gt_mapping, cache_file)
            return gt_mapping

        else:  # cache_mode == "auto"
            # 自动模式：优先使用缓存
            cached_mapping = self._load_gt_mapping_from_cache(cache_file)
            if cached_mapping is not None:
                self.logger.info(f"✓ 使用缓存的GT映射，跳过构建")
                return cached_mapping

            # 缓存不存在：构建并保存
            self.logger.info("缓存不存在，从COCO标注构建...")
            gt_mapping = self._build_gt_mapping_from_coco()
            if gt_mapping:
                self._save_gt_mapping_to_cache(gt_mapping, cache_file)
            return gt_mapping

    def _build_gt_mapping_from_coco(self) -> Dict[Tuple[str, str], Dict]:
        """
        从COCO标注构建GT真值映射（内部方法）

        🔧 新方案：存储完整的GT信息（答案、置信度、分布）

        Returns:
            GT真值映射
            {
                (image_id, question): {
                    'answer': str,
                    'gt_consistency': float,
                    'gt_dist': Dict[str, float]
                }
            }
        """
        gt_mapping = {}

        # 遍历所有图片的VQA问题
        for image_id, questions in self.vqa_data_by_image.items():
            for question_data in questions:
                question_id = question_data.get('question_id')
                question_text = question_data.get('question', '').strip()

                if not question_id or not question_text:
                    continue

                # 🔧 新方案：获取完整的GT信息（包含置信度）
                gt_info = self.vqa_gt_with_confidence.get(question_id)
                if gt_info:
                    # key: (image_id, question)
                    gt_mapping[(str(image_id), question_text)] = gt_info

        self.logger.info(f"✓ 构建了 {len(gt_mapping)} 条GT真值映射（包含置信度）")
        return gt_mapping

    def get_image_ids(self, task: Optional[str] = None) -> List[int]:
        """
        Get list of all image IDs.

        Args:
            task: Filter by task ('caption', 'instance', 'keypoints', 'vqa'). If None, return all.

        Returns:
            List of image IDs
        """
        if task == 'caption':
            return list(self.captions_data.keys())
        elif task == 'instance':
            return list(self.instances_data.keys())
        elif task == 'keypoints':
            return list(self.keypoints_data.keys())
        elif task == 'vqa':
            return list(self.vqa_data_by_image.keys())
        else:
            return list(self.images_data.keys())

    def sample_images(
        self,
        num_samples: Optional[int] = None,
        task: Optional[str] = None,
        seed: Optional[int] = None
    ) -> List[int]:
        """
        Sample subset of image IDs.

        Args:
            num_samples: Number of samples. If None, use config max_samples.
            task: Filter by task.
            seed: Random seed for reproducibility.

        Returns:
            List of sampled image IDs
        """
        if seed is not None:
            random.seed(seed)

        image_ids = self.get_image_ids(task)
        num_samples = num_samples or self.config.get("data.max_samples")

        if num_samples is None or num_samples >= len(image_ids):
            return image_ids

        sampled_ids = random.sample(image_ids, num_samples)
        self.logger.info(f"Sampled {len(sampled_ids)} images from {len(image_ids)} total")

        return sampled_ids

    def get_annotation_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of loaded annotations.

        Returns:
            Dictionary with statistics
        """
        summary = {
            'total_images': len(self.images_data),
            'images_with_captions': len(self.captions_data),
            'images_with_instances': len(self.instances_data),
            'images_with_keypoints': len(self.keypoints_data),
            'images_with_vqa': len(self.vqa_data_by_image),
            'total_captions': sum(len(caps) for caps in self.captions_data.values()),
            'total_instances': sum(len(insts) for insts in self.instances_data.values()),
            'total_keypoints_persons': sum(len(kps) for kps in self.keypoints_data.values()),
            'total_vqa_questions': sum(len(qs) for qs in self.vqa_data_by_image.values()),
            'num_categories': len(self.categories),
            'categories': list(self.categories.values()),
            'keypoint_names': self.keypoint_names,
            'num_keypoints': len(self.keypoint_names),
            'skeleton_connections': len(self.skeleton),
        }
        return summary

    def __len__(self) -> int:
        """Return total number of images."""
        return len(self.images_data)

    def __repr__(self) -> str:
        """String representation."""
        return f"COCODataLoader(images={len(self.images_data)}, initialized={self._initialized})"