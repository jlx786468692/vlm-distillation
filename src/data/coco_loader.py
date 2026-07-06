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

    def get_captions(self, image_id: int) -> List[str]:
        """
        Get all captions for an image.

        Args:
            image_id: COCO image ID

        Returns:
            List of caption strings
        """
        captions = []
        for ann in self.captions_data.get(image_id, []):
            caption = ann.get('caption', '')
            if caption:
                captions.append(caption)
        return captions

    def get_instances(self, image_id: int) -> List[Dict]:
        """
        Get all object instances for an image.

        Args:
            image_id: COCO image ID

        Returns:
            List of instance dictionaries with bbox and category
        """
        instances = []
        for ann in self.instances_data.get(image_id, []):
            instance = {
                'bbox': ann.get('bbox', []),  # [x, y, width, height]
                'category_id': ann.get('category_id'),
                'category_name': self.categories.get(ann.get('category_id'), 'unknown'),
                'area': ann.get('area', 0),
                'is_crowd': ann.get('iscrowd', 0),
                'segmentation': ann.get('segmentation', None),
            }
            instances.append(instance)
        return instances

    def get_vqa_questions(self, image_id: int) -> List[Dict]:
        """
        Get VQA questions for an image.

        Args:
            image_id: COCO image ID

        Returns:
            List of question dictionaries
        """
        return self.vqa_data_by_image.get(image_id, [])

    def get_keypoints(self, image_id: int) -> List[Dict]:
        """
        Get all person keypoints for an image.

        Args:
            image_id: COCO image ID

        Returns:
            List of person keypoints dictionaries with:
            - bbox: person bounding box [x, y, width, height]
            - keypoints: list of {name, x, y, visibility} dicts (17 keypoints)
            - num_keypoints: count of visible keypoints
            - skeleton: skeleton connections for visualization
        """
        keypoints_list = []
        for ann in self.keypoints_data.get(image_id, []):
            # Parse flat keypoints array [x1,y1,v1, x2,y2,v2, ...]
            kp_flat = ann.get('keypoints', [])
            keypoints_parsed = []

            for i, name in enumerate(self.keypoint_names):
                x = kp_flat[i * 3] if i * 3 < len(kp_flat) else 0
                y = kp_flat[i * 3 + 1] if i * 3 + 1 < len(kp_flat) else 0
                v = kp_flat[i * 3 + 2] if i * 3 + 2 < len(kp_flat) else 0
                keypoints_parsed.append({
                    'name': name,
                    'x': float(x),
                    'y': float(y),
                    'visibility': int(v)  # 0=not labeled, 1=labeled not visible, 2=visible
                })

            keypoints_list.append({
                'bbox': ann.get('bbox', []),  # [x, y, width, height]
                'keypoints': keypoints_parsed,
                'num_keypoints': ann.get('num_keypoints', 0),
                'category_id': ann.get('category_id'),
                'area': ann.get('area', 0),
                'is_crowd': ann.get('iscrowd', 0),
                'skeleton': self.skeleton,
            })

        return keypoints_list

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