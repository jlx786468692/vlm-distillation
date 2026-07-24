"""
Data Manager
============

Manages data sampling, batching, and coordination for distillation pipeline.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator
from datetime import datetime

from ..utils.config import ConfigManager
from ..utils.logger import get_logger, DistillationLogger
from .coco_loader import COCODataLoader
from .image_processor import ImageProcessor


class DataManager:
    """
    Coordinates data loading, sampling, and batching for distillation.

    Provides:
    - Stratified sampling across tasks
    - Batch creation
    - Checkpoint management
    - Progress tracking
    """

    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        coco_loader: Optional[COCODataLoader] = None,
        image_processor: Optional[ImageProcessor] = None
    ):
        """
        Initialize DataManager.

        Args:
            config: Configuration manager
            coco_loader: COCO data loader instance
            image_processor: Image processor instance
        """
        self.config = config or ConfigManager()
        self.logger = get_logger()
        self.distill_logger = DistillationLogger("data_manager")

        # Initialize components
        self.coco_loader = coco_loader or COCODataLoader(self.config)
        self.image_processor = image_processor or ImageProcessor(self.config)

        # Sampling settings
        self.max_samples = self.config.get("data.max_samples")
        self.batch_size = self.config.get("data.batch_size", 8)
        self.sampling_strategy = self.config.get("data.sampling_strategy", "balanced")
        self.seed = self.config.get("data.seed", 42)

        # Task configuration
        self.tasks = self.config.get("distillation.tasks", ["vqa", "captioning", "detection"])

        # Processed data tracking
        self.processed_ids: List[int] = []
        self.current_batch_idx = 0
        self.checkpoint_path: Optional[Path] = None

        # Initialize dataset
        self._initialize_dataset()

    def _initialize_dataset(self) -> None:
        """Initialize COCO dataset."""
        split = self.config.get("data.val_split", "val2017")
        self.coco_loader.initialize(split)

        summary = self.coco_loader.get_annotation_summary()
        self.logger.info(f"Dataset initialized: {summary}")

    def get_sample_ids(
        self,
        task: Optional[str] = None,
        strategy: Optional[str] = None
    ) -> List[int]:
        """
        Get sampled image IDs based on strategy.

        Args:
            task: Filter by specific task
            strategy: Sampling strategy (random, balanced, stratified)

        Returns:
            List of sampled image IDs
        """
        strategy = strategy or self.sampling_strategy

        # Get base image IDs
        if task:
            base_ids = self.coco_loader.get_image_ids(task)
        else:
            # Get IDs that have all required annotations
            base_ids = self._get_multi_task_ids()

        # Apply sampling
        if self.max_samples and len(base_ids) > self.max_samples:
            sampled_ids = self._apply_sampling_strategy(base_ids, strategy, self.max_samples)
        else:
            sampled_ids = base_ids

        random.seed(self.seed)
        random.shuffle(sampled_ids)

        self.logger.info(f"Selected {len(sampled_ids)} images for processing")
        return sampled_ids

    def _get_multi_task_ids(self) -> List[int]:
        """
        Get image IDs that have annotations for at least one task.

        🔧 修复：改为宽松模式（并集），而非严格模式（交集）
        原因：用户可能缺少某些任务的标注文件，不应该阻止其他任务的执行

        Returns:
            List of image IDs with at least one task annotation
        """
        # Get IDs for each task
        vqa_ids = set(self.coco_loader.get_image_ids('vqa')) if 'vqa' in self.tasks else set()

        # 🔧 改进：使用宽松模式（并集），只要有任一任务的标注即可
        all_ids = set(self.coco_loader.get_image_ids())

        # 记录每个任务的标注数量
        self.logger.info("Dataset annotations summary:")
        if 'vqa' in self.tasks:
            self.logger.info(f"  VQA: {len(vqa_ids)} images")

        # 收集所有有标注的任务集合
        task_sets = []
        if 'vqa' in self.tasks and vqa_ids:
            task_sets.append(vqa_ids)

        # 🔧 宽松模式：并集（只要有任一任务的标注）
        valid_ids = set()
        for task_set in task_sets:
            valid_ids.update(task_set)

        # 只保留在all_ids中的图片
        valid_ids = valid_ids.intersection(all_ids)

        self.logger.info(
            f"✓ Found {len(valid_ids)} images with at least one task annotation"
        )

        # 如果没有找到任何图片，给出警告
        if not valid_ids:
            self.logger.error("⚠ No images found with any task annotations!")
            self.logger.error("Please check:")
            self.logger.error("  1. Annotation files exist in: " + str(self.annotations_root))
            self.logger.error("  2. Annotation files match the split: " + self.config.get("data.val_split", "val2017"))
            self.logger.error("  3. Tasks are correctly configured: " + str(self.tasks))

        return list(valid_ids)

    def _apply_sampling_strategy(
        self,
        ids: List[int],
        strategy: str,
        num_samples: int
    ) -> List[int]:
        """
        Apply sampling strategy to select subset.

        Args:
            ids: List of image IDs
            strategy: Sampling strategy
            num_samples: Number to sample

        Returns:
            Sampled IDs
        """
        random.seed(self.seed)

        if strategy == "random":
            sampled = random.sample(ids, num_samples)

        elif strategy == "balanced":
            # Balance across tasks
            sampled = self._balanced_sampling(ids, num_samples)

        elif strategy == "stratified":
            # Stratify by image categories or characteristics
            sampled = self._stratified_sampling(ids, num_samples)

        else:
            self.logger.warning(f"Unknown sampling strategy '{strategy}', using random")
            sampled = random.sample(ids, num_samples)

        return sampled

    def _balanced_sampling(self, ids: List[int], num_samples: int) -> List[int]:
        """
        Balance sampling across different tasks.

        Args:
            ids: List of image IDs
            num_samples: Number to sample

        Returns:
            Balanced sampled IDs
        """
        # Count images per task
        task_counts = {}
        for task in self.tasks:
            task_ids = self.coco_loader.get_image_ids(task)
            task_counts[task] = len([id for id in ids if id in task_ids])

        # Sample proportionally
        samples_per_task = num_samples // len(self.tasks)
        sampled_ids = []

        for task in self.tasks:
            task_ids = [id for id in ids if id in self.coco_loader.get_image_ids(task)]
            n_samples = min(samples_per_task, len(task_ids))
            sampled_ids.extend(random.sample(task_ids, n_samples))

        # Fill remaining if needed
        remaining = num_samples - len(sampled_ids)
        if remaining > 0:
            remaining_ids = [id for id in ids if id not in sampled_ids]
            sampled_ids.extend(random.sample(remaining_ids, min(remaining, len(remaining_ids))))

        return sampled_ids[:num_samples]

    def _stratified_sampling(self, ids: List[int], num_samples: int) -> List[int]:
        """
        Stratified sampling based on image characteristics.

        Args:
            ids: List of image IDs
            num_samples: Number to sample

        Returns:
            Stratified sampled IDs
        """
        # Group by image size categories
        size_groups = self._group_by_size(ids)

        # Sample from each group proportionally
        sampled_ids = []
        total = len(ids)

        for group_name, group_ids in size_groups.items():
            group_size = len(group_ids)
            proportion = group_size / total
            n_samples = int(num_samples * proportion)

            if n_samples > 0 and group_ids:
                sampled_ids.extend(random.sample(group_ids, min(n_samples, len(group_ids))))

        # Fill if under target
        if len(sampled_ids) < num_samples:
            remaining_ids = [id for id in ids if id not in sampled_ids]
            sampled_ids.extend(random.sample(remaining_ids, num_samples - len(sampled_ids)))

        return sampled_ids[:num_samples]

    def _group_by_size(self, ids: List[int]) -> Dict[str, List[int]]:
        """
        Group images by size categories.

        Args:
            ids: List of image IDs

        Returns:
            Dictionary of size groups
        """
        groups = {
            'small': [],    # < 500x500
            'medium': [],   # 500x500 to 1000x1000
            'large': [],    # > 1000x1000
        }

        for img_id in ids:
            if img_id in self.coco_loader.images_data:
                img_data = self.coco_loader.images_data[img_id]
                width = img_data.get('width', 0)
                height = img_data.get('height', 0)
                pixels = width * height

                if pixels < 250000:  # 500x500
                    groups['small'].append(img_id)
                elif pixels < 1000000:  # 1000x1000
                    groups['medium'].append(img_id)
                else:
                    groups['large'].append(img_id)

        return groups

    def create_batches(
        self,
        image_ids: List[int],
        batch_size: Optional[int] = None
    ) -> Iterator[List[int]]:
        """
        Create batches of image IDs.

        Args:
            image_ids: List of image IDs
            batch_size: Batch size (default: from config)

        Returns:
            Iterator of batch ID lists
        """
        batch_size = batch_size or self.batch_size

        for i in range(0, len(image_ids), batch_size):
            batch_ids = image_ids[i:i + batch_size]
            yield batch_ids

    def get_batch_data(
        self,
        batch_ids: List[int],
        task: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get complete data for a batch of images.

        Args:
            batch_ids: List of image IDs
            task: Specific task to load data for

        Returns:
            Dictionary with batch data
        """
        batch_data = {
            'image_ids': batch_ids,
            'images': [],
            'annotations': {},
            'metadata': {
                'batch_size': len(batch_ids),
                'timestamp': datetime.now().isoformat(),
            }
        }

        # Initialize task annotations
        for t in self.tasks:
            batch_data['annotations'][t] = {}

        for img_id in batch_ids:
            # Load image
            image = self.coco_loader.load_image(img_id)
            if image is None:
                self.logger.warning(f"Skipping image {img_id} - could not load")
                continue

            batch_data['images'].append({
                'id': img_id,
                'image': image,
                'path': str(self.coco_loader.get_image_path(img_id)),
            })

            # Load annotations for each task
            if 'vqa' in self.tasks:
                vqa_questions = self.coco_loader.get_vqa_questions(img_id)
                batch_data['annotations']['vqa'][img_id] = vqa_questions

        return batch_data

    def save_checkpoint(
        self,
        processed_ids: List[int],
        checkpoint_path: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Save processing checkpoint.

        Args:
            processed_ids: List of processed image IDs
            checkpoint_path: Path to save checkpoint
            metadata: Additional metadata
        """
        checkpoint_dir = Path(self.config.get("output.root_dir", "./outputs"))
        checkpoint_path = checkpoint_path or str(checkpoint_dir / "checkpoint_latest.json")

        checkpoint = {
            'processed_ids': processed_ids,
            'timestamp': datetime.now().isoformat(),
            'total_processed': len(processed_ids),
            'config': {
                'max_samples': self.max_samples,
                'batch_size': self.batch_size,
                'tasks': self.tasks,
            },
        }

        if metadata:
            checkpoint['metadata'] = metadata

        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2)

        self.logger.info(f"Checkpoint saved: {checkpoint_path} ({len(processed_ids)} images)")

    def load_checkpoint(
        self,
        checkpoint_path: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load processing checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Checkpoint dictionary or None if not found
        """
        path = Path(checkpoint_path)

        if not path.exists():
            self.logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return None

        with open(path, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)

        self.logger.info(f"Checkpoint loaded: {len(checkpoint['processed_ids'])} images processed")
        return checkpoint

    def get_remaining_ids(
        self,
        all_ids: List[int],
        checkpoint_path: Optional[str] = None
    ) -> List[int]:
        """
        Get remaining image IDs to process (after checkpoint).

        Args:
            all_ids: All image IDs to process
            checkpoint_path: Path to checkpoint file

        Returns:
            List of remaining image IDs
        """
        if checkpoint_path is None:
            return all_ids

        checkpoint = self.load_checkpoint(checkpoint_path)
        if checkpoint is None:
            return all_ids

        processed_ids = set(checkpoint['processed_ids'])
        remaining_ids = [id for id in all_ids if id not in processed_ids]

        self.logger.info(f"Remaining images: {len(remaining_ids)} (total: {len(all_ids)})")
        return remaining_ids

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get data statistics and summary.

        Returns:
            Dictionary with statistics
        """
        coco_summary = self.coco_loader.get_annotation_summary()

        stats = {
            'dataset': coco_summary,
            'sampling': {
                'strategy': self.sampling_strategy,
                'max_samples': self.max_samples,
                'seed': self.seed,
            },
            'processing': {
                'batch_size': self.batch_size,
                'tasks': self.tasks,
                'processed_count': len(self.processed_ids),
            },
        }

        return stats

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DataManager(batch_size={self.batch_size}, "
            f"max_samples={self.max_samples}, "
            f"tasks={self.tasks})"
        )