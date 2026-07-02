"""
Main Distiller
==============

Orchestrates the complete distillation pipeline.
"""

import json
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

from ..models.teacher_model import TeacherModel
from ..data.data_manager import DataManager
from ..utils.config import ConfigManager
from ..utils.logger import get_logger, DistillationLogger
from ..export.json_exporter import JSONExporter

from .hard_label_gen import HardLabelGenerator
from .soft_label_gen import SoftLabelGenerator
from .cot_generator import CoTGenerator


class Distiller:
    """
    Main distillation orchestrator that coordinates all components.

    Manages the complete pipeline:
    - Data loading and sampling
    - Hard label generation
    - Soft label generation
    - Chain-of-Thought generation
    - Output merging and export
    """

    def __init__(
        self,
        teacher_model: Optional[TeacherModel] = None,
        config: Optional[ConfigManager] = None,
        data_manager: Optional[DataManager] = None
    ):
        """
        Initialize Distiller.

        Args:
            teacher_model: Teacher model instance (auto-created if None)
            config: Configuration manager
            data_manager: Data manager instance (auto-created if None)
        """
        self.config = config or ConfigManager()
        self.logger = get_logger()
        self.distill_logger = DistillationLogger("distiller")

        # Initialize components
        self.teacher = teacher_model or TeacherModel(self.config)
        self.data_manager = data_manager or DataManager(self.config)

        # Initialize generators
        self.hard_label_gen = HardLabelGenerator(self.teacher, self.config)
        self.soft_label_gen = SoftLabelGenerator(self.teacher, self.config)
        self.cot_gen = CoTGenerator(self.teacher, self.config)

        # Initialize exporter
        self.exporter = JSONExporter(self.config)

        # Distillation settings
        self.tasks = self.config.get("distillation.tasks", ["vqa", "captioning", "detection"])
        self.enable_hard_labels = self.config.get("distillation.hard_labels.enabled", True)
        self.enable_soft_labels = self.config.get("distillation.soft_labels.enabled", True)
        self.enable_cot = self.config.get("distillation.cot.enabled", True)

        # Checkpoint settings
        self.checkpoint_interval = self.config.get("distillation.checkpoint_interval", 100)
        self.output_dir = Path(self.config.get("output.root_dir", "./outputs"))

        # Processing stats
        self.stats = {
            'total_images': 0,
            'processed_images': 0,
            'failed_images': 0,
            'start_time': None,
            'end_time': None,
        }

    def run_distillation(
        self,
        max_samples: Optional[int] = None,
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run complete distillation pipeline.

        Args:
            max_samples: Maximum number of images to process (None = all)
            checkpoint_path: Path to resume from checkpoint

        Returns:
            Distillation results summary
        """
        self.logger.info("Starting distillation pipeline...")
        self.distill_logger.start_process(
            total_count=self.stats['total_images'],
            description="VLM Data Distillation"
        )

        self.stats['start_time'] = datetime.now()

        # Get sample IDs
        sample_ids = self.data_manager.get_sample_ids()
        if max_samples:
            sample_ids = sample_ids[:max_samples]

        self.stats['total_images'] = len(sample_ids)

        # Resume from checkpoint if provided
        if checkpoint_path:
            remaining_ids = self.data_manager.get_remaining_ids(sample_ids, checkpoint_path)
            sample_ids = remaining_ids
            self.logger.info(f"Resuming from checkpoint: {len(sample_ids)} images remaining")

        # Create batches
        batches = self.data_manager.create_batches(sample_ids)

        # Process each batch
        processed_ids = []
        all_results = []

        for batch_idx, batch_ids in enumerate(batches):
            self.logger.info(f"\nProcessing batch {batch_idx + 1}: {len(batch_ids)} images")

            try:
                # Get batch data
                batch_data = self.data_manager.get_batch_data(batch_ids)

                # Process batch
                batch_results = self.process_batch(batch_data)

                # Save results
                self._save_batch_results(batch_results, batch_idx)

                # Update tracking
                processed_ids.extend(batch_ids)
                all_results.append(batch_results)

                # Log progress
                self.distill_logger.log_progress(
                    current=len(processed_ids),
                    message=f"Batch {batch_idx + 1} completed"
                )

                # Save checkpoint
                if len(processed_ids) % self.checkpoint_interval == 0:
                    self._save_checkpoint(processed_ids)

                self.stats['processed_images'] += len(batch_ids)

            except Exception as e:
                self.logger.error(f"Error processing batch {batch_idx}: {e}")
                self.stats['failed_images'] += len(batch_ids)
                self.distill_logger.log_error(e, context=f"Batch {batch_idx}")

        # Final checkpoint
        self._save_checkpoint(processed_ids)

        # Merge all results
        self.logger.info("\nMerging all results...")
        merged_results = self.exporter.merge_all_results()

        # Compute final statistics
        self.stats['end_time'] = datetime.now()
        final_stats = self._compute_final_statistics(all_results)

        self.distill_logger.end_process("Distillation completed")

        return {
            'statistics': final_stats,
            'processed_count': len(processed_ids),
            'failed_count': self.stats['failed_images'],
            'merged_data_path': str(self.output_dir / "merged"),
        }

    def process_batch(
        self,
        batch_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process single batch through all generators.

        Args:
            batch_data: Batch data dictionary

        Returns:
            Complete batch results
        """
        batch_results = {
            'batch_id': batch_data['metadata']['timestamp'],
            'images': [],
        }

        for img_data in batch_data['images']:
            image_id = img_data['id']
            image_path = img_data['path']

            image_result = {
                'image_id': image_id,
                'image_path': image_path,
                'tasks': {},
            }

            self.logger.info(f"Processing image {image_id}")

            # Generate for each task
            for task in self.tasks:
                task_result = {}

                # Hard labels
                if self.enable_hard_labels:
                    start_time = time.time()
                    hard_labels = self._generate_task_hard_labels(
                        task, batch_data, image_id, image_path
                    )
                    task_result['hard_label'] = hard_labels
                    task_result['hard_label_time'] = time.time() - start_time

                # Soft labels
                if self.enable_soft_labels:
                    start_time = time.time()
                    soft_labels = self._generate_task_soft_labels(
                        task, batch_data, image_id, image_path
                    )
                    task_result['soft_label'] = soft_labels
                    task_result['soft_label_time'] = time.time() - start_time

                # Chain-of-Thought
                if self.enable_cot:
                    start_time = time.time()
                    cot = self._generate_task_cot(
                        task, batch_data, image_id, image_path
                    )
                    task_result['cot_reasoning'] = cot
                    task_result['cot_time'] = time.time() - start_time

                image_result['tasks'][task] = task_result

            # Add metadata
            image_result['metadata'] = {
                'teacher_model': self.teacher.model_name,
                'processing_timestamp': datetime.now().isoformat(),
                'total_time': sum(
                    t.get('hard_label_time', 0) +
                    t.get('soft_label_time', 0) +
                    t.get('cot_time', 0)
                    for t in image_result['tasks'].values()
                ),
            }

            batch_results['images'].append(image_result)

        return batch_results

    def _generate_task_hard_labels(
        self,
        task: str,
        batch_data: Dict,
        image_id: int,
        image_path: str
    ) -> Dict[str, Any]:
        """Generate hard labels for specific task."""
        if task == 'vqa':
            questions = batch_data['annotations']['vqa'].get(image_id, [])
            if questions:
                question = questions[0].get('question', '')
                return self.hard_label_gen.generate_vqa_hard_labels(
                    image_path=image_path,
                    question=question,
                    image_id=str(image_id)
                )
            return {}

        elif task == 'captioning':
            return self.hard_label_gen.generate_captioning_hard_labels(
                image_path=image_path,
                num_captions=3,
                image_id=str(image_id)
            )

        elif task == 'detection':
            return self.hard_label_gen.generate_detection_hard_labels(
                image_path=image_path,
                image_id=str(image_id)
            )

        elif task == 'keypoints':
            return self.hard_label_gen.generate_keypoints_hard_labels(
                image_path=image_path,
                image_id=str(image_id)
            )

        return {}

    def _generate_task_soft_labels(
        self,
        task: str,
        batch_data: Dict,
        image_id: int,
        image_path: str
    ) -> Dict[str, Any]:
        """Generate soft labels for specific task."""
        if task == 'vqa':
            questions = batch_data['annotations']['vqa'].get(image_id, [])
            if questions:
                question = questions[0].get('question', '')
                return self.soft_label_gen.generate_vqa_soft_labels(
                    image_path=image_path,
                    question=question,
                    image_id=str(image_id)
                )
            return {}

        elif task == 'captioning':
            return self.soft_label_gen.generate_captioning_soft_labels(
                image_path=image_path,
                num_captions=3,
                image_id=str(image_id)
            )

        elif task == 'detection':
            return self.soft_label_gen.generate_detection_soft_labels(
                image_path=image_path,
                image_id=str(image_id)
            )

        elif task == 'keypoints':
            return self.soft_label_gen.generate_keypoints_soft_labels(
                image_path=image_path,
                image_id=str(image_id)
            )

        return {}

    def _generate_task_cot(
        self,
        task: str,
        batch_data: Dict,
        image_id: int,
        image_path: str
    ) -> Dict[str, Any]:
        """Generate CoT for specific task."""
        if task == 'vqa':
            questions = batch_data['annotations']['vqa'].get(image_id, [])
            if questions:
                question = questions[0].get('question', '')
                return self.cot_gen.generate_vqa_cot(
                    image_path=image_path,
                    question=question,
                    image_id=str(image_id)
                )
            return {}

        elif task == 'captioning':
            return self.cot_gen.generate_captioning_cot(
                image_path=image_path,
                image_id=str(image_id)
            )

        elif task == 'detection':
            return self.cot_gen.generate_detection_cot(
                image_path=image_path,
                image_id=str(image_id)
            )

        elif task == 'keypoints':
            return self.cot_gen.generate_keypoints_cot(
                image_path=image_path,
                image_id=str(image_id)
            )

        return {}

    def _save_batch_results(
        self,
        batch_results: Dict,
        batch_idx: int
    ) -> None:
        """Save batch results to separate files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_file = self.output_dir / f"batch_{batch_idx}_{timestamp}.json"

        self.exporter.save_batch(batch_results, str(batch_file))

        # Also save to task-specific directories if enabled
        if self.config.get("output.merge_outputs", True):
            # Will be merged later
            pass
        else:
            # Save separately
            for img_result in batch_results['images']:
                for task, task_result in img_result['tasks'].items():
                    task_file = self.output_dir / task / f"{img_result['image_id']}.json"

                    task_data = {
                        'image_id': img_result['image_id'],
                        'image_path': img_result['image_path'],
                        'task': task,
                        'data': task_result,
                        'metadata': img_result['metadata'],
                    }

                    self.exporter.save_task_result(task_data, str(task_file))

    def _save_checkpoint(
        self,
        processed_ids: List[int]
    ) -> None:
        """Save processing checkpoint."""
        checkpoint_data = {
            'processed_ids': processed_ids,
            'stats': {
                'total_images': self.stats['total_images'],
                'processed_images': self.stats['processed_images'],
                'failed_images': self.stats['failed_images'],
                'start_time': self.stats['start_time'].isoformat() if self.stats['start_time'] else None,
                'end_time': self.stats['end_time'].isoformat() if self.stats['end_time'] else None,
            },
            'timestamp': datetime.now().isoformat(),
        }

        checkpoint_file = self.output_dir / "checkpoint_latest.json"

        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        self.logger.info(f"Checkpoint saved: {len(processed_ids)} images processed")

    def _compute_final_statistics(
        self,
        all_results: List[Dict]
    ) -> Dict[str, Any]:
        """Compute final statistics from all results."""
        stats = {
            'total_images': self.stats['processed_images'],
            'failed_images': self.stats['failed_images'],
            'total_batches': len(all_results),
            'processing_time': str(self.stats['end_time'] - self.stats['start_time']),
            'by_task': {},
        }

        # Aggregate by task
        for task in self.tasks:
            task_stats = {
                'count': 0,
                'hard_labels_count': 0,
                'soft_labels_count': 0,
                'cot_count': 0,
                'avg_processing_time': 0,
            }

            times = []

            for batch_result in all_results:
                for img_result in batch_result['images']:
                    if task in img_result['tasks']:
                        task_stats['count'] += 1

                        task_data = img_result['tasks'][task]

                        if 'hard_label' in task_data:
                            task_stats['hard_labels_count'] += 1

                        if 'soft_label' in task_data:
                            task_stats['soft_labels_count'] += 1

                        if 'cot_reasoning' in task_data:
                            task_stats['cot_count'] += 1

                        if 'metadata' in img_result:
                            times.append(img_result['metadata'].get('total_time', 0))

            if times:
                task_stats['avg_processing_time'] = sum(times) / len(times)

            stats['by_task'][task] = task_stats

        return stats

    def validate_results(
        self,
        results_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate generated distillation results.

        Args:
            results_path: Path to results directory

        Returns:
            Validation report
        """
        results_dir = Path(results_path or self.output_dir)

        validation_report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {},
        }

        # Check for required outputs
        required_dirs = ['merged']
        if not self.config.get("output.merge_outputs", True):
            required_dirs = ['hard_labels', 'soft_labels', 'cot_reasoning']

        for dir_name in required_dirs:
            dir_path = results_dir / dir_name
            if not dir_path.exists():
                validation_report['valid'] = False
                validation_report['errors'].append(f"Missing output directory: {dir_name}")

        # Validate JSON files
        merged_dir = results_dir / "merged"
        if merged_dir.exists():
            json_files = list(merged_dir.glob("*.json"))

            for json_file in json_files[:10]:  # Check first 10 files
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)

                    # Check required keys
                    if 'image_id' not in data:
                        validation_report['warnings'].append(f"Missing image_id in {json_file.name}")

                    if 'tasks' not in data:
                        validation_report['warnings'].append(f"Missing tasks in {json_file.name}")

                except Exception as e:
                    validation_report['errors'].append(f"Invalid JSON in {json_file.name}: {e}")
                    validation_report['valid'] = False

        validation_report['statistics']['total_files'] = len(list(merged_dir.glob("*.json")))

        return validation_report

    def get_processing_status(self) -> Dict[str, Any]:
        """
        Get current processing status.

        Returns:
            Status dictionary
        """
        elapsed = None
        if self.stats['start_time']:
            elapsed = str(datetime.now() - self.stats['start_time'])

        return {
            'total_images': self.stats['total_images'],
            'processed_images': self.stats['processed_images'],
            'failed_images': self.stats['failed_images'],
            'progress_percent': (self.stats['processed_images'] / self.stats['total_images'] * 100) if self.stats['total_images'] > 0 else 0,
            'elapsed_time': elapsed,
            'tasks': self.tasks,
            'components': {
                'hard_labels': self.enable_hard_labels,
                'soft_labels': self.enable_soft_labels,
                'cot': self.enable_cot,
            },
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Distiller(teacher={self.teacher.model_name}, "
            f"tasks={self.tasks}, "
            f"hard_labels={self.enable_hard_labels}, "
            f"soft_labels={self.enable_soft_labels}, "
            f"cot={self.enable_cot})"
        )