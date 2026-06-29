"""
JSON Exporter
=============

Handles exporting and merging distillation results in JSON format.
"""

import json
import shutil
from typing import Dict, Any, List, Optional
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import glob

from ..utils.config import ConfigManager
from ..utils.logger import get_logger

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, 'item'):
            return obj.item()
        if hasattr(obj, '__dict__'):
            return str(obj)
        return super().default(obj)

class JSONExporter:
    """
    Exports and manages distillation results in JSON format.

    Provides:
    - Individual result saving
    - Batch result saving
    - Result merging across tasks
    - Schema validation
    """

    def __init__(
        self,
        config: Optional[ConfigManager] = None
    ):
        """
        Initialize JSON Exporter.

        Args:
            config: Configuration manager
        """
        self.config = config or ConfigManager()
        self.logger = get_logger()

        # Output settings
        self.output_dir = Path(self.config.get("output.root_dir", "./outputs"))
        self.merge_outputs = self.config.get("output.merge_outputs", True)
        self.compress_outputs = self.config.get("output.compress_outputs", False)

        # Ensure output directories exist
        self._ensure_output_dirs()

    def _ensure_output_dirs(self) -> None:
        """Create output directories."""
        dirs = [
            self.output_dir,
            self.output_dir / "hard_labels",
            self.output_dir / "soft_labels",
            self.output_dir / "cot_reasoning",
            self.output_dir / "merged",
        ]

        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Output directories created: {self.output_dir}")

    def save_result(
        self,
        result: Dict[str, Any],
        output_path: str,
        validate: bool = True
    ) -> bool:
        """
        Save single result to JSON file.

        Args:
            result: Result dictionary
            output_path: Path to save
            validate: Whether to validate before saving

        Returns:
            True if successful
        """
        if validate:
            if not self._validate_result(result):
                self.logger.warning(f"Invalid result structure, skipping save to {output_path}")
                return False

        try:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Add timestamp if not present
            if 'timestamp' not in result:
                result['timestamp'] = datetime.now().isoformat()

            with open(path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, cls=CustomEncoder)

            self.logger.debug(f"Result saved to {path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save result: {e}")
            return False

    def save_batch(
        self,
        batch_results: Dict[str, Any],
        output_path: str
    ) -> bool:
        """
        Save batch results.

        Args:
            batch_results: Batch results dictionary
            output_path: Path to save

        Returns:
            True if successful
        """
        return self.save_result(batch_results, output_path, validate=False)

    def save_task_result(
        self,
        task_result: Dict[str, Any],
        output_path: str
    ) -> bool:
        """
        Save task-specific result.

        Args:
            task_result: Task result dictionary
            output_path: Path to save

        Returns:
            True if successful
        """
        return self.save_result(task_result, output_path)

    def merge_all_results(
        self,
        batch_files: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Merge all batch results into consolidated outputs.

        Args:
            batch_files: List of batch file paths. If None, finds all in output_dir.

        Returns:
            Summary of merged results
        """
        self.logger.info("Merging all batch results...")

        # Find all batch files
        if batch_files is None:
            batch_pattern = str(self.output_dir / "batch_*.json")
            batch_files = glob.glob(batch_pattern)

        if not batch_files:
            self.logger.warning("No batch files found to merge")
            return {'status': 'no_files', 'merged_count': 0}

        # Load all batch results
        all_results = []

        for batch_file in batch_files:
            try:
                with open(batch_file, 'r', encoding='utf-8') as f:
                    batch_data = json.load(f)

                # Extract image results
                if 'images' in batch_data:
                    all_results.extend(batch_data['images'])

            except Exception as e:
                self.logger.error(f"Failed to load batch file {batch_file}: {e}")

        self.logger.info(f"Loaded {len(all_results)} image results from {len(batch_files)} batch files")

        # Merge by image ID
        merged_by_image = {}

        for img_result in all_results:
            image_id = img_result.get('image_id')

            if image_id:
                if image_id in merged_by_image:
                    # Merge tasks
                    for task, task_data in img_result.get('tasks', {}).items():
                        if task in merged_by_image[image_id]['tasks']:
                            # Keep more recent/complete data
                            merged_by_image[image_id]['tasks'][task].update(task_data)
                        else:
                            merged_by_image[image_id]['tasks'][task] = task_data
                else:
                    merged_by_image[image_id] = img_result

        # Save merged results
        merged_count = 0

        for image_id, merged_data in merged_by_image.items():
            merged_file = self.output_dir / "merged" / f"{image_id}.json"

            if self.save_result(merged_data, str(merged_file), validate=False):
                merged_count += 1

        self.logger.info(f"Merged and saved {merged_count} image results")

        # Create merged summary
        summary_file = self.output_dir / "merged_summary.json"
        summary = {
            'total_images': merged_count,
            'batch_files_processed': len(batch_files),
            'tasks': self.config.get("distillation.tasks", []),
            'merge_timestamp': datetime.now().isoformat(),
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, cls=CustomEncoder)

        # Clean up batch files if successful
        if merged_count > 0:
            self._cleanup_batch_files(batch_files)

        return {
            'status': 'success',
            'merged_count': merged_count,
            'summary_path': str(summary_file),
        }

    def _cleanup_batch_files(
        self,
        batch_files: List[str]
    ) -> None:
        """
        Clean up temporary batch files after merging.

        Args:
            batch_files: List of batch file paths
        """
        archive_dir = self.output_dir / "archive"
        archive_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for batch_file in batch_files:
            try:
                batch_path = Path(batch_file)
                archived_path = archive_dir / f"{batch_path.stem}_{timestamp}.json"
                shutil.move(batch_file, str(archived_path))
            except Exception as e:
                self.logger.warning(f"Failed to archive {batch_file}: {e}")

        self.logger.info(f"Archived {len(batch_files)} batch files")

    def _validate_result(
        self,
        result: Dict[str, Any]
    ) -> bool:
        """
        Validate result structure.

        Args:
            result: Result dictionary

        Returns:
            True if valid
        """
        # Required top-level keys
        required_keys = ['image_id']

        for key in required_keys:
            if key not in result:
                self.logger.warning(f"Missing required key: {key}")
                return False

        # Validate tasks if present
        if 'tasks' in result:
            for task_name, task_data in result['tasks'].items():
                if not isinstance(task_data, dict):
                    self.logger.warning(f"Invalid task data structure for {task_name}")
                    return False

        return True

    def export_summary_report(
        self,
        statistics: Dict[str, Any]
    ) -> str:
        """
        Export summary report of distillation results.

        Args:
            statistics: Statistics dictionary

        Returns:
            Path to saved report
        """
        report_path = self.output_dir / "distillation_report.json"

        report = {
            'distillation_summary': statistics,
            'configuration': {
                'teacher_model': self.config.get("teacher.model_name"),
                'tasks': self.config.get("distillation.tasks"),
                'max_samples': self.config.get("data.max_samples"),
            },
            'output_info': {
                'output_dir': str(self.output_dir),
                'generated_timestamp': datetime.now().isoformat(),
            },
        }

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, cls=CustomEncoder)

        self.logger.info(f"Summary report saved to {report_path}")
        return str(report_path)

    def load_result(
        self,
        result_path: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load result from JSON file.

        Args:
            result_path: Path to result file

        Returns:
            Loaded result dictionary or None if failed
        """
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            return result
        except Exception as e:
            self.logger.error(f"Failed to load result from {result_path}: {e}")
            return None

    def get_output_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about generated outputs.

        Returns:
            Statistics dictionary
        """
        stats = {
            'total_files': 0,
            'by_directory': {},
        }

        dirs = ['hard_labels', 'soft_labels', 'cot_reasoning', 'merged']

        for dir_name in dirs:
            dir_path = self.output_dir / dir_name
            if dir_path.exists():
                json_files = list(dir_path.glob("*.json"))
                stats['by_directory'][dir_name] = len(json_files)
                stats['total_files'] += len(json_files)

        return stats

    def cleanup_outputs(
        self,
        keep_merged: bool = True
    ) -> None:
        """
        Clean up output directories.

        Args:
            keep_merged: Whether to keep merged results
        """
        dirs_to_clean = ['hard_labels', 'soft_labels', 'cot_reasoning']

        if not keep_merged:
            dirs_to_clean.append('merged')

        for dir_name in dirs_to_clean:
            dir_path = self.output_dir / dir_name
            if dir_path.exists():
                shutil.rmtree(dir_path)
                dir_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Cleaned up output directories: {dirs_to_clean}")

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_output_statistics()
        return f"JSONExporter(output_dir={self.output_dir}, total_files={stats['total_files']})"
