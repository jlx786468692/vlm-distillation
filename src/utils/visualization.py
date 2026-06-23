"""
Visualization Utilities
======================

Provides visualization tools for distillation results.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np

from PIL import Image

from ..utils.config import ConfigManager
from ..utils.logger import get_logger


def visualize_results(
    results_path: str,
    output_dir: Optional[str] = None,
    num_samples: int = 5
) -> Dict[str, Any]:
    """
    Visualize distillation results.

    Args:
        results_path: Path to results JSON file or directory
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize

    Returns:
        Summary of visualization
    """
    logger = get_logger()
    config = ConfigManager()

    output_dir = output_dir or str(Path(config.get("output.root_dir", "./outputs")) / "visualizations")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating visualizations in {output_path}")

    results = _load_results(results_path)

    if not results:
        logger.warning("No results to visualize")
        return {'status': 'no_results', 'visualizations_created': 0}

    # Create visualizations
    viz_count = 0

    for i, result in enumerate(results[:num_samples]):
        try:
            _visualize_single_result(result, output_path, i)
            viz_count += 1
        except Exception as e:
            logger.warning(f"Failed to visualize result {i}: {e}")

    logger.info(f"Created {viz_count} visualizations")

    return {
        'status': 'success',
        'visualizations_created': viz_count,
        'output_dir': str(output_path),
    }


def _load_results(results_path: str) -> List[Dict]:
    """Load results from file or directory."""
    path = Path(results_path)

    results = []

    if path.is_file():
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                results = data
            elif isinstance(data, dict):
                if 'images' in data:
                    results = data['images']
                else:
                    results = [data]

    elif path.is_dir():
        json_files = list(path.glob("*.json"))
        for json_file in json_files[:20]:  # Limit to 20
            with open(json_file, 'r', encoding='utf-8') as f:
                results.append(json.load(f))

    return results


def _visualize_single_result(
    result: Dict,
    output_path: Path,
    index: int
) -> None:
    """Create visualization for single result."""
    image_id = result.get('image_id', f'image_{index}')
    image_path = result.get('image_path', '')

    # Try to load image
    if image_path and Path(image_path).exists():
        image = Image.open(image_path).convert('RGB')
    else:
        # Create placeholder image
        image = Image.new('RGB', (400, 400), color='gray')

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title(f'Image ID: {image_id}')
    axes[0].axis('off')

    # Tasks results
    tasks = result.get('tasks', {})

    # VQA result
    if 'vqa' in tasks:
        vqa_data = tasks['vqa']
        hard_label = vqa_data.get('hard_label', {})
        question = hard_label.get('question', 'No question')
        answer = hard_label.get('answer', 'No answer')
        confidence = hard_label.get('confidence', 0)

        axes[1].text(
            0.5, 0.5,
            f"Question: {question}\n\nAnswer: {answer}\n\nConfidence: {confidence:.2f}",
            ha='center', va='center',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
        )
        axes[1].set_title('VQA Result')
        axes[1].axis('off')

    # Detection result
    if 'detection' in tasks:
        detection_data = tasks['detection']
        hard_label = detection_data.get('hard_label', {})
        objects = hard_label.get('objects', [])

        axes[2].imshow(image)
        axes[2].set_title('Detection Result')

        # Draw bounding boxes
        for obj in objects[:10]:  # Limit to 10 objects
            bbox = obj.get('bbox', [])
            if len(bbox) >= 4:
                # Assume [x, y, width, height] format
                x, y, w, h = bbox
                rect = patches.Rectangle(
                    (x, y), w, h,
                    linewidth=2,
                    edgecolor='red',
                    facecolor='none'
                )
                axes[2].add_patch(rect)

                # Add label
                label = obj.get('category_name', obj.get('class', 'object'))
                axes[2].text(
                    x, y - 10,
                    label,
                    color='red',
                    fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7)
                )

        axes[2].axis('off')

    plt.tight_layout()

    # Save visualization
    viz_file = output_path / f"visualization_{image_id}.png"
    plt.savefig(viz_file, dpi=100, bbox_inches='tight')
    plt.close()


def create_statistics_plots(
    statistics: Dict[str, Any],
    output_dir: str
) -> None:
    """
    Create plots for distillation statistics.

    Args:
        statistics: Statistics dictionary
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Task distribution
    if 'by_task' in statistics:
        task_counts = statistics['by_task']

        plt.figure(figsize=(10, 6))
        plt.bar(task_counts.keys(), [t['count'] for t in task_counts.values()])
        plt.title('Results by Task')
        plt.xlabel('Task')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(output_path / 'task_distribution.png')
        plt.close()

    # Processing time
    if 'by_task' in statistics:
        avg_times = {
            task: data.get('avg_processing_time', 0)
            for task, data in statistics['by_task'].items()
        }

        plt.figure(figsize=(10, 6))
        plt.bar(avg_times.keys(), avg_times.values())
        plt.title('Average Processing Time by Task')
        plt.xlabel('Task')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(output_path / 'processing_time.png')
        plt.close()


def create_comparison_visualization(
    teacher_result: Dict,
    student_result: Dict,
    output_path: str
) -> None:
    """
    Create comparison visualization between teacher and student.

    Args:
        teacher_result: Teacher model result
        student_result: Student model result
        output_path: Path to save visualization
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Teacher result
    axes[0].text(
        0.5, 0.5,
        json.dumps(teacher_result, indent=2)[:500],  # Limit text
        ha='center', va='center',
        fontsize=8,
        family='monospace'
    )
    axes[0].set_title('Teacher Model Output')
    axes[0].axis('off')

    # Student result
    axes[1].text(
        0.5, 0.5,
        json.dumps(student_result, indent=2)[:500],
        ha='center', va='center',
        fontsize=8,
        family='monospace'
    )
    axes[1].set_title('Student Model Output')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()


def __repr__() -> str:
    """Module representation."""
    return "Visualization Utilities for VLM Distillation"
