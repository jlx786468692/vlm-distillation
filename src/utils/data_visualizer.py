"""
Data Visualizer
==============

Core visualization functions:
1. Score comparison (quality + confidence)
2. Sample visualization
3. Pipeline timing with distillation breakdown

Usage:
    visualizer = DataVisualizer(config, logger)
    visualizer.visualize_all(data_list, output_dir, timing_stats)
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import random

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.patches import Rectangle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

import numpy as np


class DataVisualizer:
    """Data Visualizer - Pipeline timing with distillation breakdown"""

    def __init__(self, config: Any, logger: Any = None):
        self.config = config
        self.logger = logger

        self.viz_config = config.get('visualization', {})
        self.output_dir = self.viz_config.get('output_dir', './outputs/visualizations')
        self.plot_format = self.viz_config.get('plot_format', 'png')
        self.dpi = self.viz_config.get('dpi', 150)
        self.default_figsize = (14, 12)

        self.colors = {
            'before': '#FF6B6B',
            'after': '#2ECC71',
            'threshold': '#F39C12',
        }

        self.group_colors = {
            'distillation': '#3498DB',
            'validation': '#9B59B6',
            'cleaning': '#E74C3C',
            'visualization': '#F39C12',
        }

    def _log(self, message: str, level: str = 'info'):
        if self.logger:
            if level == 'info':
                self.logger.info(message)
            elif level == 'warning':
                self.logger.warning(message)

    def visualize_all(
        self,
        data_list: List[Dict],
        output_dir: Optional[str] = None,
        before_data: Optional[List[Dict]] = None,
        timing_stats: Optional[Dict[str, Any]] = None,
        pipeline_results: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Run all visualizations"""
        if not HAS_MATPLOTLIB:
            self._log("matplotlib not installed", 'warning')
            return {'success': False, 'error': 'matplotlib not installed'}

        if output_dir:
            self.output_dir = output_dir
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self._log(f"\nGenerating visualizations...")
        self._log(f"  Output: {self.output_dir}")
        self._log(f"  Samples: {len(data_list)}")

        start_time = datetime.now()
        generated_plots = []

        try:
            # 1. Score comparison
            if before_data and data_list:
                self._log("\n[1/3] Score comparison...")
                path = self._plot_score_comparison(before_data, data_list, output_path)
                if path:
                    generated_plots.append(('score_comparison', path))
                    self._log(f"  OK: score_comparison.{self.plot_format}")

            # 2. Sample visualization
            if data_list:
                self._log("\n[2/3] Sample visualization...")
                paths = self._visualize_samples(data_list, output_path)
                generated_plots.extend([('sample', p) for p in paths])
                self._log(f"  OK: {len(paths)} samples")

            # 3. Pipeline timing (with distillation breakdown)
            if timing_stats:
                self._log("\n[3/3] Pipeline timing...")
                path = self._plot_pipeline_timing(timing_stats, output_path)
                if path:
                    generated_plots.append(('pipeline_timing', path))
                    self._log(f"  OK: pipeline_timing.{self.plot_format}")
                else:
                    self._log("  No timing data", 'warning')

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            return {
                'success': True,
                'output_dir': str(output_path),
                'generated_plots': len(generated_plots),
                'plots': {name: str(path) for name, path in generated_plots},
                'duration_seconds': duration,
            }

        except Exception as e:
            self._log(f"Failed: {e}", 'warning')
            return {'success': False, 'error': str(e)}

    # ============================================================
    # Score Comparison
    # ============================================================

    def _plot_score_comparison(
        self,
        before_data: List[Dict],
        after_data: List[Dict],
        output_path: Path
    ) -> Optional[Path]:
        """Quality + Confidence - Point-Line style"""
        if not HAS_MATPLOTLIB:
            return None

        before_scores = [self._get_quality_score(d) for d in before_data]
        after_scores = [self._get_quality_score(d) for d in after_data]
        before_confs = [self._get_avg_confidence(d) for d in before_data]
        after_confs = [self._get_avg_confidence(d) for d in after_data]

        min_quality = self.config.get('cleaning.min_quality_score', 50.0)
        min_confidence = self.config.get('cleaning.min_confidence', 0.6)

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Quality Score
        ax1 = axes[0]
        x_before = range(1, len(before_scores)+1)
        ax1.plot(x_before, before_scores, color=self.colors['before'], linewidth=0.8, alpha=0.5)
        ax1.scatter(x_before, before_scores, c=self.colors['before'], s=30, alpha=0.7,
                   label=f'Before (avg: {np.mean(before_scores):.1f})')
        ax1.plot(range(1, len(after_scores)+1), after_scores, color=self.colors['after'],
                linewidth=1.2, alpha=0.6)
        ax1.scatter(range(1, len(after_scores)+1), after_scores, c=self.colors['after'],
                   s=50, alpha=0.8, marker='o', edgecolors='black',
                   label=f'After (avg: {np.mean(after_scores):.1f})')
        ax1.axhline(y=min_quality, color=self.colors['threshold'], linestyle='--', linewidth=2,
                   label=f'Threshold: {min_quality}')
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Quality Score')
        ax1.set_title('Quality Score Comparison', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Confidence
        ax2 = axes[1]
        ax2.plot(range(1, len(before_confs)+1), before_confs, color=self.colors['before'],
                linewidth=0.8, alpha=0.5)
        ax2.scatter(range(1, len(before_confs)+1), before_confs, c=self.colors['before'],
                   s=30, alpha=0.7, label=f'Before (avg: {np.mean(before_confs):.3f})')
        ax2.plot(range(1, len(after_confs)+1), after_confs, color=self.colors['after'],
                linewidth=1.2, alpha=0.6)
        ax2.scatter(range(1, len(after_confs)+1), after_confs, c=self.colors['after'],
                   s=50, alpha=0.8, marker='o', edgecolors='black',
                   label=f'After (avg: {np.mean(after_confs):.3f})')
        ax2.axhline(y=min_confidence, color=self.colors['threshold'], linestyle='--', linewidth=2,
                   label=f'Threshold: {min_confidence}')
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Confidence')
        ax2.set_title('Confidence Comparison', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        path = output_path / f'score_comparison.{self.plot_format}'
        plt.savefig(path, dpi=self.dpi)
        plt.close()
        return path

    # ============================================================
    # Sample Visualization
    # ============================================================

    def _visualize_samples(self, data_list: List[Dict], output_path: Path) -> List[Path]:
        """Visualize sample images with CoT"""
        if not HAS_PIL:
            return []

        sample_config = self.viz_config.get('sample_visualization', {})
        max_samples = sample_config.get('max_samples', 10)

        random.seed(42)
        selected = random.sample(data_list, min(max_samples, len(data_list)))

        paths = []
        images_root = Path(self.config.get('data.images_root', './data/coco/val2014'))

        for idx, data in enumerate(selected):
            try:
                path = self._visualize_single_sample(data, output_path, images_root, idx)
                if path:
                    paths.append(path)
            except Exception as e:
                self._log(f"  Sample {idx} failed: {e}", 'warning')

        return paths

    def _visualize_single_sample(
        self,
        data: Dict,
        output_path: Path,
        images_root: Path,
        idx: int
    ) -> Optional[Path]:
        """Single sample visualization with CoT reasoning"""
        image_id = data.get('image_id')
        file_name = data.get('file_name', '')

        image_path = None
        if file_name:
            image_path = images_root / file_name
        elif image_id:
            image_path = images_root / f"COCO_val2014_{str(image_id).zfill(12)}.jpg"

        if image_path and image_path.exists():
            img = Image.open(image_path)
            img_array = np.array(img)
        else:
            img_array = np.ones((400, 600, 3), dtype=np.uint8) * 200

        # 创建三栏布局：图像 | 标签 | CoT思维链
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1], height_ratios=[1, 1],
                             hspace=0.3, wspace=0.3)

        # ============================================================
        # 左侧：原图 + 检测框
        # ============================================================
        ax_img = fig.add_subplot(gs[:, 0])
        ax_img.imshow(img_array)
        ax_img.set_title(f'Sample {idx+1}: Image ID {image_id}', fontsize=14, fontweight='bold')
        ax_img.axis('off')
        self._draw_detection_boxes(ax_img, data)

        # ============================================================
        # 右上：硬标签 + 软标签信息
        # ============================================================
        ax_labels = fig.add_subplot(gs[0, 1:])
        ax_labels.axis('off')

        labels_text = self._format_labels_info(data)
        ax_labels.text(0.02, 0.98, labels_text, transform=ax_labels.transAxes,
                       fontsize=11, va='top', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        ax_labels.set_title('Hard & Soft Labels', fontsize=12, fontweight='bold')

        # ============================================================
        # 右下：CoT思维链推理过程
        # ============================================================
        ax_cot = fig.add_subplot(gs[1, 1:])
        ax_cot.axis('off')

        cot_text = self._format_cot_info(data)
        ax_cot.text(0.02, 0.98, cot_text, transform=ax_cot.transAxes,
                    fontsize=10, va='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
        ax_cot.set_title('Chain-of-Thought Reasoning', fontsize=12, fontweight='bold')

        plt.tight_layout()
        path = output_path / f'sample_{idx+1}.{self.plot_format}'
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close()
        return path

    def _draw_detection_boxes(self, ax: plt.Axes, data: Dict):
        """Draw detection boxes"""
        tasks = data.get('tasks', {})
        for task_name, task_data in tasks.items():
            if task_name == 'detection':
                hard_label = task_data.get('hard_label', {})
                objects = hard_label.get('objects', [])
                for obj in objects:
                    category = obj.get('category', obj.get('label', 'object'))
                    bbox = obj.get('bbox', obj.get('bbox_2d', []))
                    confidence = obj.get('confidence', 0.9)
                    if len(bbox) >= 4:
                        rect = Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                                          linewidth=2, edgecolor='#3498DB', facecolor='none')
                        ax.add_patch(rect)
                        ax.text(bbox[0], bbox[1]-5, f"{category}: {confidence:.0%}",
                               fontsize=9, color='#3498DB', fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    def _format_labels_info(self, data: Dict) -> str:
        """Format labels info (hard + soft)"""
        lines = []
        tasks = data.get('tasks', {})

        for task_name, task_data in tasks.items():
            # 硬标签
            hard_label = task_data.get('hard_label', {})
            if hard_label:
                answer = hard_label.get('answer', 'N/A')
                confidence = hard_label.get('confidence', 0)
                lines.append(f"[{task_name.upper()}] HARD:")
                lines.append(f"  Answer: {answer}")
                lines.append(f"  Confidence: {confidence:.2%}")

            # 软标签
            soft_label = task_data.get('soft_label', {})
            if soft_label:
                distribution = soft_label.get('answer_distribution', {})
                if distribution:
                    lines.append(f"\n[{task_name.upper()}] SOFT:")
                    # 显示前5个概率分布
                    sorted_dist = sorted(distribution.items(), key=lambda x: x[1], reverse=True)[:5]
                    for ans, prob in sorted_dist:
                        lines.append(f"  {ans}: {prob:.2%}")

            lines.append("")  # 空行分隔

        return "\n".join(lines) if lines else "No label info"

    def _format_cot_info(self, data: Dict) -> str:
        """Format CoT reasoning info"""
        lines = []
        tasks = data.get('tasks', {})

        for task_name, task_data in tasks.items():
            cot = task_data.get('cot_reasoning', {})
            if cot:
                lines.append(f"[{task_name.upper()}] CoT Reasoning:")

                # 原始推理文本
                raw_reasoning = cot.get('raw_reasoning', '')
                if raw_reasoning:
                    # 截断过长的推理文本
                    if len(raw_reasoning) > 300:
                        raw_reasoning = raw_reasoning[:300] + "..."
                    lines.append(f"  {raw_reasoning}")

                # 质量指标
                quality_metrics = cot.get('quality_metrics', {})
                if quality_metrics:
                    logical_flow = quality_metrics.get('logical_flow_score', 0)
                    step_count = quality_metrics.get('step_count', 0)
                    lines.append(f"\n  Quality Metrics:")
                    lines.append(f"    Logical Flow: {logical_flow:.2f}")
                    lines.append(f"    Steps: {step_count}")

                lines.append("")  # 空行分隔

        return "\n".join(lines) if lines else "No CoT reasoning available"

    # ============================================================
    # Pipeline Timing with Distillation Breakdown
    # ============================================================

    def _plot_pipeline_timing(
        self,
        timing_stats: Dict[str, Any],
        output_path: Path
    ) -> Optional[Path]:
        """Pipeline timing with distillation sub-steps breakdown"""
        if not HAS_MATPLOTLIB or not timing_stats:
            return None

        # Define step order with groups
        step_order = [
            ('data_loading', 'distillation'),
            ('preprocessing', 'distillation'),
            ('model_inference', 'distillation'),
            ('initial_validation', 'validation'),
            ('cleaning', 'cleaning'),
            ('final_validation', 'validation'),
            ('visualization', 'visualization'),
        ]

        # Extract data
        steps, durations, samples, avgs, groups = [], [], [], [], []
        for step, group in step_order:
            if step in timing_stats:
                stats = timing_stats[step]
                dur = stats.get('duration', 0) if isinstance(stats, dict) else float(stats or 0)
                n = stats.get('samples', 1) if isinstance(stats, dict) else 1
                if dur > 0:
                    steps.append(step)
                    durations.append(dur)
                    samples.append(n)
                    avgs.append(dur / max(n, 1))
                    groups.append(group)

        if not steps:
            return None

        total = sum(durations)
        total_samples = max(samples)
        bottleneck_idx = np.argmax(avgs)
        bottleneck = steps[bottleneck_idx]

        # Figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # Colors by group
        colors = [self.group_colors.get(g, '#CCCCCC') for g in groups]

        # ============================================================
        # Plot 1: Total Duration (grouped bars)
        # ============================================================
        ax1 = axes[0, 0]
        x = np.arange(len(steps))
        bars = ax1.bar(x, durations, color=colors, edgecolor='black')

        # Add group separators
        current_group = None
        for i, g in enumerate(groups):
            if g != current_group and current_group:
                ax1.axvline(x=i-0.5, color='gray', linestyle='--', alpha=0.5)
            current_group = g

        for bar, d in zip(bars, durations):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(durations)*0.02,
                    f'{d:.1f}s', ha='center', fontsize=9)

        ax1.set_xticks(x)
        ax1.set_xticklabels([s.replace('_', '\n') for s in steps], fontsize=9)
        ax1.set_ylabel('Duration (s)')
        ax1.set_title('Total Duration\n(Distillation broken down)', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # Legend
        legend_handles = [plt.Rectangle((0,0),1,1, facecolor=c, edgecolor='black', label=g.title())
                         for g, c in self.group_colors.items()]
        ax1.legend(handles=legend_handles, loc='upper right', fontsize=9)

        # ============================================================
        # Plot 2: Per-sample average (Bottleneck)
        # ============================================================
        ax2 = axes[0, 1]
        bars2 = ax2.bar(x, avgs, color=colors, edgecolor='black')

        # Highlight bottleneck
        bars2[bottleneck_idx].set_color('#FF6B6B')
        bars2[bottleneck_idx].set_edgecolor('darkred')

        for bar, a in zip(bars2, avgs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avgs)*0.02,
                    f'{a:.3f}s', ha='center', fontsize=9)

        ax2.annotate(f'BOTTLENECK\n{avgs[bottleneck_idx]:.3f}s/sample',
                    xy=(bottleneck_idx, avgs[bottleneck_idx]),
                    xytext=(bottleneck_idx+0.5, avgs[bottleneck_idx]*1.2),
                    ha='left', fontsize=10, color='red', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='red'))

        ax2.set_xticks(x)
        ax2.set_xticklabels([s.replace('_', '\n') for s in steps], fontsize=9)
        ax2.set_ylabel('Avg per Sample (s)')
        ax2.set_title('Per-Sample Timing (Bottleneck)', fontweight='bold', color='#E74C3C')
        ax2.grid(True, alpha=0.3, axis='y')

        # ============================================================
        # Plot 3: Distillation Breakdown Pie
        # ============================================================
        ax3 = axes[1, 0]

        distill_names = ['Data Loading', 'Preprocessing', 'Model Inference']
        distill_keys = ['data_loading', 'preprocessing', 'model_inference']
        distill_vals = []
        for key in distill_keys:
            if key in timing_stats:
                stats = timing_stats[key]
                d = stats.get('duration', 0) if isinstance(stats, dict) else float(stats or 0)
                if d > 0:
                    distill_vals.append(d)

        if distill_vals:
            pie_colors = ['#87CEEB', '#5DADE2', '#2E86C1']
            wedges, _, autotexts = ax3.pie(distill_vals, labels=distill_names[:len(distill_vals)],
                                          colors=pie_colors[:len(distill_vals)],
                                          autopct='%1.1f%%', startangle=90)
            ax3.set_title(f'Distillation Breakdown\nTotal: {sum(distill_vals):.1f}s', fontweight='bold')

            # Add duration labels
            for i, (wedge, dur) in enumerate(zip(wedges, distill_vals)):
                angle = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
                x, y = np.cos(np.deg2rad(angle)) * 0.6, np.sin(np.deg2rad(angle)) * 0.6
                ax3.annotate(f'{dur:.1f}s', xy=(x, y), ha='center', fontsize=11, fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No distillation data', transform=ax3.transAxes, ha='center')
            ax3.set_title('Distillation Breakdown', fontweight='bold')

        # ============================================================
        # Plot 4: Statistics Table
        # ============================================================
        ax4 = axes[1, 1]
        ax4.axis('off')

        percentages = [d/total*100 for d in durations]
        table_data = [[s.replace('_', ' ').title(), f"{d:.1f}s", str(n), f"{a:.3f}s", f"{p:.1f}%"]
                      for s, d, n, a, p in zip(steps, durations, samples, avgs, percentages)]
        table_data.append(['TOTAL', f"{total:.1f}s", str(total_samples), f"{total/total_samples:.3f}s", "100%"])

        table = ax4.table(
            cellText=table_data,
            colLabels=['Step', 'Duration', 'Samples', 'Avg/Sample', '%'],
            loc='center', cellLoc='center',
            colWidths=[0.25, 0.15, 0.12, 0.18, 0.10]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.6)

        # Highlight bottleneck
        for col in range(5):
            table[(bottleneck_idx+1, col)].set_facecolor('#FFE4E1')
        # Highlight total
        for col in range(5):
            table[(len(table_data), col)].set_facecolor('#E8F4F8')

        ax4.set_title('Statistics Summary', fontweight='bold')

        # Title
        fig.suptitle(f'Pipeline Timing Analysis\nTotal: {total:.1f}s | Bottleneck: {bottleneck.replace("_", " ").title()}',
                    fontsize=14, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        path = output_path / f'pipeline_timing.{self.plot_format}'
        plt.savefig(path, dpi=self.dpi)
        plt.close()
        return path

    # ============================================================
    # Helpers
    # ============================================================

    def _get_quality_score(self, data: Dict) -> float:
        """Get or estimate quality score"""
        score = data.get('quality_score')
        if score is not None:
            return score

        tasks = data.get('tasks', {})
        scores = []
        for task_name, task_data in tasks.items():
            s = 0
            hard_label = task_data.get('hard_label', {})
            if hard_label:
                conf = hard_label.get('confidence', 0.0)
                if conf >= 0.7:
                    s += 30
                elif conf >= 0.5:
                    s += 20
                answer = hard_label.get('answer', '')
                if 3 <= len(answer) <= 100:
                    s += 10
            scores.append(s)
        return min(np.mean(scores) if scores else 50, 100.0)

    def _get_avg_confidence(self, data: Dict) -> float:
        """Get average confidence"""
        tasks = data.get('tasks', {})
        confs = [t.get('hard_label', {}).get('confidence', 0.5)
                 for t in tasks.values() if t.get('hard_label')]
        return np.mean(confs) if confs else 0.5