"""
流程可视化生成器
===============

负责生成所有可视化图表，包括：
- 置信度分布图
- 质量分数图
- 任务分布图
- 清洗前后对比图
- 异常检测图
- 流程时间线图
- CoT质量图
- 样本可视化

Usage:
    visualizer = PipelineVisualizer(config, logger)
    plots = visualizer.generate_all_plots(data_list, output_dir, timing_stats)
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import json
import random

try:
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    np = None

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class PipelineVisualizer:
    """
    流程可视化生成器

    功能：
    1. 统计图表（置信度、质量分数、任务分布）
    2. 清洗前后对比图
    3. 异常检测可视化
    4. 流程时间线
    5. CoT质量分析图
    6. 样本可视化（原图+标签+CoT）
    7. HTML报告生成
    """

    def __init__(self, config: Any, logger: Any = None):
        """
        初始化可视化生成器

        Args:
            config: 配置管理器
            logger: 日志记录器
        """
        self.config = config
        self.logger = logger
        self.plt = plt

        # 可视化配置
        viz_config = config.get('visualization', {})
        self.output_dir = viz_config.get('output_dir', './outputs/visualizations')
        self.plot_format = viz_config.get('plot_format', 'png')
        self.dpi = viz_config.get('dpi', 150)
        self.figsize = viz_config.get('figsize', {'width': 12, 'height': 8})
        self.default_figsize = (self.figsize.get('width', 12), self.figsize.get('height', 8))

        # 颜色方案
        self.colors = {
            'before': '#FF6B6B',
            'after': '#2ECC71',
            'threshold': '#F39C12',
            'passed': '#3498DB',
            'failed': '#E74C3C',
        }

    def generate_all_plots(
        self,
        data_list: List[Dict],
        output_dir: str,
        timing_stats: Dict,
        pipeline_results: Dict,
        before_data: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        生成所有可视化图表

        Args:
            data_list: 清洗后数据
            output_dir: 输出目录
            timing_stats: 步骤耗时统计
            pipeline_results: 流程结果
            before_data: 清洗前数据（可选）

        Returns:
            可视化报告
        """
        if not HAS_MATPLOTLIB:
            if self.logger:
                self.logger.warning("Matplotlib未安装，无法生成可视化")
            return {'success': False, 'error': 'matplotlib not installed'}

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.logger:
            self.logger.info(f"\nGenerating visualizations in: {output_path}")

        generated_plots = []

        # 1. 置信度分布图
        plot_path = self._plot_confidence_distribution(
            data_list, output_path, self.plot_format, self.dpi,
            self.default_figsize, {}
        )
        if plot_path:
            generated_plots.append(('confidence_distribution', plot_path))
            if self.logger:
                self.logger.info(f"  ✓ Generated: {plot_path.name}")

        # 2. 质量分数图
        plot_path = self._plot_quality_scores(
            data_list, output_path
        )
        if plot_path:
            generated_plots.append(('quality_scores', plot_path))
            if self.logger:
                self.logger.info(f"  ✓ Generated: {plot_path.name}")

        # 3. 任务分布图
        plot_path = self._plot_task_distribution(data_list, output_path)
        if plot_path:
            generated_plots.append(('task_distribution', plot_path))
            if self.logger:
                self.logger.info(f"  ✓ Generated: {plot_path.name}")

        # 4. 清洗前后对比（如果有before_data）
        if before_data:
            plot_path = self._plot_before_after_comparison(
                before_data, data_list, output_path
            )
            if plot_path:
                generated_plots.append(('before_after_comparison', plot_path))
                if self.logger:
                    self.logger.info(f"  ✓ Generated: {plot_path.name}")

        # 5. 流程时间线
        plot_path = self._plot_pipeline_flow(timing_stats, output_path)
        if plot_path:
            generated_plots.append(('pipeline_flow', plot_path))
            if self.logger:
                self.logger.info(f"  ✓ Generated: {plot_path.name}")

        # 6. 质量分数和置信度时间线（新增）
        timeline_plots = self._plot_timeline_visualization(
            data_list, output_path, 'after'
        )
        generated_plots.extend(timeline_plots)
        if self.logger and timeline_plots:
            self.logger.info(f"  ✓ Generated {len(timeline_plots)} timeline plots")

        # 7. 清洗前后时间线对比（如果有before_data）
        if before_data:
            comparison_plots = self._plot_timeline_comparison(
                before_data, data_list, output_path
            )
            generated_plots.extend(comparison_plots)
            if self.logger and comparison_plots:
                self.logger.info(f"  ✓ Generated {len(comparison_plots)} timeline comparison plots")

        # 6. 样本可视化
        sample_plots = self._visualize_samples(
            data_list, output_path, max_samples=5
        )
        generated_plots.extend([('sample', p) for p in sample_plots])
        if self.logger:
            self.logger.info(f"  ✓ Generated {len(sample_plots)} sample visualizations")

        # 7. 生成HTML报告
        html_report = self._generate_html_report(
            generated_plots, timing_stats, output_path
        )
        if self.logger:
            self.logger.info(f"  ✓ Generated: {html_report.name}")

        return {
            'success': True,
            'generated_plots': len(generated_plots),
            'plot_list': generated_plots,
            'html_report': str(html_report)
        }

    def _plot_confidence_distribution(
        self,
        data_list: List[Dict],
        output_path: Path,
        plot_format: str = 'png',
        dpi: int = 150,
        figsize: Tuple = (12, 6),
        config: Dict = {}
    ) -> Optional[Path]:
        """
        绘制置信度分布图
        """
        if not HAS_MATPLOTLIB:
            return None

        # 收集置信度数据
        confidence_values = []
        for data in data_list:
            tasks = data.get('tasks', {})
            for task_name, task_data in tasks.items():
                hard_label = task_data.get('hard_label', {})
                confidence = hard_label.get('confidence')
                if confidence is not None:
                    confidence_values.append({
                        'value': confidence,
                        'task': task_name
                    })

        if not confidence_values:
            return None

        # 创建图表
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        values = [c['value'] for c in confidence_values]
        tasks = [c['task'] for c in confidence_values]

        # 子图1：分布直方图
        axes[0].hist(values, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].set_xlabel('Confidence Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Confidence Distribution')

        # 添加统计线
        mean_val = np.mean(values)
        median_val = np.median(values)
        axes[0].axvline(mean_val, color='red', linestyle='--',
                       label=f'Mean: {mean_val:.3f}')
        axes[0].axvline(median_val, color='green', linestyle='--',
                       label=f'Median: {median_val:.3f}')
        axes[0].legend()

        # 子图2：各任务箱线图
        unique_tasks = sorted(set(tasks))
        task_data_dict = {task: [] for task in unique_tasks}
        for c in confidence_values:
            task_data_dict[c['task']].append(c['value'])

        box_data = [task_data_dict[task] for task in unique_tasks]
        axes[1].boxplot(box_data, tick_labels=unique_tasks)
        axes[1].set_xlabel('Task Type')
        axes[1].set_ylabel('Confidence Score')
        axes[1].set_title('Confidence by Task')

        plt.tight_layout()
        plot_path = output_path / f'confidence_distribution.{plot_format}'
        plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
        plt.close()

        return plot_path

    def _plot_quality_scores(
        self,
        data_list: List[Dict],
        output_path: Path
    ) -> Optional[Path]:
        """
        绘制质量分数分布图
        """
        if not HAS_MATPLOTLIB or not data_list:
            return None

        # 收集或估算质量分数
        quality_scores = []
        for data in data_list:
            quality_score = data.get('quality_score', 50)  # 默认50分
            quality_scores.append(quality_score)

        fig, ax = plt.subplots(figsize=self.default_figsize)
        ax.hist(quality_scores, bins=30, edgecolor='black', alpha=0.7, color='green')
        ax.set_xlabel('Quality Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Quality Score Distribution')

        mean_val = np.mean(quality_scores)
        ax.axvline(mean_val, color='red', linestyle='--',
                  label=f'Mean: {mean_val:.1f}')
        ax.legend()

        plt.tight_layout()
        plot_path = output_path / f'quality_scores.{self.plot_format}'
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return plot_path

    def _plot_task_distribution(
        self,
        data_list: List[Dict],
        output_path: Path
    ) -> Optional[Path]:
        """
        绘制任务分布图
        """
        if not HAS_MATPLOTLIB or not data_list:
            return None

        # 统计任务数量
        task_counts = {}
        for data in data_list:
            tasks = data.get('tasks', {})
            for task_name in tasks.keys():
                task_counts[task_name] = task_counts.get(task_name, 0) + 1

        if not task_counts:
            return None

        fig, ax = plt.subplots(figsize=self.default_figsize)
        tasks = list(task_counts.keys())
        counts = list(task_counts.values())

        ax.bar(tasks, counts, color=['#3498DB', '#2ECC71', '#E74C3C',
                                     '#9B59B6', '#F39C12'][:len(tasks)])
        ax.set_xlabel('Task Type')
        ax.set_ylabel('Count')
        ax.set_title('Task Distribution')

        plt.tight_layout()
        plot_path = output_path / f'task_distribution.{self.plot_format}'
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return plot_path

    def _plot_before_after_comparison(
        self,
        before_data: List[Dict],
        after_data: List[Dict],
        output_path: Path
    ) -> Optional[Path]:
        """
        绘制清洗前后对比图
        """
        if not HAS_MATPLOTLIB:
            return None

        fig, axes = plt.subplots(1, 2, figsize=self.default_figsize)

        # 样本数量对比
        axes[0].bar(['Before', 'After'],
                   [len(before_data), len(after_data)],
                   color=[self.colors['before'], self.colors['after']])
        axes[0].set_ylabel('Sample Count')
        axes[0].set_title('Sample Count Comparison')

        # 质量分数对比（估算）
        before_scores = [d.get('quality_score', 50) for d in before_data]
        after_scores = [d.get('quality_score', 50) for d in after_data]

        axes[1].bar(['Before', 'After'],
                   [np.mean(before_scores), np.mean(after_scores)],
                   color=[self.colors['before'], self.colors['after']])
        axes[1].set_ylabel('Average Quality Score')
        axes[1].set_title('Quality Score Comparison')

        plt.tight_layout()
        plot_path = output_path / f'before_after_comparison.{self.plot_format}'
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return plot_path

    def _plot_pipeline_flow(
        self,
        timing_stats: Dict,
        output_path: Path
    ) -> Optional[Path]:
        """
        绘制流程时间线
        """
        if not HAS_MATPLOTLIB or not timing_stats:
            return None

        fig, ax = plt.subplots(figsize=self.default_figsize, constrained_layout=True)

        steps = list(timing_stats.keys())
        durations = list(timing_stats.values())

        colors = ['#3498DB', '#9B59B6', '#E74C3C',
                 '#2ECC71', '#F39C12'][:len(steps)]

        ax.barh(range(len(steps)), durations, color=colors)
        ax.set_yticks(range(len(steps)))
        ax.set_yticklabels([s.replace('_', ' ').title() for s in steps])
        ax.set_xlabel('Duration (seconds)')
        ax.set_title('Pipeline Flow Timeline')

        # 显示总耗时
        total_duration = sum(durations)
        ax.text(0.95, 0.95, f'Total: {total_duration:.1f}s',
               transform=ax.transAxes, ha='right', va='top',
               fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plot_path = output_path / f'pipeline_flow.{self.plot_format}'
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return plot_path

    def _visualize_samples(
        self,
        data_list: List[Dict],
        output_path: Path,
        max_samples: int = 5
    ) -> List[Path]:
        """
        可视化样本数据（原图+标签+CoT）

        Returns:
            生成的图像路径列表
        """
        if not HAS_PIL or not HAS_MATPLOTLIB:
            return []

        generated_paths = []

        # 随机选择样本
        random.seed(42)
        selected_samples = random.sample(data_list, min(max_samples, len(data_list)))

        if self.logger:
            self.logger.info(f"\n  Visualizing {len(selected_samples)} samples...")

        for idx, data in enumerate(selected_samples):
            try:
                # 创建可视化图像
                fig = plt.figure(figsize=(16, 12), constrained_layout=True)

                # 获取图像路径
                image_id = data.get('image_id')
                file_name = data.get('file_name', '')
                images_root = self.config.get('data.images_root', './data/coco/val2014')
                images_root_path = Path(images_root)

                # 尝试找到图像
                image_path = None
                if file_name:
                    image_path = images_root_path / file_name
                elif image_id:
                    image_path = images_root_path / f"COCO_val2014_{str(image_id).zfill(12)}.jpg"

                # 加载图像
                if image_path and image_path.exists():
                    img = Image.open(image_path)
                    img_array = np.array(img)
                else:
                    # 占位图
                    img_array = np.ones((400, 600, 3), dtype=np.uint8) * 200

                # 创建GridSpec布局
                gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1], height_ratios=[1, 1])

                # 子图1：原图
                ax_img = fig.add_subplot(gs[:, 0])
                ax_img.imshow(img_array)
                ax_img.set_title(f'Sample {idx+1}: Image ID {image_id}',
                                fontsize=14, fontweight='bold')
                ax_img.axis('off')

                # 子图2：标签信息
                ax_info = fig.add_subplot(gs[0, 1:])
                ax_info.axis('off')

                # 收集标签信息
                tasks = data.get('tasks', {})
                info_text = ""
                for task_name, task_data in tasks.items():
                    hard_label = task_data.get('hard_label', {})
                    answer = hard_label.get('answer', 'N/A')
                    confidence = hard_label.get('confidence', 0)
                    info_text += f"[{task_name}] Answer: {answer} (conf: {confidence:.2f})\n"

                ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                            fontsize=11, verticalalignment='top',
                            fontfamily='monospace',
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
                ax_info.set_title('Labels', fontsize=12, fontweight='bold')

                # 子图3：CoT推理
                ax_cot = fig.add_subplot(gs[1, 1:])
                ax_cot.axis('off')

                # 收集CoT信息
                cot_text = ""
                for task_name, task_data in tasks.items():
                    cot = task_data.get('cot_reasoning', {})
                    reasoning = cot.get('raw_reasoning', '')
                    if reasoning:
                        # 截断过长内容
                        if len(reasoning) > 200:
                            reasoning = reasoning[:200] + "..."
                        cot_text += f"[{task_name}] {reasoning}\n\n"

                if not cot_text:
                    cot_text = "No CoT reasoning available"

                ax_cot.text(0.05, 0.95, cot_text, transform=ax_cot.transAxes,
                            fontsize=10, verticalalignment='top',
                            fontfamily='monospace',
                            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
                ax_cot.set_title('Chain-of-Thought', fontsize=12, fontweight='bold')

                # 保存图像
                sample_path = output_path / f'sample_{idx+1}.{self.plot_format}'
                plt.savefig(sample_path, dpi=self.dpi, bbox_inches='tight')
                plt.close()

                generated_paths.append(sample_path)
                if self.logger:
                    self.logger.info(f"    ✓ Sample {idx+1}: {sample_path.name}")

            except Exception as e:
                if self.logger:
                    self.logger.warning(f"    ✗ Failed to visualize sample {idx}: {e}")
                plt.close()
                continue

        return generated_paths

    def _generate_html_report(
        self,
        generated_plots: List[Tuple[str, Path]],
        timing_stats: Dict,
        output_path: Path
    ) -> Path:
        """
        生成HTML可视化报告
        """
        html_path = output_path / 'visualization_report.html'

        # 构建HTML内容
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipeline Visualization Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2C3E50;
            text-align: center;
        }}
        .plot-container {{
            margin: 20px auto;
            padding: 15px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        .plot-title {{
            font-size: 18px;
            color: #34495E;
            margin-bottom: 10px;
            font-weight: bold;
        }}
        .stats {{
            margin: 20px auto;
            padding: 15px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stats h2 {{
            color: #2C3E50;
            margin-bottom: 10px;
        }}
        .stats table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .stats th, .stats td {{
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .stats th {{
            background-color: #3498DB;
            color: white;
        }}
        .stats tr:hover {{
            background-color: #f5f5f5;
        }}
    </style>
</head>
<body>
    <h1>VLM Distillation Pipeline Visualization Report</h1>

    <div class="stats">
        <h2>Pipeline Timing</h2>
        <table>
            <tr>
                <th>Step</th>
                <th>Duration (s)</th>
            </tr>
"""

        # 添加耗时统计
        for step, duration in timing_stats.items():
            step_name = step.replace('_', ' ').title()
            html_content += f"<tr><td>{step_name}</td><td>{duration:.1f}</td></tr>\n"

        total_duration = sum(timing_stats.values())
        html_content += f"<tr><td><strong>Total</strong></td><td><strong>{total_duration:.1f}</strong></td></tr>\n"
        html_content += """
        </table>
    </div>

"""

        # 添加所有图表
        for plot_name, plot_path in generated_plots:
            plot_title = plot_name.replace('_', ' ').title()
            html_content += f"""
    <div class="plot-container">
        <div class="plot-title">{plot_title}</div>
        <img src="{plot_path.name}" alt="{plot_title}">
    </div>
"""

        html_content += """
</body>
</html>
"""

        # 保存HTML
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return html_path

    def _plot_timeline_visualization(
        self,
        data_list: List[Dict],
        output_path: Path,
        label: str = "after"
    ) -> List[Tuple[str, Path]]:
        """
        绘制质量分数和置信度时间线折线图

        Args:
            data_list: 数据列表
            output_path: 输出目录
            label: 数据标签 ("before" 或 "after")

        Returns:
            生成的图表路径列表 [(plot_name, plot_path), ...]
        """
        if not HAS_MATPLOTLIB or not data_list:
            return []

        generated_plots = []

        # 标签映射
        label_suffix = "_before" if label == "before" else ""

        # 获取阈值配置
        min_quality = self.config.get('cleaning.min_quality_score', 50.0)
        min_confidence = self.config.get('cleaning.min_confidence', 0.6)

        # 收集每张图像的分数和置信度
        quality_scores = []
        confidence_values = []
        image_indices = []

        for idx, data in enumerate(data_list):
            # 获取质量分数
            quality_score = data.get('quality_score', 50)  # 默认50分
            quality_scores.append(quality_score)

            # 获取置信度（取各任务的平均置信度）
            tasks = data.get('tasks', {})
            task_confidences = []
            for task_name, task_data in tasks.items():
                hard_label = task_data.get('hard_label', {})
                confidence = hard_label.get('confidence')
                if confidence is not None:
                    task_confidences.append(confidence)

            avg_confidence = np.mean(task_confidences) if task_confidences else 0.5
            confidence_values.append(avg_confidence)
            image_indices.append(idx + 1)

        if not quality_scores:
            return generated_plots

        # ============================================================
        # 图1: 质量分数时间线折线图
        # ============================================================
        fig, ax = plt.subplots(figsize=self.default_figsize)

        # 绘制分数折线
        ax.plot(image_indices, quality_scores, 'b-', linewidth=1.5,
                alpha=0.7, label='Quality Score')

        # 标记每个点
        colors = ['green' if q >= min_quality else 'red' for q in quality_scores]
        ax.scatter(image_indices, quality_scores, c=colors, s=20, alpha=0.6)

        # 绘制阈值线
        ax.axhline(y=min_quality, color='red', linestyle='--', linewidth=2,
                   label=f'Threshold: {min_quality}')

        # 添加统计信息
        mean_val = np.mean(quality_scores)
        median_val = np.median(quality_scores)
        ax.axvline(mean_val, color='red', linestyle='--',
                   alpha=0.5, label=f'Mean: {mean_val:.1f}')

        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Quality Score (0-100)', fontsize=12)
        ax.set_title(f'Quality Score Timeline ({label.title()})', fontsize=14, fontweight='bold')
        ax.set_xlim(0, len(image_indices) + 1)
        ax.set_ylim(0, 105)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_path / f'quality_score_timeline{label_suffix}.{self.plot_format}'
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        generated_plots.append(('quality_score_timeline', plot_path))

        # ============================================================
        # 图2: 置信度时间线折线图
        # ============================================================
        fig, ax = plt.subplots(figsize=self.default_figsize)

        # 绘制置信度折线
        ax.plot(image_indices, confidence_values, 'purple', linewidth=1.5,
                alpha=0.7, label='Confidence')

        # 标记每个点
        colors = ['green' if c >= min_confidence else 'orange' for c in confidence_values]
        ax.scatter(image_indices, confidence_values, c=colors, s=20, alpha=0.6)

        # 绘制阈值线
        ax.axhline(y=min_confidence, color='orange', linestyle='--', linewidth=2,
                   label=f'Threshold: {min_confidence}')

        # 填充高低置信度区域
        ax.fill_between(image_indices, min_confidence, 1.0,
                        alpha=0.1, color='green', label='High Confidence')
        ax.fill_between(image_indices, 0, min_confidence,
                        alpha=0.1, color='orange', label='Low Confidence')

        # 添加统计信息
        mean_conf = np.mean(confidence_values)
        ax.text(0.02, 0.98, f'Mean: {mean_conf:.3f}\nMin: {min(confidence_values):.3f}\nMax: {max(confidence_values):.3f}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))

        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Confidence Score (0-1)', fontsize=12)
        ax.set_title(f'Confidence Timeline ({label.title()})', fontsize=14, fontweight='bold')
        ax.set_xlim(0, len(image_indices) + 1)
        ax.set_ylim(0, 1.05)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_path / f'confidence_timeline{label_suffix}.{self.plot_format}'
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        generated_plots.append(('confidence_timeline', plot_path))

        return generated_plots

    def _plot_timeline_comparison(
        self,
        before_data: List[Dict],
        after_data: List[Dict],
        output_path: Path
    ) -> List[Tuple[str, Path]]:
        """
        绘制清洗前后时间线对比折线图

        Args:
            before_data: 清洗前数据列表
            after_data: 清洗后数据列表
            output_path: 输出目录

        Returns:
            生成的图表路径列表
        """
        if not HAS_MATPLOTLIB:
            return []

        generated_plots = []

        # 获取阈值配置
        min_quality = self.config.get('cleaning.min_quality_score', 50.0)
        min_confidence = self.config.get('cleaning.min_confidence', 0.6)

        # 收集清洗前的分数和置信度
        before_scores = []
        before_confidences = []
        for data in before_data:
            score = data.get('quality_score', 50)
            before_scores.append(score)

            tasks = data.get('tasks', {})
            confs = [t.get('hard_label', {}).get('confidence', 0.5) for t in tasks.values()]
            before_confidences.append(np.mean(confs) if confs else 0.5)

        # 收集清洗后的分数和置信度
        after_scores = []
        after_confidences = []
        for data in after_data:
            score = data.get('quality_score', 50)
            after_scores.append(score)

            tasks = data.get('tasks', {})
            confs = [t.get('hard_label', {}).get('confidence', 0.5) for t in tasks.values()]
            after_confidences.append(np.mean(confs) if confs else 0.5)

        # ============================================================
        # 图1: 质量分数对比折线图
        # ============================================================
        fig, ax = plt.subplots(figsize=self.default_figsize)

        # 清洗前折线
        ax.plot(range(1, len(before_scores)+1), before_scores,
                color=self.colors['before'], linewidth=1.5, alpha=0.7,
                label=f'Before (Mean: {np.mean(before_scores):.1f})')

        # 清洗后折线
        ax.plot(range(1, len(after_scores)+1), after_scores,
                color=self.colors['after'], linewidth=1.5, alpha=0.7,
                label=f'After (Mean: {np.mean(after_scores):.1f})')

        # 阈值线
        ax.axhline(y=min_quality, color=self.colors['threshold'],
                   linestyle='--', linewidth=2,
                   label=f'Threshold: {min_quality}')

        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Quality Score', fontsize=12)
        ax.set_title('Quality Score: Before vs After Cleaning', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

        # 改进统计
        improvement = np.mean(after_scores) - np.mean(before_scores)
        ax.text(0.02, 0.02,
                f'Improvement: {improvement:.1f}\nBefore: {len(before_scores)} samples\nAfter: {len(after_scores)} samples',
                transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

        plt.tight_layout()
        plot_path = output_path / f'quality_timeline_comparison.{self.plot_format}'
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        generated_plots.append(('quality_timeline_comparison', plot_path))

        # ============================================================
        # 图2: 置信度对比折线图
        # ============================================================
        fig, ax = plt.subplots(figsize=self.default_figsize)

        # 清洗前折线
        ax.plot(range(1, len(before_confidences)+1), before_confidences,
                color=self.colors['before'], linewidth=1.5, alpha=0.7,
                label=f'Before (Mean: {np.mean(before_confidences):.3f})')

        # 清洗后折线
        ax.plot(range(1, len(after_confidences)+1), after_confidences,
                color=self.colors['after'], linewidth=1.5, alpha=0.7,
                label=f'After (Mean: {np.mean(after_confidences):.3f})')

        # 阈值线
        ax.axhline(y=min_confidence, color=self.colors['threshold'],
                   linestyle='--', linewidth=2,
                   label=f'Threshold: {min_confidence}')

        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Confidence Score', fontsize=12)
        ax.set_title('Confidence Score: Before vs After Cleaning', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

        # 改进统计
        improvement = np.mean(after_confidences) - np.mean(before_confidences)
        ax.text(0.02, 0.02,
                f'Improvement: {improvement:.3f}\nRemoval: {len(before_data) - len(after_data)} samples',
                transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

        plt.tight_layout()
        plot_path = output_path / f'confidence_timeline_comparison.{self.plot_format}'
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        generated_plots.append(('confidence_timeline_comparison', plot_path))

        return generated_plots