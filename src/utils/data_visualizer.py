"""
数据可视化类
============

将数据管道的可视化功能独立封装，包括：
- 统计图表（置信度、质量分数、任务分布等）
- 样本可视化（原图+标签+CoT）
- 时间线可视化（分数/置信度折线图）
- 步骤耗时可视化（各步骤耗时对比）
- HTML报告生成

Usage:
    visualizer = DataVisualizer(config)
    visualizer.visualize_all(data_list, output_dir, timing_stats)
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import random

# 可视化相关导入
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    from matplotlib.patches import Rectangle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# 图像处理相关导入
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

import numpy as np


class DataVisualizer:
    """
    数据可视化类

    整合所有可视化功能：
    1. 统计分析图表
    2. 样本可视化（原图+标签）
    3. 时间线可视化（分数/置信度折线）
    4. 步骤耗时可视化
    5. HTML报告生成
    """

    def __init__(self, config: Any, logger: Any = None):
        """
        初始化可视化器

        Args:
            config: 配置管理器
            logger: 日志记录器
        """
        self.config = config
        self.logger = logger

        # 获取可视化配置
        self.viz_config = config.get('visualization', {})

        # 输出设置
        self.output_dir = self.viz_config.get('output_dir', './outputs/visualizations')
        self.plot_format = self.viz_config.get('plot_format', 'png')
        self.dpi = self.viz_config.get('dpi', 150)
        self.figsize = self.viz_config.get('figsize', {'width': 12, 'height': 8})
        self.default_figsize = (self.figsize.get('width', 12), self.figsize.get('height', 8))

        # 设置matplotlib样式
        self._setup_matplotlib_style()

        # 颜色方案
        self.colors = {
            'before': '#FF6B6B',     # 清洗前（红色）
            'after': '#2ECC71',      # 清洗后（绿色）
            'threshold': '#F39C12',  # 阈值线（橙色）
            'passed': '#3498DB',     # 通过（蓝色）
            'failed': '#E74C3C',     # 失败（红色）
        }

        # 步骤耗时颜色映射
        self.step_colors = {
            'data_loading': '#3498DB',
            'preprocessing': '#9B59B6',
            'model_inference': '#E74C3C',
            'initial_validation': '#F39C12',
            'cleaning': '#1ABC9C',
            'final_validation': '#2ECC71',
            'visualization': '#E67E22',
        }

        # 步骤名称映射（英文）
        self.step_names = {
            'data_loading': 'Data Loading',
            'preprocessing': 'Preprocessing',
            'model_inference': 'Model Inference',
            'initial_validation': 'Initial Validation',
            'cleaning': 'Data Cleaning',
            'final_validation': 'Final Validation',
            'visualization': 'Visualization',
        }

        # 无效答案列表
        self.invalid_answers = ['unknown', 'n/a', 'none', 'unclear', 'cannot determine', '']

    def _setup_matplotlib_style(self):
        """设置matplotlib样式"""
        if not HAS_MATPLOTLIB:
            return

        style = self.viz_config.get('style', 'seaborn-v0_8-whitegrid')
        try:
            plt.style.use(style)
        except Exception:
            try:
                plt.style.use('seaborn-v0_8-whitegrid')
            except Exception:
                # 使用默认样式
                pass

    def _log(self, message: str, level: str = 'info'):
        """日志记录"""
        if self.logger:
            if level == 'info':
                self.logger.info(message)
            elif level == 'warning':
                self.logger.warning(message)
            elif level == 'error':
                self.logger.error(message)

    def visualize_all(
        self,
        data_list: List[Dict],
        output_dir: Optional[str] = None,
        before_data: Optional[List[Dict]] = None,
        timing_stats: Optional[Dict[str, float]] = None,
        pipeline_results: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        运行所有可视化

        Args:
            data_list: 当前数据列表（清洗后或蒸馏后）
            output_dir: 输出目录（可选，覆盖配置）
            before_data: 清洗前数据列表（可选）
            timing_stats: 各步骤耗时统计（可选）
            pipeline_results: 流程结果报告（可选）

        Returns:
            可视化报告
        """
        if not HAS_MATPLOTLIB:
            self._log("matplotlib未安装，无法进行可视化", 'warning')
            return {'success': False, 'error': 'matplotlib not installed'}

        # 设置输出目录
        if output_dir:
            self.output_dir = output_dir
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self._log(f"\nVisualization configuration:")
        self._log(f"  - Output: {self.output_dir}")
        self._log(f"  - Format: {self.plot_format}")
        self._log(f"  - Samples to visualize: {len(data_list)}")

        start_time = datetime.now()
        generated_plots = []

        try:
            # 1. 统计分析图表
            self._log("\n[1/6] Generating statistical plots...")
            stats_plots = self._generate_statistical_plots(data_list, output_path, before_data)
            generated_plots.extend(stats_plots)

            # 2. 样本可视化
            self._log("\n[2/6] Generating sample visualizations...")
            sample_plots = self._generate_sample_visualizations(data_list, output_path)
            generated_plots.extend(sample_plots)

            # 3. 时间线可视化
            self._log("\n[3/6] Generating timeline plots...")
            timeline_plots = self._generate_timeline_plots(data_list, output_path, before_data)
            generated_plots.extend(timeline_plots)

            # 4. 步骤耗时可视化
            if timing_stats:
                self._log("\n[4/6] Generating timing visualization...")
                timing_plots = self._generate_timing_plots(timing_stats, output_path)
                generated_plots.extend(timing_plots)

            # 5. 清洗前后对比（如果有）
            if before_data:
                self._log("\n[5/6] Generating before/after comparison...")
                comparison_plots = self._generate_comparison_plots(before_data, data_list, output_path)
                generated_plots.extend(comparison_plots)

            # 6. HTML报告
            self._log("\n[6/6] Generating HTML report...")
            html_path = self._generate_html_report(
                generated_plots, output_path, timing_stats, pipeline_results
            )
            if html_path:
                generated_plots.append(('html_report', html_path))

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # 生成报告
            viz_report = {
                'success': True,
                'output_dir': str(output_path),
                'generated_plots': len(generated_plots),
                'plots': {name: str(path) for name, path in generated_plots},
                'duration_seconds': duration,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
            }

            # 保存报告
            report_path = output_path / 'visualization_report.json'
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(viz_report, f, indent=2, ensure_ascii=False)

            self._log(f"\n✓ Visualization complete: {len(generated_plots)} plots generated")
            self._log(f"  Duration: {duration:.1f} seconds")

            return viz_report

        except Exception as e:
            self._log(f"Visualization failed: {e}", 'error')
            return {'success': False, 'error': str(e)}

    # ============================================================
    # 统计分析图表
    # ============================================================

    def _generate_statistical_plots(
        self,
        data_list: List[Dict],
        output_path: Path,
        before_data: Optional[List[Dict]] = None
    ) -> List[Tuple[str, Path]]:
        """生成统计分析图表"""
        plots = []
        charts_config = self.viz_config.get('charts', {})

        # 置信度分布图
        if charts_config.get('confidence_distribution', {}).get('enabled', True):
            path = self._plot_confidence_distribution(data_list, output_path)
            if path:
                plots.append(('confidence_distribution', path))
                self._log(f"  ✓ confidence_distribution.{self.plot_format}")

        # 质量分数分布图
        if charts_config.get('quality_scores', {}).get('enabled', True):
            path = self._plot_quality_distribution(data_list, output_path)
            if path:
                plots.append(('quality_scores', path))
                self._log(f"  ✓ quality_scores.{self.plot_format}")

        # 任务分布图
        if charts_config.get('task_distribution', {}).get('enabled', True):
            path = self._plot_task_distribution(data_list, output_path)
            if path:
                plots.append(('task_distribution', path))
                self._log(f"  ✓ task_distribution.{self.plot_format}")

        # CoT质量分析
        if charts_config.get('cot_quality', {}).get('enabled', True):
            path = self._plot_cot_quality(data_list, output_path)
            if path:
                plots.append(('cot_quality', path))
                self._log(f"  ✓ cot_quality.{self.plot_format}")

        return plots

    def _plot_confidence_distribution(
        self,
        data_list: List[Dict],
        output_path: Path
    ) -> Optional[Path]:
        """置信度分布图"""
        confidence_values = []
        for data in data_list:
            tasks = data.get('tasks', {})
            for task_name, task_data in tasks.items():
                hard_label = task_data.get('hard_label', {})
                confidence = hard_label.get('confidence')
                if confidence is not None:
                    confidence_values.append({'value': confidence, 'task': task_name})

        if not confidence_values:
            return None

        fig, axes = plt.subplots(1, 2, figsize=self.default_figsize)

        values = [c['value'] for c in confidence_values]
        tasks = [c['task'] for c in confidence_values]

        # 直方图
        bins = self.viz_config.get('charts', {}).get('confidence_distribution', {}).get('bins', 30)
        axes[0].hist(values, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].set_xlabel('Confidence Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Confidence Distribution')

        mean_val = np.mean(values)
        median_val = np.median(values)
        axes[0].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
        axes[0].axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.3f}')
        axes[0].legend()

        # 箱线图
        unique_tasks = sorted(set(tasks))
        task_data_dict = {task: [] for task in unique_tasks}
        for c in confidence_values:
            task_data_dict[c['task']].append(c['value'])

        box_data = [task_data_dict[task] for task in unique_tasks]
        axes[1].boxplot(box_data, labels=unique_tasks)
        axes[1].set_xlabel('Task Type')
        axes[1].set_ylabel('Confidence Score')
        axes[1].set_title('Confidence by Task')

        plt.tight_layout()
        path = output_path / f'confidence_distribution.{self.plot_format}'
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        return path

    def _plot_quality_distribution(
        self,
        data_list: List[Dict],
        output_path: Path
    ) -> Optional[Path]:
        """质量分数分布图"""
        quality_scores = []
        for data in data_list:
            score = data.get('quality_score')
            if score is None:
                score = self._estimate_quality_score(data)
            quality_scores.append(score)

        if not quality_scores:
            return None

        fig, ax = plt.subplots(figsize=self.default_figsize)

        bins = self.viz_config.get('charts', {}).get('quality_scores', {}).get('bins', 20)
        ax.hist(quality_scores, bins=bins, edgecolor='black', alpha=0.7, color='coral')

        ax.set_xlabel('Quality Score (0-100)')
        ax.set_ylabel('Frequency')
        ax.set_title('Quality Score Distribution')

        min_quality = self.config.get('cleaning.min_quality_score', 50.0)
        ax.axvline(min_quality, color='red', linestyle='--', linewidth=2,
                   label=f'Threshold: {min_quality}')
        ax.legend()

        # 质量区间标注
        ax.axvspan(0, 50, alpha=0.1, color='red')
        ax.axvspan(50, 70, alpha=0.1, color='yellow')
        ax.axvspan(70, 100, alpha=0.1, color='green')

        plt.tight_layout()
        path = output_path / f'quality_scores.{self.plot_format}'
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        return path

    def _plot_task_distribution(
        self,
        data_list: List[Dict],
        output_path: Path
    ) -> Optional[Path]:
        """任务分布图"""
        task_counts = {}
        for data in data_list:
            tasks = data.get('tasks', {})
            for task_name in tasks.keys():
                task_counts[task_name] = task_counts.get(task_name, 0) + 1

        if not task_counts:
            return None

        fig, axes = plt.subplots(1, 2, figsize=self.default_figsize)

        labels = list(task_counts.keys())
        sizes = list(task_counts.values())

        # 饼图
        axes[0].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        axes[0].set_title('Task Distribution (Pie)')

        # 柱状图
        axes[1].bar(labels, sizes, color='skyblue', edgecolor='black')
        axes[1].set_xlabel('Task Type')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Task Distribution (Bar)')

        plt.tight_layout()
        path = output_path / f'task_distribution.{self.plot_format}'
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        return path

    def _plot_cot_quality(
        self,
        data_list: List[Dict],
        output_path: Path
    ) -> Optional[Path]:
        """CoT质量分析图"""
        step_counts = []
        logical_flows = []

        for data in data_list:
            tasks = data.get('tasks', {})
            for task_name, task_data in tasks.items():
                cot = task_data.get('cot_reasoning', {})
                if cot:
                    quality_metrics = cot.get('quality_metrics', {})
                    step_count = quality_metrics.get('step_count')
                    if step_count is not None:
                        step_counts.append(step_count)
                    logical_flow = quality_metrics.get('logical_flow_score')
                    if logical_flow is not None:
                        logical_flows.append(logical_flow)

        if not step_counts and not logical_flows:
            return None

        fig, axes = plt.subplots(1, 2, figsize=self.default_figsize)

        # 步骤数分布
        if step_counts:
            unique_steps = sorted(set(step_counts))
            step_freq = [step_counts.count(s) for s in unique_steps]
            axes[0].bar(unique_steps, step_freq, color='teal', edgecolor='black')
            axes[0].set_xlabel('Step Count')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('CoT Step Count Distribution')
            axes[0].axvline(np.mean(step_counts), color='red', linestyle='--',
                           label=f'Avg: {np.mean(step_counts):.1f}')
            axes[0].legend()

        # 逻辑流畅度分布
        if logical_flows:
            axes[1].hist(logical_flows, bins=10, edgecolor='black', alpha=0.7, color='orange')
            axes[1].set_xlabel('Logical Flow Score')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('CoT Logical Flow Distribution')
            axes[1].axvline(np.mean(logical_flows), color='red', linestyle='--',
                           label=f'Avg: {np.mean(logical_flows):.3f}')
            axes[1].legend()

        plt.tight_layout()
        path = output_path / f'cot_quality.{self.plot_format}'
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        return path

    # ============================================================
    # 样本可视化
    # ============================================================

    def _generate_sample_visualizations(
        self,
        data_list: List[Dict],
        output_path: Path
    ) -> List[Tuple[str, Path]]:
        """生成样本可视化"""
        if not HAS_PIL:
            self._log("PIL未安装，跳过样本可视化", 'warning')
            return []

        sample_config = self.viz_config.get('sample_visualization', {})
        max_samples = sample_config.get('max_samples', 10)

        if not sample_config.get('enabled', True):
            return []

        # 随机选择样本
        random.seed(42)
        selected = random.sample(data_list, min(max_samples, len(data_list)))

        plots = []
        images_root = Path(self.config.get('data.images_root', './data/coco/val2014'))

        self._log(f"  Visualizing {len(selected)} samples...")

        for idx, data in enumerate(selected):
            path = self._visualize_single_sample(data, output_path, images_root, idx)
            if path:
                plots.append((f'sample_{idx}', path))

        return plots

    def _visualize_single_sample(
        self,
        data: Dict,
        output_path: Path,
        images_root: Path,
        idx: int
    ) -> Optional[Path]:
        """可视化单个样本"""
        sample_config = self.viz_config.get('sample_visualization', {})

        try:
            # 加载图像
            image_id = data.get('image_id')
            file_name = data.get('file_name', '')
            coco_url = data.get('coco_url', '')

            image_path = None
            if file_name:
                image_path = images_root / file_name
            elif image_id:
                image_path = images_root / f"COCO_val2014_{str(image_id).zfill(12)}.jpg"
            elif coco_url:
                image_path = images_root / Path(coco_url).name

            if image_path and image_path.exists():
                img = Image.open(image_path)
                img_array = np.array(img)
            else:
                img_array = np.ones((400, 600, 3), dtype=np.uint8) * 200

            # 创建可视化图
            figsize = sample_config.get('figsize', {'width': 16, 'height': 12})
            fig = plt.figure(figsize=(figsize.get('width', 16), figsize.get('height', 12)))

            gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1], height_ratios=[1, 1],
                                  hspace=0.3, wspace=0.3)

            # 左侧：原图
            ax_img = fig.add_subplot(gs[:, 0])
            ax_img.imshow(img_array)
            ax_img.set_title(f'Sample {idx+1}: Image ID {image_id}', fontsize=14, fontweight='bold')
            ax_img.axis('off')

            # 绘制检测框
            if sample_config.get('show_bbox', True):
                self._draw_detection_boxes(ax_img, data)

            # 右侧：标签信息
            ax_info = fig.add_subplot(gs[0, 1:])
            ax_info.axis('off')

            info_text = self._format_labels_info(data, sample_config)
            ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
            ax_info.set_title('Hard & Soft Labels', fontsize=12, fontweight='bold')

            # CoT信息
            ax_cot = fig.add_subplot(gs[1, 1:])
            ax_cot.axis('off')

            cot_text = self._format_cot_info(data, sample_config)
            ax_cot.text(0.05, 0.95, cot_text, transform=ax_cot.transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
            ax_cot.set_title('Chain-of-Thought', fontsize=12, fontweight='bold')

            plt.tight_layout()
            path = output_path / f'sample_{idx+1}.{self.plot_format}'
            plt.savefig(path, dpi=sample_config.get('dpi', 100), bbox_inches='tight')
            plt.close()

            self._log(f"    ✓ Sample {idx+1}")
            return path

        except Exception as e:
            self._log(f"    ✗ Sample {idx}: {e}", 'warning')
            plt.close()
            return None

    def _draw_detection_boxes(self, ax: plt.Axes, data: Dict):
        """绘制检测框"""
        bbox_colors = self.viz_config.get('sample_visualization', {}).get('bbox_colors', {
            'person': '#FF6B6B', 'car': '#4ECDC4', 'default': '#3498DB'
        })

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
                        color = bbox_colors.get(category.lower(), bbox_colors.get('default', '#3498DB'))
                        rect = Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                                          linewidth=2, edgecolor=color, facecolor='none')
                        ax.add_patch(rect)
                        ax.text(bbox[0], bbox[1]-5, f"{category}: {confidence:.0%}",
                               fontsize=9, color=color, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    def _format_labels_info(self, data: Dict, config: Dict) -> str:
        """格式化标签信息"""
        info_texts = []
        tasks = data.get('tasks', {})

        for task_name, task_data in tasks.items():
            # 硬标签
            if config.get('show_hard_label', True):
                hard_label = task_data.get('hard_label', {})
                if hard_label:
                    answer = hard_label.get('answer', 'N/A')
                    confidence = hard_label.get('confidence', 0)
                    info_texts.append(f"[{task_name.upper()}] Hard:\n  Answer: {answer}\n  Conf: {confidence:.2%}")

            # 软标签
            if config.get('show_soft_label', True):
                soft_label = task_data.get('soft_label', {})
                if soft_label:
                    distribution = soft_label.get('answer_distribution', {})
                    if distribution:
                        top_k = config.get('soft_label_display', {}).get('top_k', 5)
                        sorted_items = sorted(distribution.items(), key=lambda x: x[1], reverse=True)[:top_k]
                        prob_text = "\n  Top probs:\n"
                        for ans, prob in sorted_items:
                            prob_text += f"    {ans}: {prob:.2%}\n"
                        info_texts.append(f"[{task_name.upper()}] Soft:{prob_text}")

        return "\n\n".join(info_texts) if info_texts else "No label info"

    def _format_cot_info(self, data: Dict, config: Dict) -> str:
        """格式化CoT信息"""
        cot_texts = []
        tasks = data.get('tasks', {})
        max_length = config.get('cot_display', {}).get('max_length', 200)

        for task_name, task_data in tasks.items():
            if config.get('show_cot', True):
                cot = task_data.get('cot_reasoning', {})
                if cot:
                    reasoning = cot.get('raw_reasoning', '')
                    if reasoning:
                        if len(reasoning) > max_length:
                            reasoning = reasoning[:max_length] + "..."
                        cot_texts.append(f"[{task_name.upper()}] CoT:\n{reasoning}")

        return "\n\n".join(cot_texts) if cot_texts else "No CoT reasoning"

    # ============================================================
    # 时间线可视化
    # ============================================================

    def _generate_timeline_plots(
        self,
        data_list: List[Dict],
        output_path: Path,
        before_data: Optional[List[Dict]] = None
    ) -> List[Tuple[str, Path]]:
        """生成时间线可视化"""
        plots = []
        timeline_config = self.viz_config.get('timeline_visualization', {})

        if not timeline_config.get('enabled', True):
            return plots

        # 当前数据时间线
        timeline_plots = self._plot_timeline(data_list, output_path, "after")
        plots.extend(timeline_plots)

        # 清洗前数据时间线
        if before_data:
            before_plots = self._plot_timeline(before_data, output_path, "before")
            plots.extend([(f'{name}_before', path) for name, path in before_plots])

        return plots

    def _plot_timeline(
        self,
        data_list: List[Dict],
        output_path: Path,
        label: str = "after"
    ) -> List[Tuple[str, Path]]:
        """绘制时间线"""
        plots = []
        label_text = "Before Cleaning" if label == "before" else "After Cleaning"
        label_suffix = "_before" if label == "before" else ""

        min_quality = self.config.get('cleaning.min_quality_score', 50.0)
        min_confidence = self.config.get('cleaning.min_confidence', 0.6)

        # 收集数据
        quality_scores = []
        confidence_values = []
        for data in data_list:
            score = data.get('quality_score') or self._estimate_quality_score(data)
            quality_scores.append(score)

            tasks = data.get('tasks', {})
            confs = [t.get('hard_label', {}).get('confidence', 0.5) for t in tasks.values()]
            confidence_values.append(np.mean(confs) if confs else 0.5)

        if not quality_scores:
            return plots

        image_indices = range(1, len(quality_scores) + 1)

        # 质量分数时间线
        fig, ax = plt.subplots(figsize=(self.default_figsize[0], self.default_figsize[1] * 0.7))

        colors = ['green' if q >= min_quality else 'red' for q in quality_scores]
        ax.plot(image_indices, quality_scores, 'b-', linewidth=1.5, alpha=0.7)
        ax.scatter(image_indices, quality_scores, c=colors, s=20, alpha=0.6)
        ax.axhline(y=min_quality, color='red', linestyle='--', linewidth=2, label=f'Threshold: {min_quality}')
        ax.fill_between(image_indices, min_quality, 100, alpha=0.1, color='green')
        ax.fill_between(image_indices, 0, min_quality, alpha=0.1, color='red')

        passed = len([q for q in quality_scores if q >= min_quality])
        failed = len([q for q in quality_scores if q < min_quality])

        stats_text = f"[{label_text}]\nTotal: {len(quality_scores)}\nPassed: {passed} ({passed/len(quality_scores)*100:.1f}%)\nFailed: {failed}\nAvg: {np.mean(quality_scores):.1f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Quality Score')
        ax.set_title(f'Quality Score Timeline ({label_text})', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = output_path / f'quality_timeline{label_suffix}.{self.plot_format}'
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        plots.append(('quality_timeline', path))

        # 置信度时间线
        fig, ax = plt.subplots(figsize=(self.default_figsize[0], self.default_figsize[1] * 0.7))

        colors = ['green' if c >= min_confidence else 'orange' for c in confidence_values]
        ax.plot(image_indices, confidence_values, 'purple', linewidth=1.5, alpha=0.7)
        ax.scatter(image_indices, confidence_values, c=colors, s=20, alpha=0.6)
        ax.axhline(y=min_confidence, color='orange', linestyle='--', linewidth=2, label=f'Threshold: {min_confidence}')
        ax.fill_between(image_indices, min_confidence, 1.0, alpha=0.1, color='green')
        ax.fill_between(image_indices, 0, min_confidence, alpha=0.1, color='orange')

        stats_text = f"[{label_text}]\nAvg: {np.mean(confidence_values):.3f}\nMin: {min(confidence_values):.3f}\nMax: {max(confidence_values):.3f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))

        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Confidence')
        ax.set_title(f'Confidence Timeline ({label_text})', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = output_path / f'confidence_timeline{label_suffix}.{self.plot_format}'
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        plots.append(('confidence_timeline', path))

        return plots

    def _generate_comparison_plots(
        self,
        before_data: List[Dict],
        after_data: List[Dict],
        output_path: Path
    ) -> List[Tuple[str, Path]]:
        """生成清洗前后对比图"""
        plots = []

        before_scores = [d.get('quality_score') or self._estimate_quality_score(d) for d in before_data]
        after_scores = [d.get('quality_score') or self._estimate_quality_score(d) for d in after_data]

        before_confs = []
        for d in before_data:
            tasks = d.get('tasks', {})
            confs = [t.get('hard_label', {}).get('confidence', 0.5) for t in tasks.values()]
            before_confs.append(np.mean(confs) if confs else 0.5)

        after_confs = []
        for d in after_data:
            tasks = d.get('tasks', {})
            confs = [t.get('hard_label', {}).get('confidence', 0.5) for t in tasks.values()]
            after_confs.append(np.mean(confs) if confs else 0.5)

        # 对比折线图
        fig, axes = plt.subplots(2, 1, figsize=(self.default_figsize[0], self.default_figsize[1] * 1.2))

        ax1 = axes[0]
        ax1.plot(range(1, len(before_scores)+1), before_scores, 'r-', linewidth=1.5, alpha=0.6,
                label=f'Before ({len(before_scores)} samples)')
        ax1.plot(range(1, len(after_scores)+1), after_scores, 'g-', linewidth=2, alpha=0.8,
                label=f'After ({len(after_scores)} samples)')
        ax1.axhline(y=self.config.get('cleaning.min_quality_score', 50.0), color='orange',
                   linestyle='--', linewidth=2, label='Threshold')

        stats = f"Before: Avg={np.mean(before_scores):.1f}\nAfter: Avg={np.mean(after_scores):.1f}\nRemoved: {len(before_scores)-len(after_scores)}"
        ax1.text(0.02, 0.98, stats, transform=ax1.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax1.set_title('Quality Score: Before vs After', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.plot(range(1, len(before_confs)+1), before_confs, 'r-', linewidth=1.5, alpha=0.6, label='Before')
        ax2.plot(range(1, len(after_confs)+1), after_confs, 'g-', linewidth=2, alpha=0.8, label='After')
        ax2.axhline(y=self.config.get('cleaning.min_confidence', 0.6), color='orange',
                   linestyle='--', linewidth=2, label='Threshold')

        ax2.set_title('Confidence: Before vs After', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        path = output_path / f'before_after_comparison.{self.plot_format}'
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        plots.append(('before_after_comparison', path))

        # 统计柱状图
        fig, ax = plt.subplots(figsize=(self.default_figsize[0], self.default_figsize[1] * 0.6))

        categories = ['Total', 'Avg Quality', 'Avg Conf*100', 'Removed']
        before_vals = [len(before_scores), np.mean(before_scores), np.mean(before_confs)*100, 0]
        after_vals = [len(after_scores), np.mean(after_scores), np.mean(after_confs)*100, len(before_scores)-len(after_scores)]

        x = np.arange(len(categories))
        width = 0.35

        ax.bar(x - width/2, before_vals, width, label='Before', color='salmon', edgecolor='black')
        ax.bar(x + width/2, after_vals, width, label='After', color='lightgreen', edgecolor='black')

        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_title('Before vs After Statistics', fontweight='bold')
        ax.legend()

        survival = len(after_scores)/len(before_scores)*100 if before_scores else 0
        ax.text(0.98, 0.98, f'Survival: {survival:.1f}%\nRemoved: {100-survival:.1f}%',
               transform=ax.transAxes, fontsize=11, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        plt.tight_layout()
        path = output_path / f'before_after_stats.{self.plot_format}'
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        plots.append(('before_after_stats', path))

        return plots

    # ============================================================
    # 步骤耗时可视化
    # ============================================================

    def _generate_timing_plots(
        self,
        timing_stats: Dict[str, float],
        output_path: Path
    ) -> List[Tuple[str, Path]]:
        """生成步骤耗时可视化"""
        plots = []

        if not timing_stats:
            return plots

        # 确保所有步骤都有值（缺失的设为0）
        all_steps = ['data_loading', 'preprocessing', 'model_inference',
                    'initial_validation', 'cleaning', 'final_validation', 'visualization']

        for step in all_steps:
            if step not in timing_stats:
                timing_stats[step] = 0.0

        # 按步骤顺序排列
        ordered_steps = all_steps
        ordered_times = [timing_stats[step] for step in ordered_steps]
        ordered_names = [self.step_names.get(step, step) for step in ordered_steps]
        ordered_colors = [self.step_colors.get(step, '#CCCCCC') for step in ordered_steps]

        # ============================================================
        # 图1: 步骤耗时折线图
        # ============================================================
        fig, axes = plt.subplots(2, 1, figsize=(self.default_figsize[0], self.default_figsize[1] * 1.2))

        # 子图1: 折线图
        ax1 = axes[0]
        x_indices = range(1, len(ordered_steps) + 1)

        ax1.plot(x_indices, ordered_times, 'b-', linewidth=2, marker='o', markersize=8)
        ax1.scatter(x_indices, ordered_times, c=ordered_colors, s=100, zorder=5)

        # 标注每个点的时间
        for idx, (time_val, name) in zip(x_indices, zip(ordered_times, ordered_names)):
            ax1.annotate(f'{time_val:.1f}s', (idx, time_val),
                        textcoords="offset points", xytext=(0, 10),
                        ha='center', fontsize=10)

        # 平均耗时线
        avg_time = np.mean(ordered_times)
        ax1.axhline(y=avg_time, color='red', linestyle='--', linewidth=2,
                   label=f'Average: {avg_time:.1f}s')

        ax1.set_xticks(x_indices)
        ax1.set_xticklabels(ordered_names, rotation=15, ha='right')
        ax1.set_xlabel('Pipeline Step')
        ax1.set_ylabel('Duration (seconds)')
        ax1.set_title('Pipeline Step Duration Timeline', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # 统计信息框
        total_time = sum(ordered_times)
        stats_text = (
            f"Timing Statistics:\n"
            f"Total: {total_time:.1f}s\n"
            f"Average: {avg_time:.1f}s\n"
            f"Max: {max(ordered_times):.1f}s ({ordered_names[np.argmax(ordered_times)]})\n"
            f"Min: {min(ordered_times):.1f}s ({ordered_names[np.argmin(ordered_times)]})"
        )
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        # 子图2: 柱状图（累计耗时）
        ax2 = axes[1]

        bars = ax2.bar(ordered_names, ordered_times, color=ordered_colors, edgecolor='black')

        # 标注数值
        for bar, time_val in zip(bars, ordered_times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{time_val:.1f}s', ha='center', va='bottom', fontsize=10)

        # 累计耗时线
        cumulative = np.cumsum(ordered_times)
        ax2.plot(ordered_names, cumulative, 'r--', linewidth=2, marker='s',
                label=f'Cumulative: {total_time:.1f}s')

        ax2.set_xlabel('Pipeline Step')
        ax2.set_ylabel('Duration (seconds)')
        ax2.set_title('Pipeline Step Duration (Cumulative)', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.tick_params(axis='x', rotation=15)

        plt.tight_layout()
        path = output_path / f'step_timing.{self.plot_format}'
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        plots.append(('step_timing', path))

        # ============================================================
        # 图2: 耗时占比饼图
        # ============================================================
        fig, axes = plt.subplots(1, 2, figsize=self.default_figsize)

        # 过滤掉耗时为0的步骤
        non_zero_steps = [(s, t) for s, t in zip(ordered_names, ordered_times) if t > 0]
        if non_zero_steps:
            pie_names = [s for s, t in non_zero_steps]
            pie_times = [t for s, t in non_zero_steps]
            pie_colors = [ordered_colors[ordered_names.index(s)] for s in pie_names]

            # 饼图
            axes[0].pie(pie_times, labels=pie_names, colors=pie_colors, autopct='%1.1f%%',
                      startangle=90, explode=[0.02]*len(pie_times))
            axes[0].set_title('Time Distribution (Pie)', fontweight='bold')

            # 时间占比表格
            axes[1].axis('off')
            table_data = [[name, f"{time:.1f}s", f"{time/total_time*100:.1f}%"]
                         for name, time in non_zero_steps]
            table_data.append(['Total', f"{total_time:.1f}s", "100%"])

            axes[1].table(cellText=table_data,
                         colLabels=['Step', 'Duration', 'Percentage'],
                         loc='center', cellLoc='center',
                         colWidths=[0.4, 0.3, 0.3])
            axes[1].set_title('Time Distribution (Table)', fontweight='bold')

        plt.tight_layout()
        path = output_path / f'time_distribution.{self.plot_format}'
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        plots.append(('time_distribution', path))

        return plots

    # ============================================================
    # 质量分数估算（与DataCleaner对齐）
    # ============================================================

    def _estimate_quality_score(self, data: Dict) -> float:
        """估算质量分数"""
        score = 0.0
        tasks = data.get('tasks', {})
        task_scores = []

        for task_name, task_data in tasks.items():
            task_score = 0.0

            # 硬标签 (0-40分)
            hard_label = task_data.get('hard_label', {})
            if hard_label:
                confidence = hard_label.get('confidence', 0.0)
                if confidence >= 0.7:
                    task_score += 30
                elif confidence >= 0.5:
                    task_score += 20
                elif confidence >= 0.3:
                    task_score += 10

                answer = hard_label.get('answer', '')
                if 3 <= len(answer) <= 100:
                    task_score += 10

            # 软标签 (0-20分)
            soft_label = task_data.get('soft_label', {})
            if soft_label:
                temperature = soft_label.get('temperature', 0.0)
                if 1.5 <= temperature <= 3.0:
                    task_score += 10
                elif 1.0 <= temperature <= 5.0:
                    task_score += 5

                distribution = soft_label.get('answer_distribution', {})
                if distribution:
                    task_score += 10

            # CoT (0-30分)
            cot = task_data.get('cot_reasoning', {})
            if cot:
                quality_metrics = cot.get('quality_metrics', {})
                logical_flow = quality_metrics.get('logical_flow_score', 0.0)
                task_score += logical_flow * 15

                step_count = quality_metrics.get('step_count', 0)
                if 3 <= step_count <= 5:
                    task_score += 15
                elif 2 <= step_count <= 6:
                    task_score += 10
                elif step_count > 0:
                    task_score += step_count * 2

            task_scores.append(task_score)

        if task_scores:
            return min(np.mean(task_scores), 100.0)

        return 10.0

    # ============================================================
    # HTML报告生成
    # ============================================================

    def _generate_html_report(
        self,
        generated_plots: List[Tuple[str, Path]],
        output_path: Path,
        timing_stats: Optional[Dict[str, float]] = None,
        pipeline_results: Optional[Dict] = None
    ) -> Optional[Path]:
        """生成HTML可视化报告"""

        # 分离样本可视化和其他图表
        sample_plots = [(n, p) for n, p in generated_plots if n.startswith('sample_')]
        other_plots = [(n, p) for n, p in generated_plots if not n.startswith('sample_') and n != 'html_report']

        html_content = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VLM Data Pipeline Visualization Report</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; background-color: #f5f5f5; }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        .plot-container { background: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); padding: 15px; margin: 20px 0; }
        .plot-container img { max-width: 100%; height: auto; display: block; margin: 0 auto; }
        .plot-title { font-size: 18px; color: #2c3e50; margin-bottom: 10px; }
        .summary-box { background: #3498db; color: white; padding: 15px; border-radius: 8px; margin: 20px 0; }
        .timestamp { color: #7f8c8d; font-size: 14px; }
        .sample-section { background: #fff; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); padding: 20px; margin: 30px 0; }
        .sample-section h2 { color: #e74c3c; border-bottom: 2px solid #e74c3c; padding-bottom: 8px; }
        .sample-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(600px, 1fr)); gap: 20px; margin-top: 20px; }
        .sample-item { background: #fafafa; border-radius: 8px; padding: 10px; border: 1px solid #ddd; }
        .sample-item img { width: 100%; height: auto; border-radius: 4px; }
        .sample-item .sample-label { text-align: center; padding: 8px; font-size: 14px; color: #2c3e50; font-weight: bold; }
        .timing-section { background: #fff3cd; border-radius: 8px; padding: 15px; margin: 20px 0; }
        .timing-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        .timing-table th, .timing-table td { border: 1px solid #ddd; padding: 8px; text-align: center; }
        .timing-table th { background-color: #f39c12; color: white; }
    </style>
</head>
<body>
    <h1>VLM Data Pipeline Visualization Report</h1>
    <p class="timestamp">Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>

    <div class="summary-box">
        <h3>Summary</h3>
        <p>Total plots: """ + str(len(generated_plots)) + """ | Samples: """ + str(len(sample_plots)) + """</p>
    </div>
"""

        # 添加耗时信息
        if timing_stats:
            total_time = sum(timing_stats.values())
            html_content += """
    <div class="timing-section">
        <h2>Pipeline Timing Summary</h2>
        <p>Total duration: """ + f"{total_time:.1f}s" + """</p>
        <table class="timing-table">
            <tr><th>Step</th><th>Duration</th><th>Percentage</th></tr>
"""
            for step, time_val in timing_stats.items():
                step_name = self.step_names.get(step, step)
                percentage = time_val / total_time * 100 if total_time > 0 else 0
                html_content += f"            <tr><td>{step_name}</td><td>{time_val:.1f}s</td><td>{percentage:.1f}%</td></tr>\n"
            html_content += """        </table>
    </div>
"""

        # 统计图表
        if other_plots:
            html_content += """    <h2>Statistical Analysis</h2>\n"""
            for plot_name, plot_path in other_plots:
                if plot_path.suffix in ['.png', '.svg', '.jpg', '.jpeg']:
                    title = plot_name.replace('_', ' ').title()
                    html_content += f"""
    <div class="plot-container">
        <h3 class="plot-title">{title}</h3>
        <img src="{plot_path.name}" alt="{title}">
    </div>
"""

        # 样本可视化
        if sample_plots:
            html_content += """
    <div class="sample-section">
        <h2>Sample Visualizations</h2>
        <div class="sample-grid">
"""
            for plot_name, plot_path in sample_plots:
                num = plot_name.replace('sample_', '')
                html_content += f"""
            <div class="sample-item">
                <img src="{plot_path.name}" alt="Sample {num}">
                <div class="sample-label">Sample {num}</div>
            </div>
"""
            html_content += """        </div>
    </div>
"""

        html_content += """
</body>
</html>
"""

        html_path = output_path / 'visualization_report.html'
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return html_path