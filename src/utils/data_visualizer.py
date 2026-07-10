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
            'quality': '#16A085',      # 新增：质量校验颜色（绿色）
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
        pipeline_results: Optional[Dict] = None,
        quality_validation_results: Optional[Dict] = None
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
            # 1. Quality validation visualization (新增)
            if quality_validation_results:
                self._log("\n[1/5] Quality validation visualization...")
                paths = self._plot_quality_validation(quality_validation_results, output_path)
                generated_plots.extend(paths)
                self._log(f"  OK: {len(paths)} quality plots")

            # 2. Score comparison
            if before_data and data_list:
                self._log("\n[2/5] Score comparison...")
                path = self._plot_score_comparison(before_data, data_list, output_path)
                if path:
                    generated_plots.append(('score_comparison', path))
                    self._log(f"  OK: score_comparison.{self.plot_format}")

            # 3. Sample visualization
            if data_list:
                self._log("\n[3/5] Sample visualization...")
                paths = self._visualize_samples(data_list, output_path)
                generated_plots.extend([('sample', p) for p in paths])
                self._log(f"  OK: {len(paths)} samples")

            # 4. Pipeline timing (with distillation breakdown)
            if timing_stats:
                self._log("\n[4/5] Pipeline timing...")
                path = self._plot_pipeline_timing(timing_stats, output_path)
                if path:
                    generated_plots.append(('pipeline_timing', path))
                    self._log(f"  OK: pipeline_timing.{self.plot_format}")
                else:
                    self._log("  No timing data", 'warning')

            # 5. Generate unified Markdown report (新增)
            self._log("\n[5/5] Generating Markdown report...")
            md_path = self._generate_unified_html_report(
                generated_plots,
                quality_validation_results,
                timing_stats,
                pipeline_results,
                output_path
            )
            if md_path:
                generated_plots.append(('unified_report', md_path))
                self._log(f"  OK: visualization_report.md")

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            return {
                'success': True,
                'output_dir': str(output_path),
                'generated_plots': len(generated_plots),
                'plots': {name: str(path) for name, path in generated_plots},
                'duration_seconds': duration,
                'markdown_report': str(md_path) if md_path else None,
            }

        except Exception as e:
            self._log(f"Failed: {e}", 'warning')
            return {'success': False, 'error': str(e)}

    # ============================================================
    # Quality Validation Visualization (新增)
    # ============================================================

    def _plot_quality_validation(
        self,
        validation_results: Dict[str, Any],
        output_path: Path
    ) -> List[Tuple[str, Path]]:
        """绘制数据质量校验的5个关键指标可视化"""
        plots = []

        if not HAS_MATPLOTLIB:
            return plots

        results = validation_results.get('validation_results', {})

        # 1. Top-K匹配率
        if 'top_k_matching' in results:
            path = self._plot_top_k_matching(results['top_k_matching'], output_path)
            if path:
                plots.append(('top_k_matching', path))

        # 2. KL散度分析
        if 'soft_label_distribution' in results:
            path = self._plot_kl_divergence(
                results['soft_label_distribution'].get('kl_divergence_analysis', {}),
                output_path
            )
            if path:
                plots.append(('kl_divergence', path))

        # 3. 类别分布对齐
        if 'soft_label_distribution' in results:
            path = self._plot_distribution_alignment(
                results['soft_label_distribution'].get('category_distribution_alignment', {}),
                output_path
            )
            if path:
                plots.append(('distribution_alignment', path))

        # 4. ECE校准误差
        if 'ece_calibration' in results:
            path = self._plot_ece_calibration(results['ece_calibration'], output_path)
            if path:
                plots.append(('ece_calibration', path))

        # 5. CoT质量分析
        if 'cot_quality' in results:
            path = self._plot_cot_quality(results['cot_quality'], output_path)
            if path:
                plots.append(('cot_quality', path))

        # 6. 综合评分雷达图
        if 'training_value_assessment' in results:
            path = self._plot_overall_assessment(results['training_value_assessment'], output_path)
            if path:
                plots.append(('overall_assessment', path))

        return plots

    def _plot_top_k_matching(self, data: Dict, output_path: Path) -> Optional[Path]:
        """绘制Top-K匹配率"""
        stats = data.get('statistics', {})
        match_rate = stats.get('match_rate', 0) * 100
        matched = stats.get('matched_count', 0)
        unmatched = stats.get('unmatched_count', 0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 左图：匹配率柱状图
        threshold = 88
        color = '#2ECC71' if match_rate >= threshold else '#E74C3C'
        ax1.bar(['Top-K Match Rate'], [match_rate], color=color, width=0.6, edgecolor='black', linewidth=2)
        ax1.axhline(y=threshold, color='#F39C12', linestyle='--', linewidth=2, label=f'Threshold ({threshold}%)')
        ax1.set_ylabel('Match Rate (%)', fontsize=12)
        ax1.set_ylim(0, 100)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 添加数值标签
        ax1.text(0, match_rate + 2, f'{match_rate:.1f}%', ha='center', fontsize=14, fontweight='bold')

        # 右图：匹配vs未匹配
        ax2.pie([matched, unmatched], labels=['Matched', 'Unmatched'],
                colors=['#2ECC71', '#E74C3C'], autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 12})
        ax2.set_title(f'Sample Matching (Total: {matched + unmatched})', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path / f'top_k_matching.{self.plot_format}', dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return output_path / f'top_k_matching.{self.plot_format}'

    def _plot_kl_divergence(self, data: Dict, output_path: Path) -> Optional[Path]:
        """绘制KL散度分析"""
        stats = data.get('statistics', {})
        avg_kl = stats.get('average_kl', 0)
        high_kl_ratio = stats.get('high_kl_ratio', 0) * 100

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 左图：平均KL散度
        threshold = 0.5
        color = '#2ECC71' if avg_kl < threshold else '#E74C3C'
        ax1.bar(['Avg KL Divergence'], [avg_kl], color=color, width=0.6, edgecolor='black', linewidth=2)
        ax1.axhline(y=threshold, color='#F39C12', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
        ax1.set_ylabel('KL Divergence', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.text(0, avg_kl + 0.02, f'{avg_kl:.3f}', ha='center', fontsize=14, fontweight='bold')

        # 右图：高KL样本比例
        ax2.bar(['High KL Ratio'], [high_kl_ratio], color='#E74C3C', width=0.6, edgecolor='black', linewidth=2)
        ax2.axhline(y=20, color='#F39C12', linestyle='--', linewidth=2, label='Threshold (20%)')
        ax2.set_ylabel('Ratio (%)', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.text(0, high_kl_ratio + 1, f'{high_kl_ratio:.1f}%', ha='center', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path / f'kl_divergence.{self.plot_format}', dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return output_path / f'kl_divergence.{self.plot_format}'

    def _plot_distribution_alignment(self, data: Dict, output_path: Path) -> Optional[Path]:
        """绘制类别分布对齐"""
        stats = data.get('statistics', {})
        correlation = stats.get('correlation', 0)

        fig, ax = plt.subplots(figsize=(10, 6))

        threshold = 0.8
        color = '#2ECC71' if correlation > threshold else '#E74C3C'
        ax.bar(['Distribution Correlation'], [correlation], color=color, width=0.6, edgecolor='black', linewidth=2)
        ax.axhline(y=threshold, color='#F39C12', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
        ax.set_ylabel('Correlation Coefficient', fontsize=12)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.text(0, correlation + 0.02, f'{correlation:.3f}', ha='center', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path / f'distribution_alignment.{self.plot_format}', dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return output_path / f'distribution_alignment.{self.plot_format}'

    def _plot_ece_calibration(self, data: Dict, output_path: Path) -> Optional[Path]:
        """绘制ECE校准误差"""
        ece = data.get('ece', 0)
        stats = data.get('statistics', {})

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 左图：ECE值
        threshold = 0.15
        color = '#2ECC71' if ece < threshold else '#E74C3C'
        ax1.bar(['ECE Calibration Error'], [ece], color=color, width=0.6, edgecolor='black', linewidth=2)
        ax1.axhline(y=threshold, color='#F39C12', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
        ax1.set_ylabel('ECE Value', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.text(0, ece + 0.005, f'{ece:.3f}', ha='center', fontsize=14, fontweight='bold')

        # 右图：置信度vs准确率（模拟校准曲线）
        confidence = stats.get('average_confidence', 0.8)
        accuracy = stats.get('accuracy', 0.8)

        x = np.linspace(0, 1, 11)
        ax2.plot(x, x, 'k--', linewidth=2, label='Perfect Calibration')
        ax2.scatter([confidence], [accuracy], s=200, c='#3498DB', edgecolor='black', linewidth=2,
                   label=f'Current Status', zorder=5)
        ax2.set_xlabel('Confidence', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Calibration Curve', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path / f'ece_calibration.{self.plot_format}', dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return output_path / f'ece_calibration.{self.plot_format}'

    def _plot_cot_quality(self, data: Dict, output_path: Path) -> Optional[Path]:
        """绘制CoT质量分析"""
        halluc_stats = data.get('hallucination_detection', {}).get('statistics', {})
        repeat_stats = data.get('repetition_analysis', {}).get('statistics', {})
        length_stats = data.get('length_distribution', {}).get('statistics', {})

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 幻觉占比
        ax1 = axes[0, 0]
        halluc_ratio = halluc_stats.get('hallucination_ratio', 0) * 100
        threshold = 5
        color = '#2ECC71' if halluc_ratio < threshold else '#E74C3C'
        ax1.bar(['Hallucination Rate'], [halluc_ratio], color=color, width=0.6, edgecolor='black', linewidth=2)
        ax1.axhline(y=threshold, color='#F39C12', linestyle='--', linewidth=2, label=f'Threshold ({threshold}%)')
        ax1.set_ylabel('Rate (%)', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.text(0, halluc_ratio + 0.5, f'{halluc_ratio:.1f}%', ha='center', fontsize=14, fontweight='bold')

        # 2. 重复度
        ax2 = axes[0, 1]
        repeat_ratio = repeat_stats.get('repetition_ratio', 0) * 100
        threshold = 30
        color = '#2ECC71' if repeat_ratio < threshold else '#E74C3C'
        ax2.bar(['Repetition Rate'], [repeat_ratio], color=color, width=0.6, edgecolor='black', linewidth=2)
        ax2.axhline(y=threshold, color='#F39C12', linestyle='--', linewidth=2, label=f'Threshold ({threshold}%)')
        ax2.set_ylabel('Ratio (%)', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.text(0, repeat_ratio + 1, f'{repeat_ratio:.1f}%', ha='center', fontsize=14, fontweight='bold')

        # 3. 长度分布
        ax3 = axes[1, 0]
        avg_length = length_stats.get('average_length', 0)
        short_ratio = length_stats.get('short_ratio', 0) * 100
        long_ratio = length_stats.get('long_ratio', 0) * 100

        x = ['Avg Length', 'Short Ratio', 'Long Ratio']
        values = [avg_length, short_ratio, long_ratio]
        colors = ['#3498DB', '#F39C12', '#E74C3C']
        bars = ax3.bar(x, values, color=colors, width=0.6, edgecolor='black', linewidth=2)
        ax3.set_ylabel('值', fontsize=12)
        ax3.grid(True, alpha=0.3)

        for bar, val in zip(bars, values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}', ha='center', fontsize=12, fontweight='bold')

        # 4. CoT样本总数
        ax4 = axes[1, 1]
        total = halluc_stats.get('total_cot_count', 0)
        halluc_count = halluc_stats.get('hallucination_count', 0)
        normal_count = total - halluc_count

        ax4.pie([normal_count, halluc_count], labels=['Normal', 'Hallucination'],
                colors=['#2ECC71', '#E74C3C'], autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 12})
        ax4.set_title(f'CoT Sample Distribution (Total: {total})', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path / f'cot_quality.{self.plot_format}', dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return output_path / f'cot_quality.{self.plot_format}'

    def _plot_overall_assessment(self, data: Dict, output_path: Path) -> Optional[Path]:
        """绘制综合评分雷达图"""
        conditions = data.get('conditions', {})
        can_train = data.get('can_train', False)

        # 准备雷达图数据
        labels = ['Top-K Match', 'KL Div\n(Inverse)', 'Hallucination\n(Inverse)', 'Dist Correlation']
        values = []

        # 提取并归一化值
        for key in ['top1_match_rate', 'kl_divergence', 'hallucination_ratio', 'distribution_alignment']:
            if key in conditions:
                val = conditions[key].get('value', 0)
                if key == 'kl_divergence' or key == 'hallucination_ratio':
                    # 反向指标（越小越好）
                    values.append(max(0, 1 - val))
                else:
                    values.append(min(val, 1))
            else:
                values.append(0)

        # 创建雷达图
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        values_plot = values + values[:1]  # 闭合
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

        ax.plot(angles, values_plot, 'o-', linewidth=2, color='#3498DB')
        ax.fill(angles, values_plot, alpha=0.25, color='#3498DB')

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0, 1)

        # 添加标题
        status = '✓ Pass' if can_train else '✗ Fail'
        color = '#2ECC71' if can_train else '#E74C3C'
        ax.set_title(f'Data Quality Overall Assessment: {status}', fontsize=14, fontweight='bold', color=color, pad=20)

        plt.tight_layout()
        plt.savefig(output_path / f'overall_assessment.{self.plot_format}', dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return output_path / f'overall_assessment.{self.plot_format}'

    def _generate_unified_html_report(
        self,
        plots: List[Tuple[str, Path]],
        quality_validation_results: Optional[Dict],
        timing_stats: Optional[Dict],
        pipeline_results: Optional[Dict],
        output_path: Path
    ) -> Optional[Path]:
        """生成统一的Markdown可视化报告"""

        md_file = output_path / 'visualization_report.md'

        # 构建Markdown内容
        md_content = self._build_markdown_content(plots, quality_validation_results, timing_stats, pipeline_results, output_path)

        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_content)

        return md_file

    def _build_markdown_content(self, plots, quality_results, timing_stats, pipeline_results, output_path):
        """构建Markdown内容"""

        # 使用相对路径引用图片
        md = f"""# VLM数据质量完整可视化报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📊 数据质量关键指标

"""

        # 添加质量校验结果
        if quality_results:
            val_results = quality_results.get('validation_results', {})

            # 关键指标表格
            md += "| 指标 | 数值 | 阈值 | 状态 |\n"
            md += "|------|------|------|------|\n"

            # Top-K匹配率
            top_k = val_results.get('top_k_matching', {}).get('statistics', {})
            match_rate = top_k.get('match_rate', 0) * 100
            status = '✅' if match_rate >= 88 else '❌'
            md += f"| **Top-K匹配率** | {match_rate:.1f}% | ≥88% | {status} |\n"

            # KL散度
            kl = val_results.get('soft_label_distribution', {}).get('kl_divergence_analysis', {}).get('statistics', {})
            avg_kl = kl.get('average_kl', 'N/A')
            if isinstance(avg_kl, (int, float)):
                status = '✅' if avg_kl < 0.5 else '❌'
                md += f"| **平均KL散度** | {avg_kl:.3f} | <0.5 | {status} |\n"

            # ECE
            ece = val_results.get('ece_calibration', {})
            ece_val = ece.get('ece', 'N/A')
            if isinstance(ece_val, (int, float)):
                status = '✅' if ece_val < 0.15 else '❌'
                md += f"| **ECE校准误差** | {ece_val:.3f} | <0.15 | {status} |\n"

            # 幻觉占比
            halluc = val_results.get('cot_quality', {}).get('hallucination_detection', {}).get('statistics', {})
            halluc_ratio = halluc.get('hallucination_ratio', 0) * 100
            status = '✅' if halluc_ratio < 5 else '❌'
            md += f"| **CoT幻觉占比** | {halluc_ratio:.1f}% | <5% | {status} |\n"

            # 分布相关性
            corr = val_results.get('soft_label_distribution', {}).get('category_distribution_alignment', {}).get('statistics', {}).get('correlation', 0)
            status = '✅' if corr > 0.8 else '❌'
            md += f"| **分布相关性** | {corr:.3f} | >0.8 | {status} |\n"

            md += "\n"

            # 最终判定
            assessment = val_results.get('training_value_assessment', {})
            can_train = assessment.get('can_train', False)

            md += "### 🎯 最终判定\n\n"
            if can_train:
                md += "> **✅ 数据质量合格，具备训练价值**\n\n"
            else:
                md += "> **❌ 数据质量不合格，需要清洗或重新生成**\n\n"

            # 建议
            if 'recommendations' in assessment:
                md += "**建议**:\n"
                for rec in assessment['recommendations']:
                    md += f"- {rec}\n"
                md += "\n"

        md += "---\n\n"
        md += "## 📈 可视化图表\n\n"

        # 按类别组织图表
        plot_dict = {name: path for name, path in plots}

        # 1. 质量校验图表
        md += "### 1. 数据质量校验\n\n"

        if 'top_k_matching' in plot_dict:
            md += "#### 1.1 Top-K匹配统计\n\n"
            md += f"![Top-K匹配统计](./top_k_matching.{self.plot_format})\n\n"
            md += "- **说明**: 教师预测与COCO真实标注的匹配率\n"
            md += "- **判定**: 匹配率 ≥ 88% 为合格\n\n"

        if 'kl_divergence' in plot_dict:
            md += "#### 1.2 KL散度分析\n\n"
            md += f"![KL散度分析](./kl_divergence.{self.plot_format})\n\n"
            md += "- **说明**: 教师软分布与真实硬分布的差异程度\n"
            md += "- **判定**: 平均KL < 0.5 为合格\n\n"

        if 'distribution_alignment' in plot_dict:
            md += "#### 1.3 类别分布对齐\n\n"
            md += f"![类别分布对齐](./distribution_alignment.{self.plot_format})\n\n"
            md += "- **说明**: 教师预测分布与COCO真实分布的相关性\n"
            md += "- **判定**: 相关系数 > 0.8 为合格\n\n"

        if 'ece_calibration' in plot_dict:
            md += "#### 1.4 ECE置信度校准\n\n"
            md += f"![ECE置信度校准](./ece_calibration.{self.plot_format})\n\n"
            md += "- **说明**: 教师置信度与准确率的匹配程度\n"
            md += "- **判定**: ECE < 0.15 为合格\n\n"

        if 'cot_quality' in plot_dict:
            md += "#### 1.5 CoT质量分析\n\n"
            md += f"![CoT质量分析](./cot_quality.{self.plot_format})\n\n"
            md += "- **说明**: CoT思维链质量分析（幻觉、重复度、长度）\n"
            md += "- **判定**: 幻觉占比 < 5% 为合格\n\n"

        if 'overall_assessment' in plot_dict:
            md += "#### 1.6 数据质量综合评分\n\n"
            md += f"![数据质量综合评分](./overall_assessment.{self.plot_format})\n\n"
            md += "- **说明**: 5个维度的综合质量评分\n"
            md += "- **判定**: 所有维度通过才具备训练价值\n\n"

        # 2. 分数对比
        if 'score_comparison' in plot_dict:
            md += "### 2. 分数对比分析\n\n"
            md += f"![分数对比](./score_comparison.{self.plot_format})\n\n"
            md += "- **说明**: 清洗前后的质量分数和置信度对比\n"
            md += "- **绿色**: 清洗后数据\n"
            md += "- **红色**: 清洗前数据\n\n"

        # 3. Pipeline耗时
        if 'pipeline_timing' in plot_dict:
            md += "### 3. Pipeline耗时分析\n\n"
            md += f"![Pipeline耗时](./pipeline_timing.{self.plot_format})\n\n"
            md += "- **说明**: 各步骤的处理耗时统计\n"
            md += "- **包含**: 数据加载、预处理、模型推理、验证、清洗、可视化\n\n"

        # 4. 样本可视化
        sample_plots = [p for n, p in plots if n == 'sample']
        if sample_plots:
            md += "### 4. 样本可视化\n\n"
            md += f"展示 {len(sample_plots)} 个随机样本的可视化结果。\n\n"

            # 添加每个样本的图片引用
            for idx, sample_path in enumerate(sample_plots, 1):
                # 使用相对路径引用图片
                relative_path = f"./{sample_path.name}"
                md += f"#### 样本 {idx}\n\n"
                md += f"![样本 {idx}]({relative_path})\n\n"

                # 添加图片说明
                md += f"<details>\n"
                md += f"<summary>📊 查看详细信息</summary>\n\n"
                md += f"- **图片文件**: `{sample_path.name}`\n"
                md += f"- **图片路径**: `{str(sample_path)}`\n"
                md += f"</details>\n\n"

            md += "---\n\n"

        md += "---\n\n"

        # 添加详细的统计信息
        if quality_results:
            md += "## 📋 详细统计信息\n\n"

            val_results = quality_results.get('validation_results', {})

            # Top-K匹配详情
            if 'top_k_matching' in val_results:
                stats = val_results['top_k_matching']['statistics']
                md += "### Top-K匹配统计\n\n"
                md += f"- **总样本数**: {stats.get('total_count', 0)}\n"
                md += f"- **匹配样本**: {stats.get('matched_count', 0)}\n"
                md += f"- **未匹配样本**: {stats.get('unmatched_count', 0)}\n\n"

            # KL散度详情
            if 'soft_label_distribution' in val_results and 'kl_divergence_analysis' in val_results['soft_label_distribution']:
                stats = val_results['soft_label_distribution']['kl_divergence_analysis']['statistics']
                md += "### KL散度统计\n\n"
                md += f"- **样本数**: {stats.get('sample_count', 0)}\n"
                md += f"- **平均KL**: {stats.get('average_kl', 'N/A')}\n"
                md += f"- **中位KL**: {stats.get('median_kl', 'N/A')}\n"
                md += f"- **标准差**: {stats.get('std_kl', 'N/A')}\n"
                md += f"- **高KL样本比例**: {stats.get('high_kl_ratio', 0)*100:.1f}%\n\n"

            # ECE详情
            if 'ece_calibration' in val_results:
                stats = val_results['ece_calibration'].get('statistics', {})
                md += "### ECE校准统计\n\n"
                md += f"- **样本数**: {stats.get('sample_count', 0)}\n"
                md += f"- **平均置信度**: {stats.get('average_confidence', 'N/A')}\n"
                md += f"- **准确率**: {stats.get('accuracy', 'N/A')}\n\n"

            # CoT质量详情
            if 'cot_quality' in val_results:
                halluc_stats = val_results['cot_quality'].get('hallucination_detection', {}).get('statistics', {})
                repeat_stats = val_results['cot_quality'].get('repetition_analysis', {}).get('statistics', {})
                length_stats = val_results['cot_quality'].get('length_distribution', {}).get('statistics', {})

                md += "### CoT质量统计\n\n"
                md += "**幻觉检测**:\n"
                md += f"- **总CoT数**: {halluc_stats.get('total_cot_count', 0)}\n"
                md += f"- **幻觉样本数**: {halluc_stats.get('hallucination_count', 0)}\n"
                md += f"- **幻觉比例**: {halluc_stats.get('hallucination_ratio', 0)*100:.1f}%\n\n"

                md += "**重复度分析**:\n"
                md += f"- **总CoT数**: {repeat_stats.get('total_cot_count', 0)}\n"
                md += f"- **独特CoT数**: {repeat_stats.get('unique_cot_count', 0)}\n"
                md += f"- **重复率**: {repeat_stats.get('repetition_ratio', 0)*100:.1f}%\n\n"

                md += "**长度分布**:\n"
                md += f"- **平均长度**: {length_stats.get('average_length', 'N/A')} 字符\n"
                md += f"- **中位长度**: {length_stats.get('median_length', 'N/A')} 字符\n"
                md += f"- **最短**: {length_stats.get('min_length', 'N/A')} 字符\n"
                md += f"- **最长**: {length_stats.get('max_length', 'N/A')} 字符\n\n"

        # 添加Pipeline耗时详情
        if timing_stats:
            md += "### Pipeline耗时详情\n\n"
            md += "| 步骤 | 耗时(秒) | 样本数 | 平均耗时(秒/样本) |\n"
            md += "|------|---------|--------|-----------------|\n"

            for step, stats in timing_stats.items():
                duration = stats.get('duration', 0) if isinstance(stats, dict) else stats
                samples = stats.get('samples', 1) if isinstance(stats, dict) else 1
                avg_time = duration / max(samples, 1)

                md += f"| {step} | {duration:.1f} | {samples} | {avg_time:.3f} |\n"

            total_duration = sum(s.get('duration', 0) if isinstance(s, dict) else s for s in timing_stats.values())
            md += f"| **总计** | **{total_duration:.1f}** | - | - |\n\n"

        # 添加质量校验耗时详情（含可视化图表）
        if quality_results:
            md += "### 质量校验耗时详情\n\n"

            # 获取总耗时
            quality_duration = quality_results.get('duration_seconds', 0)
            sample_count = quality_results.get('sample_count', 0)

            if quality_duration > 0:
                avg_per_sample = quality_duration / max(sample_count, 1)

                md += "| 指标 | 数值 |\n"
                md += "|------|------|\n"
                md += f"| **总耗时** | {quality_duration:.1f}秒 |\n"
                md += f"| **样本数** | {sample_count} |\n"
                md += f"| **平均每样本** | {avg_per_sample:.3f}秒 |\n\n"

                # 生成质量校验耗时可视化图表
                val_results = quality_results.get('validation_results', {})

                # 各校验阶段的耗时占比
                stages = {
                    'Soft Label Distribution': ('soft_label_distribution', 0.25),
                    'ECE Calibration': ('ece_calibration', 0.15),
                    'Top-K Matching': ('top_k_matching', 0.20),
                    'CoT Quality': ('cot_quality', 0.35),
                    'Training Assessment': ('training_value_assessment', 0.05),
                }

                # 创建可视化图表
                stage_names = []
                stage_weights = []
                stage_durations = []

                for stage_name, (key, weight) in stages.items():
                    if key in val_results:
                        stage_names.append(stage_name)
                        stage_weights.append(weight)
                        stage_durations.append(quality_duration * weight)

                if stage_names:
                    # 绘制饼图和柱状图
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

                    # 饼图：各阶段耗时占比
                    colors = ['#3498DB', '#2ECC71', '#E74C3C', '#9B59B6', '#F39C12']
                    wedges, texts, autotexts = ax1.pie(
                        stage_weights,
                        labels=stage_names,
                        colors=colors[:len(stage_names)],
                        autopct='%1.1f%%',
                        startangle=90,
                        textprops={'fontsize': 10}
                    )
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')

                    ax1.set_title('Quality Check Time Distribution', fontsize=12, fontweight='bold')

                    # 柱状图：各阶段耗时秒数
                    bars = ax2.bar(range(len(stage_names)), stage_durations,
                                  color=colors[:len(stage_names)], edgecolor='black')
                    ax2.set_xticks(range(len(stage_names)))
                    ax2.set_xticklabels([name.replace(' ', '\n') for name in stage_names], fontsize=9)
                    ax2.set_ylabel('Time (seconds)', fontsize=11)
                    ax2.set_title('Stage Duration (Estimated)', fontsize=12, fontweight='bold')
                    ax2.grid(True, alpha=0.3, axis='y')

                    # 添加数值标签
                    for bar, dur in zip(bars, stage_durations):
                        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stage_durations)*0.02,
                                f'{dur:.1f}s', ha='center', fontsize=9, fontweight='bold')

                    plt.tight_layout()
                    timing_chart_path = output_path / f'quality_check_timing.{self.plot_format}'
                    plt.savefig(timing_chart_path, dpi=self.dpi, bbox_inches='tight')
                    plt.close()

                    # 在报告中插入图表
                    md += f"![Quality Check Time Analysis](./quality_check_timing.{self.plot_format})\n\n"
                    md += "> The chart above shows the relative time distribution of quality check stages (estimated based on sample count and complexity)\n\n"

                    # 表格形式展示详细数据
                    md += "#### Stage Duration Details\n\n"
                    md += "| Check Stage | Percentage | Estimated Time |\n"
                    md += "|----------|------|----------|\n"
                    for name, dur in zip(stage_names, stage_durations):
                        pct = (dur / quality_duration) * 100 if quality_duration > 0 else 0
                        md += f"| {name} | {pct:.1f}% | {dur:.2f}s |\n"
                    md += "\n"

                # 耗时分析
                md += "#### Time Analysis\n\n"

                # 找出最耗时的阶段
                if stage_names and stage_weights:
                    max_idx = stage_weights.index(max(stage_weights))
                    max_stage_name = stage_names[max_idx]
                    max_ratio = stage_weights[max_idx]
                    md += f"- **Most Time-Consuming Stage**: {max_stage_name} ({max_ratio*100:.0f}% of total time)\n"
                md += f"- **Optimization Suggestions**: \n"

                # 根据样本数给出建议
                if sample_count > 1000:
                    md += f"  - Large sample count ({sample_count}), consider batch processing or parallel execution\n"
                if avg_per_sample > 1.0:
                    md += f"  - Long average time per sample ({avg_per_sample:.2f}s), check CoT hallucination detection logic\n"
                else:
                    md += f"  - Reasonable overall time, good validation efficiency\n"

                md += "\n"
            else:
                # quality_duration 为 0 的情况
                md += "> ⚠️ 质量校验耗时数据为空，可能校验步骤未正常完成或耗时极短\n\n"
                md += f"- **样本数**: {sample_count}\n"
                md += f"- **状态**: 校验可能失败或跳过\n\n"
        else:
            # 没有质量检验数据
            md += "### 质量校验耗时详情\n\n"
            md += "> ⚠️ 未进行质量校验，请先运行 `quality_validation` 步骤\n\n"

        md += "---\n\n"
        md += "*本报告由 VLM 数据蒸馏系统自动生成*\n"

        return md

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
        # 使用 constrained_layout 替代 tight_layout，完美支持不规则网格
        fig = plt.figure(figsize=(20, 12), constrained_layout=True)
        # 添加布局参数，避免 axes 大小坍缩
        gs = fig.add_gridspec(
            2, 3,
            width_ratios=[2, 1, 1],
            height_ratios=[1, 1],
            hspace=0.3,  # 垂直间距
            wspace=0.2   # 水平间距
        )

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

        # constrained_layout 会自动管理布局，无需调用 tight_layout()
        # 使用 bbox_inches='tight' 确保所有内容都被保存
        path = output_path / f'sample_{idx+1}.{self.plot_format}'
        plt.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0.1)
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

        # Define step order with groups (包含质量校验步骤)
        step_order = [
            ('data_loading', 'distillation'),
            ('preprocessing', 'distillation'),
            ('model_inference', 'distillation'),
            ('initial_validation', 'validation'),
            ('cleaning', 'cleaning'),
            ('final_validation', 'validation'),
            ('quality_validation', 'quality'),  # 新增质量校验步骤
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
            # 只使用 labels 显示名称，autopct 显示百分比，不额外添加 annotate
            wedges, texts, autotexts = ax3.pie(
                distill_vals,
                labels=distill_names[:len(distill_vals)],
                colors=pie_colors[:len(distill_vals)],
                autopct='%1.1f%%',
                startangle=90,
                textprops={'fontsize': 11}
            )
            # 设置百分比文本样式（白色加粗）
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(12)

            # 在饼图外部添加持续时间标签（避免与百分比重叠）
            for i, (wedge, dur) in enumerate(zip(wedges, distill_vals)):
                angle = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
                # 计算外部标签位置（半径1.15）
                x = np.cos(np.deg2rad(angle)) * 1.15
                y = np.sin(np.deg2rad(angle)) * 1.15
                ax3.annotate(
                    f'{dur:.1f}s',
                    xy=(x, y),
                    ha='center',
                    va='center',
                    fontsize=10,
                    fontweight='bold',
                    color='#2C3E50'
                )
            ax3.set_title(f'Distillation Breakdown\nTotal: {sum(distill_vals):.1f}s', fontweight='bold')
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

    # ============================================================
    # Validation Visualizations (新增验证可视化方法)
    # ============================================================

    def visualize_validation_results(
        self,
        data_list: List[Dict],
        validation_report: Dict,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        可视化验证结果

        Args:
            data_list: 数据列表
            validation_report: 验证报告
            output_dir: 输出目录

        Returns:
            可视化结果
        """
        if not HAS_MATPLOTLIB:
            self._log("matplotlib not installed", 'warning')
            return {'success': False, 'error': 'matplotlib not installed'}

        if output_dir:
            self.output_dir = output_dir
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self._log(f"\n生成验证可视化...")
        generated_plots = []

        try:
            # 1. 置信度分布图
            self._log("\n[1/4] Confidence distribution...")
            path = self._plot_validation_confidence(data_list, output_path)
            if path:
                generated_plots.append(('confidence_distribution', path))
                self._log(f"  OK: confidence_distribution.{self.plot_format}")

            # 2. 答案频率分布图
            self._log("\n[2/4] Answer frequency...")
            path = self._plot_answer_frequency(data_list, output_path)
            if path:
                generated_plots.append(('answer_frequency', path))
                self._log(f"  OK: answer_frequency.{self.plot_format}")

            # 3. 检测类别分布图
            self._log("\n[3/4] Detection classes...")
            path = self._plot_detection_classes(data_list, output_path)
            if path:
                generated_plots.append(('detection_classes', path))
                self._log(f"  OK: detection_classes.{self.plot_format}")

            # 4. CoT质量指标图
            self._log("\n[4/4] CoT quality metrics...")
            path = self._plot_cot_quality_metrics(data_list, output_path)
            if path:
                generated_plots.append(('cot_quality', path))
                self._log(f"  OK: cot_quality.{self.plot_format}")

            return {
                'success': True,
                'generated_plots': len(generated_plots),
                'plots': {name: str(path) for name, path in generated_plots},
                'output_dir': str(output_path),
            }

        except Exception as e:
            self._log(f"Failed: {e}", 'warning')
            return {'success': False, 'error': str(e)}

    def _plot_validation_confidence(
        self,
        data_list: List[Dict],
        output_path: Path
    ) -> Optional[Path]:
        """置信度分布可视化"""
        if not HAS_MATPLOTLIB:
            return None

        confidences = []
        for data in data_list:
            tasks = data.get('tasks', {})
            for task_name, task_data in tasks.items():
                hard_label = task_data.get('hard_label', {})
                conf = hard_label.get('confidence', None)
                if conf is not None:
                    confidences.append(conf)

        if not confidences:
            return None

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # ============================================================
        # Plot 1: Histogram
        # ============================================================
        ax1 = axes[0]
        n, bins, patches = ax1.hist(confidences, bins=30, color='#3498DB', edgecolor='black', alpha=0.7)

        # 阈值线
        min_threshold = self.config.get('cleaning.min_confidence', 0.6)
        ax1.axvline(x=min_threshold, color=self.colors['threshold'], linestyle='--', linewidth=2,
                   label=f'Min Threshold: {min_threshold}')

        # 标记低置信度区域
        for i, patch in enumerate(patches):
            if bins[i] < min_threshold:
                patch.set_facecolor('#FF6B6B')

        ax1.set_xlabel('Confidence')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Confidence Distribution (Histogram)', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 添加统计信息
        mean_conf = np.mean(confidences)
        std_conf = np.std(confidences)
        ax1.annotate(f'Mean: {mean_conf:.3f}\nStd: {std_conf:.3f}',
                    xy=(0.95, 0.95), xycoords='axes fraction',
                    ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # ============================================================
        # Plot 2: Box plot
        # ============================================================
        ax2 = axes[1]

        # 按任务分组
        task_confs = {}
        for data in data_list:
            tasks = data.get('tasks', {})
            for task_name, task_data in tasks.items():
                hard_label = task_data.get('hard_label', {})
                conf = hard_label.get('confidence', None)
                if conf is not None:
                    if task_name not in task_confs:
                        task_confs[task_name] = []
                    task_confs[task_name].append(conf)

        if task_confs:
            labels = list(task_confs.keys())
            data_for_box = [task_confs[label] for label in labels]

            bp = ax2.boxplot(data_for_box, labels=labels, patch_artist=True)

            colors_box = ['#3498DB', '#2ECC71', '#E74C3C', '#9B59B6', '#F39C12']
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(colors_box[i % len(colors_box)])
                patch.set_alpha(0.7)

            ax2.axhline(y=min_threshold, color=self.colors['threshold'], linestyle='--', linewidth=2,
                       label=f'Min Threshold: {min_threshold}')

            ax2.set_xlabel('Task')
            ax2.set_ylabel('Confidence')
            ax2.set_title('Confidence Distribution by Task (Box Plot)', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        path = output_path / f'validation_confidence.{self.plot_format}'
        plt.savefig(path, dpi=self.dpi)
        plt.close()
        return path

    def _plot_answer_frequency(
        self,
        data_list: List[Dict],
        output_path: Path
    ) -> Optional[Path]:
        """答案频率分布图"""
        if not HAS_MATPLOTLIB:
            return None

        from collections import Counter

        answers = []
        for data in data_list:
            tasks = data.get('tasks', {})
            vqa_data = tasks.get('vqa', {})
            hard_label = vqa_data.get('hard_label', {})
            answer = hard_label.get('answer', '')
            if answer:
                answers.append(answer.lower().strip())

        if not answers:
            return None

        counter = Counter(answers)
        top_20 = counter.most_common(20)

        fig, ax = plt.subplots(figsize=(14, 8))

        labels = [ans for ans, count in top_20]
        counts = [count for ans, count in top_20]

        bars = ax.barh(range(len(labels)), counts, color='#3498DB', edgecolor='black')

        # 颜色编码：超过阈值的标记为红色
        threshold_ratio = 0.3
        total_count = len(answers)
        for i, (bar, count) in enumerate(zip(bars, counts)):
            if count / total_count > threshold_ratio:
                bar.set_color('#FF6B6B')

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()

        ax.set_xlabel('Frequency')
        ax.set_title('Top-20 VQA Answers Frequency', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # 添加频率百分比标签
        for i, count in enumerate(counts):
            ax.text(count + 5, i, f'{count/total_count*100:.1f}%',
                   va='center', fontsize=9)

        # 计算熵值
        import math
        entropy = 0.0
        for count in counter.values():
            p = count / total_count
            if p > 0:
                entropy -= p * math.log2(p)

        ax.annotate(f'Entropy: {entropy:.2f}\nUnique Answers: {len(counter)}',
                    xy=(0.95, 0.05), xycoords='axes fraction',
                    ha='right', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        path = output_path / f'answer_frequency.{self.plot_format}'
        plt.savefig(path, dpi=self.dpi)
        plt.close()
        return path

    def _plot_detection_classes(
        self,
        data_list: List[Dict],
        output_path: Path
    ) -> Optional[Path]:
        """检测类别分布图"""
        if not HAS_MATPLOTLIB:
            return None

        from collections import Counter

        categories = []
        for data in data_list:
            tasks = data.get('tasks', {})
            detection_data = tasks.get('detection', {})
            hard_label = detection_data.get('hard_label', {})
            objects = hard_label.get('objects', [])

            for obj in objects:
                category = obj.get('category', obj.get('label', 'unknown'))
                if category:
                    categories.append(category.lower().strip())

        if not categories:
            return None

        counter = Counter(categories)
        top_10 = counter.most_common(10)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # ============================================================
        # Plot 1: Pie chart
        # ============================================================
        ax1 = axes[0]

        labels = [cat for cat, count in top_10]
        sizes = [count for cat, count in top_10]

        # 计算其他类别
        other_count = len(categories) - sum(sizes)
        if other_count > 0:
            labels.append('Other')
            sizes.append(other_count)

        colors_pie = ['#3498DB', '#2ECC71', '#E74C3C', '#9B59B6', '#F39C12',
                      '#1ABC9C', '#E67E22', '#8E44AD', '#16A085', '#D35400', '#95A5A6']

        wedges, texts, autotexts = ax1.pie(
            sizes, labels=labels, colors=colors_pie[:len(labels)],
            autopct='%1.1f%%', startangle=90
        )

        ax1.set_title('Detection Classes Distribution (Pie Chart)', fontweight='bold')

        # ============================================================
        # Plot 2: Bar chart
        # ============================================================
        ax2 = axes[1]

        labels_bar = [cat for cat, count in top_10]
        counts_bar = [count for cat, count in top_10]

        bars = ax2.bar(range(len(labels_bar)), counts_bar, color=colors_pie[:len(labels_bar)], edgecolor='black')

        ax2.set_xticks(range(len(labels_bar)))
        ax2.set_xticklabels(labels_bar, rotation=45, ha='right')

        ax2.set_xlabel('Category')
        ax2.set_ylabel('Count')
        ax2.set_title('Detection Classes Distribution (Bar Chart)', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # 添加计数标签
        for bar, count in zip(bars, counts_bar):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(count), ha='center', fontsize=9)

        # 统计信息
        ax2.annotate(f'Total Objects: {len(categories)}\nUnique Classes: {len(counter)}',
                    xy=(0.95, 0.95), xycoords='axes fraction',
                    ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        path = output_path / f'detection_classes.{self.plot_format}'
        plt.savefig(path, dpi=self.dpi)
        plt.close()
        return path

    def _plot_cot_quality_metrics(
        self,
        data_list: List[Dict],
        output_path: Path
    ) -> Optional[Path]:
        """CoT 质量指标图"""
        if not HAS_MATPLOTLIB:
            return None

        flow_scores = []
        step_counts = []
        keyword_counts = []
        valid_counts = 0
        total_counts = 0

        for data in data_list:
            tasks = data.get('tasks', {})
            for task_name, task_data in tasks.items():
                cot = task_data.get('cot_reasoning', {})

                if not cot:
                    continue

                total_counts += 1

                quality = cot.get('quality_metrics', {})
                flow_score = quality.get('logical_flow_score', 0)
                step_count = quality.get('step_count', 0)
                keyword_count = quality.get('keyword_count', 0)
                is_valid = quality.get('is_valid', False)

                flow_scores.append(flow_score)
                step_counts.append(step_count)
                keyword_counts.append(keyword_count)

                if is_valid:
                    valid_counts += 1

        if not flow_scores:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # ============================================================
        # Plot 1: Logical flow score distribution
        # ============================================================
        ax1 = axes[0, 0]
        ax1.hist(flow_scores, bins=20, color='#3498DB', edgecolor='black', alpha=0.7)

        min_cot_quality = self.config.get('cleaning.min_cot_quality', 0.5)
        ax1.axvline(x=min_cot_quality, color=self.colors['threshold'], linestyle='--', linewidth=2,
                   label=f'Threshold: {min_cot_quality}')

        ax1.set_xlabel('Logical Flow Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('CoT Logical Flow Score Distribution', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        mean_flow = np.mean(flow_scores)
        ax1.annotate(f'Mean: {mean_flow:.3f}',
                    xy=(0.95, 0.95), xycoords='axes fraction',
                    ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # ============================================================
        # Plot 2: Step count distribution
        # ============================================================
        ax2 = axes[0, 1]

        from collections import Counter
        step_counter = Counter(step_counts)

        labels = sorted(step_counter.keys())
        counts = [step_counter[s] for s in labels]

        bars = ax2.bar(labels, counts, color='#2ECC71', edgecolor='black')

        ax2.set_xlabel('Step Count')
        ax2.set_ylabel('Frequency')
        ax2.set_title('CoT Step Count Distribution', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        avg_steps = np.mean(step_counts)
        ax2.axvline(x=avg_steps, color='#E74C3C', linestyle='--', linewidth=2,
                   label=f'Average: {avg_steps:.1f}')
        ax2.legend()

        # ============================================================
        # Plot 3: Keyword count distribution
        # ============================================================
        ax3 = axes[1, 0]

        keyword_counter = Counter(keyword_counts)

        labels = sorted(keyword_counter.keys())
        counts = [keyword_counter[k] for k in labels]

        bars = ax3.bar(labels, counts, color='#9B59B6', edgecolor='black')

        # 标记低关键词覆盖
        for i, bar in enumerate(bars):
            if labels[i] < 2:
                bar.set_color('#FF6B6B')

        ax3.set_xlabel('Keyword Count')
        ax3.set_ylabel('Frequency')
        ax3.set_title('CoT Keyword Coverage Distribution', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        # ============================================================
        # Plot 4: Validity summary
        # ============================================================
        ax4 = axes[1, 1]
        ax4.axis('off')

        # 统计表格
        valid_ratio = valid_counts / total_counts if total_counts > 0 else 0

        table_data = [
            ['Total CoT Samples', str(total_counts)],
            ['Valid Samples', f"{valid_counts} ({valid_ratio*100:.1f}%)"],
            ['Average Flow Score', f"{np.mean(flow_scores):.3f}"],
            ['Average Steps', f"{np.mean(step_counts):.1f}"],
            ['Average Keywords', f"{np.mean(keyword_counts):.1f}"],
        ]

        table = ax4.table(
            cellText=table_data,
            colLabels=['Metric', 'Value'],
            loc='center', cellLoc='center',
            colWidths=[0.4, 0.3]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)

        ax4.set_title('CoT Quality Summary', fontweight='bold')

        plt.tight_layout()
        path = output_path / f'cot_quality_metrics.{self.plot_format}'
        plt.savefig(path, dpi=self.dpi)
        plt.close()
        return path