"""
数据质量分析脚本
================

全面分析蒸馏数据的准确性，包括：
1. 统计分析（置信度分布、质量分布）
2. 异常检测（离群点、异常值）
3. 与原始标注对比（COCO ground truth）
4. 可视化报告

Usage:
    python scripts/analyze_data_quality.py --input outputs/cleaned/cleaned
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

# 兼容导入
try:
    from utils import setup_logger
except ImportError:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root / "src"))
    from utils import setup_logger


def load_data_files(input_dir: str) -> List[Dict[str, Any]]:
    """加载所有数据文件"""
    input_path = Path(input_dir)
    json_files = list(input_path.glob("*.json"))

    data_list = []
    for json_file in json_files:
        if json_file.name.startswith(('cleaning_report', 'merged_summary', 'validation', 'checkpoint', 'pipeline')):
            continue

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data_list.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")

    return data_list


def analyze_confidence_distribution(data_list: List[Dict]) -> Dict[str, Any]:
    """分析置信度分布"""
    confidence_values = []

    for data in data_list:
        tasks = data.get('tasks', {})
        for task_name, task_data in tasks.items():
            hard_label = task_data.get('hard_label', {})
            confidence = hard_label.get('confidence')
            if confidence is not None:
                confidence_values.append(confidence)

    if not confidence_values:
        return {'error': 'No confidence values found'}

    return {
        'count': len(confidence_values),
        'mean': np.mean(confidence_values),
        'median': np.median(confidence_values),
        'std': np.std(confidence_values),
        'min': np.min(confidence_values),
        'max': np.max(confidence_values),
        'distribution': {
            'high (≥0.7)': len([c for c in confidence_values if c >= 0.7]),
            'medium (0.5-0.7)': len([c for c in confidence_values if 0.5 <= c < 0.7]),
            'low (<0.5)': len([c for c in confidence_values if c < 0.5]),
        },
        'values': confidence_values,
    }


def analyze_quality_distribution(data_list: List[Dict]) -> Dict[str, Any]:
    """分析质量分数分布"""
    quality_scores = []

    for data in data_list:
        quality_score = data.get('quality_score')
        if quality_score is not None:
            quality_scores.append(quality_score)

    if not quality_scores:
        return {'error': 'No quality scores found'}

    return {
        'count': len(quality_scores),
        'mean': np.mean(quality_scores),
        'median': np.median(quality_scores),
        'std': np.std(quality_scores),
        'min': np.min(quality_scores),
        'max': np.max(quality_scores),
        'distribution': {
            'high (70-100)': len([q for q in quality_scores if q >= 70]),
            'medium (50-70)': len([q for q in quality_scores if 50 <= q < 70]),
            'low (30-50)': len([q for q in quality_scores if 30 <= q < 50]),
            'very low (<30)': len([q for q in quality_scores if q < 30]),
        },
        'values': quality_scores,
    }


def analyze_task_distribution(data_list: List[Dict]) -> Dict[str, Any]:
    """分析任务分布"""
    task_counts = {}

    for data in data_list:
        tasks = data.get('tasks', {})
        for task_name in tasks.keys():
            task_counts[task_name] = task_counts.get(task_name, 0) + 1

    return {
        'total_samples': len(data_list),
        'tasks': task_counts,
        'avg_tasks_per_sample': np.mean([len(d.get('tasks', {})) for d in data_list]),
    }


def analyze_cot_quality(data_list: List[Dict]) -> Dict[str, Any]:
    """分析CoT质量"""
    cot_samples = 0
    logical_flows = []
    step_counts = []

    for data in data_list:
        tasks = data.get('tasks', {})
        for task_name, task_data in tasks.items():
            cot = task_data.get('cot_reasoning', {})
            if cot:
                cot_samples += 1
                quality_metrics = cot.get('quality_metrics', {})

                logical_flow = quality_metrics.get('logical_flow_score')
                if logical_flow is not None:
                    logical_flows.append(logical_flow)

                step_count = quality_metrics.get('step_count')
                if step_count is not None:
                    step_counts.append(step_count)

    return {
        'cot_coverage': cot_samples,
        'cot_rate': cot_samples / len(data_list) if data_list else 0,
        'avg_logical_flow': np.mean(logical_flows) if logical_flows else None,
        'avg_step_count': np.mean(step_counts) if step_counts else None,
        'logical_flows': logical_flows,
        'step_counts': step_counts,
    }


def detect_anomalies(data_list: List[Dict]) -> Dict[str, Any]:
    """检测异常数据"""
    anomalies = {
        'low_confidence': [],
        'short_answers': [],
        'long_answers': [],
        'empty_cot': [],
        'format_errors': [],
    }

    for data in data_list:
        image_id = data.get('image_id', 'unknown')
        tasks = data.get('tasks', {})

        for task_name, task_data in tasks.items():
            # 低置信度
            hard_label = task_data.get('hard_label', {})
            confidence = hard_label.get('confidence', 1.0)
            if confidence < 0.5:
                anomalies['low_confidence'].append({
                    'image_id': image_id,
                    'task': task_name,
                    'confidence': confidence,
                })

            # 答案长度异常
            answer = hard_label.get('answer', '')
            if len(answer) < 3:
                anomalies['short_answers'].append({
                    'image_id': image_id,
                    'task': task_name,
                    'answer': answer,
                    'length': len(answer),
                })
            elif len(answer) > 100:
                anomalies['long_answers'].append({
                    'image_id': image_id,
                    'task': task_name,
                    'answer': answer[:50] + '...',
                    'length': len(answer),
                })

            # 空CoT
            cot = task_data.get('cot_reasoning', {})
            if not cot or not cot.get('raw_reasoning'):
                anomalies['empty_cot'].append({
                    'image_id': image_id,
                    'task': task_name,
                })

    return {
        'total_anomalies': sum(len(v) for v in anomalies.values()),
        'by_type': {k: len(v) for k, v in anomalies.items()},
        'details': anomalies,
    }


def compare_with_ground_truth(
    data_list: List[Dict],
    coco_annotations_path: str
) -> Dict[str, Any]:
    """与COCO原始标注对比（验证准确性）"""

    # 加载COCO标注
    try:
        with open(coco_annotations_path, 'r') as f:
            coco_data = json.load(f)

        # 提取VQA答案
        ground_truth = {}
        for ann in coco_data.get('annotations', []):
            question_id = ann.get('question_id')
            answer = ann.get('multiple_choice_answer') or ann.get('answers', [{}])[0].get('answer')
            if question_id and answer:
                ground_truth[question_id] = answer

        # 对比
        matches = 0
        total = 0
        mismatches = []

        for data in data_list:
            tasks = data.get('tasks', {})
            vqa = tasks.get('vqa', {})
            hard_label = vqa.get('hard_label', {})

            # 注意：这里需要根据实际数据结构调整
            predicted_answer = hard_label.get('answer', '')
            question_id = hard_label.get('question_id')  # 如果有的话

            if question_id and question_id in ground_truth:
                gt_answer = ground_truth[question_id]
                total += 1

                # 标准化答案进行比较
                if predicted_answer.lower().strip() == gt_answer.lower().strip():
                    matches += 1
                else:
                    mismatches.append({
                        'question_id': question_id,
                        'predicted': predicted_answer,
                        'ground_truth': gt_answer,
                    })

        accuracy = matches / total if total > 0 else None

        return {
            'total_compared': total,
            'matches': matches,
            'accuracy': accuracy,
            'mismatch_count': len(mismatches),
            'mismatch_examples': mismatches[:10],  # 前10个示例
        }

    except Exception as e:
        return {'error': f'Failed to load COCO annotations: {e}'}


def generate_visualization(
    confidence_analysis: Dict,
    quality_analysis: Dict,
    output_dir: str
):
    """生成可视化图表"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 置信度分布图
    if 'values' in confidence_analysis:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(confidence_analysis['values'], bins=20, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Count')
        ax.set_title('Confidence Distribution')
        ax.axvline(x=0.5, color='red', linestyle='--', label='Threshold (0.5)')
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_path / 'confidence_distribution.png')
        plt.close()

    # 2. 质量分数分布图
    if 'values' in quality_analysis:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(quality_analysis['values'], bins=20, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Quality Score')
        ax.set_ylabel('Count')
        ax.set_title('Quality Score Distribution')
        ax.axvline(x=50, color='orange', linestyle='--', label='Medium threshold (50)')
        ax.axvline(x=70, color='green', linestyle='--', label='High threshold (70)')
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_path / 'quality_distribution.png')
        plt.close()

    print(f"\n✓ Visualization saved to {output_path}")


def generate_report(
    confidence_analysis: Dict,
    quality_analysis: Dict,
    task_analysis: Dict,
    cot_analysis: Dict,
    anomaly_analysis: Dict,
    gt_comparison: Dict,
    output_path: str
):
    """生成完整分析报告"""
    report = {
        'summary': {
            'total_samples': task_analysis['total_samples'],
            'data_quality': 'GOOD' if quality_analysis.get('mean', 0) >= 60 else 'NEEDS_IMPROVEMENT',
        },
        'confidence_analysis': confidence_analysis,
        'quality_analysis': quality_analysis,
        'task_analysis': task_analysis,
        'cot_analysis': cot_analysis,
        'anomaly_analysis': anomaly_analysis,
        'ground_truth_comparison': gt_comparison,
        'recommendations': [],
    }

    # 生成建议
    avg_quality = quality_analysis.get('mean', 0)
    if avg_quality < 50:
        report['recommendations'].append('⚠️ 平均质量分数较低，建议使用更严格的清洗参数')
    elif avg_quality < 70:
        report['recommendations'].append('✓ 数据质量中等，可以使用，建议关注低质量数据')
    else:
        report['recommendations'].append('✓ 数据质量良好，可以直接用于训练')

    low_conf_count = anomaly_analysis.get('by_type', {}).get('low_confidence', 0)
    if low_conf_count > 0:
        report['recommendations'].append(f'⚠️ 发现{low_conf_count}个低置信度样本，建议人工检查或移除')

    # 保存报告
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Analysis report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze distilled data quality")

    parser.add_argument(
        '--input',
        type=str,
        default='./outputs/cleaned/cleaned',
        help='Input directory with cleaned data'
    )

    parser.add_argument(
        '--coco-annotations',
        type=str,
        default=None,
        help='COCO annotations path for ground truth comparison'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='./outputs/analysis',
        help='Output directory for analysis results'
    )

    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization plots'
    )

    args = parser.parse_args()

    print("="*60)
    print("Data Quality Analysis")
    print("="*60)

    # 加载数据
    print(f"\nLoading data from: {args.input}")
    data_list = load_data_files(args.input)
    print(f"Loaded {len(data_list)} samples")

    # 分析
    print("\n[1] Analyzing confidence distribution...")
    confidence_analysis = analyze_confidence_distribution(data_list)

    print("\n[2] Analyzing quality score distribution...")
    quality_analysis = analyze_quality_distribution(data_list)

    print("\n[3] Analyzing task distribution...")
    task_analysis = analyze_task_distribution(data_list)

    print("\n[4] Analyzing CoT quality...")
    cot_analysis = analyze_cot_quality(data_list)

    print("\n[5] Detecting anomalies...")
    anomaly_analysis = detect_anomalies(data_list)

    # 与ground truth对比
    gt_comparison = {'error': 'No COCO annotations provided'}
    if args.coco_annotations:
        print("\n[6] Comparing with ground truth...")
        gt_comparison = compare_with_ground_truth(data_list, args.coco_annotations)

    # 打印摘要
    print("\n" + "="*60)
    print("Analysis Summary")
    print("="*60)

    print(f"\nTotal samples: {task_analysis['total_samples']}")
    print(f"Tasks distribution: {task_analysis['tasks']}")

    if 'mean' in confidence_analysis:
        print(f"\nConfidence:")
        print(f"  Mean: {confidence_analysis['mean']:.3f}")
        print(f"  High (≥0.7): {confidence_analysis['distribution']['high (≥0.7)']}")
        print(f"  Medium (0.5-0.7): {confidence_analysis['distribution']['medium (0.5-0.7)']}")
        print(f"  Low (<0.5): {confidence_analysis['distribution']['low (<0.5)']}")

    if 'mean' in quality_analysis:
        print(f"\nQuality Score:")
        print(f"  Mean: {quality_analysis['mean']:.2f}")
        print(f"  High (70-100): {quality_analysis['distribution']['high (70-100)']}")
        print(f"  Medium (50-70): {quality_analysis['distribution']['medium (50-70)']}")
        print(f"  Low (<50): {quality_analysis['distribution']['low (<30)'] + quality_analysis['distribution']['very low (<30)']}")

    print(f"\nCoT Coverage:")
    print(f"  Rate: {cot_analysis['cot_rate']*100:.1f}%")
    if cot_analysis['avg_logical_flow']:
        print(f"  Avg logical flow: {cot_analysis['avg_logical_flow']:.3f}")

    print(f"\nAnomalies:")
    print(f"  Total: {anomaly_analysis['total_anomalies']}")
    for anomaly_type, count in anomaly_analysis['by_type'].items():
        if count > 0:
            print(f"  {anomaly_type}: {count}")

    # 生成报告
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    generate_report(
        confidence_analysis,
        quality_analysis,
        task_analysis,
        cot_analysis,
        anomaly_analysis,
        gt_comparison,
        output_path / 'quality_analysis_report.json'
    )

    # 生成可视化
    if args.visualize:
        generate_visualization(
            confidence_analysis,
            quality_analysis,
            args.output
        )

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()