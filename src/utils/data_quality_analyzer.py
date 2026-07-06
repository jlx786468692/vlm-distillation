"""
数据质量分析器
=============

负责分析数据质量，包括：
- 置信度分析
- 质量分数分析
- 任务分布分析
- CoT质量分析
- 异常检测
- 质量建议生成

Usage:
    analyzer = DataQualityAnalyzer(logger)
    quality_report = analyzer.analyze_data_quality(data_list)
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import json


class DataQualityAnalyzer:
    """
    数据质量分析器

    功能：
    1. 置信度分析（分布、低置信度样本）
    2. 质量分数分析（分布、平均分）
    3. 任务分布统计
    4. CoT推理质量评估
    5. 异常数据检测
    6. 生成质量建议
    """

    def __init__(self, logger: Any = None):
        """
        初始化数据质量分析器

        Args:
            logger: 日志记录器
        """
        self.logger = logger

        # 无效答案列表（用于质量评估）
        self.invalid_answers = [
            'unknown', 'n/a', 'none', 'unclear',
            'cannot determine', ''
        ]

    def analyze_data_quality(self, input_dir: str) -> Dict[str, Any]:
        """
        分析数据质量（完整分析）

        Args:
            input_dir: 数据目录路径

        Returns:
            数据质量分析报告
        """
        if self.logger:
            self.logger.info("\n" + "="*70)
            self.logger.info("DATA QUALITY ANALYSIS")
            self.logger.info("="*70)

        # 加载数据
        data_list = self._load_data_from_dir(input_dir)

        if not data_list:
            if self.logger:
                self.logger.warning("No data found for analysis")
            return {
                'success': False,
                'error': 'No data found',
                'data_count': 0
            }

        if self.logger:
            self.logger.info(f"Analyzing {len(data_list)} data samples...")

        # 执行各项分析
        confidence_analysis = self._analyze_confidence(data_list)
        quality_analysis = self._analyze_quality_scores(data_list)
        task_analysis = self._analyze_task_distribution(data_list)
        cot_analysis = self._analyze_cot_quality(data_list)
        anomaly_analysis = self._detect_anomalies_summary(data_list)

        # 生成建议
        recommendations = self._generate_quality_recommendations(
            confidence_analysis,
            quality_analysis,
            cot_analysis,
            anomaly_analysis
        )

        # 整合报告
        report = {
            'success': True,
            'data_count': len(data_list),
            'confidence_analysis': confidence_analysis,
            'quality_analysis': quality_analysis,
            'task_distribution': task_analysis,
            'cot_analysis': cot_analysis,
            'anomaly_analysis': anomaly_analysis,
            'recommendations': recommendations,
            'overall_score': self._calculate_overall_score(
                confidence_analysis,
                quality_analysis,
                cot_analysis,
                anomaly_analysis
            )
        }

        # 输出报告摘要
        if self.logger:
            self._display_analysis_summary(report)

        return report

    def _load_data_from_dir(self, input_dir: str) -> List[Dict]:
        """
        从目录加载所有数据文件

        Args:
            input_dir: 数据目录路径

        Returns:
            数据列表
        """
        input_path = Path(input_dir)
        json_files = list(input_path.glob("*.json"))

        # 过滤掉报告文件
        data_files = [
            f for f in json_files
            if not f.name.startswith((
                'cleaning_report', 'merged_summary', 'validation',
                'checkpoint', 'pipeline', 'visualization',
                'data_quality', 'timing'
            ))
        ]

        data_list = []
        for json_file in data_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                data_list.append(data)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to load {json_file}: {e}")

        return data_list

    def _analyze_confidence(self, data_list: List[Dict]) -> Dict[str, Any]:
        """
        分析置信度分布

        Args:
            data_list: 数据列表

        Returns:
            置信度分析结果
        """
        import numpy as np

        confidence_values = []
        low_confidence_samples = []
        high_confidence_samples = []

        for data in data_list:
            tasks = data.get('tasks', {})
            for task_name, task_data in tasks.items():
                hard_label = task_data.get('hard_label', {})
                confidence = hard_label.get('confidence')

                if confidence is not None:
                    confidence_values.append(confidence)

                    # 分类高/低置信度样本
                    if confidence < 0.5:
                        low_confidence_samples.append({
                            'image_id': data.get('image_id'),
                            'task': task_name,
                            'confidence': confidence
                        })
                    else:
                        high_confidence_samples.append({
                            'image_id': data.get('image_id'),
                            'task': task_name,
                            'confidence': confidence
                        })

        if not confidence_values:
            return {
                'count': 0,
                'mean': None,
                'median': None,
                'std': None,
                'min': None,
                'max': None,
                'low_confidence_count': 0,
                'high_confidence_count': 0,
                'low_rate': 0.0
            }

        return {
            'count': len(confidence_values),
            'mean': float(np.mean(confidence_values)),
            'median': float(np.median(confidence_values)),
            'std': float(np.std(confidence_values)),
            'min': float(np.min(confidence_values)),
            'max': float(np.max(confidence_values)),
            'low_confidence_count': len(low_confidence_samples),
            'high_confidence_count': len(high_confidence_samples),
            'low_rate': len(low_confidence_samples) / len(confidence_values),
            'low_confidence_samples': low_confidence_samples[:10],  # 只保留前10个
            'distribution': {
                'below_0.3': len([c for c in confidence_values if c < 0.3]),
                '0.3_to_0.5': len([c for c in confidence_values if 0.3 <= c < 0.5]),
                '0.5_to_0.7': len([c for c in confidence_values if 0.5 <= c < 0.7]),
                '0.7_to_0.9': len([c for c in confidence_values if 0.7 <= c < 0.9]),
                'above_0.9': len([c for c in confidence_values if c >= 0.9])
            }
        }

    def _analyze_quality_scores(self, data_list: List[Dict]) -> Dict[str, Any]:
        """
        分析质量分数分布

        Args:
            data_list: 数据列表

        Returns:
            质量分数分析结果
        """
        import numpy as np

        quality_scores = []
        low_quality_samples = []

        for data in data_list:
            score = self._estimate_quality_score(data)
            quality_scores.append(score)

            if score < 30:
                low_quality_samples.append({
                    'image_id': data.get('image_id'),
                    'score': score
                })

        if not quality_scores:
            return {
                'count': 0,
                'mean': None,
                'median': None,
                'std': None,
                'min': None,
                'max': None,
                'low_quality_count': 0,
                'low_rate': 0.0
            }

        return {
            'count': len(quality_scores),
            'mean': float(np.mean(quality_scores)),
            'median': float(np.median(quality_scores)),
            'std': float(np.std(quality_scores)),
            'min': float(np.min(quality_scores)),
            'max': float(np.max(quality_scores)),
            'low_quality_count': len(low_quality_samples),
            'low_rate': len(low_quality_samples) / len(quality_scores),
            'distribution': {
                'below_30': len([s for s in quality_scores if s < 30]),
                '30_to_50': len([s for s in quality_scores if 30 <= s < 50]),
                '50_to_70': len([s for s in quality_scores if 50 <= s < 70]),
                '70_to_90': len([s for s in quality_scores if 70 <= s < 90]),
                'above_90': len([s for s in quality_scores if s >= 90])
            }
        }

    def _estimate_quality_score(self, data: Dict[str, Any]) -> float:
        """
        估算单个样本的质量分数

        Args:
            data: 单个数据样本

        Returns:
            质量分数 (0-100)
        """
        score = 0.0
        weights = {
            'confidence': 0.3,
            'answer_validity': 0.25,
            'cot_quality': 0.25,
            'task_coverage': 0.2
        }

        tasks = data.get('tasks', {})

        # 1. 置信度分数 (30%)
        confidence_scores = []
        for task_name, task_data in tasks.items():
            hard_label = task_data.get('hard_label', {})
            confidence = hard_label.get('confidence')
            if confidence is not None:
                confidence_scores.append(confidence)

        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            score += avg_confidence * 100 * weights['confidence']

        # 2. 答案有效性分数 (25%)
        valid_answer_count = 0
        total_answers = 0
        for task_name, task_data in tasks.items():
            hard_label = task_data.get('hard_label', {})
            answer = hard_label.get('answer', '')

            total_answers += 1
            if answer and answer.lower() not in self.invalid_answers:
                valid_answer_count += 1

        if total_answers > 0:
            answer_validity = valid_answer_count / total_answers
            score += answer_validity * 100 * weights['answer_validity']

        # 3. CoT质量分数 (25%)
        cot_scores = []
        for task_name, task_data in tasks.items():
            cot = task_data.get('cot_reasoning', {})
            raw_reasoning = cot.get('raw_reasoning', '')

            if raw_reasoning:
                # 基于长度、逻辑性、步骤数估算
                length_score = min(len(raw_reasoning) / 200, 1.0) * 30

                # 逻辑性标记
                logic_markers = ['because', 'since', 'therefore', 'thus',
                                '首先', '然后', '最后', '因此']
                logic_count = sum(1 for marker in logic_markers
                                 if marker in raw_reasoning.lower())
                logic_score = min(logic_count / 3, 1.0) * 40

                # 步骤数
                step_count = raw_reasoning.count('\n') + 1
                step_score = min(step_count / 5, 1.0) * 30

                cot_scores.append(length_score + logic_score + step_score)

        if cot_scores:
            avg_cot_score = sum(cot_scores) / len(cot_scores)
            score += avg_cot_score * weights['cot_quality']

        # 4. 任务覆盖分数 (20%)
        expected_tasks = ['vqa', 'captioning', 'detection']
        task_count = len([t for t in expected_tasks if t in tasks])
        task_coverage = task_count / len(expected_tasks)
        score += task_coverage * 100 * weights['task_coverage']

        return min(score, 100.0)

    def _analyze_task_distribution(self, data_list: List[Dict]) -> Dict[str, Any]:
        """
        分析任务分布

        Args:
            data_list: 数据列表

        Returns:
            任务分布统计
        """
        task_counts = {}
        task_confidence = {}

        for data in data_list:
            tasks = data.get('tasks', {})
            for task_name, task_data in tasks.items():
                task_counts[task_name] = task_counts.get(task_name, 0) + 1

                hard_label = task_data.get('hard_label', {})
                confidence = hard_label.get('confidence')
                if confidence is not None:
                    if task_name not in task_confidence:
                        task_confidence[task_name] = []
                    task_confidence[task_name].append(confidence)

        # 计算各任务平均置信度
        import numpy as np
        task_avg_confidence = {}
        for task_name, conf_values in task_confidence.items():
            task_avg_confidence[task_name] = float(np.mean(conf_values))

        return {
            'total_samples': len(data_list),
            'task_counts': task_counts,
            'task_avg_confidence': task_avg_confidence,
            'task_coverage_rate': len(task_counts) / 3  # 预期3个任务
        }

    def _analyze_cot_quality(self, data_list: List[Dict]) -> Dict[str, Any]:
        """
        分析CoT推理质量

        Args:
            data_list: 数据列表

        Returns:
            CoT质量分析结果
        """
        import numpy as np

        cot_samples = 0
        logical_flows = []
        step_counts = []

        for data in data_list:
            tasks = data.get('tasks', {})
            for task_name, task_data in tasks.items():
                cot = task_data.get('cot_reasoning', {})
                raw_reasoning = cot.get('raw_reasoning', '')

                if raw_reasoning and len(raw_reasoning) > 20:
                    cot_samples += 1

                    # 逻辑性评估
                    logic_markers = ['because', 'since', 'therefore', 'thus',
                                    '首先', '然后', '最后', '因此']
                    logic_count = sum(1 for marker in logic_markers
                                     if marker in raw_reasoning.lower())
                    logical_flows.append(min(logic_count / 3, 1.0))

                    # 步骤数
                    step_count = raw_reasoning.count('\n') + 1
                    step_counts.append(step_count)

        return {
            'cot_samples': cot_samples,
            'cot_rate': cot_samples / len(data_list) if data_list else 0,
            'avg_logical_flow': float(np.mean(logical_flows)) if logical_flows else None,
            'avg_step_count': float(np.mean(step_counts)) if step_counts else None,
        }

    def _detect_anomalies_summary(self, data_list: List[Dict]) -> Dict[str, Any]:
        """
        检测异常数据（汇总）

        Args:
            data_list: 数据列表

        Returns:
            异常检测结果
        """
        anomalies = {
            'low_confidence': 0,
            'short_answers': 0,
            'long_answers': 0,
            'empty_cot': 0,
            'format_errors': 0,
        }

        for data in data_list:
            tasks = data.get('tasks', {})

            for task_name, task_data in tasks.items():
                # 低置信度
                hard_label = task_data.get('hard_label', {})
                confidence = hard_label.get('confidence', 1.0)
                if confidence < 0.5:
                    anomalies['low_confidence'] += 1

                # 答案长度异常
                answer = hard_label.get('answer', '')
                if len(answer) < 3:
                    anomalies['short_answers'] += 1
                elif len(answer) > 100:
                    anomalies['long_answers'] += 1

                # 空CoT
                cot = task_data.get('cot_reasoning', {})
                if not cot or not cot.get('raw_reasoning'):
                    anomalies['empty_cot'] += 1

        return {
            'total_anomalies': sum(anomalies.values()),
            'by_type': anomalies,
        }

    def _generate_quality_recommendations(
        self,
        confidence_analysis: Dict,
        quality_analysis: Dict,
        cot_analysis: Dict,
        anomaly_analysis: Dict
    ) -> List[str]:
        """
        生成数据质量建议

        Args:
            confidence_analysis: 置信度分析结果
            quality_analysis: 质量分数分析结果
            cot_analysis: CoT分析结果
            anomaly_analysis: 异常检测结果

        Returns:
            建议列表
        """
        recommendations = []

        # 1. 整体质量评估
        avg_quality = quality_analysis.get('mean', 0)
        if avg_quality >= 70:
            recommendations.append(
                f"✓ 数据质量优秀（平均{avg_quality:.1f}分），可直接用于训练"
            )
        elif avg_quality >= 60:
            recommendations.append(
                f"✓ 数据质量良好（平均{avg_quality:.1f}分），建议使用前检查低质量样本"
            )
        elif avg_quality >= 50:
            recommendations.append(
                f"⚠️ 数据质量中等（平均{avg_quality:.1f}分），建议使用更严格的清洗参数"
            )
        else:
            recommendations.append(
                f"❌ 数据质量较差（平均{avg_quality:.1f}分），建议重新生成数据或调整参数"
            )

        # 2. 置信度评估
        low_conf_rate = confidence_analysis.get('low_rate', 0)
        if low_conf_rate > 0.2:
            recommendations.append(
                f"⚠️ 低置信度样本占比{low_conf_rate*100:.1f}%，建议检查教师模型生成质量"
            )

        # 3. CoT覆盖率
        cot_rate = cot_analysis.get('cot_rate', 0)
        if cot_rate < 0.8:
            recommendations.append(
                f"⚠️ CoT覆盖率仅{cot_rate*100:.1f}%，部分样本缺少推理过程"
            )
        elif cot_rate >= 0.9:
            recommendations.append(
                f"✓ CoT覆盖率{cot_rate*100:.1f}%，推理数据完整"
            )

        # 4. 异常评估
        anomaly_count = anomaly_analysis.get('total_anomalies', 0)
        if anomaly_count > 0:
            anomaly_rate = anomaly_count / (confidence_analysis.get('count', 1) or 1)
            if anomaly_rate > 0.1:
                recommendations.append(
                    f"⚠️ 异常率{anomaly_rate*100:.1f}%，建议人工抽查验证"
                )

        # 5. 建议
        if avg_quality >= 60 and low_conf_rate < 0.1 and anomaly_count < 5:
            recommendations.append(
                "✓ 数据整体可信，建议随机抽查100个样本进行人工验证"
            )

        return recommendations

    def _calculate_overall_score(
        self,
        confidence_analysis: Dict,
        quality_analysis: Dict,
        cot_analysis: Dict,
        anomaly_analysis: Dict
    ) -> float:
        """
        计算整体质量分数

        Returns:
            整体分数 (0-100)
        """
        weights = {
            'quality': 0.4,
            'confidence': 0.3,
            'cot': 0.2,
            'anomaly': 0.1
        }

        # 质量分数
        quality_score = quality_analysis.get('mean', 50)

        # 置信度分数
        avg_confidence = confidence_analysis.get('mean', 0.5)
        confidence_score = avg_confidence * 100

        # CoT分数
        cot_rate = cot_analysis.get('cot_rate', 0)
        cot_score = cot_rate * 100

        # 异常分数（越少越好）
        anomaly_count = anomaly_analysis.get('total_anomalies', 0)
        total_count = confidence_analysis.get('count', 1)
        anomaly_rate = anomaly_count / total_count if total_count > 0 else 0
        anomaly_score = (1 - anomaly_rate) * 100

        overall_score = (
            quality_score * weights['quality'] +
            confidence_score * weights['confidence'] +
            cot_score * weights['cot'] +
            anomaly_score * weights['anomaly']
        )

        return round(overall_score, 2)

    def _display_analysis_summary(self, report: Dict[str, Any]):
        """
        显示分析摘要

        Args:
            report: 分析报告
        """
        self.logger.info("\n" + "-"*70)
        self.logger.info("ANALYSIS SUMMARY")
        self.logger.info("-"*70)

        self.logger.info(f"\nData Count: {report['data_count']}")
        self.logger.info(f"Overall Score: {report['overall_score']:.1f}/100")

        # 置信度
        conf = report['confidence_analysis']
        self.logger.info(f"\nConfidence Analysis:")
        self.logger.info(f"  Average: {conf['mean']:.3f}")
        self.logger.info(f"  Low Confidence: {conf['low_confidence_count']} ({conf['low_rate']*100:.1f}%)")

        # 质量分数
        qual = report['quality_analysis']
        self.logger.info(f"\nQuality Score Analysis:")
        self.logger.info(f"  Average: {qual['mean']:.1f}")
        self.logger.info(f"  Low Quality (<30): {qual['low_quality_count']} ({qual['low_rate']*100:.1f}%)")

        # CoT
        cot = report['cot_analysis']
        self.logger.info(f"\nCoT Analysis:")
        self.logger.info(f"  Coverage: {cot['cot_rate']*100:.1f}%")
        if cot['avg_logical_flow']:
            self.logger.info(f"  Logical Flow: {cot['avg_logical_flow']:.2f}")

        # 异常
        anom = report['anomaly_analysis']
        self.logger.info(f"\nAnomaly Detection:")
        self.logger.info(f"  Total: {anom['total_anomalies']}")
        for anomaly_type, count in anom['by_type'].items():
            if count > 0:
                self.logger.info(f"    {anomaly_type}: {count}")

        # 建议
        self.logger.info(f"\nRecommendations:")
        for rec in report['recommendations']:
            self.logger.info(f"  {rec}")

        self.logger.info("\n" + "="*70)

    def calculate_average_quality(self, input_dir: str) -> float:
        """
        计算数据集的平均质量分数

        Args:
            input_dir: 数据目录路径

        Returns:
            平均质量分数
        """
        data_list = self._load_data_from_dir(input_dir)

        if not data_list:
            return 0.0

        quality_scores = [
            self._estimate_quality_score(data)
            for data in data_list
        ]

        import numpy as np
        return float(np.mean(quality_scores))