"""
深度数据验证模块
================

提供蒸馏数据的深度质量验证功能：
1. TeacherOutputValidator - 7B 老师输出可靠性验证
2. LabelDistributionValidator - 标签分布无偏移验证
3. CoTHallucinationValidator - CoT 幻觉检测
4. DataValidator - 整合入口

Usage:
    validator = DataValidator(config, logger)
    report = validator.run_all_validations('./outputs/merged')
    validator.export_visualizations()
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import json
import math
from collections import Counter

from .config import ConfigManager
from .logger import get_logger


# ============================================================
# Teacher Output Validator - 7B 老师输出可靠性验证
# ============================================================

class TeacherOutputValidator:
    """
    7B 老师输出可靠性验证

    检查项：
    - 置信度分布统计（均值、方差、分位数）
    - 异常置信度检测（异常高/异常低）
    - 答案合理性检查（无效答案、过短/过长答案）
    - 任务间置信度差异
    """

    # 无效答案列表
    INVALID_ANSWERS = [
        'unknown', 'n/a', 'none', 'null', 'nothing',
        'i don\'t know', 'cannot determine', 'unclear',
        'no answer', 'not sure', ''
    ]

    def __init__(self, config: ConfigManager, logger: Any = None):
        self.config = config
        self.logger = logger

        # 阈值配置
        self.min_confidence = config.get('cleaning.min_confidence', 0.6)
        self.max_confidence = config.get('cleaning.max_confidence', 0.95)
        self.min_answer_length = config.get('cleaning.min_answer_length', 3)
        self.max_answer_length = config.get('cleaning.max_answer_length', 100)

    def validate(self, data_list: List[Dict]) -> Dict[str, Any]:
        """
        执行所有老师输出验证

        Args:
            data_list: 数据列表

        Returns:
            验证报告
        """
        report = {
            'validator': 'TeacherOutputValidator',
            'timestamp': datetime.now().isoformat(),
            'sample_count': len(data_list),
            'checks': {}
        }

        # 1. 置信度分布
        report['checks']['confidence_distribution'] = self._check_confidence_distribution(data_list)

        # 2. 异常置信度检测
        report['checks']['confidence_anomaly'] = self._check_confidence_anomaly(data_list)

        # 3. 答案合理性
        report['checks']['answer_reasonability'] = self._check_answer_reasonability(data_list)

        # 4. 任务置信度差异
        report['checks']['task_confidence_diff'] = self._check_task_confidence_diff(data_list)

        # 计算总体通过状态
        report['passed'] = all(
            check.get('passed', True)
            for check in report['checks'].values()
        )

        return report

    def _check_confidence_distribution(self, data_list: List[Dict]) -> Dict[str, Any]:
        """置信度分布统计"""
        confidences = []

        for data in data_list:
            tasks = data.get('tasks', {})
            for task_name, task_data in tasks.items():
                hard_label = task_data.get('hard_label', {})
                conf = hard_label.get('confidence', None)
                if conf is not None:
                    confidences.append(conf)

        if not confidences:
            return {
                'passed': False,
                'error': 'No confidence data found',
                'statistics': {}
            }

        # 统计计算
        mean_conf = sum(confidences) / len(confidences)
        variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)
        std_dev = math.sqrt(variance)

        # 分位数
        sorted_conf = sorted(confidences)
        n = len(sorted_conf)
        p25 = sorted_conf[n // 4] if n >= 4 else sorted_conf[0]
        p50 = sorted_conf[n // 2]
        p75 = sorted_conf[3 * n // 4] if n >= 4 else sorted_conf[-1]

        # 判断通过状态
        passed = mean_conf >= self.min_confidence

        warnings = []
        if mean_conf < 0.5:
            warnings.append('平均置信度过低 (< 0.5)，老师输出可能不可靠')
        if std_dev > 0.3:
            warnings.append('置信度方差过大 (> 0.3)，输出稳定性差')

        return {
            'passed': passed,
            'statistics': {
                'count': len(confidences),
                'mean': round(mean_conf, 4),
                'std_dev': round(std_dev, 4),
                'min': min(confidences),
                'max': max(confidences),
                'p25': p25,
                'p50': p50,
                'p75': p75,
            },
            'warnings': warnings,
            'threshold': self.min_confidence
        }

    def _check_confidence_anomaly(self, data_list: List[Dict]) -> Dict[str, Any]:
        """异常置信度检测"""
        low_conf_count = 0  # < 0.3
        high_conf_count = 0  # > 0.95
        total_count = 0

        anomaly_samples = []

        for data in data_list:
            tasks = data.get('tasks', {})
            for task_name, task_data in tasks.items():
                hard_label = task_data.get('hard_label', {})
                conf = hard_label.get('confidence', None)
                if conf is not None:
                    total_count += 1

                    if conf < 0.3:
                        low_conf_count += 1
                        anomaly_samples.append({
                            'image_id': data.get('image_id'),
                            'task': task_name,
                            'confidence': conf,
                            'type': 'low'
                        })

                    if conf > 0.95:
                        high_conf_count += 1

        # 计算比例
        low_conf_ratio = low_conf_count / total_count if total_count > 0 else 0
        high_conf_ratio = high_conf_count / total_count if total_count > 0 else 0

        # 判断通过状态
        passed = low_conf_ratio < 0.2 and high_conf_ratio < 0.1

        warnings = []
        if low_conf_ratio > 0.2:
            warnings.append(f'低置信度样本比例过高 ({low_conf_ratio*100:.1f}% > 20%)')
        if high_conf_ratio > 0.1:
            warnings.append(f'异常高置信度样本比例过高 ({high_conf_ratio*100:.1f}% > 10%)，可能过于自信')

        return {
            'passed': passed,
            'statistics': {
                'total_count': total_count,
                'low_confidence_count': low_conf_count,
                'high_confidence_count': high_conf_count,
                'low_confidence_ratio': round(low_conf_ratio, 4),
                'high_confidence_ratio': round(high_conf_ratio, 4),
            },
            'warnings': warnings,
            'anomaly_samples': anomaly_samples[:20]  # 只保留前20个样本
        }

    def _check_answer_reasonability(self, data_list: List[Dict]) -> Dict[str, Any]:
        """答案合理性检查"""
        invalid_count = 0
        short_count = 0
        long_count = 0
        total_count = 0

        invalid_samples = []

        for data in data_list:
            tasks = data.get('tasks', {})
            for task_name, task_data in tasks.items():
                hard_label = task_data.get('hard_label', {})

                # VQA 答案检查
                answer = hard_label.get('answer', '')

                if answer:
                    total_count += 1

                    # 无效答案检查
                    if answer.lower().strip() in self.INVALID_ANSWERS:
                        invalid_count += 1
                        invalid_samples.append({
                            'image_id': data.get('image_id'),
                            'task': task_name,
                            'answer': answer,
                            'type': 'invalid'
                        })

                    # 长度检查
                    answer_len = len(answer)
                    if answer_len < self.min_answer_length:
                        short_count += 1

                    if answer_len > self.max_answer_length:
                        long_count += 1

        # 计算比例
        invalid_ratio = invalid_count / total_count if total_count > 0 else 0
        short_ratio = short_count / total_count if total_count > 0 else 0
        long_ratio = long_count / total_count if total_count > 0 else 0

        # 判断通过状态
        passed = invalid_ratio < 0.05 and short_ratio < 0.1

        warnings = []
        if invalid_ratio > 0.05:
            warnings.append(f'无效答案比例过高 ({invalid_ratio*100:.1f}% > 5%)')
        if short_ratio > 0.1:
            warnings.append(f'过短答案比例过高 ({short_ratio*100:.1f}% > 10%)')

        return {
            'passed': passed,
            'statistics': {
                'total_count': total_count,
                'invalid_count': invalid_count,
                'short_count': short_count,
                'long_count': long_count,
                'invalid_ratio': round(invalid_ratio, 4),
                'short_ratio': round(short_ratio, 4),
                'long_ratio': round(long_ratio, 4),
            },
            'warnings': warnings,
            'invalid_samples': invalid_samples[:20]
        }

    def _check_task_confidence_diff(self, data_list: List[Dict]) -> Dict[str, Any]:
        """任务间置信度差异"""
        task_confidences = {}

        for data in data_list:
            tasks = data.get('tasks', {})
            for task_name, task_data in tasks.items():
                hard_label = task_data.get('hard_label', {})
                conf = hard_label.get('confidence', None)

                if conf is not None:
                    if task_name not in task_confidences:
                        task_confidences[task_name] = []
                    task_confidences[task_name].append(conf)

        # 计算各任务平均置信度
        task_stats = {}
        for task_name, confs in task_confidences.items():
            mean_conf = sum(confs) / len(confs)
            task_stats[task_name] = {
                'count': len(confs),
                'mean_confidence': round(mean_conf, 4),
            }

        # 检查差异
        if len(task_stats) > 1:
            mean_values = [s['mean_confidence'] for s in task_stats.values()]
            max_diff = max(mean_values) - min(mean_values)

            passed = max_diff < 0.2

            warnings = []
            if max_diff > 0.2:
                warnings.append(f'任务间置信度差异过大 ({max_diff:.2f} > 0.2)')
        else:
            passed = True
            max_diff = 0
            warnings = []

        return {
            'passed': passed,
            'statistics': task_stats,
            'max_difference': round(max_diff, 4),
            'warnings': warnings
        }


# ============================================================
# Label Distribution Validator - 标签分布无偏移验证
# ============================================================

class LabelDistributionValidator:
    """
    标签分布无偏移验证

    检查项：
    - VQA 答案频率统计（Top-20 高频答案）
    - 答案熵值计算（分布均匀度）
    - Caption 多样性/长度检查
    - 检测类别分布（类别频率统计）
    - 偏移预警报告
    """

    def __init__(self, config: ConfigManager, logger: Any = None):
        self.config = config
        self.logger = logger

    def validate(self, data_list: List[Dict]) -> Dict[str, Any]:
        """执行所有标签分布验证"""
        report = {
            'validator': 'LabelDistributionValidator',
            'timestamp': datetime.now().isoformat(),
            'sample_count': len(data_list),
            'checks': {}
        }

        # 1. VQA 答案分布
        report['checks']['vqa_distribution'] = self._check_vqa_distribution(data_list)

        # 计算总体通过状态
        report['passed'] = all(
            check.get('passed', True)
            for check in report['checks'].values()
        )

        return report

    def _check_vqa_distribution(self, data_list: List[Dict]) -> Dict[str, Any]:
        """VQA 答案分布检查"""
        answers = []
        yes_no_count = 0
        total_count = 0

        for data in data_list:
            tasks = data.get('tasks', {})
            vqa_data = tasks.get('vqa', {})
            hard_label = vqa_data.get('hard_label', {})
            answer = hard_label.get('answer', '')

            if answer:
                answers.append(answer.lower().strip())
                total_count += 1

                # Yes/No 统计
                if answer.lower().strip() in ['yes', 'no', 'yes.', 'no.']:
                    yes_no_count += 1

        if not answers:
            return {
                'passed': True,
                'statistics': {},
                'warnings': ['No VQA data found']
            }

        # 频率统计
        counter = Counter(answers)
        top_20 = counter.most_common(20)

        # 熵值计算
        entropy = self._calculate_entropy(counter, total_count)

        # Yes/No 比例
        yes_no_ratio = yes_no_count / total_count if total_count > 0 else 0

        # Top-1 频率
        top1_answer, top1_count = top_20[0] if top_20 else ('', 0)
        top1_ratio = top1_count / total_count if total_count > 0 else 0

        # 判断通过状态
        passed = entropy >= 3.0 and top1_ratio < 0.3

        warnings = []
        if entropy < 3.0:
            warnings.append(f'答案分布熵值过低 ({entropy:.2f} < 3.0)，分布过于集中')
        if top1_ratio > 0.3:
            warnings.append(f'Top-1 答案 "{top1_answer}" 占比过高 ({top1_ratio*100:.1f}% > 30%)')
        if yes_no_ratio > 0.5:
            warnings.append(f'Yes/No 类答案占比过高 ({yes_no_ratio*100:.1f}% > 50%)')

        return {
            'passed': passed,
            'statistics': {
                'total_count': total_count,
                'unique_answers': len(counter),
                'entropy': round(entropy, 2),
                'yes_no_ratio': round(yes_no_ratio, 4),
                'top_1_answer': top1_answer,
                'top_1_ratio': round(top1_ratio, 4),
            },
            'top_20_answers': top_20,
            'warnings': warnings
        }

    def _calculate_entropy(self, counter: Counter, total: int) -> float:
        """计算熵值"""
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in counter.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy


# ============================================================
# CoT Hallucination Validator - CoT 幻觉检测
# ============================================================

class CoTHallucinationValidator:
    """
    CoT 幻觉检测（基于规则检查）

    检查项：
    - 逻辑自洽性检查（利用 quality_metrics.logical_flow_score）
    - 推理步骤质量检查
    - 关键词覆盖率检查
    - 幻觉风险评分计算
    """

    # 关键词列表（来自 CoTGenerator）
    REQUIRED_KEYWORDS = ['first', 'next', 'then', 'finally', 'therefore', 'because', 'so']

    def __init__(self, config: ConfigManager, logger: Any = None):
        self.config = config
        self.logger = logger

        self.min_cot_quality = config.get('cleaning.min_cot_quality', 0.5)

    def validate(self, data_list: List[Dict]) -> Dict[str, Any]:
        """执行所有 CoT 幻觉检测"""
        report = {
            'validator': 'CoTHallucinationValidator',
            'timestamp': datetime.now().isoformat(),
            'sample_count': len(data_list),
            'checks': {}
        }

        # 1. 逻辑自洽性
        report['checks']['logical_consistency'] = self._check_logical_consistency(data_list)

        # 2. 推理步骤质量
        report['checks']['step_quality'] = self._check_step_quality(data_list)

        # 3. 关键词覆盖率
        report['checks']['keyword_coverage'] = self._check_keyword_coverage(data_list)

        # 4. 幻觉风险评分
        report['checks']['hallucination_risk'] = self._calculate_hallucination_risk(data_list)

        # 计算总体通过状态
        report['passed'] = all(
            check.get('passed', True)
            for check in report['checks'].values()
        )

        return report

    def _check_logical_consistency(self, data_list: List[Dict]) -> Dict[str, Any]:
        """逻辑自洽性检查"""
        flow_scores = []
        empty_count = 0
        total_count = 0

        for data in data_list:
            tasks = data.get('tasks', {})
            for task_name, task_data in tasks.items():
                cot = task_data.get('cot_reasoning', {})

                total_count += 1

                if not cot or not cot.get('raw_reasoning'):
                    empty_count += 1
                    continue

                quality = cot.get('quality_metrics', {})
                flow_score = quality.get('logical_flow_score', 0)
                flow_scores.append(flow_score)

        if not flow_scores:
            return {
                'passed': empty_count / total_count < 0.2 if total_count > 0 else True,
                'statistics': {
                    'total_count': total_count,
                    'empty_count': empty_count,
                    'empty_ratio': round(empty_count / total_count, 4) if total_count > 0 else 0,
                },
                'warnings': ['No valid CoT with quality metrics found']
            }

        # 统计计算
        avg_flow_score = sum(flow_scores) / len(flow_scores)
        low_flow_count = sum(1 for s in flow_scores if s < 0.5)

        # 判断通过状态
        passed = avg_flow_score >= self.min_cot_quality and empty_count / total_count < 0.2

        warnings = []
        if avg_flow_score < self.min_cot_quality:
            warnings.append(f'平均逻辑流畅度分数过低 ({avg_flow_score:.2f} < {self.min_cot_quality})')
        if empty_count / total_count > 0.2:
            warnings.append(f'空推理比例过高 ({empty_count/total_count*100:.1f}% > 20%)')

        return {
            'passed': passed,
            'statistics': {
                'total_count': total_count,
                'valid_count': len(flow_scores),
                'empty_count': empty_count,
                'avg_flow_score': round(avg_flow_score, 4),
                'low_flow_count': low_flow_count,
                'empty_ratio': round(empty_count / total_count, 4) if total_count > 0 else 0,
            },
            'warnings': warnings
        }

    def _check_step_quality(self, data_list: List[Dict]) -> Dict[str, Any]:
        """推理步骤质量检查"""
        step_counts = []
        low_step_count = 0  # < 2 steps
        total_count = 0

        for data in data_list:
            tasks = data.get('tasks', {})
            for task_name, task_data in tasks.items():
                cot = task_data.get('cot_reasoning', {})

                if not cot:
                    continue

                total_count += 1

                # 从 quality_metrics 获取步骤数
                quality = cot.get('quality_metrics', {})
                step_count = quality.get('step_count', 0)

                if step_count > 0:
                    step_counts.append(step_count)

                if step_count < 2:
                    low_step_count += 1

        if not step_counts:
            return {
                'passed': True,
                'statistics': {},
                'warnings': ['No step data found']
            }

        # 统计计算
        avg_steps = sum(step_counts) / len(step_counts)

        # 判断通过状态
        low_step_ratio = low_step_count / total_count if total_count > 0 else 0
        passed = low_step_ratio < 0.3

        warnings = []
        if low_step_ratio > 0.3:
            warnings.append(f'少步骤（<2）推理比例过高 ({low_step_ratio*100:.1f}% > 30%)')

        return {
            'passed': passed,
            'statistics': {
                'total_count': total_count,
                'avg_steps': round(avg_steps, 2),
                'min_steps': min(step_counts),
                'max_steps': max(step_counts),
                'low_step_count': low_step_count,
                'low_step_ratio': round(low_step_ratio, 4),
            },
            'warnings': warnings
        }

    def _check_keyword_coverage(self, data_list: List[Dict]) -> Dict[str, Any]:
        """关键词覆盖率检查"""
        keyword_counts = {kw: 0 for kw in self.REQUIRED_KEYWORDS}
        low_coverage_count = 0  # < 2 keywords
        total_count = 0

        for data in data_list:
            tasks = data.get('tasks', {})
            for task_name, task_data in tasks.items():
                cot = task_data.get('cot_reasoning', {})
                raw_reasoning = cot.get('raw_reasoning', '')

                if not raw_reasoning:
                    continue

                total_count += 1

                # 统计关键词出现
                count = 0
                for kw in self.REQUIRED_KEYWORDS:
                    if kw in raw_reasoning.lower():
                        keyword_counts[kw] += 1
                        count += 1

                if count < 2:
                    low_coverage_count += 1

        if total_count == 0:
            return {
                'passed': True,
                'statistics': {},
                'warnings': ['No CoT data found']
            }

        # 计算覆盖率
        keyword_coverage = {
            kw: round(count / total_count, 4) for kw, count in keyword_counts.items()
        }

        low_coverage_ratio = low_coverage_count / total_count

        # 判断通过状态
        passed = low_coverage_ratio < 0.4

        warnings = []
        if low_coverage_ratio > 0.4:
            warnings.append(f'低关键词覆盖率（<2）比例过高 ({low_coverage_ratio*100:.1f}% > 40%)')

        return {
            'passed': passed,
            'statistics': {
                'total_count': total_count,
                'keyword_coverage': keyword_coverage,
                'low_coverage_count': low_coverage_count,
                'low_coverage_ratio': round(low_coverage_ratio, 4),
            },
            'warnings': warnings
        }

    def _calculate_hallucination_risk(self, data_list: List[Dict]) -> Dict[str, Any]:
        """幻觉风险评分计算"""
        risk_scores = []
        high_risk_samples = []
        total_count = 0

        for data in data_list:
            tasks = data.get('tasks', {})
            for task_name, task_data in tasks.items():
                cot = task_data.get('cot_reasoning', {})

                if not cot:
                    continue

                total_count += 1

                # 获取指标
                quality = cot.get('quality_metrics', {})
                flow_score = quality.get('logical_flow_score', 0)
                step_count = quality.get('step_count', 0)
                keyword_count = quality.get('keyword_count', 0)

                # 计算步骤得分（归一化到 0-1）
                step_score = min(step_count / 5.0, 1.0)  # 假设5步为满分

                # 计算关键词得分
                keyword_score = min(keyword_count / 4.0, 1.0)  # 假设4个关键词为满分

                # 综合幻觉风险评分
                # risk_score = 1 - (flow * 0.4 + keyword * 0.3 + step * 0.3)
                quality_score = flow_score * 0.4 + keyword_score * 0.3 + step_score * 0.3
                risk_score = 1.0 - quality_score

                risk_scores.append(risk_score)

                if risk_score > 0.5:
                    high_risk_samples.append({
                        'image_id': data.get('image_id'),
                        'task': task_name,
                        'risk_score': round(risk_score, 4),
                        'flow_score': flow_score,
                        'step_count': step_count,
                        'keyword_count': keyword_count,
                    })

        if not risk_scores:
            return {
                'passed': True,
                'statistics': {},
                'warnings': ['No CoT data found']
            }

        # 统计计算
        avg_risk = sum(risk_scores) / len(risk_scores)
        high_risk_count = sum(1 for r in risk_scores if r > 0.5)
        high_risk_ratio = high_risk_count / len(risk_scores)

        # 判断通过状态
        passed = high_risk_ratio < 0.1

        warnings = []
        if high_risk_ratio > 0.1:
            warnings.append(f'高风险样本比例过高 ({high_risk_ratio*100:.1f}% > 10%)')

        return {
            'passed': passed,
            'statistics': {
                'total_count': len(risk_scores),
                'avg_risk_score': round(avg_risk, 4),
                'high_risk_count': high_risk_count,
                'high_risk_ratio': round(high_risk_ratio, 4),
            },
            'warnings': warnings,
            'high_risk_samples': high_risk_samples[:20]
        }


# ============================================================
# Data Validator - 整合入口
# ============================================================

class DataValidator:
    """
    数据验证整合入口

    功能：
    - 执行所有验证模块
    - 生成综合报告
    - 导出可视化
    """

    def __init__(self, config: ConfigManager = None, logger: Any = None):
        self.config = config or ConfigManager()
        self.logger = logger or get_logger()

        # 初始化子验证器
        self.teacher_validator = TeacherOutputValidator(self.config, self.logger)
        self.distribution_validator = LabelDistributionValidator(self.config, self.logger)
        self.hallucination_validator = CoTHallucinationValidator(self.config, self.logger)

        # 输出目录
        self.output_dir = Path(self.config.get('output.root_dir', './outputs'))

    def run_all_validations(
        self,
        input_dir: str,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        执行所有数据验证

        Args:
            input_dir: 数据目录路径
            output_dir: 输出报告目录

        Returns:
            综合验证报告
        """
        if output_dir:
            self.output_dir = Path(output_dir)

        self.logger.info("\n" + "="*70)
        self.logger.info("深度数据验证开始")
        self.logger.info("="*70)

        start_time = datetime.now()

        # 加载数据
        data_list = self._load_data(input_dir)

        if not data_list:
            self.logger.warning("No data found for validation")
            return {
                'success': False,
                'error': 'No data found',
                'input_dir': input_dir
            }

        self.logger.info(f"加载 {len(data_list)} 个数据样本")

        # 执行验证
        validation_results = {}

        self.logger.info("\n[1/3] Teacher Output Validation...")
        validation_results['teacher_output'] = self.teacher_validator.validate(data_list)
        self._display_check_result(validation_results['teacher_output'])

        self.logger.info("\n[2/3] Label Distribution Validation...")
        validation_results['label_distribution'] = self.distribution_validator.validate(data_list)
        self._display_check_result(validation_results['label_distribution'])

        self.logger.info("\n[3/3] CoT Hallucination Validation...")
        validation_results['cot_hallucination'] = self.hallucination_validator.validate(data_list)
        self._display_check_result(validation_results['cot_hallucination'])

        # 整合报告
        report = {
            'success': True,
            'input_dir': input_dir,
            'sample_count': len(data_list),
            'validation_results': validation_results,
            'overall_passed': all(
                result.get('passed', True)
                for result in validation_results.values()
            ),
            'start_time': start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_seconds': (datetime.now() - start_time).total_seconds()
        }

        # 保存报告
        self._save_report(report)

        # 显示最终结果
        self._display_final_result(report)

        return report

    def _load_data(self, input_dir: str) -> List[Dict]:
        """从目录加载所有数据文件"""
        input_path = Path(input_dir)
        json_files = list(input_path.glob("*.json"))

        # 过滤掉报告文件
        data_files = [
            f for f in json_files
            if not f.name.startswith((
                'cleaning_report', 'merged_summary', 'validation',
                'checkpoint', 'pipeline', 'visualization',
                'data_quality', 'timing', 'deep_validation'
            ))
        ]

        data_list = []
        for json_file in data_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                data_list.append(data)
            except Exception as e:
                self.logger.warning(f"Failed to load {json_file}: {e}")

        return data_list

    def _display_check_result(self, result: Dict):
        """显示单个验证结果"""
        validator = result.get('validator', 'Unknown')
        passed = result.get('passed', False)

        status = "✓ PASSED" if passed else "⚠️ FAILED"
        self.logger.info(f"  {validator}: {status}")

        # 显示警告
        for check_name, check_data in result.get('checks', {}).items():
            warnings = check_data.get('warnings', [])
            if warnings:
                self.logger.warning(f"    [{check_name}] {len(warnings)} warnings")
                for w in warnings[:3]:
                    self.logger.warning(f"      - {w}")

    def _display_final_result(self, report: Dict):
        """显示最终验证结果"""
        self.logger.info("\n" + "="*70)
        self.logger.info("验证完成")
        self.logger.info("="*70)

        self.logger.info(f"\n样本数量: {report['sample_count']}")
        self.logger.info(f"耗时: {report['duration_seconds']:.2f} 秒")

        overall = "✓ 所有验证通过" if report['overall_passed'] else "⚠️ 验证发现问题"
        self.logger.info(f"\n总体结果: {overall}")

        if not report['overall_passed']:
            self.logger.warning("\n建议检查以下问题:")
            for validator_name, result in report['validation_results'].items():
                if not result.get('passed', True):
                    for check_name, check_data in result.get('checks', {}).items():
                        if not check_data.get('passed', True):
                            warnings = check_data.get('warnings', [])
                            for w in warnings:
                                self.logger.warning(f"  - [{validator_name}/{check_name}] {w}")

        self.logger.info(f"\n报告保存位置: {self.output_dir / 'deep_validation_report.json'}")

    def _save_report(self, report: Dict):
        """保存验证报告"""
        output_path = self.output_dir
        output_path.mkdir(parents=True, exist_ok=True)

        report_file = output_path / 'deep_validation_report.json'

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.logger.info(f"\n报告已保存: {report_file}")

    def get_validation_summary(self, report: Dict) -> str:
        """生成验证摘要文本"""
        lines = []

        lines.append("="*60)
        lines.append("深度数据验证报告摘要")
        lines.append("="*60)

        lines.append(f"\n样本数量: {report['sample_count']}")
        lines.append(f"总体状态: {'通过' if report['overall_passed'] else '有问题'}")

        # Teacher Output
        teacher = report['validation_results'].get('teacher_output', {})
        checks = teacher.get('checks', {})

        if 'confidence_distribution' in checks:
            stats = checks['confidence_distribution'].get('statistics', {})
            lines.append(f"\n[老师输出可靠性]")
            lines.append(f"  平均置信度: {stats.get('mean', 'N/A')}")
            lines.append(f"  置信度标准差: {stats.get('std_dev', 'N/A')}")

        if 'answer_reasonability' in checks:
            stats = checks['answer_reasonability'].get('statistics', {})
            lines.append(f"  无效答案比例: {stats.get('invalid_ratio', 'N/A')*100:.1f}%")

        # Label Distribution
        distribution = report['validation_results'].get('label_distribution', {})
        checks = distribution.get('checks', {})

        if 'vqa_distribution' in checks:
            stats = checks['vqa_distribution'].get('statistics', {})
            lines.append(f"\n[标签分布]")
            lines.append(f"  VQA答案熵值: {stats.get('entropy', 'N/A')}")
            lines.append(f"  Top-1答案占比: {stats.get('top_1_ratio', 'N/A')*100:.1f}%")

        # CoT Hallucination
        cot = report['validation_results'].get('cot_hallucination', {})
        checks = cot.get('checks', {})

        if 'logical_consistency' in checks:
            stats = checks['logical_consistency'].get('statistics', {})
            lines.append(f"\n[CoT幻觉检测]")
            lines.append(f"  平均逻辑流畅度: {stats.get('avg_flow_score', 'N/A')}")
            lines.append(f"  空推理比例: {stats.get('empty_ratio', 'N/A')*100:.1f}%")

        if 'hallucination_risk' in checks:
            stats = checks['hallucination_risk'].get('statistics', {})
            lines.append(f"  高风险样本比例: {stats.get('high_risk_ratio', 'N/A')*100:.1f}%")

        lines.append("\n" + "="*60)

        return "\n".join(lines)


def run_validation(input_dir: str, output_dir: str = None, visualize: bool = False) -> Dict:
    """
    快速验证函数

    Args:
        input_dir: 数据目录
        output_dir: 输出目录
        visualize: 是否生成可视化

    Returns:
        验证报告
    """
    config = ConfigManager()
    logger = get_logger()

    validator = DataValidator(config, logger)
    report = validator.run_all_validations(input_dir, output_dir)

    if visualize:
        # 可视化将在 data_visualizer.py 中实现
        logger.info("\n可视化功能请使用 DataVisualizer 类")

    return report