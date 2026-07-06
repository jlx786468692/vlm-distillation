"""
验证比较器
=========

负责数据验证和比较，包括：
- 初步验证（清洗前）
- 最终验证（清洗后）
- 验证结果对比

Usage:
    comparator = ValidationComparator(logger)
    before_report = comparator.run_validation(before_dir, 'before')
    after_report = comparator.run_validation(after_dir, 'after')
    comparison = comparator.compare_results(before_report, after_report)
"""

from typing import Dict, Any, List
from pathlib import Path
import json
from datetime import datetime


class ValidationComparator:
    """
    验证比较器

    功能：
    1. 数据完整性验证
    2. 数据格式验证
    3. 数据质量验证
    4. 清洗前后对比分析
    5. 生成对比报告
    """

    def __init__(self, logger: Any = None):
        """
        初始化验证比较器

        Args:
            logger: 日志记录器
        """
        self.logger = logger

    def run_validation(
        self,
        input_dir: str,
        validation_type: str = 'initial'
    ) -> Dict[str, Any]:
        """
        运行数据验证

        Args:
            input_dir: 数据目录路径
            validation_type: 验证类型 ('initial' 或 'final')

        Returns:
            验证报告
        """
        validation_name = {
            'initial': 'Initial Validation (Before Cleaning)',
            'final': 'Final Validation (After Cleaning)'
        }.get(validation_type, 'Validation')

        if self.logger:
            self.logger.info("\n" + "="*70)
            self.logger.info(f"STEP: {validation_name.upper()}")
            self.logger.info("="*70)

        step_start = datetime.now()

        # 加载数据
        data_list = self._load_data_from_dir(input_dir)

        if not data_list:
            if self.logger:
                self.logger.warning("No data found for validation")
            return {
                'success': False,
                'error': 'No data found',
                'validation_type': validation_type,
                'data_count': 0
            }

        if self.logger:
            self.logger.info(f"Validating {len(data_list)} data samples...")

        # 执行验证
        integrity_check = self._check_data_integrity(data_list)
        format_check = self._check_data_format(data_list)
        quality_check = self._check_data_quality(data_list)

        # 整合报告
        report = {
            'success': True,
            'validation_type': validation_type,
            'data_count': len(data_list),
            'integrity_check': integrity_check,
            'format_check': format_check,
            'quality_check': quality_check,
            'overall_passed': (
                integrity_check['passed'] and
                format_check['passed'] and
                quality_check['passed']
            ),
            'start_time': step_start.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_seconds': (datetime.now() - step_start).total_seconds()
        }

        # 显示验证结果
        if self.logger:
            self._display_validation_results(report, validation_name)

        return report

    def compare_validation_results(
        self,
        before_report: Dict[str, Any],
        after_report: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        对比清洗前后验证结果

        Args:
            before_report: 清洗前验证报告
            after_report: 清洗后验证报告

        Returns:
            对比报告
        """
        if self.logger:
            self.logger.info("\n" + "="*70)
            self.logger.info("VALIDATION COMPARISON: BEFORE vs AFTER")
            self.logger.info("="*70)

        comparison = {
            'before': before_report,
            'after': after_report,
            'improvements': {},
            'changes': {}
        }

        # 对比数据量变化
        before_count = before_report.get('data_count', 0)
        after_count = after_report.get('data_count', 0)
        removed_count = before_count - after_count
        removal_rate = removed_count / before_count if before_count > 0 else 0

        comparison['changes']['data_count'] = {
            'before': before_count,
            'after': after_count,
            'removed': removed_count,
            'removal_rate': removal_rate
        }

        # 对比质量变化
        before_quality = before_report.get('quality_check', {})
        after_quality = after_report.get('quality_check', {})

        quality_changes = {
            'low_confidence': {
                'before': before_quality.get('low_confidence_count', 0),
                'after': after_quality.get('low_confidence_count', 0),
                'improved': before_quality.get('low_confidence_count', 0) >
                           after_quality.get('low_confidence_count', 0)
            },
            'invalid_answers': {
                'before': before_quality.get('invalid_answer_count', 0),
                'after': after_quality.get('invalid_answer_count', 0),
                'improved': before_quality.get('invalid_answer_count', 0) >
                           after_quality.get('invalid_answer_count', 0)
            },
            'empty_cot': {
                'before': before_quality.get('empty_cot_count', 0),
                'after': after_quality.get('empty_cot_count', 0),
                'improved': before_quality.get('empty_cot_count', 0) >
                           after_quality.get('empty_cot_count', 0)
            }
        }

        comparison['changes']['quality'] = quality_changes

        # 计算改进率
        improvements = {
            'data_quality_improved': after_report.get('overall_passed', False) and
                                     not before_report.get('overall_passed', False),
            'removal_rate': removal_rate,
            'quality_improvements_count': sum(
                1 for change in quality_changes.values()
                if change['improved']
            )
        }

        comparison['improvements'] = improvements

        # 显示对比结果
        if self.logger:
            self._display_comparison_results(comparison)

        # 保存对比报告
        comparison_report_path = Path('./outputs/validation_comparison.json')
        comparison_report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(comparison_report_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)

        if self.logger:
            self.logger.info(f"\n✓ Comparison report saved to: {comparison_report_path}")

        return comparison

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

    def _check_data_integrity(self, data_list: List[Dict]) -> Dict[str, Any]:
        """
        检查数据完整性

        Args:
            data_list: 数据列表

        Returns:
            完整性检查结果
        """
        checks = {
            'has_image_id': True,
            'has_tasks': True,
            'has_valid_json': True,
            'errors': []
        }

        for idx, data in enumerate(data_list):
            # 检查image_id
            if not data.get('image_id'):
                checks['has_image_id'] = False
                checks['errors'].append(f"Sample {idx}: missing image_id")

            # 检查tasks
            if not data.get('tasks'):
                checks['has_tasks'] = False
                checks['errors'].append(f"Sample {idx}: missing tasks")

        checks['passed'] = (
            checks['has_image_id'] and
            checks['has_tasks'] and
            checks['has_valid_json']
        )

        checks['error_count'] = len(checks['errors'])

        return checks

    def _check_data_format(self, data_list: List[Dict]) -> Dict[str, Any]:
        """
        检查数据格式

        Args:
            data_list: 数据列表

        Returns:
            格式检查结果
        """
        checks = {
            'valid_task_structure': True,
            'valid_hard_label': True,
            'valid_soft_label': True,
            'valid_cot': True,
            'errors': []
        }

        for idx, data in enumerate(data_list):
            tasks = data.get('tasks', {})

            for task_name, task_data in tasks.items():
                # 检查hard_label
                hard_label = task_data.get('hard_label', {})
                if not hard_label.get('answer'):
                    checks['valid_hard_label'] = False
                    checks['errors'].append(
                        f"Sample {idx}, task {task_name}: missing hard_label answer"
                    )

                # 检查soft_label
                soft_label = task_data.get('soft_label', {})
                if soft_label and not soft_label.get('answer_distribution'):
                    checks['valid_soft_label'] = False
                    checks['errors'].append(
                        f"Sample {idx}, task {task_name}: invalid soft_label"
                    )

                # 检查cot
                cot = task_data.get('cot_reasoning', {})
                if cot and not cot.get('raw_reasoning'):
                    checks['valid_cot'] = False
                    checks['errors'].append(
                        f"Sample {idx}, task {task_name}: invalid cot"
                    )

        checks['passed'] = (
            checks['valid_task_structure'] and
            len(checks['errors']) < len(data_list) * 0.1  # 允许10%错误率
        )

        checks['error_count'] = len(checks['errors'])

        return checks

    def _check_data_quality(self, data_list: List[Dict]) -> Dict[str, Any]:
        """
        检查数据质量

        Args:
            data_list: 数据列表

        Returns:
            质量检查结果
        """
        checks = {
            'low_confidence_count': 0,
            'invalid_answer_count': 0,
            'empty_cot_count': 0,
            'short_answer_count': 0,
            'passed': True
        }

        invalid_answers = ['unknown', 'n/a', 'none', 'unclear', 'cannot determine', '']

        for data in data_list:
            tasks = data.get('tasks', {})

            for task_name, task_data in tasks.items():
                hard_label = task_data.get('hard_label', {})

                # 低置信度
                confidence = hard_label.get('confidence', 1.0)
                if confidence < 0.5:
                    checks['low_confidence_count'] += 1

                # 无效答案
                answer = hard_label.get('answer', '')
                if answer.lower() in invalid_answers:
                    checks['invalid_answer_count'] += 1

                # 过短答案
                if len(answer) < 3:
                    checks['short_answer_count'] += 1

                # 空CoT
                cot = task_data.get('cot_reasoning', {})
                if not cot or not cot.get('raw_reasoning'):
                    checks['empty_cot_count'] += 1

        # 判断是否通过（阈值可调整）
        total_checks = len(data_list) * 3  # 假设平均3个任务
        checks['passed'] = (
            checks['low_confidence_count'] < total_checks * 0.1 and
            checks['invalid_answer_count'] < total_checks * 0.05 and
            checks['empty_cot_count'] < total_checks * 0.2
        )

        checks['low_confidence_rate'] = (
            checks['low_confidence_count'] / total_checks
            if total_checks > 0 else 0
        )

        return checks

    def _display_validation_results(
        self,
        report: Dict[str, Any],
        validation_name: str
    ):
        """
        显示验证结果

        Args:
            report: 验证报告
            validation_name: 验证名称
        """
        self.logger.info(f"\n{validation_name} Results:")
        self.logger.info("-"*70)

        self.logger.info(f"\nData Count: {report['data_count']}")

        # 完整性检查
        integrity = report['integrity_check']
        self.logger.info(f"\nIntegrity Check:")
        self.logger.info(f"  ✓ Passed: {integrity['passed']}")
        self.logger.info(f"  - Errors: {integrity['error_count']}")

        # 格式检查
        format_check = report['format_check']
        self.logger.info(f"\nFormat Check:")
        self.logger.info(f"  ✓ Passed: {format_check['passed']}")
        self.logger.info(f"  - Errors: {format_check['error_count']}")

        # 质量检查
        quality = report['quality_check']
        self.logger.info(f"\nQuality Check:")
        self.logger.info(f"  ✓ Passed: {quality['passed']}")
        self.logger.info(f"  - Low Confidence: {quality['low_confidence_count']}")
        self.logger.info(f"  - Invalid Answers: {quality['invalid_answer_count']}")
        self.logger.info(f"  - Empty CoT: {quality['empty_cot_count']}")

        # 整体结果
        self.logger.info(f"\n{'='*70}")
        if report['overall_passed']:
            self.logger.info("✓ VALIDATION PASSED")
        else:
            self.logger.info("⚠️ VALIDATION FAILED - Issues detected")
        self.logger.info(f"{'='*70}")

    def _display_comparison_results(self, comparison: Dict[str, Any]):
        """
        显示对比结果

        Args:
            comparison: 对比报告
        """
        self.logger.info("\n" + "-"*70)
        self.logger.info("Comparison Results:")
        self.logger.info("-"*70)

        # 数据量变化
        data_changes = comparison['changes']['data_count']
        self.logger.info(f"\nData Count Changes:")
        self.logger.info(f"  Before: {data_changes['before']}")
        self.logger.info(f"  After: {data_changes['after']}")
        self.logger.info(f"  Removed: {data_changes['removed']} ({data_changes['removal_rate']*100:.1f}%)")

        # 质量变化
        quality_changes = comparison['changes']['quality']
        self.logger.info(f"\nQuality Changes:")

        for metric, change in quality_changes.items():
            metric_name = metric.replace('_', ' ').title()
            self.logger.info(f"  {metric_name}:")
            self.logger.info(f"    Before: {change['before']}")
            self.logger.info(f"    After: {change['after']}")
            if change['improved']:
                self.logger.info(f"    ✓ Improved")
            else:
                self.logger.info(f"    ⚠️ No improvement")

        # 改进总结
        improvements = comparison['improvements']
        self.logger.info(f"\nImprovements Summary:")
        self.logger.info(f"  Quality Improvements: {improvements['quality_improvements_count']}")
        self.logger.info(f"  Removal Rate: {improvements['removal_rate']*100:.1f}%")

        self.logger.info("\n" + "="*70)