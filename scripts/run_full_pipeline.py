"""
完整数据管道脚本（重构版）
=========================

简化后的流程协调脚本，主要负责：
1. 流程步骤协调
2. 参数解析和配置
3. 结果汇总和报告生成

具体功能已提取到独立模块：
- 数据质量分析 → DataQualityAnalyzer
- 验证比较 → ValidationComparator
- 可视化生成 → PipelineVisualizer

Usage:
    # 运行完整流程（使用配置文件参数）
    python scripts/run_full_pipeline.py --samples 5000

    # 仅运行特定步骤
    python scripts/run_full_pipeline.py --steps distillation cleaning

    # 自定义参数（覆盖配置文件）
    python scripts/run_full_pipeline.py \
        --samples 1000 \
        --tasks vqa captioning \
        --min-quality 40 \
        --min-confidence 0.6

    # 启用可视化
    python scripts/run_full_pipeline.py --samples 1000 --enable-visualization
"""

import argparse
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from collections import OrderedDict

# 兼容两种导入方式
try:
    from src import (
        ConfigManager, TeacherModel, Distiller, COCODataLoader,
        setup_logger, DataCleaner
    )
    from src.utils import (
        DataQualityAnalyzer, ValidationComparator, PipelineVisualizer
    )
except ImportError:
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    from src import (
        ConfigManager, TeacherModel, Distiller, COCODataLoader,
        setup_logger, DataCleaner
    )
    from src.utils import (
        DataQualityAnalyzer, ValidationComparator, PipelineVisualizer
    )


class FullPipelineRunner:
    """
    Complete VLM Data Pipeline Runner

    Main responsibilities:
    1. Flow coordination - call steps in sequence
    2. Parameter management - parse CLI and config parameters
    3. Result aggregation - generate final report

    Modules:
    - Data Quality → DataQualityAnalyzer
    - Validation → ValidationComparator
    - Visualization → PipelineVisualizer
    """

    def __init__(self, config_path: str = 'configs/default.yaml'):
        """
        Initialize pipeline runner

        Args:
            config_path: Configuration file path
        """
        self.config_path = config_path
        self.config = ConfigManager(config_path)
        self.logger = setup_logger(
            name="full_pipeline",
            level="INFO",
            log_file="./logs/full_pipeline.log",
            console_output=True
        )

        # Pipeline status tracking
        self.pipeline_status = {
            'start_time': None,
            'end_time': None,
            'steps_completed': [],
            'steps_failed': [],
            'results': {},
        }

        # Step timing tracking
        self.timing_stats = OrderedDict()
        self.timing_stats['data_loading'] = 0.0
        self.timing_stats['preprocessing'] = 0.0
        self.timing_stats['model_inference'] = 0.0
        self.timing_stats['initial_validation'] = 0.0
        self.timing_stats['cleaning'] = 0.0
        self.timing_stats['final_validation'] = 0.0
        self.timing_stats['visualization'] = 0.0

        # Module instances
        self.quality_analyzer = DataQualityAnalyzer(self.logger)
        self.validation_comparator = ValidationComparator(self.logger)
        self.pipeline_visualizer = None  # Initialize on demand

        # 默认步骤
        self.DEFAULT_STEPS = ['distillation', 'initial_validation', 'cleaning', 'final_validation']
        self.ALL_STEPS = ['distillation', 'initial_validation', 'cleaning', 'final_validation', 'visualization']

    def run_full_pipeline(
        self,
        steps: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        tasks: Optional[List[str]] = None,
        min_quality: Optional[float] = None,
        min_confidence: Optional[float] = None,
        skip_validation: bool = False,
        dry_run: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        运行完整数据管道

        Args:
            steps: 要运行的步骤列表
            max_samples: 最大处理样本数
            tasks: 任务列表
            min_quality: 最小质量分数阈值
            min_confidence: 最小置信度阈值
            skip_validation: 是否跳过验证
            dry_run: 测试运行
            output_dir: 输出目录

        Returns:
            流程报告
        """
        self.pipeline_status['start_time'] = datetime.now()

        # 确定运行步骤
        if steps is None:
            steps = self.DEFAULT_STEPS.copy()
            if skip_validation:
                steps = [s for s in steps if 'validation' not in s]

        # 确定参数（优先级：命令行 > 配置文件 > 默认值）
        if tasks is None:
            tasks = self.config.get('distillation.tasks', ['vqa', 'captioning', 'detection'])

        if min_quality is None:
            min_quality = self.config.get('cleaning.min_quality_score', 30.0)
            self.logger.info(f"Using min_quality from config: {min_quality}")

        if min_confidence is None:
            min_confidence = self.config.get('cleaning.min_confidence', 0.5)
            self.logger.info(f"Using min_confidence from config: {min_confidence}")

        # 验证步骤名称
        for step in steps:
            if step not in self.ALL_STEPS:
                self.logger.error(f"Unknown step: {step}")
                return {'success': False, 'error': f"Unknown step: {step}"}

        self.logger.info("="*70)
        self.logger.info("VLM Data Full Pipeline")
        self.logger.info("="*70)
        self.logger.info(f"Steps to run: {steps}")
        self.logger.info(f"Configuration: {self.config_path}")
        self.logger.info(f"Dry run mode: {dry_run}")

        if dry_run:
            success = self._test_configuration(max_samples, tasks)
            return {'success': success, 'dry_run': True}

        # 按顺序执行步骤
        step_results = {}
        current_input_dir = None

        for step in steps:
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"Running Step: {step.upper()}")
            self.logger.info(f"{'='*70}")

            step_start = datetime.now()

            try:
                if step == 'distillation':
                    result = self._run_distillation(
                        max_samples, tasks, output_dir
                    )
                    current_input_dir = result.get('merged_output')

                elif step == 'initial_validation':
                    result = self.validation_comparator.run_validation(
                        current_input_dir, 'initial'
                    )

                elif step == 'cleaning':
                    result = self._run_cleaning(
                        current_input_dir, min_quality, min_confidence, output_dir
                    )
                    current_input_dir = result.get('cleaned_output')

                elif step == 'final_validation':
                    before_dir = step_results.get('distillation', {}).get('merged_output')
                    result = self.validation_comparator.run_validation(
                        current_input_dir, 'final'
                    )
                    # 对比验证结果
                    if 'initial_validation' in step_results:
                        comparison = self.validation_comparator.compare_validation_results(
                            step_results['initial_validation'],
                            result
                        )
                        result['comparison'] = comparison

                elif step == 'visualization':
                    result = self._run_visualization(
                        current_input_dir, step_results.get('distillation', {}).get('merged_output')
                    )

                step_end = datetime.now()
                duration = (step_end - step_start).total_seconds()
                self.timing_stats[step] = duration

                result['duration_seconds'] = duration
                result['start_time'] = step_start.isoformat()
                result['end_time'] = step_end.isoformat()

                step_results[step] = result
                self.pipeline_status['steps_completed'].append(step)

                self.logger.info(f"\n✓ Step {step} completed in {duration:.1f}s")

            except Exception as e:
                self.logger.error(f"\n✗ Step {step} failed: {e}")
                self.pipeline_status['steps_failed'].append(step)
                step_results[step] = {'success': False, 'error': str(e)}

        # 生成最终报告
        self.pipeline_status['end_time'] = datetime.now()
        self.pipeline_status['results'] = step_results

        final_report = self._finalize_pipeline()

        return final_report

    def _run_distillation(
        self,
        max_samples: Optional[int],
        tasks: List[str],
        output_dir: Optional[str]
    ) -> Dict[str, Any]:
        """
        Run distillation step

        Returns:
            Distillation result
        """
        self.logger.info("\nSTEP: DISTILLATION")

        step_start = datetime.now()

        try:
            # Update configuration
            if max_samples:
                self.config.set('data.max_samples', max_samples)
            self.config.set('distillation.tasks', tasks)
            if output_dir:
                self.config.set('output.root_dir', output_dir)

            # Initialize COCO data loader
            self.logger.info("\n[1/3] Initializing COCO dataset loader...")
            coco_loader = COCODataLoader(self.config)
            coco_loader.initialize(self.config.get('data.val_split', 'val2017'))

            dataset_summary = coco_loader.get_annotation_summary()
            self.logger.info(f"Dataset loaded successfully")
            self.logger.info(f"  - Images: {dataset_summary.get('total_images', 0)}")

            # Initialize teacher model
            self.logger.info("\n[2/3] Loading teacher model...")
            self.logger.info(f"  Model: {self.config.get('teacher.model_name')}")

            teacher = TeacherModel(self.config)
            model_info = teacher.get_model_info()
            self.logger.info(f"  Device: {model_info.get('device', 'unknown')}")
            self.logger.info(f"  Precision: {model_info.get('precision', 'unknown')}")

            # Initialize distiller
            self.logger.info("\n[3/3] Creating distiller...")
            distiller = Distiller(
                teacher_model=teacher,
                config=self.config
            )

            # Run distillation
            self.logger.info("\n" + "-"*70)
            self.logger.info("Running distillation...")
            self.logger.info("-"*70)

            result = distiller.run_distillation(max_samples=max_samples)

            step_end = datetime.now()
            duration = (step_end - step_start).total_seconds()

            # Prepare result report
            result_report = {
                'success': True,
                'processed_count': result.get('processed_count', 0),
                'failed_count': result.get('failed_count', 0),
                'merged_output': result.get('merged_data_path', './outputs/merged'),
                'merged_data_path': result.get('merged_data_path', './outputs/merged'),
                'statistics': result.get('statistics', {}),
                'duration_seconds': duration,
                'start_time': step_start.isoformat(),
                'end_time': step_end.isoformat(),
            }

            # Display summary
            self.logger.info("\n" + "-"*70)
            self.logger.info("Distillation Summary:")
            self.logger.info("-"*70)
            self.logger.info(f"  ✓ Processed: {result_report['processed_count']} images")
            self.logger.info(f"  ✓ Failed: {result_report['failed_count']} errors")
            self.logger.info(f"  ✓ Output: {result_report['merged_output']}")
            self.logger.info(f"  ✓ Time: {duration:.1f}s")

            return result_report

        except Exception as e:
            self.logger.error(f"\n✗ Distillation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'duration_seconds': (datetime.now() - step_start).total_seconds()
            }

    def _run_cleaning(
        self,
        input_dir: str,
        min_quality: float,
        min_confidence: float,
        output_dir: Optional[str]
    ) -> Dict[str, Any]:
        """
        Run cleaning step

        Args:
            input_dir: Input data directory (蒸馏输出)
            min_quality: Minimum quality score
            min_confidence: Minimum confidence
            output_dir: Output directory

        Returns:
            Cleaning result
        """
        self.logger.info("\nSTEP: DATA CLEANING")
        self.logger.info(f"Input: {input_dir}")
        self.logger.info(f"Parameters: min_quality={min_quality}, min_confidence={min_confidence}")

        step_start = datetime.now()

        try:
            # Validate input directory
            if not input_dir:
                raise ValueError("Input directory is required for cleaning step")

            output_path = output_dir or './outputs/cleaned'

            # Update cleaning config
            self.config.set('cleaning.min_quality_score', min_quality)
            self.config.set('cleaning.min_confidence', min_confidence)

            # Initialize cleaner
            cleaner = DataCleaner(self.config)

            # Run cleaning
            self.logger.info("\n" + "-"*70)
            self.logger.info("Running data cleaning...")
            self.logger.info("-"*70)

            result = cleaner.clean_directory(
                data_dir=input_dir,
                output_dir=output_path
            )

            step_end = datetime.now()
            duration = (step_end - step_start).total_seconds()

            # Build output paths (DataCleaner creates 'cleaned' and 'removed' subdirs)
            output_path_obj = Path(output_path)
            cleaned_output = str(output_path_obj / "cleaned")
            removed_output = str(output_path_obj / "removed")

            # Get stats from result
            summary = result.get('summary', {})

            # Prepare result report
            result_report = {
                'success': True,
                'cleaned_output': cleaned_output,
                'removed_output': removed_output,
                'stats': {
                    'cleaned_count': summary.get('cleaned_count', 0),
                    'removed_count': summary.get('removed_count', 0),
                    'total_input': summary.get('total_input', 0),
                },
                'duration_seconds': duration,
                'start_time': step_start.isoformat(),
                'end_time': step_end.isoformat(),
            }

            # Display summary
            self.logger.info("\n" + "-"*70)
            self.logger.info("Cleaning Summary:")
            self.logger.info("-"*70)
            stats = result_report.get('stats', {})
            self.logger.info(f"  ✓ Cleaned: {stats.get('cleaned_count', 0)} samples")
            self.logger.info(f"  ✓ Removed: {stats.get('removed_count', 0)} samples")
            self.logger.info(f"  ✓ Output: {result_report['cleaned_output']}")
            self.logger.info(f"  ✓ Time: {duration:.1f}s")

            return result_report

        except Exception as e:
            self.logger.error(f"\n✗ Cleaning failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'duration_seconds': (datetime.now() - step_start).total_seconds()
            }

    def _run_visualization(
        self,
        input_dir: str,
        before_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run visualization step

        Args:
            input_dir: Input data directory (cleaned output)
            before_dir: Before cleaning data directory (for comparison)

        Returns:
            Visualization result
        """
        self.logger.info("\nSTEP: DATA VISUALIZATION")

        step_start = datetime.now()

        try:
            # Validate input directory
            if not input_dir:
                raise ValueError("Input directory is required for visualization step")

            # Initialize visualizer
            if self.pipeline_visualizer is None:
                self.pipeline_visualizer = PipelineVisualizer(self.config, self.logger)

            # Load current data
            data_list = self._load_data_from_dir(input_dir)

            if not data_list:
                self.logger.warning("No data files found for visualization")
                return {
                    'success': False,
                    'error': 'No data files',
                    'duration_seconds': 0
                }

            self.logger.info(f"Loaded {len(data_list)} data samples")

            # Load before cleaning data (if available)
            before_data = None
            if before_dir:
                before_data = self._load_data_from_dir(before_dir)
                self.logger.info(f"Loaded {len(before_data) if before_data else 0} before-cleaning samples")

            # Run visualization
            viz_report = self.pipeline_visualizer.generate_all_plots(
                data_list=data_list,
                output_dir=self.config.get('visualization.output_dir', './outputs/visualizations'),
                timing_stats=dict(self.timing_stats),
                pipeline_results=self.pipeline_status['results'],
                before_data=before_data
            )

            step_end = datetime.now()
            duration = (step_end - step_start).total_seconds()

            # Display timing summary
            self._display_timing_summary()

            viz_report['duration_seconds'] = duration
            viz_report['start_time'] = step_start.isoformat()
            viz_report['end_time'] = step_end.isoformat()

            self.logger.info(f"\n✓ Visualization completed")
            self.logger.info(f"  Generated {viz_report.get('generated_plots', 0)} plots")
            self.logger.info(f"  Time: {duration:.1f}s")

            return viz_report

        except Exception as e:
            self.logger.error(f"\n✗ Visualization failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'duration_seconds': (datetime.now() - step_start).total_seconds()
            }

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
                self.logger.warning(f"Failed to load {json_file}: {e}")

        return data_list

    def _test_configuration(
        self,
        max_samples: Optional[int],
        tasks: List[str]
    ) -> bool:
        """
        测试配置是否正确（dry run）

        Returns:
            配置是否有效
        """
        self.logger.info("\n[DRY RUN] Testing configuration...")

        try:
            # 测试数据加载器
            loader = COCODataLoader(self.config)
            self.logger.info("✓ COCODataLoader initialized")

            # 测试教师模型
            teacher = TeacherModel(self.config)
            self.logger.info("✓ TeacherModel initialized")

            # 测试蒸馏器
            distiller = Distiller(self.config, teacher)
            self.logger.info("✓ Distiller initialized")

            # 测试清洗器
            cleaner = DataCleaner(self.config)
            self.logger.info("✓ DataCleaner initialized")

            # 显示配置参数
            self.logger.info(f"\nConfiguration Summary:")
            self.logger.info(f"  Max Samples: {max_samples or 'all'}")
            self.logger.info(f"  Tasks: {tasks}")
            self.logger.info(f"  Min Quality: {self.config.get('cleaning.min_quality_score', 30.0)}")
            self.logger.info(f"  Min Confidence: {self.config.get('cleaning.min_confidence', 0.5)}")

            self.logger.info("\n✓ All components initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"\n✗ Configuration test failed: {e}")
            return False

    def _display_timing_summary(self):
        """
        显示耗时汇总
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("PIPELINE TIMING SUMMARY")
        self.logger.info("="*70)

        total_time = sum(self.timing_stats.values())

        self.logger.info("\nStep-by-step Duration:")
        self.logger.info("-"*50)

        for step, duration in self.timing_stats.items():
            percentage = (duration / total_time * 100) if total_time > 0 else 0
            self.logger.info(f"  {step:25s}: {duration:8.1f}s ({percentage:5.1f}%)")

        self.logger.info("-"*50)
        self.logger.info(f"  {'TOTAL':25s}: {total_time:8.1f}s (100.0%)")
        self.logger.info("="*70)

    def _finalize_pipeline(self) -> Dict[str, Any]:
        """
        流程完成，生成最终报告

        Returns:
            最终流程报告
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("PIPELINE COMPLETE")
        self.logger.info("="*70)

        total_duration = sum(self.timing_stats.values())

        # 检查是否有失败的步骤
        success = len(self.pipeline_status['steps_failed']) == 0

        if success:
            self.logger.info("\n✓ Pipeline completed successfully!")
        else:
            self.logger.warning(f"\n⚠️ Pipeline completed with {len(self.pipeline_status['steps_failed'])} failed steps")

        self.logger.info(f"\nSteps completed: {self.pipeline_status['steps_completed']}")
        self.logger.info(f"Steps failed: {self.pipeline_status['steps_failed']}")
        self.logger.info(f"Total duration: {total_duration:.1f} seconds")

        # 显示输出位置
        self.logger.info("\nOutput Locations:")
        distillation_result = self.pipeline_status['results'].get('distillation', {})
        if distillation_result.get('success'):
            self.logger.info(f"  Distilled data: {distillation_result.get('merged_output')}")

        cleaning_result = self.pipeline_status['results'].get('cleaning', {})
        if cleaning_result.get('success'):
            self.logger.info(f"  Cleaned data: {cleaning_result.get('cleaned_output')}")
            self.logger.info(f"  Removed data: {cleaning_result.get('removed_output')}")

        quality_analysis = distillation_result.get('quality_analysis', {})
        if quality_analysis:
            self.logger.info(f"  Quality analysis report: ./outputs/data_quality_analysis.json")

        # 保存流程报告
        pipeline_report_path = Path('./outputs/pipeline_report.json')
        pipeline_report_path.parent.mkdir(parents=True, exist_ok=True)

        final_report = {
            'success': success,
            'start_time': self.pipeline_status['start_time'].isoformat(),
            'end_time': self.pipeline_status['end_time'].isoformat(),
            'total_duration_seconds': total_duration,
            'steps_completed': self.pipeline_status['steps_completed'],
            'steps_failed': self.pipeline_status['steps_failed'],
            'timing_stats': dict(self.timing_stats),
            'results_summary': self._clean_results_for_json(self.pipeline_status['results'])
        }

        with open(pipeline_report_path, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)

        self.logger.info(f"\n✓ Pipeline report saved to: {pipeline_report_path}")
        self.logger.info("="*70)

        return final_report

    def _clean_results_for_json(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean results for JSON serialization (remove circular references and complex objects)

        Args:
            results: Original results dictionary

        Returns:
            Cleaned results with only basic data types
        """
        cleaned = {}

        for step_name, step_result in results.items():
            if isinstance(step_result, dict):
                # Only keep basic serializable fields
                cleaned[step_name] = {
                    'success': step_result.get('success'),
                    'duration_seconds': step_result.get('duration_seconds'),
                    'start_time': step_result.get('start_time'),
                    'end_time': step_result.get('end_time'),
                    'error': step_result.get('error'),
                }

                # Add specific output paths if available
                if 'merged_output' in step_result:
                    cleaned[step_name]['merged_output'] = step_result.get('merged_output')
                if 'merged_data_path' in step_result:
                    cleaned[step_name]['merged_data_path'] = step_result.get('merged_data_path')
                if 'cleaned_output' in step_result:
                    cleaned[step_name]['cleaned_output'] = step_result.get('cleaned_output')
                if 'removed_output' in step_result:
                    cleaned[step_name]['removed_output'] = step_result.get('removed_output')
                if 'processed_count' in step_result:
                    cleaned[step_name]['processed_count'] = step_result.get('processed_count')
                if 'failed_count' in step_result:
                    cleaned[step_name]['failed_count'] = step_result.get('failed_count')
                if 'generated_plots' in step_result:
                    cleaned[step_name]['generated_plots'] = step_result.get('generated_plots')
                if 'stats' in step_result and isinstance(step_result.get('stats'), dict):
                    # Only keep basic stats
                    stats = step_result.get('stats')
                    cleaned[step_name]['stats'] = {
                        'cleaned_count': stats.get('cleaned_count'),
                        'removed_count': stats.get('removed_count'),
                        'total_input': stats.get('total_input'),
                    }
            else:
                # Skip non-dict results (likely complex objects)
                cleaned[step_name] = {'success': False, 'error': 'Non-serializable result'}

        return cleaned


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="Run the full VLM data distillation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Configuration file path'
    )

    parser.add_argument(
        '--samples',
        type=int,
        default=None,
        help='Maximum number of samples to process'
    )

    parser.add_argument(
        '--tasks',
        nargs='+',
        default=None,
        choices=['vqa', 'captioning', 'detection'],
        help='Tasks to include (vqa, captioning, detection)'
    )

    parser.add_argument(
        '--steps',
        nargs='+',
        default=None,
        choices=['distillation', 'initial_validation', 'cleaning',
                 'final_validation', 'visualization'],
        help='Steps to run (default: all except visualization)'
    )

    parser.add_argument(
        '--min-quality',
        type=float,
        default=None,
        help='Minimum quality score threshold (0-100)'
    )

    parser.add_argument(
        '--min-confidence',
        type=float,
        default=None,
        help='Minimum confidence threshold (0-1)'
    )

    parser.add_argument(
        '--enable-visualization',
        action='store_true',
        help='Enable visualization step'
    )

    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip validation steps'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Test configuration without running pipeline'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for all results'
    )

    args = parser.parse_args()

    # 初始化管道运行器
    runner = FullPipelineRunner(config_path=args.config)

    # 确定步骤列表
    steps = args.steps
    if steps is None:
        steps = runner.DEFAULT_STEPS.copy()
        if args.enable_visualization:
            steps.append('visualization')

    # 运行管道
    result = runner.run_full_pipeline(
        steps=steps,
        max_samples=args.samples,
        tasks=args.tasks,
        min_quality=args.min_quality,
        min_confidence=args.min_confidence,
        skip_validation=args.skip_validation,
        dry_run=args.dry_run,
        output_dir=args.output_dir
    )

    # 返回状态码
    if result.get('success'):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()