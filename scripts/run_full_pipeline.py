"""
完整数据管道脚本（重构版）
=========================

简化后的流程协调脚本，主要负责：
1. 流程步骤协调
2. 参数解析和配置
3. 结果汇总和报告生成

所有参数从配置文件读取，只有 --config 和 --steps 可通过命令行覆盖

具体功能已提取到独立模块：
- 数据质量分析 → DataQualityAnalyzer
- 验证比较 → ValidationComparator
- 可视化生成 → DataVisualizer

配置文件说明：
    所有参数在 configs/default.yaml 中配置：

    核心参数：
    - data.max_samples: 最大样本数（默认 5000）
    - distillation.tasks: 任务列表（默认 ['vqa', 'detection']）
    - cleaning.min_quality_score: 最小质量分数（默认 50.0）
    - cleaning.min_confidence: 最小置信度（默认 0.6）

    输出目录（单步运行时自动使用）：
    - output.merged_dir: Distillation 输出（默认 './outputs/merged'）
    - output.cleaned_dir: Cleaning 输出（默认 './outputs/cleaned'）
    - output.visualization_dir: Visualization 输出（默认 './outputs/visualizations'')

    Pipeline 控制：
    - pipeline.default_steps: 默认步骤列表
    - pipeline.checkpoint_path: 断点续运行的 checkpoint 文件路径
    - pipeline.dry_run: 测试运行模式

Usage:
    # 运行完整流程（使用配置文件中的参数）
    python scripts/run_full_pipeline.py

    # 指定配置文件
    python scripts/run_full_pipeline.py --config configs/custom.yaml

    # 覆盖步骤
    python scripts/run_full_pipeline.py --steps distillation cleaning

    # 单独运行某个步骤（自动使用配置文件中的默认输入目录）
    python scripts/run_full_pipeline.py --steps cleaning
    # 输入目录自动使用 output.merged_dir ('./outputs/merged')

    python scripts/run_full_pipeline.py --steps visualization
    # 输入目录自动使用 output.cleaned_dir ('./outputs/cleaned')

步骤依赖关系：
    - distillation: 从数据集加载，无前置依赖
    - initial_validation: 输入来自 merged_dir
    - cleaning: 输入来自 merged_dir
    - final_validation: 输入来自 cleaned_dir
    - quality_validation: 输入来自 cleaned_dir
    - visualization: 输入来自 cleaned_dir，对比数据来自 merged_dir
"""

# 🔧 关键：在所有导入之前设置 multiprocessing 为 spawn 模式
# vLLM 多 GPU 推理（tensor_parallel）需要 spawn 模式，不能使用默认的 fork
# 必须在主程序入口设置，不能在被导入的模块中设置
import multiprocessing
try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass  # Already set

# 🔧 可选：设置 vLLM 日志级别（抑制大量内部日志）
# 如需调试，可改为 INFO 或 DEBUG
import os
os.environ.setdefault('VLLM_LOGGING_LEVEL', 'WARNING')

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
    from src.utils.data_visualizer import DataVisualizer
    from src.utils.validation_comparator import ValidationComparator
    from src.utils.data_quality_validator import DataQualityValidator, compare_cleaning_effect
except ImportError:
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    from src import (
        ConfigManager, TeacherModel, Distiller, COCODataLoader,
        setup_logger, DataCleaner
    )
    from src.utils.data_visualizer import DataVisualizer
    from src.utils.validation_comparator import ValidationComparator
    from src.utils.data_quality_validator import DataQualityValidator, compare_cleaning_effect


class FullPipelineRunner:
    """
    Complete VLM Data Pipeline Runner

    Main responsibilities:
    1. Flow coordination - call steps in sequence
    2. Parameter management - parse CLI and config parameters
    3. Result aggregation - generate final report

    Modules:
    - Validation → ValidationComparator
    - Visualization → DataVisualizer
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

        # Step timing tracking (包含蒸馏子步骤)
        self.timing_stats = OrderedDict()
        # 蒸馏子步骤
        self.timing_stats['data_loading'] = {'duration': 0.0, 'samples': 0}
        self.timing_stats['preprocessing'] = {'duration': 0.0, 'samples': 0}
        self.timing_stats['model_inference'] = {'duration': 0.0, 'samples': 0}
        # 质量校验步骤
        self.timing_stats['quality_validation'] = {'duration': 0.0, 'samples': 0}
        # 其他步骤
        self.timing_stats['initial_validation'] = {'duration': 0.0, 'samples': 0}
        self.timing_stats['cleaning'] = {'duration': 0.0, 'samples': 0}
        self.timing_stats['final_validation'] = {'duration': 0.0, 'samples': 0}
        self.timing_stats['visualization'] = {'duration': 0.0, 'samples': 0}

        # Module instances (initialize on demand)
        self.visualizer = None
        self.validation_comparator = None
        self.quality_validator = None

        # Default steps (可视化已包含在默认流程中)
        self.DEFAULT_STEPS = ['distillation', 'initial_validation', 'cleaning', 'final_validation', 'quality_validation', 'visualization']
        self.ALL_STEPS = ['distillation', 'initial_validation', 'cleaning', 'final_validation', 'quality_validation', 'visualization']

    def run_full_pipeline(
        self,
        steps: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        运行完整数据管道

        所有参数从配置文件读取，只有 steps 可以通过命令行覆盖

        Args:
            steps: 要运行的步骤列表（可覆盖配置文件中的 pipeline.default_steps）

        Returns:
            流程报告
        """
        self.pipeline_status['start_time'] = datetime.now()

        # 确定运行步骤（优先级：参数 > 配置文件）
        if steps is None:
            steps = self.config.get('pipeline.default_steps', self.DEFAULT_STEPS)

        # 从配置文件读取所有参数
        max_samples = self.config.get('data.max_samples', None)
        tasks = self.config.get('distillation.tasks', ['vqa'])
        min_quality = self.config.get('cleaning.min_quality_score', 30.0)
        min_confidence = self.config.get('cleaning.min_confidence', 0.6)
        output_dir = self.config.get('output.root_dir', './outputs')
        checkpoint_path = self.config.get('pipeline.checkpoint_path', None)
        dry_run = self.config.get('pipeline.dry_run', False)

        # 从配置文件读取各步骤的默认输出目录
        merged_dir = self.config.get('output.merged_dir', './outputs/merged')
        cleaned_dir = self.config.get('output.cleaned_dir', './outputs/cleaned')
        visualization_dir = self.config.get('output.visualization_dir', './outputs/visualizations')

        self.logger.info(f"Using config parameters:")
        self.logger.info(f"  max_samples: {max_samples}")
        self.logger.info(f"  tasks: {tasks}")
        self.logger.info(f"  min_quality: {min_quality}")
        self.logger.info(f"  min_confidence: {min_confidence}")

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

        # 确定初始输入目录（根据步骤依赖关系自动确定）
        current_input_dir = None

        if 'distillation' in steps and steps[0] == 'distillation':
            # 以 distillation 开始，初始输入为空（distillation 会从数据集加载）
            pass
        else:
            # 根据步骤类型确定默认输入目录
            if 'cleaning' in steps and 'distillation' not in steps:
                # 单独运行 cleaning，默认输入是 merged_dir
                current_input_dir = merged_dir
            elif 'visualization' in steps and 'distillation' not in steps:
                # 单独运行 visualization，默认输入是 cleaned_dir
                current_input_dir = cleaned_dir
            elif any(s in steps for s in ['initial_validation', 'final_validation', 'quality_validation']) \
                 and 'distillation' not in steps and 'cleaning' not in steps:
                # 单独运行 validation，根据步骤类型确定输入
                if 'final_validation' in steps or 'quality_validation' in steps:
                    current_input_dir = cleaned_dir
                elif 'initial_validation' in steps:
                    current_input_dir = merged_dir

            if current_input_dir:
                self.logger.info(f"Using default input directory for step: {current_input_dir}")

        for step in steps:
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"Running Step: {step.upper()}")
            self.logger.info(f"{'='*70}")

            step_start = datetime.now()

            # 检查步骤依赖
            if step in ['initial_validation', 'cleaning', 'final_validation', 'quality_validation', 'visualization']:
                if current_input_dir is None:
                    self.logger.error(f"✗ Step {step} requires input from previous step, but input directory not found")
                    self.logger.error("")
                    self.logger.error("Solution:")
                    self.logger.error("  1. Run 'distillation' step first to generate data")
                    self.logger.error("  2. Or ensure the default input directory exists:")
                    self.logger.error(f"     - For cleaning/initial_validation: {merged_dir}")
                    self.logger.error(f"     - For visualization/final_validation/quality_validation: {cleaned_dir}")
                    self.logger.error("")
                    step_results[step] = {'success': False, 'error': 'Missing input directory'}
                    continue

            try:
                if step == 'distillation':
                    result = self._run_distillation(
                        max_samples, tasks, output_dir, checkpoint_path
                    )
                    current_input_dir = result.get('merged_output') or merged_dir

                elif step == 'initial_validation':
                    # Initialize validation_comparator on demand
                    if self.validation_comparator is None:
                        self.validation_comparator = ValidationComparator(self.logger)
                    result = self.validation_comparator.run_validation(
                        current_input_dir, 'initial'
                    )

                elif step == 'cleaning':
                    result = self._run_cleaning(
                        current_input_dir, min_quality, min_confidence, output_dir
                    )
                    current_input_dir = result.get('cleaned_output') or cleaned_dir

                elif step == 'final_validation':
                    # Initialize validation_comparator on demand
                    if self.validation_comparator is None:
                        self.validation_comparator = ValidationComparator(self.logger)
                    result = self.validation_comparator.run_validation(
                        current_input_dir, 'final'
                    )
                    # Compare validation results
                    if 'initial_validation' in step_results:
                        comparison = self.validation_comparator.compare_validation_results(
                            step_results['initial_validation'],
                            result
                        )
                        result['comparison'] = comparison

                elif step == 'quality_validation':
                    # 最终质量校验，在清洗后进行深度质量评估
                    result = self._run_quality_validation(current_input_dir)

                elif step == 'visualization':
                    # 默认使用 cleaned_dir 作为输入，merged_dir 作为 before_dir（用于对比）
                    viz_before_dir = step_results.get('distillation', {}).get('merged_output') or merged_dir

                    # 获取质量校验结果用于可视化（传递完整的结果，包括 duration_seconds）
                    quality_validation_results = step_results.get('quality_validation', {})

                    result = self._run_visualization(
                        current_input_dir, viz_before_dir,
                        quality_validation_results=quality_validation_results
                    )

                step_end = datetime.now()
                duration = (step_end - step_start).total_seconds()

                # 记录耗时和样本数
                if step == 'distillation':
                    sample_count = result.get('processed_count', 1)
                elif step == 'cleaning':
                    sample_count = result.get('summary', {}).get('total_input', 1)
                elif 'validation' in step and 'quality' not in step:
                    sample_count = result.get('total_files', 1)
                elif step == 'quality_validation':
                    sample_count = result.get('sample_count', 1)
                elif step == 'visualization':
                    sample_count = result.get('generated_plots', 1)
                else:
                    sample_count = 1

                self.timing_stats[step] = {
                    'duration': duration,
                    'samples': sample_count if sample_count > 0 else 1
                }

                result['duration_seconds'] = duration
                result['start_time'] = step_start.isoformat()
                result['end_time'] = step_end.isoformat()

                step_results[step] = result
                self.pipeline_status['steps_completed'].append(step)

                # 计算每样本平均耗时
                avg_per_sample = duration / max(sample_count, 1)
                self.logger.info(f"\n✓ Step {step} completed in {duration:.1f}s")
                self.logger.info(f"  Samples: {sample_count}, Avg per sample: {avg_per_sample:.3f}s")

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
        output_dir: Optional[str],
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run distillation step with sub-timing tracking

        Args:
            max_samples: Maximum samples to process
            tasks: Tasks to run
            output_dir: Output directory
            checkpoint_path: Path to checkpoint file for resuming

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

            # ============================================================
            # [Phase 1] Data Loading
            # ============================================================
            data_loading_start = datetime.now()
            self.logger.info("\n[1/3] Loading COCO dataset...")

            coco_loader = COCODataLoader(self.config)
            coco_loader.initialize(self.config.get('data.val_split', 'val2017'))

            dataset_summary = coco_loader.get_annotation_summary()
            self.logger.info(f"Dataset loaded: {dataset_summary.get('total_images', 0)} images")

            data_loading_end = datetime.now()
            data_loading_duration = (data_loading_end - data_loading_start).total_seconds()

            # ============================================================
            # [Phase 2] Preprocessing (Model loading)
            # ============================================================
            preprocessing_start = datetime.now()
            self.logger.info("\n[2/3] Loading teacher model...")
            self.logger.info(f"  Model: {self.config.get('teacher.model_name')}")

            teacher = TeacherModel(self.config)
            model_info = teacher.get_model_info()
            self.logger.info(f"  Device: {model_info.get('device', 'unknown')}")
            self.logger.info(f"  Precision: {model_info.get('precision', 'unknown')}")

            self.logger.info("\n[3/3] Creating distiller...")
            distiller = Distiller(teacher_model=teacher, config=self.config)

            preprocessing_end = datetime.now()
            preprocessing_duration = (preprocessing_end - preprocessing_start).total_seconds()

            # ============================================================
            # [Phase 3] Model Inference
            # ============================================================
            inference_start = datetime.now()

            # Resume info
            if checkpoint_path:
                self.logger.info(f"\n⚠️  Resume mode: Using checkpoint {checkpoint_path}")
                checkpoint_file = Path(checkpoint_path)
                if not checkpoint_file.exists():
                    self.logger.warning(f"Checkpoint file not found: {checkpoint_path}")
                    checkpoint_path = None

            self.logger.info("\n" + "-"*70)
            self.logger.info("Running model inference...")
            self.logger.info("-"*70)

            result = distiller.run_distillation(
                max_samples=max_samples,
                checkpoint_path=checkpoint_path
            )

            inference_end = datetime.now()
            inference_duration = (inference_end - inference_start).total_seconds()

            step_end = datetime.now()
            total_duration = (step_end - step_start).total_seconds()

            # 记录子步骤耗时
            self.timing_stats['data_loading'] = {'duration': data_loading_duration, 'samples': max_samples or 1}
            self.timing_stats['preprocessing'] = {'duration': preprocessing_duration, 'samples': max_samples or 1}
            self.timing_stats['model_inference'] = {'duration': inference_duration, 'samples': result.get('processed_count', max_samples or 1)}

            # Prepare result report
            result_report = {
                'success': True,
                'processed_count': result.get('processed_count', 0),
                'failed_count': result.get('failed_count', 0),
                'merged_output': result.get('merged_data_path', './outputs/merged'),
                'merged_data_path': result.get('merged_data_path', './outputs/merged'),
                'statistics': result.get('statistics', {}),
                'duration_seconds': total_duration,
                'timing_breakdown': {
                    'data_loading': data_loading_duration,
                    'preprocessing': preprocessing_duration,
                    'model_inference': inference_duration,
                },
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
            self.logger.info(f"\n  Timing Breakdown:")
            self.logger.info(f"    Data Loading:    {data_loading_duration:.1f}s")
            self.logger.info(f"    Preprocessing:   {preprocessing_duration:.1f}s")
            self.logger.info(f"    Model Inference: {inference_duration:.1f}s")
            self.logger.info(f"    Total:           {total_duration:.1f}s")

            return result_report

        except Exception as e:
            self.logger.error(f"\n✗ Distillation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'duration_seconds': (datetime.now() - step_start).total_seconds()
            }

    def _run_quality_validation(
        self,
        input_dir: str
    ) -> Dict[str, Any]:
        """
        Run data quality validation step (最终质量把关)

        在清洗后、训练前进行深度质量校验，包括：
        - 软标签分布校验（KL散度、分布对齐）
        - ECE置信度校准
        - Top-K匹配统计
        - CoT质量校验（幻觉检测、重复度）
        - 数据阶段判定（是否具备训练价值）

        Args:
            input_dir: Input data directory (清洗后的数据)

        Returns:
            Quality validation result
        """
        self.logger.info("\nSTEP: FINAL QUALITY VALIDATION")
        self.logger.info(f"Input: {input_dir} (清洗后的数据)")
        self.logger.info(f"Purpose: 训练前的最终质量把关")

        step_start = datetime.now()

        try:
            # Validate input directory
            if not input_dir:
                raise ValueError("Input directory is required for quality validation step")

            # Initialize quality validator on demand
            if self.quality_validator is None:
                coco_annotations_dir = self.config.get('data.annotations_root', './data/coco/annotations')
                self.quality_validator = DataQualityValidator(
                    config=self.config,
                    logger=self.logger,
                    coco_annotations_dir=coco_annotations_dir
                )

            # Run quality validation
            self.logger.info("\n" + "-"*70)
            self.logger.info("Running comprehensive data quality validation...")
            self.logger.info("-"*70)

            # 直接保存到 outputs 目录（不需要 validation 子目录）
            output_dir = self.config.get('output.root_dir', './outputs')
            result = self.quality_validator.run_full_validation(
                input_dir=input_dir,
                output_dir=output_dir
            )

            step_end = datetime.now()
            duration = (step_end - step_start).total_seconds()

            # Prepare result report
            result_report = {
                'success': result.get('success', False),
                'overall_passed': result.get('overall_passed', False),
                'sample_count': result.get('sample_count', 0),
                'validation_results': result.get('validation_results', {}),
                'duration_seconds': duration,
                'start_time': step_start.isoformat(),
                'end_time': step_end.isoformat(),
            }

            # Display summary
            self.logger.info("\n" + "-"*70)
            self.logger.info("Quality Validation Summary:")
            self.logger.info("-"*70)

            # 显示关键指标
            val_results = result.get('validation_results', {})

            # Top-K匹配
            top_k = val_results.get('top_k_matching', {}).get('statistics', {})
            self.logger.info(f"  Top-K匹配率: {top_k.get('match_rate', 0)*100:.1f}%")

            # KL散度
            kl = val_results.get('soft_label_distribution', {}).get('kl_divergence_analysis', {}).get('statistics', {})
            self.logger.info(f"  平均KL散度: {kl.get('average_kl', 'N/A')}")

            # ECE
            ece = val_results.get('ece_calibration', {})
            self.logger.info(f"  ECE校准误差: {ece.get('ece', 'N/A')}")

            # 幻觉检测
            halluc = val_results.get('cot_quality', {}).get('hallucination_detection', {}).get('statistics', {})
            self.logger.info(f"  CoT幻觉占比: {halluc.get('hallucination_ratio', 0)*100:.1f}%")

            # 最终判定
            assessment = val_results.get('training_value_assessment', {})
            if assessment.get('can_train'):
                self.logger.info(f"\n  ✓ 数据质量合格，具备训练价值")
            else:
                self.logger.info(f"\n  ✗ 数据质量不合格，需要清洗或重新生成")

            self.logger.info(f"\n  Time: {duration:.1f}s")

            return result_report

        except Exception as e:
            self.logger.error(f"\n✗ Quality validation failed: {e}")
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
        before_dir: Optional[str] = None,
        quality_validation_results: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Run visualization step

        Args:
            input_dir: Input data directory (cleaned output)
            before_dir: Before cleaning data directory (for comparison)
            quality_validation_results: Quality validation results for visualization

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
            if self.visualizer is None:
                self.visualizer = DataVisualizer(self.config, self.logger)

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

            # Run visualization with quality validation results
            # 设置visualization的初始timing
            self.timing_stats['visualization'] = {'duration': 0.0, 'samples': len(data_list)}

            viz_report = self.visualizer.visualize_all(
                data_list=data_list,
                before_data=before_data,
                timing_stats=dict(self.timing_stats),
                pipeline_results=self.pipeline_status['results'],
                quality_validation_results=quality_validation_results
            )

            step_end = datetime.now()
            duration = (step_end - step_start).total_seconds()

            # 更新visualization的实际耗时
            self.timing_stats['visualization'] = {
                'duration': duration,
                'samples': viz_report.get('generated_plots', len(data_list))
            }

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
        Display timing summary with per-sample averages
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("PIPELINE TIMING SUMMARY")
        self.logger.info("="*70)

        self.logger.info("\nStep-by-step Duration:")
        self.logger.info("-"*60)
        self.logger.info(f"  {'Step':20s} {'Duration':12s} {'Samples':10s} {'Avg/sample':12s}")

        total_duration = 0
        total_samples = 0

        for step, stats in self.timing_stats.items():
            duration = stats.get('duration', 0)
            samples = stats.get('samples', 0)
            avg_per_sample = duration / max(samples, 1)

            self.logger.info(f"  {step:20s} {duration:10.1f}s {samples:8d} {avg_per_sample:10.3f}s")

            total_duration += duration
            total_samples += samples

        self.logger.info("-"*60)
        self.logger.info(f"  {'TOTAL':20s} {total_duration:10.1f}s {total_samples:8d}")
        self.logger.info("="*70)

        # 找出瓶颈
        max_avg = 0
        bottleneck = None
        for step, stats in self.timing_stats.items():
            avg = stats.get('duration', 0) / max(stats.get('samples', 1), 1)
            if avg > max_avg:
                max_avg = avg
                bottleneck = step

        if bottleneck:
            self.logger.info(f"\n⚠️  Bottleneck detected: {bottleneck} ({max_avg:.3f}s/sample)")

    def _finalize_pipeline(self) -> Dict[str, Any]:
        """
        流程完成，生成最终报告

        Returns:
            最终流程报告
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("PIPELINE COMPLETE")
        self.logger.info("="*70)

        # 正确计算总耗时（从 dict 结构中提取 duration）
        total_duration = sum(
            stats.get('duration', 0) if isinstance(stats, dict) else stats
            for stats in self.timing_stats.values()
        )

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

    # 只保留必要的命令行参数，其他参数从配置文件读取
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Configuration file path (default: configs/default.yaml)'
    )

    parser.add_argument(
        '--steps',
        nargs='+',
        default=None,
        choices=['distillation', 'quality_validation', 'initial_validation', 'cleaning',
                 'final_validation', 'visualization'],
        help='Steps to run (overrides pipeline.default_steps in config)'
    )

    args = parser.parse_args()

    # 初始化管道运行器
    runner = FullPipelineRunner(config_path=args.config)

    # 确定步骤列表（优先级：命令行 > 配置文件）
    steps = args.steps
    if steps is None:
        steps = runner.config.get('pipeline.default_steps', runner.DEFAULT_STEPS)

    # 运行管道（所有参数从配置文件读取）
    result = runner.run_full_pipeline(
        steps=steps
    )

    # 返回状态码
    if result.get('success'):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()