"""
完整数据管道脚本
================

一次性运行蒸馏、清洗、验证三个步骤的完整流程。

Usage:
    # 运行完整流程
    python scripts/run_full_pipeline.py --samples 5000

    # 仅运行特定步骤
    python scripts/run_full_pipeline.py --steps distillation cleaning

    # 自定义参数
    python scripts/run_full_pipeline.py \
        --samples 1000 \
        --tasks vqa captioning \
        --min-quality 40 \
        --skip-validation

    # Dry run（测试配置）
    python scripts/run_full_pipeline.py --dry-run

流程步骤:
    Step 1: 数据蒸馏 - 生成三重标签（硬标签/软标签/CoT）
    Step 2: 数据清洗 - 异常检测+质量评分+过滤+修复
    Step 3: 数据验证 - Schema验证+质量检查
"""

import argparse
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# 兼容两种导入方式：安装后和未安装
try:
    # 安装后的导入方式
    from src import ConfigManager, TeacherModel, Distiller, COCODataLoader, setup_logger, DataCleaner
except ImportError:
    # 未安装时的导入方式（开发模式）
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    from src import ConfigManager, TeacherModel, Distiller, COCODataLoader, setup_logger, DataCleaner


class FullPipelineRunner:
    """
    完整数据管道运行器

    整合三个步骤:
    1. 数据蒸馏 (Distillation)
    2. 数据清洗 (Cleaning)
    3. 数据验证 (Validation)
    """

    def __init__(self, config_path: str = 'configs/default.yaml'):
        """
        初始化管道运行器

        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = ConfigManager(config_path)
        self.logger = setup_logger(
            name="full_pipeline",
            level="INFO",
            log_file="./logs/full_pipeline.log",
            console_output=True
        )

        # 流程状态追踪
        self.pipeline_status = {
            'start_time': None,
            'end_time': None,
            'steps_completed': [],
            'steps_failed': [],
            'results': {},
        }

        # 步骤名称
        self.STEPS = ['distillation', 'cleaning', 'validation']

    def run_full_pipeline(
        self,
        steps: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        tasks: Optional[List[str]] = None,
        min_quality: float = 30.0,
        min_confidence: float = 0.5,
        skip_validation: bool = False,
        dry_run: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        运行完整数据管道

        Args:
            steps: 要运行的步骤列表（None表示运行全部）
            max_samples: 最大处理样本数
            tasks: 任务列表 ['vqa', 'captioning', 'detection']
            min_quality: 清洗最小质量分数阈值
            min_confidence: 清洗最小置信度阈值
            skip_validation: 是否跳过验证步骤
            dry_run: 测试运行（不实际处理）
            output_dir: 输出目录

        Returns:
            完整流程报告
        """
        self.pipeline_status['start_time'] = datetime.now()

        # 确定运行步骤
        if steps is None:
            steps = self.STEPS.copy()
            if skip_validation and 'validation' in steps:
                steps.remove('validation')

        # 验证步骤名称
        for step in steps:
            if step not in self.STEPS:
                self.logger.error(f"Unknown step: {step}")
                self.logger.error(f"Valid steps: {self.STEPS}")
                return {'success': False, 'error': f"Unknown step: {step}"}

        self.logger.info("="*70)
        self.logger.info("VLM Data Full Pipeline")
        self.logger.info("="*70)
        self.logger.info(f"Steps to run: {steps}")
        self.logger.info(f"Configuration: {self.config_path}")

        if dry_run:
            self.logger.info("\n[DRY RUN] Testing configuration...")
            self._test_configuration(
                max_samples=max_samples,
                tasks=tasks,
                min_quality=min_quality,
                min_confidence=min_confidence
            )
            return {'success': True, 'dry_run': True}

        # 执行各步骤
        try:
            # Step 1: 数据蒸馏
            if 'distillation' in steps:
                distillation_result = self._run_distillation(
                    max_samples=max_samples,
                    tasks=tasks,
                    output_dir=output_dir
                )

                if distillation_result['success']:
                    self.pipeline_status['steps_completed'].append('distillation')
                    self.pipeline_status['results']['distillation'] = distillation_result
                else:
                    self.pipeline_status['steps_failed'].append('distillation')
                    self.logger.error(f"Distillation failed: {distillation_result.get('error')}")
                    return self._finalize_pipeline(success=False)

            # Step 2: 数据清洗
            if 'cleaning' in steps:
                # 确定输入目录
                if 'distillation' in self.pipeline_status['steps_completed']:
                    input_dir = self.pipeline_status['results']['distillation']['merged_output']
                else:
                    input_dir = self.config.get('output.merged_dir', './outputs/merged')

                cleaning_result = self._run_cleaning(
                    input_dir=input_dir,
                    min_quality=min_quality,
                    min_confidence=min_confidence,
                    output_dir=output_dir
                )

                if cleaning_result['success']:
                    self.pipeline_status['steps_completed'].append('cleaning')
                    self.pipeline_status['results']['cleaning'] = cleaning_result
                else:
                    self.pipeline_status['steps_failed'].append('cleaning')
                    self.logger.error(f"Cleaning failed: {cleaning_result.get('error')}")
                    # 清洗失败不中断流程，继续验证

            # Step 3: 数据验证
            if 'validation' in steps:
                # 确定验证目录
                validation_dirs = []
                if 'distillation' in self.pipeline_status['steps_completed']:
                    validation_dirs.append(
                        self.pipeline_status['results']['distillation']['merged_output']
                    )
                if 'cleaning' in self.pipeline_status['steps_completed']:
                    validation_dirs.append(
                        self.pipeline_status['results']['cleaning']['cleaned_output']
                    )

                if not validation_dirs:
                    validation_dirs = [self.config.get('output.merged_dir', './outputs/merged')]

                validation_result = self._run_validation(validation_dirs)

                if validation_result['success']:
                    self.pipeline_status['steps_completed'].append('validation')
                    self.pipeline_status['results']['validation'] = validation_result
                else:
                    self.pipeline_status['steps_failed'].append('validation')
                    self.logger.warning(f"Validation found issues: {validation_result.get('issues')}")

            # 生成最终报告
            return self._finalize_pipeline(success=True)

        except Exception as e:
            self.logger.error(f"\nPipeline failed with exception: {e}")
            return self._finalize_pipeline(success=False, error=str(e))

    def _test_configuration(
        self,
        max_samples: Optional[int],
        tasks: Optional[List[str]],
        min_quality: float,
        min_confidence: float
    ) -> bool:
        """
        测试配置是否正确（Dry Run）
        """
        self.logger.info("\nTesting configuration parameters:")

        # 测试配置加载
        self.logger.info(f"  ✓ Config loaded: {self.config_path}")

        # 测试参数覆盖
        test_config = ConfigManager(self.config_path)
        if max_samples:
            test_config.set('data.max_samples', max_samples)
            self.logger.info(f"  ✓ max_samples: {max_samples}")

        if tasks:
            test_config.set('distillation.tasks', tasks)
            self.logger.info(f"  ✓ tasks: {tasks}")

        test_config.set('cleaning.min_quality_score', min_quality)
        test_config.set('cleaning.min_confidence', min_confidence)
        self.logger.info(f"  ✓ min_quality: {min_quality}")
        self.logger.info(f"  ✓ min_confidence: {min_confidence}")

        # 测试配置验证
        if not test_config.validate():
            self.logger.error("  ✗ Configuration validation failed!")
            return False

        self.logger.info("  ✓ Configuration validation passed")

        # 检查必要目录
        self.logger.info("\nChecking directories:")
        dirs_to_check = [
            ('configs/', Path('configs/')),
            ('outputs/', Path('outputs/')),
            ('logs/', Path('logs/')),
        ]

        for name, path in dirs_to_check:
            if path.exists():
                self.logger.info(f"  ✓ {name} exists")
            else:
                self.logger.info(f"  ! {name} will be created")

        self.logger.info("\n" + "="*70)
        self.logger.info("DRY RUN COMPLETE - Configuration is valid!")
        self.logger.info("="*70)

        return True

    def _run_distillation(
        self,
        max_samples: Optional[int],
        tasks: Optional[List[str]],
        output_dir: Optional[str]
    ) -> Dict[str, Any]:
        """
        运行数据蒸馏步骤

        Returns:
            蒸馏结果报告
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("STEP 1: DATA DISTILLATION")
        self.logger.info("="*70)

        step_start = datetime.now()

        try:
            # 更新配置
            if max_samples:
                self.config.set('data.max_samples', max_samples)
            if tasks:
                self.config.set('distillation.tasks', tasks)
            if output_dir:
                self.config.set('output.root_dir', output_dir)

            # 初始化数据加载器
            self.logger.info("\n[1/3] Initializing COCO dataset loader...")
            coco_loader = COCODataLoader(self.config)
            coco_loader.initialize(self.config.get('data.val_split', 'val2014'))

            dataset_summary = coco_loader.get_annotation_summary()
            self.logger.info(f"Dataset loaded successfully")
            self.logger.info(f"  - Images: {dataset_summary.get('total_images', 0)}")
            self.logger.info(f"  - VQA questions: {dataset_summary.get('vqa_count', 0)}")

            # 初始化教师模型
            self.logger.info("\n[2/3] Loading teacher model...")
            self.logger.info(f"  Model: {self.config.get('teacher.model_name')}")

            teacher = TeacherModel(self.config)
            model_info = teacher.get_model_info()
            self.logger.info(f"  Device: {model_info.get('device', 'unknown')}")
            self.logger.info(f"  Precision: {model_info.get('precision', 'unknown')}")

            # 初始化蒸馏器
            self.logger.info("\n[3/3] Creating distiller...")
            distiller = Distiller(
                teacher_model=teacher,
                config=self.config
            )

            # 运行蒸馏
            self.logger.info("\n" + "-"*70)
            self.logger.info("Running distillation...")
            self.logger.info("-"*70)

            results = distiller.run_distillation(max_samples=max_samples)

            step_end = datetime.now()
            duration = (step_end - step_start).total_seconds()

            # 提取关键结果
            result_report = {
                'success': True,
                'processed_count': results.get('processed_count', 0),
                'failed_count': results.get('failed_count', 0),
                'merged_output': results.get('merged_data_path', './outputs/merged'),
                'statistics': results.get('statistics', {}),
                'duration_seconds': duration,
                'start_time': step_start.isoformat(),
                'end_time': step_end.isoformat(),
            }

            # 显示摘要
            self.logger.info("\n" + "-"*70)
            self.logger.info("Distillation Summary:")
            self.logger.info("-"*70)
            self.logger.info(f"  ✓ Processed: {result_report['processed_count']} images")
            self.logger.info(f"  ✓ Failed: {result_report['failed_count']} images")
            self.logger.info(f"  ✓ Output: {result_report['merged_output']}")
            self.logger.info(f"  ✓ Duration: {duration:.1f} seconds")

            return result_report

        except Exception as e:
            self.logger.error(f"\nDistillation failed: {e}")
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
        运行数据清洗步骤

        Returns:
            清洗结果报告
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("STEP 2: DATA CLEANING")
        self.logger.info("="*70)

        step_start = datetime.now()

        try:
            # 更新清洗配置
            self.config.set('cleaning.min_quality_score', min_quality)
            self.config.set('cleaning.min_confidence', min_confidence)
            self.config.set('cleaning.enabled', True)

            self.logger.info(f"\nCleaning configuration:")
            self.logger.info(f"  - Input: {input_dir}")
            self.logger.info(f"  - Min quality score: {min_quality}")
            self.logger.info(f"  - Min confidence: {min_confidence}")

            # 初始化清洗器
            cleaner = DataCleaner(self.config)

            # 确定输出目录
            if output_dir:
                cleaning_output = str(Path(output_dir) / "cleaned")
            else:
                cleaning_output = str(Path(input_dir).parent / "cleaned")

            # 检查输入数据
            input_path = Path(input_dir)
            json_files = list(input_path.glob("*.json"))
            valid_files = [
                f for f in json_files
                if not f.name.startswith('checkpoint')
                and not f.name.startswith('merged_summary')
                and not f.name.startswith('cleaning_report')
            ]

            if not valid_files:
                self.logger.error(f"No valid data files found in {input_dir}")
                return {
                    'success': False,
                    'error': 'No input data files',
                    'duration_seconds': 0
                }

            self.logger.info(f"  - Files to clean: {len(valid_files)}")

            # 运行清洗
            self.logger.info("\n" + "-"*70)
            self.logger.info("Running cleaning...")
            self.logger.info("-"*70)

            report = cleaner.clean_directory(input_dir, cleaning_output)

            step_end = datetime.now()
            duration = (step_end - step_start).total_seconds()

            # 提取关键结果
            result_report = {
                'success': True,
                'total_input': report['summary']['total_input'],
                'cleaned_count': report['summary']['cleaned_count'],
                'removed_count': report['summary']['removed_count'],
                'removal_rate': report['summary']['removal_rate'],
                'cleaned_output': f"{cleaning_output}/cleaned",
                'removed_output': f"{cleaning_output}/removed",
                'report_file': f"{cleaning_output}/cleaning_report.json",
                'average_quality': report['quality_statistics']['average_quality_score'],
                'duration_seconds': duration,
                'start_time': step_start.isoformat(),
                'end_time': step_end.isoformat(),
                'recommendations': report['recommendations'],
            }

            # 显示摘要
            self.logger.info("\n" + "-"*70)
            self.logger.info("Cleaning Summary:")
            self.logger.info("-"*70)
            self.logger.info(f"  ✓ Input: {result_report['total_input']} files")
            self.logger.info(f"  ✓ Cleaned: {result_report['cleaned_count']} files")
            self.logger.info(f"  ✓ Removed: {result_report['removed_count']} files")
            self.logger.info(f"  ✓ Removal rate: {result_report['removal_rate']*100:.1f}%")
            self.logger.info(f"  ✓ Avg quality: {result_report['average_quality']:.1f}/100")
            self.logger.info(f"  ✓ Output: {result_report['cleaned_output']}")
            self.logger.info(f"  ✓ Duration: {duration:.1f} seconds")

            # 显示建议
            if result_report['recommendations']:
                self.logger.info("\nRecommendations:")
                for i, rec in enumerate(result_report['recommendations'][:3], 1):
                    self.logger.info(f"  {i}. {rec[:100]}...")

            return result_report

        except Exception as e:
            self.logger.error(f"\nCleaning failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'duration_seconds': (datetime.now() - step_start).total_seconds()
            }

    def _run_validation(self, validation_dirs: List[str]) -> Dict[str, Any]:
        """
        运行数据验证步骤

        Args:
            validation_dirs: 要验证的目录列表

        Returns:
            验证结果报告
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("STEP 3: DATA VALIDATION")
        self.logger.info("="*70)

        step_start = datetime.now()

        validation_results = []
        all_valid = True

        for data_dir in validation_dirs:
            self.logger.info(f"\nValidating: {data_dir}")

            try:
                # 调用validate_data.py的逻辑
                from scripts.validate_data import validate_directory, generate_validation_report

                report = validate_directory(data_dir)

                validation_results.append({
                    'directory': data_dir,
                    'valid': report['valid'],
                    'total_files': report['total_files'],
                    'valid_files': report['valid_files'],
                    'invalid_files': report['invalid_files'],
                })

                if not report['valid']:
                    all_valid = False

                self.logger.info(f"  Total: {report['total_files']}")
                self.logger.info(f"  Valid: {report['valid_files']}")
                self.logger.info(f"  Invalid: {report['invalid_files']}")

            except Exception as e:
                self.logger.warning(f"  Validation error: {e}")
                validation_results.append({
                    'directory': data_dir,
                    'valid': False,
                    'error': str(e)
                })
                all_valid = False

        step_end = datetime.now()
        duration = (step_end - step_start).total_seconds()

        result_report = {
            'success': all_valid,
            'validation_results': validation_results,
            'total_directories': len(validation_dirs),
            'duration_seconds': duration,
            'start_time': step_start.isoformat(),
            'end_time': step_end.isoformat(),
        }

        # 显示摘要
        self.logger.info("\n" + "-"*70)
        self.logger.info("Validation Summary:")
        self.logger.info("-"*70)

        for v_result in validation_results:
            status = "✓ VALID" if v_result['valid'] else "✗ ISSUES FOUND"
            self.logger.info(f"  {v_result['directory']}: {status}")

        self.logger.info(f"  Duration: {duration:.1f} seconds")

        return result_report

    def _finalize_pipeline(
        self,
        success: bool,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        生成最终流程报告并保存

        Args:
            success: 流程是否成功
            error: 错误信息（如果有）

        Returns:
            最终流程报告
        """
        self.pipeline_status['end_time'] = datetime.now()

        total_duration = (
            self.pipeline_status['end_time'] -
            self.pipeline_status['start_time']
        ).total_seconds()

        final_report = {
            'success': success,
            'error': error,
            'start_time': self.pipeline_status['start_time'].isoformat(),
            'end_time': self.pipeline_status['end_time'].isoformat(),
            'total_duration_seconds': total_duration,
            'steps_completed': self.pipeline_status['steps_completed'],
            'steps_failed': self.pipeline_status['steps_failed'],
            'results': self.pipeline_status['results'],
        }

        # 保存报告
        report_path = Path('./outputs/pipeline_report.json')
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)

        # 显示最终摘要
        self.logger.info("\n" + "="*70)
        self.logger.info("PIPELINE COMPLETE")
        self.logger.info("="*70)

        if success:
            self.logger.info("\n✓ Pipeline completed successfully!")
        else:
            self.logger.error(f"\n✗ Pipeline failed: {error}")

        self.logger.info(f"\nSteps completed: {self.pipeline_status['steps_completed']}")
        self.logger.info(f"Steps failed: {self.pipeline_status['steps_failed']}")
        self.logger.info(f"Total duration: {total_duration:.1f} seconds")

        # 显示各步骤输出
        self.logger.info("\nOutput Locations:")

        if 'distillation' in self.pipeline_status['steps_completed']:
            dist_out = self.pipeline_status['results']['distillation']['merged_output']
            self.logger.info(f"  Distilled data: {dist_out}")

        if 'cleaning' in self.pipeline_status['steps_completed']:
            clean_out = self.pipeline_status['results']['cleaning']['cleaned_output']
            self.logger.info(f"  Cleaned data: {clean_out}")
            self.logger.info(f"  Removed data: {self.pipeline_status['results']['cleaning']['removed_output']}")

        self.logger.info(f"  Pipeline report: {report_path}")

        return final_report


def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(
        description="Run complete VLM data pipeline (distillation + cleaning + validation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with default settings
  python scripts/run_full_pipeline.py

  # Process 1000 samples
  python scripts/run_full_pipeline.py --samples 1000

  # Run only distillation and cleaning
  python scripts/run_full_pipeline.py --steps distillation cleaning

  # Custom thresholds
  python scripts/run_full_pipeline.py --min-quality 40 --min-confidence 0.6

  # Dry run to test configuration
  python scripts/run_full_pipeline.py --dry-run

  # Specific tasks only
  python scripts/run_full_pipeline.py --tasks vqa captioning

Pipeline Steps:
  1. Distillation: Generate hard/soft labels + CoT using teacher model
  2. Cleaning: Detect anomalies, score quality, filter and repair data
  3. Validation: Verify schema and data integrity

Output:
  - outputs/merged/*.json          - Raw distilled data
  - outputs/cleaned/cleaned/*.json - High-quality cleaned data
  - outputs/cleaned/removed/*.json - Low-quality removed data
  - outputs/pipeline_report.json   - Complete pipeline report
        """
    )

    # 基本参数
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Configuration file path (default: configs/default.yaml)'
    )

    parser.add_argument(
        '--samples',
        type=int,
        default=None,
        help='Maximum number of samples to process'
    )

    parser.add_argument(
        '--tasks',
        type=str,
        nargs='+',
        choices=['vqa', 'captioning', 'detection'],
        default=['vqa', 'captioning', 'detection'],
        help='Tasks to run: vqa, captioning, detection'
    )

    # 步骤选择
    parser.add_argument(
        '--steps',
        type=str,
        nargs='+',
        choices=['distillation', 'cleaning', 'validation'],
        default=['distillation', 'cleaning', 'validation'],
        help='Steps to run (default: all three steps)'
    )

    # 清洗参数
    parser.add_argument(
        '--min-quality',
        type=float,
        default=30.0,
        help='Minimum quality score for cleaning (0-100, default: 30.0)'
    )

    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.5,
        help='Minimum confidence threshold (default: 0.5)'
    )

    parser.add_argument(
        '--keep-invalid',
        action='store_true',
        default=True,
        help='Keep invalid data instead of removing (mark only)'
    )

    # 验证参数
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        default=False,
        help='Skip validation step'
    )

    # 其他参数
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override output directory'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=False,
        help='Test configuration without actual processing'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Show detailed logging'
    )

    args = parser.parse_args()

    # 创建运行器
    runner = FullPipelineRunner(config_path=args.config)

    # 运行管道
    result = runner.run_full_pipeline(
        steps=args.steps,
        max_samples=args.samples,
        tasks=args.tasks,
        min_quality=args.min_quality,
        min_confidence=args.min_confidence,
        skip_validation=args.skip_validation,
        dry_run=args.dry_run,
        output_dir=args.output_dir
    )

    # 返回状态码
    if result['success']:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
