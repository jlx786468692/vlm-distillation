"""
完整数据管道脚本
================

一次性运行蒸馏、初步验证、清洗、最终验证、数据质量分析五个步骤的完整流程。

参数优先级（命令行 > 配置文件 > 默认值）:
    - min_quality: 命令行 --min-quality > configs/default.yaml cleaning.min_quality_score > 30.0
    - min_confidence: 命令行 --min-confidence > configs/default.yaml cleaning.min_confidence > 0.5
    - tasks: 命令行 --tasks > configs/default.yaml distillation.tasks > ['vqa', 'captioning', 'detection']

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

    # Dry run（测试配置）
    python scripts/run_full_pipeline.py --dry-run

流程步骤（正确顺序）:
    Step 1: 数据蒸馏 - 生成三重标签（硬标签/软标签/CoT）
    Step 2: 初步验证 - 发现数据质量问题
    Step 3: 数据清洗 - 异常检测+质量评分+过滤+修复
    Step 4: 最终验证 - 确认清洗效果并对比前后差异
    Step 5: 数据质量分析 - 深度统计分析和准确性验证
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

    整合五个步骤（正确顺序）:
    1. 数据蒸馏 (Distillation)
    2. 初步验证 (Initial Validation) - 清洗前
    3. 数据清洗 (Cleaning) - 解决初步验证发现的问题
    4. 最终验证 (Final Validation) - 清洗后，对比效果
    5. 数据质量分析 (Quality Analysis) - 深度统计分析验证准确性
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

        # 步骤名称（正确顺序：蒸馏→初步验证→清洗→最终验证）
        self.STEPS = ['distillation', 'initial_validation', 'cleaning', 'final_validation']

        # 无效答案列表（用于质量评估）
        self.invalid_answers = ['unknown', 'n/a', 'none', 'unclear', 'cannot determine', '']

    def run_full_pipeline(
        self,
        steps: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        tasks: Optional[List[str]] = None,
        min_quality: Optional[float] = None,      # ← 改为None，从配置文件读取
        min_confidence: Optional[float] = None,   # ← 改为None，从配置文件读取
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

        # 确定运行步骤（正确顺序）
        if steps is None:
            steps = self.STEPS.copy()
            # skip_validation 跳过所有验证步骤
            if skip_validation:
                steps = [s for s in steps if 'validation' not in s]

        # 确定运行任务（优先级：命令行 > 配置文件 > 默认值）
        if tasks is None:
            # 从配置文件读取任务列表
            tasks = self.config.get('distillation.tasks', ['vqa', 'captioning', 'detection'])

        # 确定清洗参数（优先级：命令行 > 配置文件 > 默认值）
        if min_quality is None:
            # 从配置文件读取，如果没有则使用默认值30.0
            min_quality = self.config.get('cleaning.min_quality_score', 30.0)
            self.logger.info(f"Using min_quality from config: {min_quality}")
        else:
            self.logger.info(f"Using min_quality from command line: {min_quality}")

        if min_confidence is None:
            # 从配置文件读取，如果没有则使用默认值0.5
            min_confidence = self.config.get('cleaning.min_confidence', 0.5)
            self.logger.info(f"Using min_confidence from config: {min_confidence}")
        else:
            self.logger.info(f"Using min_confidence from command line: {min_confidence}")

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
        self.logger.info(f"Dry run mode: {dry_run}")  # 添加调试信息

        if dry_run:
            self.logger.info("\n[DRY RUN] Testing configuration...")
            self._test_configuration(
                max_samples=max_samples,
                tasks=tasks,
                min_quality=min_quality,
                min_confidence=min_confidence
            )
            return {'success': True, 'dry_run': True}

        # 执行各步骤（正确顺序：蒸馏→初步验证→清洗→最终验证）
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

            # Step 2: 初步验证（清洗前）✅ 关键改进
            if 'initial_validation' in steps:
                # 燃定验证目录（蒸馏输出）
                if 'distillation' in self.pipeline_status['steps_completed']:
                    input_dir = self.pipeline_status['results']['distillation']['merged_output']
                else:
                    input_dir = self.config.get('output.merged_dir', './outputs/merged')

                initial_validation_result = self._run_initial_validation(input_dir)

                if initial_validation_result['success']:
                    self.pipeline_status['steps_completed'].append('initial_validation')
                    self.pipeline_status['results']['initial_validation'] = initial_validation_result

                    # ✅ 根据初步验证结果动态调整清洗参数（重要改进）
                    avg_quality = initial_validation_result.get('average_quality', 50)
                    invalid_rate = initial_validation_result.get('invalid_rate', 0)

                    if avg_quality < 20:
                        # 虞量极低 → 更严格清洗
                        self.logger.warning("⚠️ 数据质量极低，将使用更严格的清洗参数")
                        min_quality = max(min_quality, 40)
                        min_confidence = max(min_confidence, 0.6)
                    elif avg_quality > 80 and invalid_rate < 0.05:
                        # 虞量很好且无效数据很少 → 可跳过清洗
                        self.logger.info("✓ 数据质量很好，可以考虑跳过清洗")
                        if 'cleaning' in steps:
                            self.logger.info("将继续执行清洗步骤以确保质量")
                else:
                    self.pipeline_status['steps_failed'].append('initial_validation')
                    self.logger.warning(f"Initial validation found issues: {initial_validation_result.get('issues')}")
                    self.logger.info("将继续执行清洗步骤以解决这些问题")

            # Step 3: 数据清洗
            if 'cleaning' in steps:
                # 燃定输入目录
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

            # Step 4: 最终验证（清洗后）✅ 关键改进
            if 'final_validation' in steps:
                # 燃定验证目录（清洗输出）
                if 'cleaning' in self.pipeline_status['steps_completed']:
                    input_dir = self.pipeline_status['results']['cleaning']['cleaned_output']
                else:
                    # 如果没有清洗，验证蒸馏数据
                    if 'distillation' in self.pipeline_status['steps_completed']:
                        input_dir = self.pipeline_status['results']['distillation']['merged_output']
                    else:
                        input_dir = self.config.get('output.merged_dir', './outputs/merged')

                final_validation_result = self._run_final_validation(input_dir)

                if final_validation_result['success']:
                    self.pipeline_status['steps_completed'].append('final_validation')
                    self.pipeline_status['results']['final_validation'] = final_validation_result

                    # ✅ 对比清洗前后效果（重要改进）
                    if 'initial_validation' in self.pipeline_status['steps_completed']:
                        self._compare_validation_results(
                            self.pipeline_status['results']['initial_validation'],
                            final_validation_result
                        )
                else:
                    self.pipeline_status['steps_failed'].append('final_validation')
                    self.logger.warning(f"Final validation found issues: {final_validation_result.get('issues')}")

            # Step 5: 数据质量分析（深度验证）
            if 'final_validation' in self.pipeline_status['steps_completed']:
                # 对清洗后的数据进行深度分析
                cleaned_dir = self.pipeline_status['results']['cleaning']['cleaned_output'] \
                    if 'cleaning' in self.pipeline_status['steps_completed'] \
                    else self.pipeline_status['results']['distillation']['merged_output']

                quality_analysis_result = self._analyze_data_quality(cleaned_dir)
                self.pipeline_status['results']['quality_analysis'] = quality_analysis_result

            # 生成最终报告
            return self._finalize_pipeline(success=True)

        except Exception as e:
            self.logger.error(f"\nPipeline failed with exception: {e}")
            return self._finalize_pipeline(success=False, error=str(e))

    def _test_configuration(
        self,
        max_samples: Optional[int],
        tasks: Optional[List[str]],
        min_quality: Optional[float],
        min_confidence: Optional[float]
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

        # 显示清洗参数来源
        final_min_quality = min_quality if min_quality is not None else test_config.get('cleaning.min_quality_score', 30.0)
        final_min_confidence = min_confidence if min_confidence is not None else test_config.get('cleaning.min_confidence', 0.5)

        test_config.set('cleaning.min_quality_score', final_min_quality)
        test_config.set('cleaning.min_confidence', final_min_confidence)

        if min_quality is not None:
            self.logger.info(f"  ✓ min_quality: {final_min_quality} (from command line)")
        else:
            self.logger.info(f"  ✓ min_quality: {final_min_quality} (from config file)")

        if min_confidence is not None:
            self.logger.info(f"  ✓ min_confidence: {final_min_confidence} (from command line)")
        else:
            self.logger.info(f"  ✓ min_confidence: {final_min_confidence} (from config file)")

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
            # tasks已经在run_full_pipeline中设置了默认值，直接使用
            self.config.set('distillation.tasks', tasks)
            if output_dir:
                self.config.set('output.root_dir', output_dir)

            # 初始化数据加载器
            self.logger.info("\n[1/3] Initializing COCO dataset loader...")
            coco_loader = COCODataLoader(self.config)
            coco_loader.initialize(self.config.get('data.val_split', 'val2017'))

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
        self.logger.info("STEP 3: DATA CLEANING")
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

    def _run_initial_validation(self, input_dir: str) -> Dict[str, Any]:
        """
        运行初步验证（清洗前）- 发现数据质量问题

        Args:
            input_dir: 要验证的输入目录（蒸馏输出）

        Returns:
            初步验证结果报告
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("STEP 2: INITIAL VALIDATION (Before Cleaning)")
        self.logger.info("="*70)

        step_start = datetime.now()

        try:
            # 调用validate_data.py的逻辑
            from scripts.validate_data import validate_directory

            self.logger.info(f"\nValidating: {input_dir}")
            report = validate_directory(input_dir)

            # 提取质量统计
            total_files = report['total_files']
            valid_files = report['valid_files']
            invalid_files = report['invalid_files']
            invalid_rate = invalid_files / total_files if total_files > 0 else 0

            # 计算平均质量分数（从清洗报告或数据文件中读取）
            avg_quality = self._calculate_average_quality(input_dir)

            self.logger.info(f"  Total: {total_files}")
            self.logger.info(f"  Valid: {valid_files}")
            self.logger.info(f"  Invalid: {invalid_files}")
            self.logger.info(f"  Invalid rate: {invalid_rate*100:.1f}%")
            self.logger.info(f"  Average quality: {avg_quality:.1f}/100")

            # 保存验证报告
            validation_report_path = Path('./outputs/validation_initial.json')
            validation_report_path.parent.mkdir(parents=True, exist_ok=True)

            import json
            validation_report = {
                'input_dir': input_dir,
                'total_files': total_files,
                'valid_files': valid_files,
                'invalid_files': invalid_files,
                'invalid_rate': invalid_rate,
                'average_quality': avg_quality,
                'issues': report.get('issues', []),
                'timestamp': step_start.isoformat()
            }

            with open(validation_report_path, 'w', encoding='utf-8') as f:
                json.dump(validation_report, f, indent=2, ensure_ascii=False)

            step_end = datetime.now()
            duration = (step_end - step_start).total_seconds()

            result_report = {
                'success': True,
                'total_files': total_files,
                'valid_files': valid_files,
                'invalid_files': invalid_files,
                'invalid_rate': invalid_rate,
                'average_quality': avg_quality,
                'issues': report.get('issues', []),
                'report_file': str(validation_report_path),
                'duration_seconds': duration,
                'start_time': step_start.isoformat(),
                'end_time': step_end.isoformat(),
            }

            # 显示摘要和建议
            self.logger.info("\n" + "-"*70)
            self.logger.info("Initial Validation Summary:")
            self.logger.info("-"*70)

            if invalid_rate > 0.5:
                self.logger.warning("⚠️ 数据质量极低，建议使用更严格的清洗参数")
                self.logger.info("  推荐参数: --min-quality 40 --min-confidence 0.6")
            elif invalid_rate > 0.1:
                self.logger.info("⚠️ 数据质量中等，建议使用标准清洗参数")
                self.logger.info("  推荐参数: --min-quality 30 --min-confidence 0.5")
            else:
                self.logger.info("✓ 数据质量较好，可以使用宽松的清洗参数")
                self.logger.info("  推荐参数: --min-quality 25 --min-confidence 0.3")

            self.logger.info(f"\n  Report saved: {validation_report_path}")
            self.logger.info(f"  Duration: {duration:.1f} seconds")

            return result_report

        except Exception as e:
            self.logger.warning(f"Initial validation error: {e}")
            self.logger.info("将继续执行清洗步骤以解决这些问题")
            return {
                'success': False,
                'error': str(e),
                'issues': [str(e)],
                'duration_seconds': (datetime.now() - step_start).total_seconds()
            }

    def _run_final_validation(self, input_dir: str) -> Dict[str, Any]:
        """
        运行最终验证（清洗后）- 确认清洗效果

        Args:
            input_dir: 要验证的输入目录（清洗输出）

        Returns:
            最终验证结果报告
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("STEP 4: FINAL VALIDATION (After Cleaning)")
        self.logger.info("="*70)

        step_start = datetime.now()

        try:
            # 调用validate_data.py的逻辑
            from scripts.validate_data import validate_directory

            self.logger.info(f"\nValidating: {input_dir}")
            report = validate_directory(input_dir)

            # 提取质量统计
            total_files = report['total_files']
            valid_files = report['valid_files']
            invalid_files = report['invalid_files']
            invalid_rate = invalid_files / total_files if total_files > 0 else 0

            # 计算平均质量分数（从清洗报告或数据文件中读取）
            avg_quality = self._calculate_average_quality(input_dir)

            self.logger.info(f"  Total: {total_files}")
            self.logger.info(f"  Valid: {valid_files}")
            self.logger.info(f"  Invalid: {invalid_files}")
            self.logger.info(f"  Invalid rate: {invalid_rate*100:.1f}%")
            self.logger.info(f"  Average quality: {avg_quality:.1f}/100")

            # 保存验证报告
            validation_report_path = Path('./outputs/validation_final.json')
            validation_report_path.parent.mkdir(parents=True, exist_ok=True)

            import json
            validation_report = {
                'input_dir': input_dir,
                'total_files': total_files,
                'valid_files': valid_files,
                'invalid_files': invalid_files,
                'invalid_rate': invalid_rate,
                'average_quality': avg_quality,
                'issues': report.get('issues', []),
                'timestamp': step_start.isoformat()
            }

            with open(validation_report_path, 'w', encoding='utf-8') as f:
                json.dump(validation_report, f, indent=2, ensure_ascii=False)

            step_end = datetime.now()
            duration = (step_end - step_start).total_seconds()

            result_report = {
                'success': invalid_rate < 0.05,  # 无效数据少于5%视为成功
                'total_files': total_files,
                'valid_files': valid_files,
                'invalid_files': invalid_files,
                'invalid_rate': invalid_rate,
                'average_quality': avg_quality,
                'issues': report.get('issues', []),
                'report_file': str(validation_report_path),
                'duration_seconds': duration,
                'start_time': step_start.isoformat(),
                'end_time': step_end.isoformat(),
            }

            # 显示摘要
            self.logger.info("\n" + "-"*70)
            self.logger.info("Final Validation Summary:")
            self.logger.info("-"*70)

            if result_report['success']:
                self.logger.info("✓ 数据质量达标！")
            else:
                self.logger.warning(f"⚠️ 数据质量仍不达标（无效数据率: {invalid_rate*100:.1f}%）")

            self.logger.info(f"\n  Report saved: {validation_report_path}")
            self.logger.info(f"  Duration: {duration:.1f} seconds")

            return result_report

        except Exception as e:
            self.logger.warning(f"Final validation error: {e}")
            return {
                'success': False,
                'error': str(e),
                'issues': [str(e)],
                'duration_seconds': (datetime.now() - step_start).total_seconds()
            }

    def _compare_validation_results(
        self,
        initial_result: Dict[str, Any],
        final_result: Dict[str, Any]
    ) -> None:
        """
        对比清洗前后验证结果

        Args:
            initial_result: 初步验证结果
            final_result: 最终验证结果
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("Comparing Before/After Validation Results")
        self.logger.info("="*70)

        # 计算改善情况
        initial_invalid_rate = initial_result.get('invalid_rate', 0)
        final_invalid_rate = final_result.get('invalid_rate', 0)
        invalid_rate_reduction = initial_invalid_rate - final_invalid_rate

        initial_quality = initial_result.get('average_quality', 50)
        final_quality = final_result.get('average_quality', 50)
        quality_improvement = final_quality - initial_quality

        # 获取移除率（如果有）
        cleaning_result = self.pipeline_status['results'].get('cleaning', {})
        removal_rate = cleaning_result.get('removal_rate', 0) if cleaning_result else 0

        # 显示对比结果
        self.logger.info("\nBefore Cleaning:")
        self.logger.info(f"  Total files: {initial_result.get('total_files', 0)}")
        self.logger.info(f"  Valid files: {initial_result.get('valid_files', 0)}")
        self.logger.info(f"  Invalid files: {initial_result.get('invalid_files', 0)}")
        self.logger.info(f"  Invalid rate: {initial_invalid_rate*100:.1f}%")
        self.logger.info(f"  Average quality: {initial_quality:.1f}/100")

        self.logger.info("\nAfter Cleaning:")
        self.logger.info(f"  Total files: {final_result.get('total_files', 0)}")
        self.logger.info(f"  Valid files: {final_result.get('valid_files', 0)}")
        self.logger.info(f"  Invalid files: {final_result.get('invalid_files', 0)}")
        self.logger.info(f"  Invalid rate: {final_invalid_rate*100:.1f}%")
        self.logger.info(f"  Average quality: {final_quality:.1f}/100")

        self.logger.info("\nImprovement:")
        self.logger.info(f"  Invalid rate reduction: {invalid_rate_reduction*100:.1f}%")
        self.logger.info(f"  Quality score increase: +{quality_improvement:.1f}")
        if removal_rate > 0:
            self.logger.info(f"  Data removal rate: {removal_rate*100:.1f}%")

        # 保存对比报告
        comparison_report_path = Path('./outputs/validation_comparison.json')
        comparison_report_path.parent.mkdir(parents=True, exist_ok=True)

        import json
        comparison_report = {
            'before': {
                'total_files': initial_result.get('total_files', 0),
                'valid_files': initial_result.get('valid_files', 0),
                'invalid_files': initial_result.get('invalid_files', 0),
                'invalid_rate': initial_invalid_rate,
                'average_quality': initial_quality,
            },
            'after': {
                'total_files': final_result.get('total_files', 0),
                'valid_files': final_result.get('valid_files', 0),
                'invalid_files': final_result.get('invalid_files', 0),
                'invalid_rate': final_invalid_rate,
                'average_quality': final_quality,
            },
            'improvement': {
                'invalid_rate_reduction': invalid_rate_reduction,
                'quality_score_increase': quality_improvement,
                'removal_rate': removal_rate,
            },
            'timestamp': datetime.now().isoformat()
        }

        with open(comparison_report_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_report, f, indent=2, ensure_ascii=False)

        self.logger.info(f"\n  Comparison report saved: {comparison_report_path}")

        # 综合评估清洗效果（改进判断逻辑）
        # 评估标准1: 质量分数提升 ≥ 5分
        quality_improved = quality_improvement >= 5.0

        # 评估标准2: 无效数据率降低 ≥ 10%
        invalid_rate_improved = invalid_rate_reduction >= 0.1

        # 评估标准3: 移除了低质量数据（移除率 > 0%）
        data_removed = removal_rate > 0

        # 评估标准4: 最终质量分数达标（≥ 60分）
        quality_达标 = final_quality >= 60.0

        # 综合判断
        if quality_improved or invalid_rate_improved or (data_removed and quality_达标):
            if quality_improved and invalid_rate_improved:
                self.logger.info("\n✓ 清洗效果显著，数据质量大幅提升（质量+{}分，无效数据率降低{}%）".format(
                    quality_improvement, invalid_rate_reduction*100))
            elif quality_improved:
                self.logger.info("\n✓ 清洗有效，质量分数提升{}分".format(quality_improvement))
            elif invalid_rate_improved:
                self.logger.info("\n✓ 清洗有效，无效数据率降低{}%".format(invalid_rate_reduction*100))
            elif data_removed and quality_达标:
                self.logger.info("\n✓ 清洗有效，移除了{}%低质量数据，最终质量达标（{}分）".format(
                    removal_rate*100, final_quality))
            else:
                self.logger.info("\n✓ 清洗效果良好")
        else:
            # 判断是否数据本身就很好
            if initial_invalid_rate < 0.05 and initial_quality >= 60:
                self.logger.info("\n✓ 数据质量本身已达标，清洗保持了数据质量（{}分）".format(final_quality))
                self.logger.info("  提示: 这是正常情况，数据质量良好无需大幅改善")
            else:
                self.logger.warning("\n⚠️ 清洗效果不明显，建议调整清洗参数或检查数据源")
                self.logger.info("  当前状态: 质量分数 {}, 移除率 {}%, 无效数据率 {}%".format(
                    final_quality, removal_rate*100, final_invalid_rate*100))
                self.logger.info("  建议: 尝试更严格的参数（--min-quality 60 --min-confidence 0.7）")

    def _calculate_average_quality(self, input_dir: str) -> float:
        """
        计算目录中数据的平均质量分数

        Args:
            input_dir: 数据目录路径

        Returns:
            平均质量分数 (0-100)
        """
        input_path = Path(input_dir)

        # 方法1: 尝试读取清洗报告（优先级最高，因为这是最准确的）
        # 检查多种可能的清洗报告位置
        possible_report_paths = [
            input_path.parent / 'cleaning_report.json',  # outputs/cleaned/cleaning_report.json
            Path('./outputs/cleaned/cleaning_report.json'),  # 固定位置
            input_path / 'cleaning_report.json',  # 如果报告在数据目录内
        ]

        for report_path in possible_report_paths:
            if report_path.exists():
                try:
                    with open(report_path, 'r', encoding='utf-8') as f:
                        cleaning_report = json.load(f)
                    if 'quality_statistics' in cleaning_report:
                        avg_quality = cleaning_report['quality_statistics'].get('average_quality_score')
                        if avg_quality is not None:
                            self.logger.debug(f"Read average quality from cleaning report: {avg_quality}")
                            return avg_quality
                except Exception as e:
                    self.logger.debug(f"Failed to read cleaning report {report_path}: {e}")

        # 方法2: 从数据文件中读取或计算
        json_files = list(input_path.glob("*.json"))
        quality_scores = []

        for json_file in json_files:
            # 跳过报告文件和摘要文件
            if json_file.name.startswith(('cleaning_report', 'merged_summary', 'validation', 'checkpoint', 'pipeline')):
                continue

            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 如果数据已有quality_score字段（清洗后的数据），直接读取
                if 'quality_score' in data:
                    quality_scores.append(data['quality_score'])
                    self.logger.debug(f"Read quality_score from {json_file.name}: {data['quality_score']}")
                else:
                    # 如果没有quality_score字段（未清洗的数据），使用估算算法
                    temp_quality = self._estimate_quality_score(data)
                    quality_scores.append(temp_quality)
                    self.logger.debug(f"Estimated quality for {json_file.name}: {temp_quality}")

            except Exception as e:
                self.logger.debug(f"Failed to process {json_file}: {e}")
                continue

        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            self.logger.debug(f"Calculated average quality from {len(quality_scores)} files: {avg_quality}")
            return avg_quality

        # 方法3: 如果都无法获取，返回默认值
        self.logger.warning(f"Unable to calculate average quality for {input_dir}, using default 50.0")
        return 50.0

    def _estimate_quality_score(self, data: Dict[str, Any]) -> float:
        """
        估算单个数据文件的质量分数（用于未清洗的数据）

        ⚠️ 评分标准完全对齐 DataCleaner._compute_task_quality()
        以确保清洗前后对比的一致性

        Args:
            data: 数据字典

        Returns:
            估算的质量分数 (0-100)
        """
        score = 0.0
        min_answer_length = 3
        max_answer_length = 100

        tasks = data.get('tasks', {})

        # 对每个任务计算质量分数，然后取平均
        task_scores = []

        for task_name, task_data in tasks.items():
            task_score = 0.0

            # 1. 硬标签质量 (0-40分) - 完全对齐DataCleaner
            hard_label = task_data.get('hard_label', {})
            if hard_label:
                confidence = hard_label.get('confidence', 0.0)

                # 置信度贡献 (最高30分) - 对齐DataCleaner
                if confidence >= 0.7:
                    task_score += 30
                elif confidence >= 0.5:
                    task_score += 20
                elif confidence >= 0.3:
                    task_score += 10

                # 答案完整性 (最高10分) - 对齐DataCleaner
                answer = hard_label.get('answer', '')
                if min_answer_length <= len(answer) <= max_answer_length:
                    task_score += 10

            # 2. 软标签质量 (0-20分) - 完全对齐DataCleaner
            soft_label = task_data.get('soft_label', {})
            if soft_label:
                # 温度参数合理性 (最高10分) - 对齐DataCleaner
                temperature = soft_label.get('temperature', 0.0)
                if 1.5 <= temperature <= 3.0:  # 推荐范围
                    task_score += 10
                elif 1.0 <= temperature <= 5.0:  # 可接受范围
                    task_score += 5

                # 分布完整性 (最高10分) - 对齐DataCleaner
                distribution = soft_label.get('answer_distribution', {})
                if distribution and len(distribution) > 0:
                    task_score += 10

            # 3. CoT质量 (0-30分) - 完全对齐DataCleaner
            cot = task_data.get('cot_reasoning', {})
            if cot:
                quality_metrics = cot.get('quality_metrics', {})

                # 逻辑流畅度 (最高15分) - 对齐DataCleaner
                logical_flow = quality_metrics.get('logical_flow_score', 0.0)
                task_score += logical_flow * 15

                # 步骤数量合理性 (最高15分) - 对齐DataCleaner
                step_count = quality_metrics.get('step_count', 0)
                if 3 <= step_count <= 5:  # 最佳步骤数
                    task_score += 15
                elif 2 <= step_count <= 6:  # 可接受
                    task_score += 10
                elif step_count > 0:  # 有步骤但不理想
                    task_score += step_count * 2

                # 长度合理性 (额外加分) - 对齐DataCleaner
                reasoning_length = len(cot.get('raw_reasoning', ''))
                if 50 <= reasoning_length <= 300:  # 合理长度
                    task_score += 5

            # 4. 任务特定加分 - 对齐DataCleaner
            if task_name == 'vqa':
                answer = hard_label.get('answer', '')
                if answer and answer.lower() not in self.invalid_answers:
                    task_score += 5  # VQA有效答案加分

            task_scores.append(task_score)

        # 计算平均分数（与DataCleaner一致）
        if task_scores:
            avg_score = sum(task_scores) / len(task_scores)
            return min(avg_score, 100.0)  # 最高100分

        # 如果没有任务，返回默认低分
        return 10.0

    def _analyze_data_quality(self, input_dir: str) -> Dict[str, Any]:
        """
        分析数据质量（最终验证后的深度分析）

        Args:
            input_dir: 数据目录路径

        Returns:
            数据质量分析报告
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("DATA QUALITY ANALYSIS")
        self.logger.info("="*70)

        input_path = Path(input_dir)
        json_files = list(input_path.glob("*.json"))

        # 过滤掉报告文件
        data_files = [
            f for f in json_files
            if not f.name.startswith(('cleaning_report', 'merged_summary', 'validation', 'checkpoint', 'pipeline'))
        ]

        if not data_files:
            self.logger.warning("No data files found for analysis")
            return {'error': 'No data files'}

        self.logger.info(f"Analyzing {len(data_files)} data files...")

        # 加载所有数据
        data_list = []
        for json_file in data_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                data_list.append(data)
            except Exception as e:
                self.logger.warning(f"Failed to load {json_file}: {e}")

        # 1. 置信度分析
        confidence_analysis = self._analyze_confidence(data_list)

        # 2. 质量分数分析
        quality_analysis = self._analyze_quality_scores(data_list)

        # 3. 任务分布分析
        task_analysis = self._analyze_task_distribution(data_list)

        # 4. CoT质量分析
        cot_analysis = self._analyze_cot_quality(data_list)

        # 5. 异常检测
        anomaly_analysis = self._detect_anomalies_summary(data_list)

        # 显示分析结果
        self.logger.info("\n" + "-"*70)
        self.logger.info("Analysis Results:")
        self.logger.info("-"*70)

        # 置信度统计
        if 'mean' in confidence_analysis:
            self.logger.info(f"\nConfidence Statistics:")
            self.logger.info(f"  Mean: {confidence_analysis['mean']:.3f}")
            self.logger.info(f"  Median: {confidence_analysis['median']:.3f}")
            self.logger.info(f"  Std: {confidence_analysis['std']:.3f}")
            self.logger.info(f"  Distribution:")
            self.logger.info(f"    High (≥0.7): {confidence_analysis['high_count']} ({confidence_analysis['high_rate']*100:.1f}%)")
            self.logger.info(f"    Medium (0.5-0.7): {confidence_analysis['medium_count']} ({confidence_analysis['medium_rate']*100:.1f}%)")
            self.logger.info(f"    Low (<0.5): {confidence_analysis['low_count']} ({confidence_analysis['low_rate']*100:.1f}%)")

        # 质量分数统计
        if 'mean' in quality_analysis:
            self.logger.info(f"\nQuality Score Statistics:")
            self.logger.info(f"  Mean: {quality_analysis['mean']:.2f}")
            self.logger.info(f"  Median: {quality_analysis['median']:.2f}")
            self.logger.info(f"  Std: {quality_analysis['std']:.2f}")
            self.logger.info(f"  Distribution:")
            self.logger.info(f"    High (70-100): {quality_analysis['high_count']} ({quality_analysis['high_rate']*100:.1f}%)")
            self.logger.info(f"    Medium (50-70): {quality_analysis['medium_count']} ({quality_analysis['medium_rate']*100:.1f}%)")
            self.logger.info(f"    Low (<50): {quality_analysis['low_count']} ({quality_analysis['low_rate']*100:.1f}%)")

        # 任务统计
        self.logger.info(f"\nTask Distribution:")
        self.logger.info(f"  Total samples: {task_analysis['total_samples']}")
        for task_name, count in task_analysis['tasks'].items():
            self.logger.info(f"  {task_name}: {count} samples")

        # CoT统计
        if cot_analysis['cot_rate'] > 0:
            self.logger.info(f"\nCoT Quality:")
            self.logger.info(f"  Coverage: {cot_analysis['cot_rate']*100:.1f}%")
            if cot_analysis['avg_logical_flow']:
                self.logger.info(f"  Avg logical flow: {cot_analysis['avg_logical_flow']:.3f}")
            if cot_analysis['avg_step_count']:
                self.logger.info(f"  Avg step count: {cot_analysis['avg_step_count']:.1f}")

        # 异常统计
        if anomaly_analysis['total_anomalies'] > 0:
            self.logger.warning(f"\nAnomalies Detected:")
            self.logger.warning(f"  Total: {anomaly_analysis['total_anomalies']}")
            for anomaly_type, count in anomaly_analysis['by_type'].items():
                if count > 0:
                    self.logger.warning(f"  {anomaly_type}: {count}")

        # 生成建议
        recommendations = self._generate_quality_recommendations(
            confidence_analysis, quality_analysis, cot_analysis, anomaly_analysis
        )

        self.logger.info(f"\nQuality Assessment:")
        for rec in recommendations:
            self.logger.info(f"  {rec}")

        # 保存分析报告
        analysis_report = {
            'total_samples': len(data_list),
            'confidence_analysis': confidence_analysis,
            'quality_analysis': quality_analysis,
            'task_analysis': task_analysis,
            'cot_analysis': cot_analysis,
            'anomaly_analysis': anomaly_analysis,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat(),
        }

        report_path = Path('./outputs/data_quality_analysis.json')
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_report, f, indent=2, ensure_ascii=False)

        self.logger.info(f"\n  Analysis report saved: {report_path}")

        return analysis_report

    def _analyze_confidence(self, data_list: List[Dict]) -> Dict[str, Any]:
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
            return {'error': 'No confidence values'}

        import numpy as np

        high_count = len([c for c in confidence_values if c >= 0.7])
        medium_count = len([c for c in confidence_values if 0.5 <= c < 0.7])
        low_count = len([c for c in confidence_values if c < 0.5])
        total = len(confidence_values)

        return {
            'count': total,
            'mean': float(np.mean(confidence_values)),
            'median': float(np.median(confidence_values)),
            'std': float(np.std(confidence_values)),
            'min': float(np.min(confidence_values)),
            'max': float(np.max(confidence_values)),
            'high_count': high_count,
            'medium_count': medium_count,
            'low_count': low_count,
            'high_rate': high_count / total,
            'medium_rate': medium_count / total,
            'low_rate': low_count / total,
        }

    def _analyze_quality_scores(self, data_list: List[Dict]) -> Dict[str, Any]:
        """分析质量分数分布"""
        quality_scores = []

        for data in data_list:
            quality_score = data.get('quality_score')
            if quality_score is not None:
                quality_scores.append(quality_score)

        if not quality_scores:
            return {'error': 'No quality scores'}

        import numpy as np

        high_count = len([q for q in quality_scores if q >= 70])
        medium_count = len([q for q in quality_scores if 50 <= q < 70])
        low_count = len([q for q in quality_scores if q < 50])
        total = len(quality_scores)

        return {
            'count': total,
            'mean': float(np.mean(quality_scores)),
            'median': float(np.median(quality_scores)),
            'std': float(np.std(quality_scores)),
            'min': float(np.min(quality_scores)),
            'max': float(np.max(quality_scores)),
            'high_count': high_count,
            'medium_count': medium_count,
            'low_count': low_count,
            'high_rate': high_count / total,
            'medium_rate': medium_count / total,
            'low_rate': low_count / total,
        }

    def _analyze_task_distribution(self, data_list: List[Dict]) -> Dict[str, Any]:
        """分析任务分布"""
        task_counts = {}

        for data in data_list:
            tasks = data.get('tasks', {})
            for task_name in tasks.keys():
                task_counts[task_name] = task_counts.get(task_name, 0) + 1

        import numpy as np

        return {
            'total_samples': len(data_list),
            'tasks': task_counts,
            'avg_tasks_per_sample': float(np.mean([len(d.get('tasks', {})) for d in data_list])),
        }

    def _analyze_cot_quality(self, data_list: List[Dict]) -> Dict[str, Any]:
        """分析CoT质量"""
        import numpy as np

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
            'cot_samples': cot_samples,
            'cot_rate': cot_samples / len(data_list) if data_list else 0,
            'avg_logical_flow': float(np.mean(logical_flows)) if logical_flows else None,
            'avg_step_count': float(np.mean(step_counts)) if step_counts else None,
        }

    def _detect_anomalies_summary(self, data_list: List[Dict]) -> Dict[str, Any]:
        """检测异常数据（汇总）"""
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
        """生成数据质量建议"""
        recommendations = []

        # 1. 整体质量评估
        avg_quality = quality_analysis.get('mean', 0)
        if avg_quality >= 70:
            recommendations.append("✓ 数据质量优秀（平均{}分），可直接用于训练".format(avg_quality))
        elif avg_quality >= 60:
            recommendations.append("✓ 数据质量良好（平均{}分），建议使用前检查低质量样本".format(avg_quality))
        elif avg_quality >= 50:
            recommendations.append("⚠️ 数据质量中等（平均{}分），建议使用更严格的清洗参数".format(avg_quality))
        else:
            recommendations.append("❌ 数据质量较差（平均{}分），建议重新生成数据或调整参数".format(avg_quality))

        # 2. 置信度评估
        low_conf_rate = confidence_analysis.get('low_rate', 0)
        if low_conf_rate > 0.2:
            recommendations.append("⚠️ 低置信度样本占比{}%，建议检查教师模型生成质量".format(low_conf_rate*100))

        # 3. CoT覆盖率
        cot_rate = cot_analysis.get('cot_rate', 0)
        if cot_rate < 0.8:
            recommendations.append("⚠️ CoT覆盖率仅{}%，部分样本缺少推理过程".format(cot_rate*100))
        elif cot_rate >= 0.9:
            recommendations.append("✓ CoT覆盖率{}%，推理数据完整".format(cot_rate*100))

        # 4. 异常评估
        anomaly_count = anomaly_analysis.get('total_anomalies', 0)
        if anomaly_count > 0:
            anomaly_rate = anomaly_count / (confidence_analysis.get('count', 1) or 1)
            if anomaly_rate > 0.1:
                recommendations.append("⚠️ 异常率{}%，建议人工抽查验证".format(anomaly_rate*100))

        # 5. 建议
        if avg_quality >= 60 and low_conf_rate < 0.1 and anomaly_count < 5:
            recommendations.append("✓ 数据整体可信，建议随机抽查100个样本进行人工验证")

        return recommendations

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

        if 'quality_analysis' in self.pipeline_status['results']:
            self.logger.info(f"  Quality analysis report: outputs/data_quality_analysis.json")

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

Pipeline Steps (正确顺序):
  1. Distillation: Generate hard/soft labels + CoT using teacher model
  2. Initial Validation: Detect quality issues before cleaning
  3. Cleaning: Detect anomalies, score quality, filter and repair data
  4. Final Validation: Verify cleaning results and compare before/after
  5. Quality Analysis: Deep statistical analysis and accuracy validation

Output:
  - outputs/merged/*.json          - Raw distilled data
  - outputs/cleaned/cleaned/*.json - High-quality cleaned data
  - outputs/cleaned/removed/*.json - Low-quality removed data
  - outputs/validation_initial.json   - Initial validation report
  - outputs/validation_final.json     - Final validation report
  - outputs/validation_comparison.json - Before/after comparison
  - outputs/data_quality_analysis.json - Deep quality analysis
  - outputs/pipeline_report.json      - Complete pipeline report
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
        choices=['vqa', 'captioning', 'detection', 'keypoints'],
        default=None,  # None表示使用YAML配置文件的值
        help='Tasks to run: vqa, captioning, detection, keypoints (default: from config file)'
    )

    # 步骤选择
    parser.add_argument(
        '--steps',
        type=str,
        nargs='+',
        choices=['distillation', 'initial_validation', 'cleaning', 'final_validation'],
        default=None,
        help='Steps to run (default: all four steps in correct order: distillation → initial_validation → cleaning → final_validation)'
    )

    # 清洗参数
    parser.add_argument(
        '--min-quality',
        type=float,
        default=None,           # ← 改为None，从配置文件读取
        help='Minimum quality score for cleaning (0-100, default: from config file)'
    )

    parser.add_argument(
        '--min-confidence',
        type=float,
        default=None,           # ← 改为None，从配置文件读取
        help='Minimum confidence threshold (default: from config file)'
    )

    parser.add_argument(
        '--keep-invalid',
        action='store_true',
        help='Keep invalid data instead of removing (mark only)'
    )

    # 验证参数
    parser.add_argument(
        '--skip-validation',
        action='store_true',
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
        default=False,  # 明确设置默认值为False
        help='Test configuration without actual processing'
    )

    parser.add_argument(
        '--no-dry-run',
        action='store_false',
        dest='dry_run',
        help='Explicitly disable dry run mode and run actual processing'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
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