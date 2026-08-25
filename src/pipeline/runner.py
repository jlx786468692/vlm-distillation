"""
Pipeline Runner (编排逻辑置于 src)
==================================

六步流水线编排器：
  1. distillation          —— 数据蒸馏（src/distillation/Distiller）
  2. cleaning              —— 数据清洗（src/cleaning）
  3. prepare_training_data —— 生成训练 jsonl（src/export/TrainingDataExporter）
  4. training              —— 学生模型 SFT 训练（src/training/train.run_training）
  5. evaluation            —— 学生模型评估（src/evaluation/evaluator.run_evaluation）
  6. visualization         —— 可视化（src/utils/data_visualizer）

脚本 scripts/run_full_pipeline.py 只做 argparse + 调用本类的 run_full_pipeline()。
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import OrderedDict

from ..utils.config import ConfigManager
from ..utils.logger import setup_logger
from ..data import COCODataLoader
from ..models import TeacherModel
from ..distillation import Distiller
from ..cleaning import RewardModelScorer, DataPartitioner
from ..export import TrainingDataExporter
from ..utils.data_visualizer import DataVisualizer
from ..utils.data_quality_validator import DataQualityValidator


class PipelineRunner:
    """
    六步流水线编排器。仅负责步骤调度与报告，具体实现均位于 src 各模块。
    """

    # 六步默认流程 + 可选 quality_validation（不在默认流程中）
    DEFAULT_STEPS = [
        "distillation", "cleaning", "prepare_training_data",
        "training", "evaluation", "visualization",
    ]
    ALL_STEPS = DEFAULT_STEPS + ["quality_validation"]

    def __init__(self, config_path: str = "configs/default.yaml"):
        self.config_path = config_path
        self.config = ConfigManager(config_path)
        self.logger = setup_logger(
            name="full_pipeline", level="INFO",
            log_file="./logs/full_pipeline.log", console_output=True,
        )

        self.pipeline_status = {
            "start_time": None, "end_time": None,
            "steps_completed": [], "steps_failed": [], "results": {},
        }
        self.timing_stats = OrderedDict()
        for k in ("data_loading", "preprocessing", "model_inference",
                  "distillation", "cleaning", "prepare_training_data",
                  "training", "evaluation", "quality_validation",
                  "visualization"):
            self.timing_stats[k] = {"duration": 0.0, "samples": 0}

        self.visualizer = None
        self.quality_validator = None

    # ============================================================
    # 主调度
    # ============================================================
    def run_full_pipeline(self, steps: Optional[List[str]] = None) -> Dict[str, Any]:
        self.pipeline_status["start_time"] = datetime.now()

        if steps is None:
            steps = self.config.get("pipeline.default_steps", self.DEFAULT_STEPS)

        # 公共参数
        max_samples = self.config.get("data.max_samples", None)
        tasks = self.config.get("distillation.tasks", ["vqa"])
        output_dir = self.config.get("output.root_dir", "./outputs")
        checkpoint_path = self.config.get("pipeline.checkpoint_path", None)
        dry_run = self.config.get("pipeline.dry_run", False)

        merged_dir = self.config.get("output.merged_dir", "./outputs/merged")
        cleaned_dir = self.config.get("output.cleaned_dir", "./outputs/cleaned")
        training_data_dir = self.config.get("output.training_data_dir", "./outputs/training")
        training_dir = self.config.get("output.training_dir", "./outputs/student_ckpt")
        evaluation_dir = self.config.get("output.evaluation_dir", "./outputs/evaluation")
        visualization_dir = self.config.get("output.visualization_dir", "./outputs/visualizations")

        self.logger.info("=" * 70)
        self.logger.info("VLM Data Full Pipeline (6 steps)")
        self.logger.info("=" * 70)
        self.logger.info(f"Steps: {steps}")
        self.logger.info(f"Config: {self.config_path}  dry_run={dry_run}")

        # 校验步骤名
        for step in steps:
            if step not in self.ALL_STEPS:
                msg = f"Unknown step: {step}"
                self.logger.error(msg)
                return {"success": False, "error": msg}

        if dry_run:
            ok = self._test_configuration(max_samples, tasks)
            return {"success": ok, "dry_run": True}

        step_results: Dict[str, Any] = {}
        # 各步骤默认输入目录（单步运行时回退用）
        default_inputs = {
            "cleaning": merged_dir,
            "prepare_training_data": str(Path(cleaned_dir) / "clean_valid"),
            "training": str(Path(training_data_dir) / "train.jsonl"),
            "evaluation": str(Path(training_data_dir) / "train.jsonl"),
            "quality_validation": cleaned_dir,
            "visualization": evaluation_dir,
        }
        # 是否从 distillation 起步（决定是否用前步产物）
        starts_at_distillation = bool(steps) and steps[0] == "distillation"

        # 当前输入目录：若从 distillation 起步则逐步由产物填充；否则用默认回退
        current_input = None if starts_at_distillation else \
            default_inputs.get(steps[0]) if steps else None

        if current_input:
            self.logger.info(f"Using default input for step '{steps[0]}': {current_input}")

        for step in steps:
            self.logger.info(f"\n{'=' * 70}\nRunning Step: {step.upper()}\n{'=' * 70}")
            step_start = datetime.now()

            try:
                if step == "distillation":
                    result = self._run_distillation(max_samples, tasks, output_dir, checkpoint_path)
                    current_input = result.get("merged_output") or merged_dir

                elif step == "cleaning":
                    in_dir = current_input or merged_dir
                    result = self._run_cleaning(in_dir, cleaned_dir)
                    current_input = result.get("cleaned_output") or str(Path(cleaned_dir) / "clean_valid")

                elif step == "prepare_training_data":
                    in_dir = current_input or str(Path(cleaned_dir) / "clean_valid")
                    if not Path(in_dir).is_dir():  # 清洗跳过则回退 merged
                        in_dir = merged_dir
                    result = self._run_prepare_training_data(in_dir, training_data_dir)
                    current_input = result.get("output_path") or str(Path(training_data_dir) / "train.jsonl")

                elif step == "training":
                    train_data = current_input or str(Path(training_data_dir) / "train.jsonl")
                    result = self._run_training(train_data, training_dir)
                    current_input = result.get("output_dir") or training_dir

                elif step == "evaluation":
                    train_data = current_input or str(Path(training_data_dir) / "train.jsonl")
                    result = self._run_evaluation(train_data, evaluation_dir)

                elif step == "quality_validation":
                    in_dir = current_input or cleaned_dir
                    result = self._run_quality_validation(in_dir)

                elif step == "visualization":
                    in_dir = current_input or evaluation_dir
                    before_dir = step_results.get("distillation", {}).get("merged_output") or merged_dir
                    result = self._run_visualization(in_dir, before_dir,
                                                     step_results.get("evaluation", {}))
                else:
                    result = {"success": False, "error": f"Unhandled step {step}"}

                duration = (datetime.now() - step_start).total_seconds()
                sample_count = result.get("_stat_samples", 1) or 1
                self.timing_stats[step] = {"duration": duration, "samples": sample_count}
                result["duration_seconds"] = duration
                result["start_time"] = step_start.isoformat()
                result["end_time"] = datetime.now().isoformat()

                step_results[step] = result
                self.pipeline_status["steps_completed"].append(step)
                self.logger.info(f"\n✓ Step {step} done in {duration:.1f}s")

            except Exception as e:
                self.logger.error(f"\n✗ Step {step} failed: {e}")
                self.pipeline_status["steps_failed"].append(step)
                step_results[step] = {"success": False, "error": str(e)}

        self.pipeline_status["end_time"] = datetime.now()
        self.pipeline_status["results"] = step_results
        return self._finalize_pipeline()

    # ============================================================
    # Step 1: 数据蒸馏
    # ============================================================
    def _run_distillation(self, max_samples, tasks, output_dir, checkpoint_path):
        self.logger.info("\nSTEP: DISTILLATION")
        step_start = datetime.now()
        try:
            if max_samples:
                self.config.set("data.max_samples", max_samples)
            self.config.set("distillation.tasks", tasks)
            if output_dir:
                self.config.set("output.root_dir", output_dir)

            self.logger.info("[1/3] Loading COCO dataset...")
            coco_loader = COCODataLoader(self.config)
            coco_loader.initialize(self.config.get("data.val_split", "val2014"))
            self.logger.info(f"Dataset: {coco_loader.get_annotation_summary().get('total_images', 0)} images")

            self.logger.info("[2/3] Loading teacher model...")
            teacher = TeacherModel(self.config)
            mi = teacher.get_model_info()
            self.logger.info(f"  device={mi.get('device')}, precision={mi.get('precision')}")

            self.logger.info("[3/3] Creating distiller...")
            distiller = Distiller(teacher_model=teacher, config=self.config)

            if checkpoint_path:
                self.logger.info(f"Resume mode: checkpoint={checkpoint_path}")
                if not Path(checkpoint_path).exists():
                    checkpoint_path = None

            result = distiller.run_distillation(max_samples=max_samples,
                                                checkpoint_path=checkpoint_path)
            total = (datetime.now() - step_start).total_seconds()
            merged_output = result.get("merged_data_path", "./outputs/merged")
            return {
                "success": True,
                "processed_count": result.get("processed_count", 0),
                "failed_count": result.get("failed_count", 0),
                "merged_output": merged_output,
                "_stat_samples": result.get("processed_count", 1),
                "statistics": result.get("statistics", {}),
                "duration_seconds": total,
            }
        except Exception as e:
            self.logger.error(f"Distillation failed: {e}")
            return {"success": False, "error": str(e),
                    "duration_seconds": (datetime.now() - step_start).total_seconds()}

    # ============================================================
    # Step 2: 数据清洗
    # ============================================================
    def _run_cleaning(self, input_dir, cleaned_dir):
        self.logger.info(f"\nSTEP: CLEANING\nInput: {input_dir}")
        step_start = datetime.now()
        try:
            if not input_dir:
                raise ValueError("cleaning 需要输入目录")
            data_list = self._load_data_from_dir(input_dir)
            if not data_list:
                return {"success": False, "error": "No data files"}

            scorer = RewardModelScorer(self.config, logger=self.logger)
            scored = scorer.score_batch(data_list)
            stats = {
                "total": len(scored),
                "valid": sum(1 for s in scored if s.get("quality_score", {}).get("is_valid", False)),
            }

            partitioner = DataPartitioner(self.config, logger=self.logger)
            partition_report = partitioner.partition(
                samples=scored, cleaning_metadata={"scoring_stats": stats}
            )
            summary = partition_report.get("summary", {})
            clean_valid = str(Path(cleaned_dir) / "clean_valid")
            return {
                "success": True,
                "cleaned_output": clean_valid,
                "need_fix_output": str(Path(cleaned_dir) / "need_fix"),
                "discard_output": str(Path(cleaned_dir) / "discard"),
                "stats": {
                    "clean_valid_count": summary.get("clean_valid_count", 0),
                    "need_fix_count": summary.get("need_fix_count", 0),
                    "discard_count": summary.get("discard_count", 0),
                    "total_input": summary.get("total_samples", len(data_list)),
                },
                "_stat_samples": summary.get("total_samples", len(data_list)),
                "duration_seconds": (datetime.now() - step_start).total_seconds(),
            }
        except Exception as e:
            self.logger.error(f"Cleaning failed: {e}")
            return {"success": False, "error": str(e),
                    "duration_seconds": (datetime.now() - step_start).total_seconds()}

    # ============================================================
    # Step 3: 准备训练数据 (jsonl)
    # ============================================================
    def _run_prepare_training_data(self, input_dir, training_data_dir):
        self.logger.info(f"\nSTEP: PREPARE TRAINING DATA\nInput: {input_dir}")
        step_start = datetime.now()
        try:
            Path(training_data_dir).mkdir(parents=True, exist_ok=True)
            output_path = str(Path(training_data_dir) / "train.jsonl")
            exporter = TrainingDataExporter(config=self.config, logger=self.logger)
            stats = exporter.run(
                input_dir=input_dir, output_path=output_path,
                split=False, min_conf=0.0,
            )
            self.logger.info(
                f"  open={stats['open']} closed={stats['closed']} "
                f"skipped={stats['skipped']} -> {output_path}"
            )
            return {
                "success": True,
                "output_path": output_path,
                "stats": stats,
                "_stat_samples": stats["open"] + stats["closed"],
                "duration_seconds": (datetime.now() - step_start).total_seconds(),
            }
        except Exception as e:
            self.logger.error(f"Prepare training data failed: {e}")
            return {"success": False, "error": str(e),
                    "duration_seconds": (datetime.now() - step_start).total_seconds()}

    # ============================================================
    # Step 4: 学生模型训练
    # ============================================================
    def _run_training(self, train_data_path, training_dir):
        self.logger.info(f"\nSTEP: TRAINING\nTrain data: {train_data_path}")
        step_start = datetime.now()
        try:
            # 加载独立训练配置
            train_cfg_path = self.config.get("pipeline.train_config", "configs/train.yaml")
            cm = ConfigManager(train_cfg_path)
            # 用主配置产物覆盖训练数据路径与输出目录
            cm.set("train.train_data_path", train_data_path)
            cm.set("train.output_dir", training_dir)
            merged_out = self.config.get("output.training_dir", training_dir)
            cm.set("train.merged_output_dir",
                   str(Path(merged_out).parent / "student_merged"))

            # 延迟导入，避免在未安装 torch/transformers 时阻塞其它步骤
            from ..training.train import run_training
            output_dir = run_training(cm.config)
            self.logger.info(f"  student model saved -> {output_dir}")
            return {
                "success": True,
                "output_dir": output_dir,
                "merged_output_dir": cm.get("train.merged_output_dir"),
                "_stat_samples": 1,
                "duration_seconds": (datetime.now() - step_start).total_seconds(),
            }
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return {"success": False, "error": str(e),
                    "duration_seconds": (datetime.now() - step_start).total_seconds()}

    # ============================================================
    # Step 5: 学生模型评估
    # ============================================================
    def _run_evaluation(self, train_data_path, evaluation_dir):
        self.logger.info(f"\nSTEP: EVALUATION\nTrain data: {train_data_path}")
        step_start = datetime.now()
        try:
            # 把主配置里的产物路径注入 evaluation 配置
            self.config.set("evaluation.train_data_path", train_data_path)
            self.config.set("evaluation.output_dir", evaluation_dir)
            # 默认指向训练合并输出
            merged_student = str(
                Path(self.config.get("output.training_dir", "./outputs/student_ckpt")).parent
                / "student_merged"
            )
            if not self.config.get("evaluation.student_model_path"):
                self.config.set("evaluation.student_model_path", merged_student)

            from ..evaluation.evaluator import run_evaluation
            report = run_evaluation(self.config)
            self.logger.info(f"  evaluation report -> {evaluation_dir}/report.json")
            return {
                "success": True,
                "report": report,
                "output_dir": evaluation_dir,
                "_stat_samples": report.get("parameter_count", {}).get("total", 1),
                "duration_seconds": (datetime.now() - step_start).total_seconds(),
            }
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            return {"success": False, "error": str(e),
                    "duration_seconds": (datetime.now() - step_start).total_seconds()}

    # ============================================================
    # 可选：质量校验（不列入默认六步）
    # ============================================================
    def _run_quality_validation(self, input_dir):
        self.logger.info(f"\nSTEP: QUALITY VALIDATION\nInput: {input_dir}")
        step_start = datetime.now()
        try:
            if self.quality_validator is None:
                ann_dir = self.config.get("data.annotations_root", "./data/coco/annotations")
                self.quality_validator = DataQualityValidator(
                    config=self.config, logger=self.logger,
                    coco_annotations_dir=ann_dir,
                )
            output_dir = self.config.get("output.root_dir", "./outputs")
            result = self.quality_validator.run_full_validation(
                input_dir=input_dir, output_dir=output_dir
            )
            return {
                "success": result.get("success", False),
                "overall_passed": result.get("overall_passed", False),
                "validation_results": result.get("validation_results", {}),
                "_stat_samples": result.get("sample_count", 1),
                "duration_seconds": (datetime.now() - step_start).total_seconds(),
            }
        except Exception as e:
            self.logger.error(f"Quality validation failed: {e}")
            return {"success": False, "error": str(e),
                    "duration_seconds": (datetime.now() - step_start).total_seconds()}

    # ============================================================
    # Step 6: 可视化
    # ============================================================
    def _run_visualization(self, input_dir, before_dir=None, evaluation_results=None):
        self.logger.info(f"\nSTEP: VISUALIZATION\nInput: {input_dir}")
        step_start = datetime.now()
        try:
            if self.visualizer is None:
                self.visualizer = DataVisualizer(self.config, self.logger)
            data_list = self._load_data_from_dir(input_dir)
            before_data = self._load_data_from_dir(before_dir) if before_dir else None
            self.timing_stats["visualization"] = {"duration": 0.0, "samples": len(data_list)}
            report = self.visualizer.visualize_all(
                data_list=data_list, before_data=before_data,
                timing_stats=dict(self.timing_stats),
                pipeline_results=self.pipeline_status["results"],
                quality_validation_results=evaluation_results,
            )
            self._display_timing_summary()
            return {**report,
                    "duration_seconds": (datetime.now() - step_start).total_seconds()}
        except Exception as e:
            self.logger.error(f"Visualization failed: {e}")
            return {"success": False, "error": str(e),
                    "duration_seconds": (datetime.now() - step_start).total_seconds()}

    # ============================================================
    # 工具方法
    # ============================================================
    def _load_data_from_dir(self, input_dir: str) -> List[Dict]:
        input_path = Path(input_dir) if input_dir else Path(".")
        if not input_path.is_dir():
            return []
        data_files = [
            f for f in input_path.glob("*.json")
            if not f.name.startswith((
                "cleaning_report", "merged_summary", "validation",
                "checkpoint", "pipeline", "visualization",
                "data_quality", "timing", "report",
            ))
        ]
        out = []
        for jf in data_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    out.append(json.load(f))
            except Exception as e:
                self.logger.warning(f"Failed to load {jf}: {e}")
        return out

    def _test_configuration(self, max_samples, tasks) -> bool:
        self.logger.info("\n[DRY RUN] Testing configuration...")
        try:
            COCODataLoader(self.config); self.logger.info("✓ COCODataLoader")
            TeacherModel(self.config); self.logger.info("✓ TeacherModel")
            Distiller(teacher_model=TeacherModel(self.config), config=self.config); self.logger.info("✓ Distiller")
            RewardModelScorer(self.config, logger=self.logger); self.logger.info("✓ RewardModelScorer")
            DataPartitioner(config=self.config, logger=self.logger); self.logger.info("✓ DataPartitioner")
            TrainingDataExporter(config=self.config, logger=self.logger); self.logger.info("✓ TrainingDataExporter")
            self.logger.info(f"max_samples={max_samples or 'all'} tasks={tasks}")
            self.logger.info("✓ All components initialized")
            return True
        except Exception as e:
            self.logger.error(f"Configuration test failed: {e}")
            return False

    def _display_timing_summary(self):
        self.logger.info("\n" + "=" * 70 + "\nPIPELINE TIMING SUMMARY\n" + "=" * 70)
        self.logger.info(f"  {'Step':22s} {'Duration':12s} {'Samples':10s}")
        total = 0
        for step, st in self.timing_stats.items():
            d = st.get("duration", 0); s = st.get("samples", 0)
            self.logger.info(f"  {step:22s} {d:10.1f}s {s:8d}")
            total += d
        self.logger.info(f"  {'TOTAL':22s} {total:10.1f}s")
        self.logger.info("=" * 70)

    def _finalize_pipeline(self) -> Dict[str, Any]:
        self.logger.info("\n" + "=" * 70 + "\nPIPELINE COMPLETE\n" + "=" * 70)
        total = sum(st.get("duration", 0) if isinstance(st, dict) else st
                    for st in self.timing_stats.values())
        success = len(self.pipeline_status["steps_failed"]) == 0
        (self.logger.info("\n✓ Pipeline completed successfully!") if success
         else self.logger.warning(f"\n⚠ Pipeline completed with {len(self.pipeline_status['steps_failed'])} failed steps"))
        self.logger.info(f"Completed: {self.pipeline_status['steps_completed']}")
        self.logger.info(f"Failed: {self.pipeline_status['steps_failed']}")
        self.logger.info(f"Total duration: {total:.1f}s")

        report_path = Path("./outputs/pipeline_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        final_report = {
            "success": success,
            "start_time": self.pipeline_status["start_time"].isoformat(),
            "end_time": self.pipeline_status["end_time"].isoformat(),
            "total_duration_seconds": total,
            "steps_completed": self.pipeline_status["steps_completed"],
            "steps_failed": self.pipeline_status["steps_failed"],
            "timing_stats": dict(self.timing_stats),
            "results_summary": self._clean_results_for_json(self.pipeline_status["results"]),
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Pipeline report -> {report_path}")
        self.logger.info("=" * 70)
        return final_report

    def _clean_results_for_json(self, results: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = {}
        for name, res in results.items():
            if isinstance(res, dict):
                entry = {
                    "success": res.get("success"),
                    "duration_seconds": res.get("duration_seconds"),
                    "start_time": res.get("start_time"),
                    "end_time": res.get("end_time"),
                    "error": res.get("error"),
                }
                for k in ("merged_output", "cleaned_output", "output_path",
                          "output_dir", "processed_count", "failed_count",
                          "generated_plots", "report"):
                    if k in res:
                        entry[k] = res.get(k)
                cleaned[name] = entry
            else:
                cleaned[name] = {"success": False, "error": "Non-serializable"}
        return cleaned
