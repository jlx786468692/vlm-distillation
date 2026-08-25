"""
评估总入口
==========

run_evaluation(cfg)：
  1. 加载训练 JSONL 记录（教师标签）
  2. 加载学生模型（merged 全权重，贪婪推理）
  3. 依次运行三个维度评估
  4. 聚合写入 outputs/evaluation/report.json
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from .distillation_quality import evaluate as eval_distillation_quality
from .driving_scenario_eval import evaluate as eval_driving_scenario
from .deployment_efficiency import evaluate as eval_deployment_efficiency


def _load_records(train_data_path: str, max_samples: Optional[int]) -> List[Dict[str, Any]]:
    """读取训练 JSONL 的扁平记录（教师标签）。"""
    records = []
    with open(train_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
            if max_samples is not None and len(records) >= max_samples:
                break
    return records


class StudentInferencer:
    """
    学生模型贪婪推理器（评估用）。

    加载 merged 全权重模型；若指向 LoRA adapter 目录会回退挂载 base+adapter。
    """

    def __init__(self, model_path: str, device: str = "cuda",
                 dtype: str = "bfloat16", max_new_tokens: int = 256):
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        dt = getattr(torch, dtype, torch.bfloat16)

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        try:
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path, torch_dtype=dt, trust_remote_code=True,
                attn_implementation="sdpa",
            ).to(device)
        except Exception as e:
            # 可能是 LoRA adapter 目录：尝试用 base + adapter
            try:
                from peft import PeftModel
                base_cfg = self.processor
                # 用 processor 所基于的 base 名称（兼容性兜底）
                base_name = getattr(base_cfg, "name_or_path", None) or "models/Qwen2.5-VL-3B-Instruct"
                base = AutoModelForVision2Seq.from_pretrained(
                    base_name, torch_dtype=dt, trust_remote_code=True
                ).to(device)
                self.model = PeftModel.from_pretrained(base, model_path).to(device)
            except Exception:
                raise RuntimeError(f"加载学生模型失败 {model_path}: {e}") from e

        self.model.eval()
        self.dtype = dt

    @torch.inference_mode()
    def infer(self, image_path: str, question: str) -> str:
        """贪婪解码，返回学生回答文本。"""
        image = Image.open(image_path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        # 去掉 prompt 部分
        gen_ids = out[0, inputs["input_ids"].shape[1]:]
        return self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


def run_evaluation(cfg: Any) -> Dict[str, Any]:
    """
    评估主入口。

    Args:
        cfg: ConfigManager（含 evaluation.* / data.* / output.* 键）

    Returns:
        评估报告 dict（同时写入 outputs/evaluation/report.json）
    """
    eval_cfg = cfg.get("evaluation", {}) or {}
    if isinstance(eval_cfg, list):  # 容错：被解析成 list
        eval_cfg = {}

    train_data_path = eval_cfg.get(
        "train_data_path", cfg.get("output.training_data_dir", "./outputs/training") + "/train.jsonl"
    )
    images_root = eval_cfg.get("images_root", cfg.get("data.images_root", "./data/filter_coco/images/val2014"))
    student_model_path = eval_cfg.get(
        "student_model_path", cfg.get("output.training_dir", "./outputs/student_ckpt")
    )
    max_samples = eval_cfg.get("max_samples")
    max_new_tokens = int(eval_cfg.get("max_new_tokens", 256))
    device = eval_cfg.get("device", cfg.get("student.device", "cuda"))

    output_dir = Path(eval_cfg.get("output_dir", cfg.get("output.evaluation_dir", "./outputs/evaluation")))
    output_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "train_data_path": train_data_path,
        "student_model_path": student_model_path,
        "max_samples": max_samples,
        "dimensions": {},
    }

    print(f"[evaluation] train_data_path={train_data_path}")
    print(f"[evaluation] student_model_path={student_model_path}")

    # 1) 加载教师标签记录
    records = _load_records(train_data_path, max_samples)
    print(f"[evaluation] loaded {len(records)} teacher-labeled samples")
    if not records:
        report["error"] = "无训练记录，无法评估"
        _save_report(output_dir / "report.json", report)
        return report

    # 2) 加载学生模型
    t0 = time.time()
    inferencer = StudentInferencer(
        model_path=student_model_path, device=device,
        max_new_tokens=max_new_tokens,
    )
    report["model_load_seconds"] = round(time.time() - t0, 2)
    report["parameter_count"] = inferencer.count_parameters()

    # 3) 依次运行三维度评估
    dims = eval_cfg.get("dimensions", {}) or {}

    if dims.get("distillation_quality", {}).get("enabled", True):
        report["dimensions"]["distillation_quality"] = eval_distillation_quality(
            inferencer, records, images_root, eval_cfg.get("distillation_quality", {})
        )

    if dims.get("driving_scenario", {}).get("enabled", True):
        report["dimensions"]["driving_scenario"] = eval_driving_scenario(
            inferencer, records, images_root,
            eval_cfg.get("driving_scenario", {}), full_cfg=cfg,
        )

    if dims.get("deployment_efficiency", {}).get("enabled", True):
        report["dimensions"]["deployment_efficiency"] = eval_deployment_efficiency(
            inferencer, records, images_root, eval_cfg.get("deployment_efficiency", {})
        )

    _save_report(output_dir / "report.json", report)
    print(f"[evaluation] report saved -> {output_dir / 'report.json'}")
    return report


def _save_report(path: Path, report: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
