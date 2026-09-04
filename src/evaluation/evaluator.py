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


def _first_token_id(tokenizer, text: str, leading_space: bool = True) -> Optional[int]:
    """取文本(可选前缀加空格)的首个 token id，多 token 返回 None。

    镜像 `src/training/distill_dataset.py::_first_token_id`，保证评估侧候选
    token 映射与训练侧 KL 完全一致。
    """
    t = (" " + text) if leading_space else text
    ids = tokenizer.encode(t, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    return None


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

    @torch.inference_mode()
    def sequence_ce(self, image_path: str, question: str, target_text: str):
        """
        学生对「教师目标文本」的自回归 Cross-Entropy（蒸馏质量指标）。

        以教师输出文本（[Reasoning]...[Answer]... 或开集 reasoning_paragraph）
        作为学生要预测的 token 序列；学生仅做一次前向，计算目标段每个
        token 的平均 NLL。

        温度：推理阶段强制 T=1（不对 logits 做温度缩放），与训练阶段
        师生同 T 的设置解耦——这里测的是学生在原始尺度下对教师目标
        序列的似然。

        Returns:
            (mean_ce, n_tokens)：目标段平均负对数似然 / 目标 token 数。
            target_text 为空或前向失败时返回 (None, 0)。
        """
        import torch.nn.functional as F

        if not target_text:
            return None, 0

        image = Image.open(image_path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }]
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + target_text

        prompt_inputs = self.processor(text=[prompt_text], images=[image], return_tensors="pt")
        full_inputs = self.processor(text=[full_text], images=[image], return_tensors="pt")
        full_inputs = {k: v.to(self.device) for k, v in full_inputs.items()}

        # 目标 token 区间 = prompt 之后的 token
        prompt_len = prompt_inputs["input_ids"].shape[1]
        total_len = full_inputs["input_ids"].shape[1]
        if total_len <= prompt_len:
            return None, 0

        out = self.model(**full_inputs)
        logits = out.logits[0]  # [L, V]

        # next-token 预测：位置 i 的 logits 预测 token i+1
        # 目标 token 在 [prompt_len, total_len)，由 [prompt_len-1, total_len-1) 预测
        shift_logits = logits[prompt_len - 1:total_len - 1, :]   # [T, V]
        shift_labels = full_inputs["input_ids"][0, prompt_len:total_len]  # [T]

        # T=1：直接 log_softmax，不做温度缩放
        logp = F.log_softmax(shift_logits.float(), dim=-1)
        nll = -logp.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
        return nll.mean().item(), nll.numel()

    @torch.inference_mode()
    def sequence_ce_and_distribution(
        self, image_path: str, question: str, target_text: str,
        short_answer: Optional[str] = None, dist: Optional[Dict[str, float]] = None,
        temperature: float = 1.0,
    ):
        """合并前向：序列 CE + 软标签分布匹配（一次 forward 算两个指标）。

        与 sequence_ce 共用同一套 prompt+target 前向，额外在「答案 token 位置」
        上抽取学生对候选集的分布，与教师软标签 answer_distribution 比较。

        Args:
            short_answer: 闭集短答案（用于定位答案 token 位置）
            dist: soft_label.answer_distribution {answer_str: prob}
            temperature: 评估温度（默认 1.0，不缩放，与序列 CE 一致）
        Returns:
            (ce, n_tok, dist_res)
            dist_res = {"kl","cos","top1_match","n_cand"} 或 None（不可计算时）
        """
        import torch.nn.functional as F

        if not target_text:
            return None, 0, None

        tok = self.processor.tokenizer
        image = Image.open(image_path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }]
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + target_text

        prompt_inputs = self.processor(text=[prompt_text], images=[image], return_tensors="pt")
        full_inputs = self.processor(text=[full_text], images=[image], return_tensors="pt")
        full_inputs = {k: v.to(self.device) for k, v in full_inputs.items()}

        prompt_len = prompt_inputs["input_ids"].shape[1]
        total_len = full_inputs["input_ids"].shape[1]
        if total_len <= prompt_len:
            return None, 0, None

        out = self.model(**full_inputs)
        logits = out.logits[0]  # [L, V]

        # ---- 序列 CE（同 sequence_ce）----
        shift_logits = logits[prompt_len - 1:total_len - 1, :]   # [T, V]
        shift_labels = full_inputs["input_ids"][0, prompt_len:total_len]  # [T]
        logp = F.log_softmax(shift_logits.float(), dim=-1)
        nll = -logp.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
        ce_val = nll.mean().item()
        n_tok = nll.numel()

        # ---- 软标签分布匹配（镜像训练 _build_kl_meta）----
        dist_res = None
        if short_answer and dist:
            cand = self._candidate_token_ids(dist)
            if cand is not None:
                cand_ids, tprobs = cand
                # 定位答案 token：short_answer 首 token 在 response 内最后一次出现
                gold_tid = _first_token_id(tok, short_answer, leading_space=True)
                if gold_tid is None:
                    gold_tid = _first_token_id(tok, short_answer, leading_space=False)
                if gold_tid is not None:
                    resp = full_inputs["input_ids"][0, prompt_len:]
                    positions = (resp == gold_tid).nonzero(as_tuple=True)[0]
                    if positions.numel() > 0:
                        answer_pos = prompt_len + positions[-1].item()
                        if 0 < answer_pos < total_len:
                            pred = logits[answer_pos - 1, :]              # (V,)
                            cand_idx = torch.tensor(cand_ids, device=pred.device)
                            cand_logits = pred[cand_idx]                  # (C,)
                            T = float(temperature)
                            log_s = F.log_softmax(cand_logits.float() / T, dim=-1)
                            t = torch.tensor(tprobs, device=pred.device, dtype=log_s.dtype)
                            eps = 1e-9
                            kl = (t * (torch.log(t + eps) - log_s)).sum().item()
                            s_probs = torch.softmax(cand_logits.float() / T, dim=-1)
                            cos = F.cosine_similarity(s_probs, t, dim=0).item()
                            top1_match = bool(
                                s_probs.argmax().item() == t.argmax().item()
                            )
                            dist_res = {
                                "kl": kl, "cos": cos, "top1_match": top1_match,
                                "n_cand": len(cand_ids),
                            }
        return ce_val, n_tok, dist_res

    def _candidate_token_ids(self, dist: Dict[str, float]):
        """候选答案 -> 首 token id（单 token 才接受）。

        镜像 DistillDataset._build_kl_meta 的候选映射 + 跳过规则：
        多 token 候选 / token 冲突 / 概率和<=0 均返回 None（与训练一致）。
        Returns: (cand_ids, probs) 已归一化，或 None。
        """
        tok = self.processor.tokenizer
        cands = list(dist.keys())
        cand_ids = []
        for c in cands:
            tid = _first_token_id(tok, c, leading_space=True)
            if tid is None:
                tid = _first_token_id(tok, c, leading_space=False)
            if tid is None:
                return None  # 多 token 候选 -> 跳过
            cand_ids.append(tid)
        if len(set(cand_ids)) != len(cand_ids):
            return None  # token 冲突 -> 跳过
        probs = [float(dist[c]) for c in cands]
        s = sum(probs)
        if s <= 0:
            return None
        probs = [p / s for p in probs]
        return cand_ids, probs

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

    # held-out 评估：优先 eval_data_path，不存在则回退训练集并告警
    eval_data_path = eval_cfg.get("eval_data_path")
    is_heldout = False
    if eval_data_path and Path(eval_data_path).is_file():
        records_path = eval_data_path
        is_heldout = True
    else:
        records_path = train_data_path
        print(f"[evaluation] ⚠️ eval_data_path 未提供或不存在 ({eval_data_path})，"
              f"回退训练集评估（无 held-out，指标会高估泛化）")

    report: Dict[str, Any] = {
        "train_data_path": train_data_path,
        "eval_data_path": records_path if is_heldout else None,
        "is_heldout": is_heldout,
        "student_model_path": student_model_path,
        "max_samples": max_samples,
        "dimensions": {},
    }

    print(f"[evaluation] records_path={records_path} (held_out={is_heldout})")
    print(f"[evaluation] student_model_path={student_model_path}")

    # 1) 加载评估记录
    records = _load_records(records_path, max_samples)
    print(f"[evaluation] loaded {len(records)} samples for evaluation")
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
