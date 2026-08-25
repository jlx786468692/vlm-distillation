"""
Student Training Entrypoint
===========================

加载 Qwen2.5-VL-3B-Instruct 学生模型，构建蒸馏数据集，
用 HuggingFace Trainer 进行 SFT (LoRA 或全量微调)。

用法 (CLI):
    python scripts/train_student.py --config configs/train.yaml
    python scripts/train_student.py --use_lora false --learning_rate 1e-5 --epochs 5
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    Trainer,
    TrainingArguments,
)

from .distill_dataset import DistillDataset
from .collator import QwenVLDistillCollator


# ============================================================
# 蒸馏 Trainer: SFT(CE) + 软标签 KL
# ============================================================
_KL_KEYS = ("has_kl", "kl_answer_pos", "kl_candidate_ids", "kl_teacher_probs", "kl_candidate_mask")


class DistillTrainer(Trainer):
    """
    在标准 SFT 交叉熵基础上，对闭集样本额外施加软标签 KL 蒸馏。

    loss = CE(标签) + kl_weight * mean_i KL(teacher_dist_i || student_dist_i)

    student_dist_i: 答案 token 前一位置的 logits 在候选 token 上做
                    softmax(logits / temperature)。
    teacher_dist_i: 来自 soft_label.answer_distribution (已归一化)。
    """

    def __init__(self, *args, kl_weight: float = 0.0, kl_temperature: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_weight = float(kl_weight)
        self.kl_temperature = float(kl_temperature)
        # 子损失累计缓冲 (按 logging_steps 窗口取均值, 与内置 loss 对齐)
        self._ce_sum = 0.0
        self._kl_sum = 0.0
        self._n = 0
        # 单卡保护：无论 torch 当前可见几张卡，都拆除 Trainer 自动包的
        # nn.DataParallel。Qwen2.5-VL 视觉塔为变长 grid_thw，DataParallel
        # 按 batch 切分 pixel_values 会与 rotary_pos_emb 错位，导致 reshape
        # 报错 (shape '[N,4,-1]' is invalid for input of size M)。
        if isinstance(self.model, torch.nn.DataParallel):
            self.model = self.model.module
            self.args.n_gpu = 1
            print("[train] 检测到 nn.DataParallel，已拆除 -> 强制单卡 (cuda:0) 训练")

    def log(self, logs, *args, **kwargs):
        # 在 Trainer 的 logging_steps 边界注入 ce/kl 分项均值
        if self._n > 0:
            logs["ce_loss"] = round(self._ce_sum / self._n, 6)
            logs["kl_loss"] = round(self._kl_sum / self._n, 6)
            self._ce_sum = 0.0
            self._kl_sum = 0.0
            self._n = 0
        super().log(logs, *args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 剥离 KL 专属键 (模型 forward 不认识)
        kl_inputs = {}
        for k in _KL_KEYS:
            if k in inputs:
                kl_inputs[k] = inputs.pop(k)

        outputs = model(**inputs)          # inputs 含 labels -> outputs.loss 为 SFT CE
        ce_loss = outputs.loss

        loss = ce_loss
        kl_loss = torch.zeros((), device=ce_loss.device)

        if self.kl_weight > 0 and kl_inputs.get("has_kl") is not None:
            has_kl = kl_inputs["has_kl"].to(ce_loss.device)
            if has_kl.any():
                logits = outputs.logits               # (B, L, V)
                answer_pos = kl_inputs["kl_answer_pos"].to(ce_loss.device)
                cand_all = kl_inputs["kl_candidate_ids"].to(ce_loss.device)
                prob_all = kl_inputs["kl_teacher_probs"].to(ce_loss.device).to(logits.dtype)
                mask_all = kl_inputs["kl_candidate_mask"].to(ce_loss.device)
                T = self.kl_temperature

                kl_terms = []
                for i in range(has_kl.shape[0]):
                    if not bool(has_kl[i]):
                        continue
                    pos = int(answer_pos[i].item())
                    if pos <= 0:
                        continue
                    m = mask_all[i]
                    cand = cand_all[i][m]               # (C_i,)
                    tprobs = prob_all[i][m]             # (C_i,)
                    if cand.numel() == 0 or tprobs.numel() == 0:
                        continue
                    # logits[i, pos-1] 预测 token at pos (答案首 token)
                    pred_logits = logits[i, pos - 1, :]            # (V,)
                    cand_logits = pred_logits[cand]                # (C_i,)
                    log_s = F.log_softmax(cand_logits / T, dim=-1)  # 学生分布 (温度缩放)
                    # KL(t||s) = sum t*(log t - log s)
                    kl = (tprobs * (torch.log(tprobs + 1e-9) - log_s)).sum()
                    kl_terms.append(kl)

                if kl_terms:
                    kl_loss = torch.stack(kl_terms).mean()
                    loss = loss + self.kl_weight * kl_loss

        # 累计子损失用于日志 (detach + 转主机浮点, 不影响反向)
        self._ce_sum += float(ce_loss.detach().float().item())
        self._kl_sum += float(kl_loss.detach().float().item())
        self._n += 1

        # 日志
        if return_outputs:
            outputs.loss = loss
            return (loss, outputs)
        return loss


# ============================================================
# 模型构建
# ============================================================
def build_student_model(cfg: Dict[str, Any]):
    """
    加载学生 VLM，可选挂载 LoRA。

    Args:
        cfg: 训练配置 (含 student.* 与 train.* 键)
    Returns:
        (model, processor)
    """
    student_cfg = cfg.get("student", {})
    train_cfg = cfg.get("train", {})

    model_name = student_cfg.get("model_name") or student_cfg.get("model_name_path")
    dtype_str = student_cfg.get("torch_dtype", "bfloat16")
    dtype = getattr(torch, dtype_str, torch.bfloat16)
    attn_impl = train_cfg.get("attn_implementation", "sdpa")

    print(f"[train] loading student model: {model_name} (dtype={dtype_str}, attn={attn_impl})")

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation=attn_impl,
        # 训练阶段关闭 KV cache (与 gradient checkpointing 配合)
    )

    # ---- LoRA ----
    use_lora = train_cfg.get("use_lora", True)
    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as e:
            raise ImportError("use_lora=True 但未安装 peft，请 pip install peft 或设 use_lora=false") from e

        lora_cfg = LoraConfig(
            r=train_cfg.get("lora_r", 64),
            lora_alpha=train_cfg.get("lora_alpha", 128),
            lora_dropout=train_cfg.get("lora_dropout", 0.05),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=train_cfg.get(
                "lora_target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    else:
        # 全量微调：启用梯度
        for p in model.parameters():
            p.requires_grad = True

    # ---- 梯度检查点 ----
    # 训练阶段关闭 KV cache；梯度检查点由 TrainingArguments 统一开启，
    # 这里仅补 PEFT 必需的 input_require_grads (冻结基座下 checkpoint 重算需要)
    model.config.use_cache = False
    if train_cfg.get("gradient_checkpointing", True) and use_lora:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    return model, processor


# ============================================================
# 训练主流程
# ============================================================
def run_training(cfg: Dict[str, Any]) -> str:
    """
    运行学生模型 SFT 蒸馏训练。

    Args:
        cfg: 完整配置 dict (含 student, train 两个 section)
    Returns:
        output_dir
    """
    train_cfg = cfg["train"]
    student_cfg = cfg.get("student", {})
    output_dir = train_cfg["output_dir"]

    print(f"[train] output_dir = {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # 强制单卡训练：Qwen2.5-VL 视觉塔为变长 grid_thw，nn.DataParallel 按batch 切分会
    # 使 pixel_values 与 rotary_pos_emb 错位，触发 reshape 报错
    # (shape '[N,4,-1]' is invalid for input of size M)。单卡训练无此问题。
    # 仅在 CUDA 尚未初始化时设置环境变量——否则“程序启动后改动
    # CUDA_VISIBLE_DEVICES”会让 torch 把可见设备数清零；该情形改由
    # DistillTrainer 内部的去 DataParallel 保护兜底。
    cuda_devices = train_cfg.get("cuda_devices", "0")
    if cuda_devices and not torch.cuda.is_initialized():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)
        print(f"[train] CUDA_VISIBLE_DEVICES={cuda_devices} (单卡, 训练前设置)")

    # 1) 模型 / 处理器
    model, processor = build_student_model(cfg)

    # 2) 数据集
    dataset = DistillDataset(
        train_data_path=train_cfg.get("train_data_path") or train_cfg.get("merged_dir"),
        images_root=train_cfg["images_root"],
        processor=processor,
        max_length=train_cfg.get("max_length", 1280),
        max_pixels=train_cfg.get("max_pixels", 1003520),
        min_pixels=train_cfg.get("min_pixels", 3136),
        target_mode=train_cfg.get("target_mode", "cot"),
        system_prompt=train_cfg.get("system_prompt"),
        max_samples=train_cfg.get("max_samples"),
    )

    if len(dataset) == 0:
        raise RuntimeError(
            f"数据集为空: train_data_path="
            f"{train_cfg.get('train_data_path') or train_cfg.get('merged_dir')} "
            f"images_root={train_cfg['images_root']}，请检查标签与图像路径"
        )

    collator = QwenVLDistillCollator(processor=processor)

    # 3) 训练参数
    # 显存估算: per_device_batch * grad_accum = 等效 batch
    targs = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 16),
        learning_rate=train_cfg.get("learning_rate", 1e-4),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 500),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
        seed=train_cfg.get("seed", 42),
        report_to=train_cfg.get("report_to", "tensorboard"),
        remove_unused_columns=False,   # 保留 pixel_values/image_grid_thw
        dataloader_pin_memory=True,
    )

    # 4) Trainer (SFT CE + 软标签 KL)
    trainer = DistillTrainer(
        model=model,
        args=targs,
        train_dataset=dataset,
        data_collator=collator,
        kl_weight=train_cfg.get("kl_weight", 0.5),
        kl_temperature=train_cfg.get("kl_temperature", 2.0),
    )

    # 5) 训练
    print("[train] start training ...")
    trainer.train()

    # 6) 保存
    print(f"[train] saving to {output_dir}")
    trainer.save_model(output_dir)
    try:
        processor.save_pretrained(output_dir)
    except Exception as e:
        print(f"[train] processor save warning: {e}")

    # 7) LoRA 合并导出 (可选)
    if train_cfg.get("use_lora", True) and train_cfg.get("merged_output_dir"):
        _merge_and_save(model, processor, train_cfg["merged_output_dir"])

    return output_dir


def _merge_and_save(model, processor, merged_dir: str) -> None:
    """LoRA 权重合并回基座，导出可独立部署的全权重模型。"""
    print(f"[train] merging LoRA -> {merged_dir}")
    try:
        os.makedirs(merged_dir, exist_ok=True)
        # peft 模型合并
        merged = model.merge_and_unload()
        merged.save_pretrained(merged_dir, safe_serialization=True)
        processor.save_pretrained(merged_dir)
        print(f"[train] merged model saved to {merged_dir}")
    except Exception as e:
        print(f"[train] merge failed (非致命): {e}")
