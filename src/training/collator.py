"""
Qwen2.5-VL Distillation Collator (with KL metadata)
====================================================

Qwen2.5-VL 每张图 patch 数随分辨率变化 (image_grid_thw 不同)，
无法直接 torch.stack。本 collator 按以下方式拼装一个 batch:

  - input_ids / attention_mask / labels : 右填充到 batch 最大长度
        input_ids 填 pad_token_id ; attention_mask 填 0 ; labels 填 -100
  - pixel_values : 沿 dim=0 拼接 (P_total, C, H, W)
  - image_grid_thw : 沿 dim=0 拼接 (N_total, 3)
  - KL 元数据 (per-sample, 不 pad):
        has_kl           (B,) bool
        kl_answer_pos    (B,) long
        kl_candidate_ids List[Tensor]  每元素 (C_i,)
        kl_teacher_probs List[Tensor]  每元素 (C_i,)

模型按 input_ids 中 image_pad 占位符顺序消费 pixel_values，
本 collator 保持按样本顺序拼接，故两者对齐。
"""

from typing import Any, Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence


class QwenVLDistillCollator:
    def __init__(self, processor: Any, pad_token_id: int = None):
        self.processor = processor
        if pad_token_id is not None:
            self.pad_token_id = pad_token_id
        elif processor is not None and getattr(processor, "tokenizer", None) is not None:
            tok = processor.tokenizer
            self.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        else:
            self.pad_token_id = 0

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [b["input_ids"] for b in batch]
        labels = [b["labels"] for b in batch]
        attention_masks = [b["attention_mask"] for b in batch]

        # 右填充
        input_ids_pad = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels_pad = pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_pad = pad_sequence(attention_masks, batch_first=True, padding_value=0)

        out = {
            "input_ids": input_ids_pad,
            "attention_mask": attention_pad,
            "labels": labels_pad,
        }

        # 像素 / 网格拼接 (所有样本都有 1 张图)
        pixel_values = [b["pixel_values"] for b in batch]
        grids = [b["image_grid_thw"] for b in batch]
        if all(p is not None for p in pixel_values):
            out["pixel_values"] = torch.cat(pixel_values, dim=0)
            out["image_grid_thw"] = torch.cat(grids, dim=0)

        # KL 元数据 (padded 成纯张量, 便于 Trainer 设备迁移)
        if "has_kl" in batch[0]:
            B = len(batch)
            out["has_kl"] = torch.tensor([bool(b["has_kl"]) for b in batch], dtype=torch.bool)
            out["kl_answer_pos"] = torch.tensor(
                [int(b["kl_answer_pos"]) for b in batch], dtype=torch.long
            )
            max_c = max(b["kl_candidate_ids"].shape[0] for b in batch)
            cand_pad = torch.zeros(B, max_c, dtype=torch.long)
            prob_pad = torch.zeros(B, max_c, dtype=torch.float)
            mask = torch.zeros(B, max_c, dtype=torch.bool)
            for i, b in enumerate(batch):
                c = b["kl_candidate_ids"]
                n = c.shape[0]
                if n > 0:
                    cand_pad[i, :n] = c
                    prob_pad[i, :n] = b["kl_teacher_probs"]
                    mask[i, :n] = True
            out["kl_candidate_ids"] = cand_pad
            out["kl_teacher_probs"] = prob_pad
            out["kl_candidate_mask"] = mask

        return out
