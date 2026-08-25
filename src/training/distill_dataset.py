"""
Distillation SFT Dataset (with soft-label KL support)
=====================================================

解析训练 JSONL（由 TrainingDataExporter 产出的统一两段式记录），构建
Qwen2.5-VL 对话样本，生成 prompt-mask 标签 (仅在 assistant 回复 token 上
计算 SFT 损失)，并为闭集样本附加软标签 KL 蒸馏所需的元数据。

每行一条扁平记录，字段：
  - 开集 (question_category=="open"):
      hard_label.answer / cot_reasoning.reasoning_paragraph(完整推理)
      cot_reasoning.answer(抽取的短答案)
  - 闭集 (question_category=="closed"):
      hard_label.answer / soft_label.primary_answer
      soft_label.answer_distribution {answer_str: prob}   <- 软标签 KL 用
      cot_reasoning.{reasoning_paragraph, answer}          <- 两段式 CoT

assistant 目标文本由 build_target_text() 构造，模式：
  - cot    : 闭集 -> [Reasoning]reasoning_paragraph + [Answer]answer；开集 -> reasoning_paragraph
  - answer : 闭集 -> 仅短答案；开集 -> reasoning_paragraph(完整)

软标签 KL (可选):
  对每个闭集样本，把 candidate 答案映射到 student 词表 token，
  在答案 token 前一位置的 logits 上与 teacher 概率分布做 KL。
  多 token 答案 / 候选 token 冲突 / 缺 answer_distribution 的样本自动跳过 KL。
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# 目标文本构造
# ============================================================
def _cot_reasoning_of(rec: Dict[str, Any]) -> Dict[str, Any]:
    """取记录的 cot_reasoning（统一两段式），兼容旧版三段式。"""
    cot = rec.get("cot_reasoning") or {}
    if not isinstance(cot, dict):
        return {}
    # 兼容旧版三段式：若没有两段式字段但有 structured_reasoning，转换为两段式
    if not cot.get("reasoning_paragraph") and not cot.get("answer"):
        sr = cot.get("structured_reasoning") or {}
        if isinstance(sr, dict) and sr:
            parts = [sr.get(k, "").strip()
                     for k in ("observation", "analysis")
                     if sr.get(k) and str(sr.get(k)).strip()]
            return {
                "reasoning_paragraph": "\n".join(parts),
                "answer": (sr.get("conclusion") or "").strip(),
            }
    return cot


def _short_answer_of(rec: Dict[str, Any]) -> Optional[str]:
    """闭集简短答案优先级: hard_label.answer > cot_reasoning.answer > soft_label.primary_answer。"""
    short = None
    hl = rec.get("hard_label")
    if isinstance(hl, dict):
        short = hl.get("answer")
    if not short:
        short = (_cot_reasoning_of(rec).get("answer"))
    if not short:
        soft = rec.get("soft_label")
        if isinstance(soft, dict):
            short = soft.get("primary_answer")
    return str(short).strip() if short else None


def get_closed_meta(rec: Dict[str, Any]) -> Dict[str, Any]:
    """提取闭集样本的软标签元信息 (用于 KL)。"""
    soft = rec.get("soft_label") or {}
    return {
        "answer_distribution": soft.get("answer_distribution"),
        "short_answer": _short_answer_of(rec),
    }


def build_target_text(rec: Dict[str, Any], mode: str = "cot") -> Optional[str]:
    """
    根据训练记录构造 assistant 目标文本。

    Args:
        rec: 扁平训练记录（含 question_category/question_type/hard_label/
             soft_label/cot_reasoning，或旧版 tasks.vqa 字段）
        mode: "cot" | "answer"
    Returns:
        目标文本；无法构造时返回 None
    """
    qtype = rec.get("question_type", "")
    qcat = rec.get("question_category")

    # 判定开集：优先 question_category，回退 question_type / 旧式 answer-only
    is_open = (qcat == "open") or (qtype == "open_descriptive") \
        or ("answer" in rec and "hard_label" not in rec and "soft_label" not in rec)

    # ---------- 开集：直接用教师完整推理 ----------
    if is_open:
        cot = _cot_reasoning_of(rec)
        rp = cot.get("reasoning_paragraph")
        if rp and str(rp).strip():
            return str(rp).strip()
        # 旧版 merged 记录（tasks.vqa.answer）兜底
        ans = rec.get("answer")
        if isinstance(ans, str) and ans.strip():
            return ans.strip()
        return None

    # ---------- 闭集 ----------
    short_answer = _short_answer_of(rec)
    if not short_answer:
        return None

    if mode == "answer":
        return short_answer

    # cot 模式：复现教师 [Reasoning]+[Answer] 两段式
    cot = _cot_reasoning_of(rec)
    reasoning_text = (cot.get("reasoning_paragraph") or "").strip()
    if reasoning_text:
        return f"[Reasoning] {reasoning_text}\n[Answer] {short_answer}"
    return f"[Answer] {short_answer}"


# ============================================================
# 路径解析
# ============================================================
def resolve_image_path(stored_path: str, image_id, images_root: str) -> Optional[str]:
    """
    标签里 image_path 指向 data/coco/val2014/xxx.jpg (旧路径，不存在)，
    真实图像在 images_root。优先级:
      1) 原始路径存在
      2) images_root / basename(stored_path)
      3) images_root / COCO_val2014_{image_id:012d}.jpg
    """
    candidates = []
    if stored_path:
        candidates.append(stored_path)
        candidates.append(os.path.join(images_root, os.path.basename(stored_path)))
    if image_id is not None:
        candidates.append(os.path.join(images_root, f"COCO_val2014_{int(image_id):012d}.jpg"))
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None


# ============================================================
# Dataset
# ============================================================
class DistillDataset(Dataset):
    """
    蒸馏 SFT 数据集 (可选软标签 KL)。

    __getitem__ 返回:
        input_ids        (L,)
        attention_mask   (L,)
        labels           (L,)          prompt 部分 = -100
        pixel_values     (P, C, H, W)
        image_grid_thw   (1, 3)
    闭集样本额外返回 KL 元数据 (has_kl=True 时有效):
        has_kl           bool
        kl_answer_pos    int           答案 token 在 full_ids 中的下标
        kl_candidate_ids (C,)          候选答案 token id
        kl_teacher_probs (C,)          teacher 概率 (与候选对齐, 已归一化)
    """

    def __init__(
        self,
        train_data_path: str,
        images_root: str,
        processor: Any,
        max_length: int = 1280,
        max_pixels: int = 802816,
        min_pixels: int = 3136,
        target_mode: str = "cot",
        system_prompt: Optional[str] = None,
        max_samples: Optional[int] = None,
        # 向后兼容：旧代码可能传 merged_dir=
        merged_dir: Optional[str] = None,
    ):
        self.train_data_path = Path(train_data_path) if train_data_path \
            else (Path(merged_dir) if merged_dir else None)
        self.images_root = images_root
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.target_mode = target_mode
        self.system_prompt = system_prompt

        # 在 image_processor 上设置像素上限/下限 (最稳健)
        try:
            self.processor.image_processor.max_pixels = self.max_pixels
            self.processor.image_processor.min_pixels = self.min_pixels
        except Exception as e:
            print(f"[DistillDataset] set image_processor pixels failed (忽略): {e}")

        # 收集样本
        self.samples: List[Dict[str, Any]] = []
        skipped = 0

        if self.train_data_path is None:
            raise ValueError("需要提供 train_data_path (jsonl) 或 merged_dir")

        if self.train_data_path.is_file():
            # JSONL 模式（推荐）：每行一条扁平训练记录
            skipped = self._load_jsonl(self.train_data_path)
        elif self.train_data_path.is_dir():
            # 兼容模式：目录下每张图一个 JSON（旧 merged 格式）
            skipped = self._load_merged_dir(self.train_data_path)
        else:
            raise FileNotFoundError(
                f"训练数据不存在: {self.train_data_path} "
                f"(应为 jsonl 文件或含 *.json 的目录)"
            )

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        n_kl = sum(1 for s in self.samples if s["is_closed"])
        print(f"[DistillDataset] loaded {len(self.samples)} samples "
              f"(skipped {skipped}), mode={target_mode}, "
              f"KL-eligible(closed)={n_kl}, src={self.train_data_path}")

    def _load_jsonl(self, path: Path) -> int:
        """读取 JSONL（每行一条扁平记录）。"""
        skipped = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if not rec.get("question"):
                    continue
                target = build_target_text(rec, self.target_mode)
                if not target:
                    skipped += 1
                    continue
                img_path = resolve_image_path(
                    rec.get("image_path", ""), rec.get("image_id"), self.images_root
                )
                if not img_path:
                    skipped += 1
                    continue
                meta = get_closed_meta(rec)
                self.samples.append({
                    "image_path": img_path,
                    "question": rec["question"],
                    "target": target,
                    "answer_distribution": meta["answer_distribution"],
                    "short_answer": meta["short_answer"],
                    "is_closed": meta["answer_distribution"] is not None,
                })
        return skipped

    def _load_merged_dir(self, dir_path: Path) -> int:
        """兼容旧版 merged 目录（每张图一个 JSON，tasks.vqa 结构）。"""
        skipped = 0
        files = sorted(dir_path.glob("*.json"))
        files = [f for f in files if f.name != "merged_summary.json"]
        for fp in files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    rec = json.load(f)
            except Exception:
                continue
            vqa = (rec.get("tasks") or {}).get("vqa")
            if not isinstance(vqa, dict) or not vqa.get("question"):
                continue
            target = build_target_text(vqa, self.target_mode)
            if not target:
                skipped += 1
                continue
            img_path = resolve_image_path(
                rec.get("image_path", ""), rec.get("image_id"), self.images_root
            )
            if not img_path:
                skipped += 1
                continue
            meta = get_closed_meta(vqa)
            self.samples.append({
                "image_path": img_path,
                "question": vqa["question"],
                "target": target,
                "answer_distribution": meta["answer_distribution"],
                "short_answer": meta["short_answer"],
                "is_closed": meta["answer_distribution"] is not None,
            })
        return skipped

    def __len__(self) -> int:
        return len(self.samples)

    # ---------- 候选答案 -> token id ----------
    def _first_token_id(self, text: str, leading_space: bool = True) -> Optional[int]:
        """取文本(前缀加空格)的首个 token id，多 token 返回 None。"""
        t = (" " + text) if leading_space else text
        ids = self.tokenizer.encode(t, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
        return None

    # ---------- KL 元数据 ----------
    def _build_kl_meta(
        self, sample: Dict[str, Any], full_ids: torch.Tensor, prompt_len: int
    ) -> Dict[str, Any]:
        """
        构建软标签 KL 所需元数据。失败则返回 has_kl=False。

        Args:
            sample: 数据样本 (含 answer_distribution, short_answer)
            full_ids: 完整 token 序列 (L,)
            prompt_len: prompt token 数
        """
        empty = {"has_kl": False, "kl_answer_pos": -1,
                 "kl_candidate_ids": torch.empty(0, dtype=torch.long),
                 "kl_teacher_probs": torch.empty(0, dtype=torch.float)}

        dist = sample["answer_distribution"]
        short = sample["short_answer"]
        if not dist or not short:
            return empty

        # 候选答案 -> 首 token id (单 token 才接受)
        cands = list(dist.keys())
        cand_ids = []
        for c in cands:
            tid = self._first_token_id(c, leading_space=True)
            if tid is None:
                # 退一步：不加前导空格再试
                tid = self._first_token_id(c, leading_space=False)
            if tid is None:
                return empty  # 多 token 候选，跳过 KL
            cand_ids.append(tid)
        if len(set(cand_ids)) != len(cand_ids):
            return empty  # 候选 token 冲突，跳过
        cand_ids_t = torch.tensor(cand_ids, dtype=torch.long)

        # teacher 概率归一化
        probs = [float(dist[c]) for c in cands]
        s = sum(probs)
        if s <= 0:
            return empty
        probs_t = torch.tensor([p / s for p in probs], dtype=torch.float)

        # 定位答案 token: 在 response 区间找 short 的首 token (最后一次出现)
        gold_tid = self._first_token_id(short, leading_space=True)
        if gold_tid is None:
            gold_tid = self._first_token_id(short, leading_space=False)
        if gold_tid is None:
            return empty
        resp = full_ids[prompt_len:]
        positions = (resp == gold_tid).nonzero(as_tuple=True)[0]
        if positions.numel() == 0:
            return empty  # tokenization 不一致，跳过
        last = positions[-1].item()
        answer_pos = prompt_len + last

        # 截断保护: 答案 token 被截掉则跳过
        if answer_pos >= full_ids.shape[0]:
            return empty
        if answer_pos == 0:
            return empty

        return {
            "has_kl": True,
            "kl_answer_pos": answer_pos,
            "kl_candidate_ids": cand_ids_t,
            "kl_teacher_probs": probs_t,
        }

    # ---------- 消息构建 ----------
    def _build_messages(self, image: Image.Image, question: str, target: str):
        user_content = [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]
        messages_full = []
        messages_prompt = []
        if self.system_prompt:
            messages_full.append({"role": "system", "content": self.system_prompt})
            messages_prompt.append({"role": "system", "content": self.system_prompt})
        messages_full.append({"role": "user", "content": user_content})
        messages_full.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
        messages_prompt.append({"role": "user", "content": user_content})
        return messages_full, messages_prompt

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        image = Image.open(s["image_path"]).convert("RGB")

        messages_full, messages_prompt = self._build_messages(image, s["question"], s["target"])

        full_text = self.processor.apply_chat_template(
            messages_full, tokenize=False, add_generation_prompt=False
        )
        prompt_text = self.processor.apply_chat_template(
            messages_prompt, tokenize=False, add_generation_prompt=True
        )

        full = self.processor(text=[full_text], images=[image], return_tensors="pt", padding=False)
        prompt = self.processor(text=[prompt_text], images=[image], return_tensors="pt", padding=False)

        full_ids = full["input_ids"][0]                  # (L,)
        prompt_len = prompt["input_ids"].shape[1]        # prompt token 数 (含图像展开)

        labels = full_ids.clone()
        if prompt_len < labels.shape[0]:
            labels[:prompt_len] = -100
        else:
            labels[:] = -100

        # 右截断
        truncated = full_ids.shape[0] > self.max_length
        if truncated:
            full_ids = full_ids[: self.max_length]
            labels = labels[: self.max_length]

        attention_mask = torch.ones(full_ids.shape[0], dtype=torch.long)

        # 像素张量规整为 (P, C, H, W)
        pixel_values = full["pixel_values"]
        if pixel_values.ndim == 5 and pixel_values.shape[0] == 1:
            pixel_values = pixel_values.squeeze(0)
        image_grid_thw = full["image_grid_thw"]
        if image_grid_thw.ndim == 3 and image_grid_thw.shape[0] == 1:
            image_grid_thw = image_grid_thw.squeeze(0)

        item = {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

        # 软标签 KL 元数据 (仅闭集样本)
        if s["is_closed"]:
            kl = self._build_kl_meta(s, full_ids, prompt_len)
        else:
            kl = {"has_kl": False, "kl_answer_pos": -1,
                  "kl_candidate_ids": torch.empty(0, dtype=torch.long),
                  "kl_teacher_probs": torch.empty(0, dtype=torch.float)}
        item["has_kl"] = kl["has_kl"]
        item["kl_answer_pos"] = kl["kl_answer_pos"]
        item["kl_candidate_ids"] = kl["kl_candidate_ids"]
        item["kl_teacher_probs"] = kl["kl_teacher_probs"]

        return item
