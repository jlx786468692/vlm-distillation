"""
维度1：蒸馏质量评估
====================

学生是否学到教师精髓。学生在每个样本上贪婪推理，与教师标签对比：
  - 教师答案精确匹配率（闭集：student 短答案 == teacher 短答案）
  - 闭集 primary_answer 一致率（学生 top-1 == soft_label.primary_answer）
  - CoT 文本相似度（学生推理 vs 教师推理 reasoning_paragraph，词级 Jaccard）
  - 开集答案文本相似度（学生输出 vs 教师推理，词级 Jaccard）

注：分布 KL 需要学生在答案 token 处的 logits，开销大且实现复杂，
本实现以 top-1 一致率为主，KL 留接口（返回 None）。
"""

import os
from typing import Any, Dict

from ._common import (
    extract_student_answer,
    extract_student_reasoning,
    normalize_for_match,
    teacher_reasoning,
    teacher_short_answer,
    token_overlap,
)


def evaluate(inferencer, records, images_root: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """运行蒸馏质量评估。"""
    n = len(records)
    closed_match = 0
    closed_total = 0
    primary_match = 0
    primary_total = 0
    cot_sim_sum = 0.0
    cot_sim_n = 0
    open_sim_sum = 0.0
    open_sim_n = 0

    detail = []
    for rec in records:
        img = _resolve_image(rec, images_root)
        if not img:
            continue
        q = rec.get("question", "")
        out = inferencer.infer(img, q)

        is_closed = (rec.get("question_category") == "closed")
        t_short = teacher_short_answer(rec)
        s_short = extract_student_answer(out)

        if is_closed:
            closed_total += 1
            if t_short and normalize_for_match(s_short) == normalize_for_match(t_short):
                closed_match += 1
            soft = rec.get("soft_label") or {}
            pa = soft.get("primary_answer")
            if pa:
                primary_total += 1
                if normalize_for_match(s_short) == normalize_for_match(pa):
                    primary_match += 1
        else:
            # 开集：整段输出 vs 教师推理文本相似度
            t_rp = teacher_reasoning(rec)
            if t_rp:
                open_sim_sum += token_overlap(out, t_rp)
                open_sim_n += 1

        # CoT 相似度（学生推理段 vs 教师推理段）
        s_reason = extract_student_reasoning(out)
        t_reason = teacher_reasoning(rec)
        if t_reason and s_reason:
            cot_sim_sum += token_overlap(s_reason, t_reason)
            cot_sim_n += 1

        if len(detail) < cfg.get("detail_samples", 5):
            detail.append({
                "image_id": rec.get("image_id"),
                "question": q,
                "teacher_answer": t_short,
                "student_answer": s_short,
                "student_output": out[:200],
            })

    return {
        "dimension": "distillation_quality",
        "samples_evaluated": n,
        "closed_answer_match_rate": round(closed_match / max(closed_total, 1), 4),
        "closed_primary_match_rate": round(primary_match / max(primary_total, 1), 4),
        "open_text_similarity": round(open_sim_sum / max(open_sim_n, 1), 4),
        "cot_similarity": round(cot_sim_sum / max(cot_sim_n, 1), 4),
        "distribution_kl": None,  # 接口预留：需学生 logits，暂不实现
        "detail": detail,
    }


def _resolve_image(rec, images_root: str):
    import os
    p = rec.get("image_path", "")
    iid = rec.get("image_id")
    candidates = []
    if p:
        candidates.append(p)
        candidates.append(os.path.join(images_root, os.path.basename(p)))
    if iid is not None:
        try:
            candidates.append(os.path.join(images_root, f"COCO_val2014_{int(iid):012d}.jpg"))
        except (ValueError, TypeError):
            pass
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None
