"""
维度1：蒸馏质量评估
====================

学生是否学到教师精髓。学生在每个样本上贪婪推理，与教师标签对比：
  - 教师答案精确匹配率（闭集：[Answer] 标记后整行 == 教师短答案）
  - 闭集 primary_answer 一致率（学生 top-1 == soft_label.primary_answer）
  - CoT 文本相似度（学生推理 vs 教师推理 reasoning_paragraph，词级 Jaccard）
  - 开集答案文本相似度（学生输出 vs 教师推理，词级 Jaccard）
  - 序列 Cross-Entropy：学生对「教师目标文本」的自回归 NLL（分闭集/开集）
    ——取代原来的 distribution_kl（next-token 分布 KL 需要候选集对齐，
    实现复杂且开集无候选分布；序列 CE 是工业更常用的蒸馏质量度量，
    开闭集均适用）。
  - 软标签分布匹配（仅闭集）：在答案 token 位置抽取学生对候选集的分布，
    与教师 answer_distribution 比较（KL / 余弦 / top1 一致率）。
    补回训练侧 KL 蒸馏的可观测性，判断学生是否学到分布形状而非仅 top-1。

注：推理阶段学生 CE 强制 T=1，不做温度缩放（训练阶段师生同 T）。
"""

import os
from collections import defaultdict
from typing import Any, Dict

from ._common import (
    extract_student_answer,
    extract_student_reasoning,
    normalize_for_match,
    score_match,
    teacher_reasoning,
    teacher_short_answer,
    token_overlap,
)


def _classify_scenario(question_type: str, question: str = "") -> str:
    """
    按 question_type(回退 question 文本)近似分桶到驾驶场景维度。

    覆盖本数据集出现的模板型 question_type：is/are*、how many、what color、
    none of the above、what/where/who…。返回: yes_no/counting/color/choice/open/other。
    """
    qt = (question_type or "").strip().lower()
    q = (question or "").strip().lower()
    text = qt or q
    if not text:
        return "other"
    if "how many" in text:
        return "counting"
    if "color" in text or "colour" in text:
        return "color"
    if "none of the above" in text or "multiple choice" in text:
        return "choice"
    if text.startswith("is") or text.startswith("are"):
        return "yes_no"
    if text.startswith("what") or text.startswith("where") \
            or text.startswith("who") or text.startswith("which") \
            or text.startswith("why"):
        return "open"
    return "other"



def evaluate(inferencer, records, images_root: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """运行蒸馏质量评估。"""
    from ..training.distill_dataset import build_target_text, get_closed_meta

    target_mode = cfg.get("target_mode", "cot")
    n = len(records)
    closed_match = 0
    closed_total = 0
    primary_match = 0
    primary_total = 0
    cot_sim_sum = 0.0
    cot_sim_n = 0
    open_sim_sum = 0.0
    open_sim_n = 0

    # 序列 CE 分桶累计（按 closed / open 分开统计）
    ce_sum = {"closed": 0.0, "open": 0.0}
    ce_tok = {"closed": 0, "open": 0}
    ce_n = {"closed": 0, "open": 0}

    # 软标签分布匹配累计（仅闭集且有 answer_distribution 的样本）
    dist_kl_sum = 0.0
    dist_cos_sum = 0.0
    dist_top1 = 0
    dist_n = 0          # 可计算分布的闭集样本数
    dist_skipped = 0    # 多 token/冲突/定位失败 跳过数

    # 场景分桶绝对准确率（学生 vs 人工 ground_truth，复用同一轮生成，无额外开销）
    # 原 driving_scenario 维度独立生成会翻倍耗时，且依赖 COCO (image_id,question) 查表
    # ——本数据 question_type 为模板型、且自带 ground_truth 字段，故并入此处。
    scen_buckets: Dict[str, list] = defaultdict(lambda: [0, 0])  # bucket -> [correct, total]
    scen_total = 0
    scen_correct = 0

    detail = []
    n_recs = len(records)
    for idx, rec in enumerate(records):
        if idx % 100 == 0:
            print(f"[distill_quality] {idx}/{n_recs}", flush=True)
        img = _resolve_image(rec, images_root)
        if not img:
            continue
        q = rec.get("question", "")
        out = inferencer.infer(img, q)

        # 闭集判定：与训练侧 DistillDataset 对齐——以 answer_distribution 存在为准，
        # 而非 question_category=="closed"（教师标注 jsonl 的 question_category 常为 None，
        # 否则全部误判为 open，closed_* 与 closed_distribution 悉数归零）
        soft_rec = rec.get("soft_label") or {}
        is_closed = bool(soft_rec.get("answer_distribution"))
        bucket = "closed" if is_closed else "open"
        t_short = teacher_short_answer(rec)
        s_short = extract_student_answer(out)

        # 场景分桶绝对准确率：学生答案 vs 记录自带的 ground_truth（人工 GT）
        gt = rec.get("ground_truth")
        if gt is not None and str(gt).strip() != "":
            scen_bucket = _classify_scenario(rec.get("question_type", ""), q)
            is_open_bucket = (scen_bucket == "open")
            scen_buckets[scen_bucket][1] += 1
            scen_total += 1
            if score_match(out, str(gt), is_open=is_open_bucket):
                scen_buckets[scen_bucket][0] += 1
                scen_correct += 1


        if is_closed:
            closed_total += 1
            if score_match(out, t_short, is_open=False):
                closed_match += 1
            soft = rec.get("soft_label") or {}
            pa = soft.get("primary_answer")
            if pa:
                primary_total += 1
                if normalize_for_match(s_short) == normalize_for_match(pa):
                    primary_match += 1
        else:
            # 开集：学生推理段（抽去标签/答案行噪声）vs 教师推理文本相似度
            t_rp = teacher_reasoning(rec)
            if t_rp:
                s_rp = extract_student_reasoning(out)
                open_sim_sum += token_overlap(s_rp, t_rp)
                open_sim_n += 1

        # CoT 相似度（学生推理段 vs 教师推理段）
        s_reason = extract_student_reasoning(out)
        t_reason = teacher_reasoning(rec)
        if t_reason and s_reason:
            cot_sim_sum += token_overlap(s_reason, t_reason)
            cot_sim_n += 1

        # 序列 Cross-Entropy + 软标签分布匹配（一次前向）
        target_text = build_target_text(rec, target_mode)
        short_answer = None
        dist = None
        if is_closed:
            meta = get_closed_meta(rec)
            short_answer = meta.get("short_answer")
            dist = meta.get("answer_distribution")
        ce_val, n_tok, dist_res = inferencer.sequence_ce_and_distribution(
            img, q, target_text, short_answer=short_answer, dist=dist,
        )
        if ce_val is not None and n_tok > 0:
            ce_sum[bucket] += ce_val * n_tok
            ce_tok[bucket] += n_tok
            ce_n[bucket] += 1
        if is_closed and short_answer and dist:
            # 仅闭集有软标签；统计可计算 vs 跳过
            if dist_res is not None:
                dist_kl_sum += dist_res["kl"]
                dist_cos_sum += dist_res["cos"]
                dist_top1 += int(dist_res["top1_match"])
                dist_n += 1
            else:
                dist_skipped += 1

        if len(detail) < cfg.get("detail_samples", 5):
            detail.append({
                "image_id": rec.get("image_id"),
                "question": q,
                "teacher_answer": t_short,
                "student_answer": s_short if s_short else (out[:60] + "..." if len(out) > 60 else out),
                "student_output": out[:200],
            })

    def _ce_mean(bucket_key):
        if ce_tok[bucket_key] == 0:
            return None
        return round(ce_sum[bucket_key] / ce_tok[bucket_key], 4)

    return {
        "dimension": "distillation_quality",
        "samples_evaluated": n,
        "closed_answer_match_rate": round(closed_match / max(closed_total, 1), 4),
        "closed_primary_match_rate": round(primary_match / max(primary_total, 1), 4),
        "open_text_similarity": round(open_sim_sum / max(open_sim_n, 1), 4),
        "cot_similarity": round(cot_sim_sum / max(cot_sim_n, 1), 4),
        "answer_sequence_ce": {
            "closed": {"mean": _ce_mean("closed"), "samples": ce_n["closed"]},
            "open": {"mean": _ce_mean("open"), "samples": ce_n["open"]},
            "temperature": 1.0,
            "note": "学生对教师目标文本的自回归 NLL（越低越好）；推理 T=1，无温度缩放；分闭集/开集",
        },
        "closed_distribution": {
            "samples": dist_n,
            "skipped": dist_skipped,
            "kl_mean": round(dist_kl_sum / max(dist_n, 1), 6),
            "cosine_mean": round(dist_cos_sum / max(dist_n, 1), 4),
            "top1_distribution_match_rate": round(dist_top1 / max(dist_n, 1), 4),
            "temperature": 1.0,
            "note": "学生在答案 token 位置对候选集的分布 vs 教师 answer_distribution；"
                    "KL(teacher||student) 越低越好；cosine 越高越好；"
                    "top1 为候选集内 argmax 一致率；skipped=多token/冲突/定位失败",
        },
        "scenario_buckets": {
            "overall_accuracy": round(scen_correct / max(scen_total, 1), 4),
            "overall_total": scen_total,
            "overall_correct": scen_correct,
            "buckets": {
                b: {"correct": c, "total": t, "accuracy": round(c / max(t, 1), 4)}
                for b, (c, t) in sorted(scen_buckets.items(), key=lambda x: -x[1][1])
            },
            "note": "学生 vs 记录自带 ground_truth(人工GT)的绝对准确率，按 question_type 模板分桶；"
                    "复用 distillation_quality 同一轮生成，无额外开销；"
                    "原 driving_scenario 维度(独立生成+COCO查表)已并入此处",
        },
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
