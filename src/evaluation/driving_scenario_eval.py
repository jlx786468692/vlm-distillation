"""
维度2：驾驶场景适配评估
========================

按 question_type 把样本分桶（安全关键/否定、空间、计数、颜色、开放），
对每桶用 COCO GT 计算学生答案准确率，衡量学生在安全关键与空间/否定
等场景上的适配情况。

说明：COCO 上无显式“安全关键”标注，本实现按 question_type 近似分桶，
口径在报告中标注。
"""

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

from ._common import extract_student_answer, score_match


# question_type -> 场景桶
SCENARIO_BUCKET = {
    "yes_no": "negation_safety",   # 否定/安全关键（yes-no 判定）
    "location": "spatial",          # 空间方位
    "counting": "counting",         # 计数
    "color": "color",               # 颜色
    "choice": "multiple_choice",    # 选择
    "open_descriptive": "open",     # 开放描述
}


class GTResolver:
    """(image_id, question) -> COCO GT 答案。"""

    def __init__(self, annotations_root: str, split: str = "val2014"):
        self.q2gt: Dict[tuple, str] = {}
        loaded = False
        # 尝试常见文件名
        q_files = [
            os.path.join(annotations_root, f"v2_mscoco_{split}_questions.json"),
            os.path.join(annotations_root, f"mscoco_{split}_questions.json"),
        ]
        a_files = [
            os.path.join(annotations_root, f"v2_mscoco_{split}_annotations.json"),
            os.path.join(annotations_root, f"mscoco_{split}_annotations.json"),
        ]
        qpath = next((p for p in q_files if os.path.isfile(p)), None)
        apath = next((p for p in a_files if os.path.isfile(p)), None)
        if qpath and apath:
            with open(qpath, "r", encoding="utf-8") as f:
                questions = json.load(f).get("questions", [])
            with open(apath, "r", encoding="utf-8") as f:
                ann_by_qid = {a["question_id"]: a for a in json.load(f).get("annotations", [])}
            for q in questions:
                ann = ann_by_qid.get(q["question_id"])
                if ann:
                    gt = ann.get("multiple_choice_answer")
                    if gt:
                        self.q2gt[(q["image_id"], q["question"].strip())] = str(gt).strip()
            loaded = True
        self.loaded = loaded

    def get(self, image_id, question: str) -> Optional[str]:
        return self.q2gt.get((image_id, question.strip()))


def evaluate(inferencer, records, images_root: str, cfg: Dict[str, Any],
             full_cfg: Any = None) -> Dict[str, Any]:
    """运行驾驶场景适配评估（full_cfg 传入主 ConfigManager 以取 annotations_root）。"""
    # 取 COCO 标注目录
    ann_root = None
    if full_cfg is not None:
        ann_root = full_cfg.get("data.annotations_root", None)
    if ann_root is None:
        ann_root = cfg.get("annotations_root", "./data/filter_coco/annotations")
    split = cfg.get("eval_split", "val2014")

    resolver = GTResolver(ann_root, split)
    if not resolver.loaded:
        return {
            "dimension": "driving_scenario",
            "note": f"未找到 COCO 标注 ({ann_root})，跳过 GT 准确率评估",
            "buckets": {},
        }

    # 桶 -> (correct, total)
    buckets: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    detail = []

    for rec in records:
        img = _resolve_image(rec, images_root)
        if not img:
            continue
        q = rec.get("question", "")
        gt = resolver.get(rec.get("image_id"), q)
        if gt is None:
            continue

        qtype = rec.get("question_type", "open_descriptive")
        bucket = SCENARIO_BUCKET.get(qtype, "other")
        out = inferencer.infer(img, q)
        s_ans = extract_student_answer(out)

        # 开放描述桶：统一两段式目标后学生应吐 [Answer] 短答案；
        # score_match 内部优先精确匹配 [Answer]，抽不到再退回 token 子集。
        is_open = (bucket == "open")
        buckets[bucket][1] += 1
        if score_match(out, gt, is_open=is_open):
            buckets[bucket][0] += 1

        if len(detail) < cfg.get("detail_samples", 5):
            detail.append({
                "image_id": rec.get("image_id"),
                "bucket": bucket,
                "question": q,
                "gt": gt,
                "student_answer": s_ans if s_ans else (out[:60] + "..." if len(out) > 60 else out),
            })

    bucket_stats = {
        b: {"correct": c, "total": t, "accuracy": round(c / max(t, 1), 4)}
        for b, (c, t) in buckets.items()
    }
    total_c = sum(v[0] for v in buckets.values())
    total_t = sum(v[1] for v in buckets.values())

    return {
        "dimension": "driving_scenario",
        "note": "按 question_type 近似分桶；COCO 无显式安全关键标注",
        "annotations_root": ann_root,
        "overall_accuracy": round(total_c / max(total_t, 1), 4),
        "overall_total": total_t,
        "buckets": bucket_stats,
        "detail": detail,
    }


def _resolve_image(rec, images_root: str):
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
