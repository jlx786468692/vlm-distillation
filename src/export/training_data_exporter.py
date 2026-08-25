"""
训练数据导出器 (JSONL)
======================

读取蒸馏/清洗结果目录（每张图一个 JSON），按问题类型组装为学生模型训练用
JSONL，CoT 统一为两段式 cot_reasoning.{reasoning_paragraph, answer}。

- 开放问题 (open_descriptive)：输出 硬标签 + CoT
- 闭合问题 (yes_no / counting / color / location / choice)：输出 软标签 + 硬标签 + CoT

既可被流水线 (PipelineRunner) 调用，也可被 scripts/export_jsonl.py CLI 调用。
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# 问题类型 -> 大类映射
OPEN_TYPES = {"open_descriptive"}
CLOSED_TYPES = {"yes_no", "counting", "color", "location", "choice"}


def question_category(question_type: Optional[str]) -> str:
    """返回问题大类：open / closed。"""
    if question_type in OPEN_TYPES:
        return "open"
    if question_type in CLOSED_TYPES:
        return "closed"
    return "closed"


def _clean(text: str) -> str:
    """清洗文本：去除首尾空白与多余 Markdown 星号。"""
    if not text:
        return ""
    return text.strip().strip("*").strip()


def build_cot_reasoning(cot_reasoning: Dict[str, Any],
                        fallback_answer: Optional[str] = None) -> Dict[str, str]:
    """
    将 merged 中的 cot_reasoning 统一为两段式
    {reasoning_paragraph, answer}。

    优先级：
      1. 新版两段式：reasoning_paragraph + answer
      2. 旧版三段式：structured_reasoning.{observation, analysis, conclusion}
         reasoning_paragraph = observation + analysis
         answer              = conclusion
      3. 兜底：空对象

    fallback_answer: 当 answer 抽取为空时使用（通常是 hard_label.answer）。
    """
    result = {"reasoning_paragraph": "", "answer": ""}
    if not isinstance(cot_reasoning, dict) or not cot_reasoning:
        if fallback_answer:
            result["answer"] = _clean(fallback_answer)
        return result

    # 1) 新版两段式
    rp = cot_reasoning.get("reasoning_paragraph")
    ans = cot_reasoning.get("answer")
    if rp or ans:
        result["reasoning_paragraph"] = _clean(rp or "")
        result["answer"] = _clean(ans or "")
        if not result["answer"] and fallback_answer:
            result["answer"] = _clean(fallback_answer)
        return result

    # 2) 旧版三段式
    sr = cot_reasoning.get("structured_reasoning") or {}
    if isinstance(sr, dict) and sr:
        parts = []
        for key in ("observation", "analysis"):
            seg = sr.get(key)
            if seg and isinstance(seg, str) and seg.strip():
                parts.append(seg.strip())
        conclusion = sr.get("conclusion")
        result["reasoning_paragraph"] = "\n".join(parts)
        result["answer"] = _clean(conclusion or "")
        if not result["answer"] and fallback_answer:
            result["answer"] = _clean(fallback_answer)
        return result

    # 3) 兜底
    if fallback_answer:
        result["answer"] = _clean(fallback_answer)
    return result


def extract_open_short_answer(text: str) -> str:
    """
    尝试从开放问题的自然语言回答中抽取一个简短的"最终答案"。

    匹配 "Final Answer: X" / "**Final Answer**: X" 等模式。
    抽取失败时返回原文（开放问题的 answer 本身即监督目标）。
    """
    if not text:
        return ""
    m = re.search(
        r"\*?\*?Final Answer\*?\*?\s*[:：]\s*(.+?)\s*(?:\n|$)",
        text,
        re.IGNORECASE,
    )
    if m:
        cand = _clean(m.group(1))
        if cand:
            return cand
    return text.strip()


def build_open_record(image_id, image_path, question, question_type, vqa):
    """组装开放问题记录：硬标签 + CoT（两段式）。

    开放问题的答案不在顶层 vqa.answer，而是分布在：
      - hard_label.answer        （确定性硬标签）
      - cot_reasoning.answer      （CoT 推理结论）
      - ground_truth             （COCO GT，兜底）
    优先用 cot_reasoning（已为两段式），hard_label 作为硬标签与置信度来源。
    """
    cot = build_cot_reasoning(vqa.get("cot_reasoning") or {})

    hl = vqa.get("hard_label")
    hard_answer = None
    confidence = None
    if isinstance(hl, dict) and hl.get("answer"):
        hard_answer = hl["answer"]
        confidence = hl.get("confidence")

    # answer 兜底优先级：hard_label.answer > cot_reasoning.answer > ground_truth > vqa.answer
    if hard_answer is None:
        hard_answer = (cot.get("answer")
                       or vqa.get("ground_truth")
                       or vqa.get("answer")
                       or "")
    hard_answer = (hard_answer or "").strip()
    if not hard_answer:
        # 没有任何可用答案，无法作为监督样本，跳过
        return None

    # 若 cot 两段式均为空（既无 reasoning_paragraph 也无 answer），
    # 用自然语言 answer 兜底构造两段式（开放问题 answer 本身即监督目标）
    if not cot["reasoning_paragraph"] and not cot["answer"]:
        answer_text = (vqa.get("answer") or hard_answer).strip()
        short = extract_open_short_answer(answer_text)
        cot = {"reasoning_paragraph": answer_text, "answer": short or hard_answer}

    # 确保 cot.answer 非空（兜底用 hard_answer）
    if not cot["answer"]:
        cot["answer"] = hard_answer

    record = {
        "image_id": image_id,
        "image_path": image_path,
        "question": question,
        "question_type": question_type,
        "question_category": "open",
        "label_type": "hard+cot",
        "hard_label": {"answer": hard_answer},
        "cot_reasoning": cot,
    }
    if confidence is not None:
        record["hard_label"]["confidence"] = confidence
    return record


def build_closed_record(image_id, image_path, question, question_type, vqa, min_conf):
    """组装闭合问题记录：软标签 + 硬标签 + CoT（两段式）。"""
    hard = vqa.get("hard_label") or {}
    soft = vqa.get("soft_label") or {}

    hard_answer = hard.get("answer")
    confidence = hard.get("confidence")

    if not hard_answer:
        return None
    if confidence is not None and confidence < min_conf:
        return None

    dist = soft.get("answer_distribution") if isinstance(soft, dict) else None
    primary = soft.get("primary_answer") if isinstance(soft, dict) else None
    if not dist:
        return None

    pool = vqa.get("candidate_pool") or soft.get("allowed_answers")

    cot = build_cot_reasoning(vqa.get("cot_reasoning") or {},
                              fallback_answer=hard_answer)

    soft_label = {"answer_distribution": dist}
    if primary:
        soft_label["primary_answer"] = primary
    if pool:
        soft_label["candidate_pool"] = pool

    record = {
        "image_id": image_id,
        "image_path": image_path,
        "question": question,
        "question_type": question_type,
        "question_category": "closed",
        "label_type": "soft+hard+cot",
        "hard_label": {"answer": hard_answer},
        "soft_label": soft_label,
        "cot_reasoning": cot,
    }
    if confidence is not None:
        record["hard_label"]["confidence"] = confidence
    return record


def process_file(path: Path, min_conf: float) -> Tuple[Optional[Dict], Optional[str]]:
    """处理单个 JSON 文件，返回 (record, category)。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None, None

    image_id = data.get("image_id")
    image_path = data.get("image_path", "")
    tasks = data.get("tasks") or {}
    vqa = tasks.get("vqa")
    if not isinstance(vqa, dict):
        return None, None

    question = vqa.get("question", "")
    question_type = vqa.get("question_type")

    cat = question_category(question_type)
    if cat == "open":
        return build_open_record(image_id, image_path, question, question_type, vqa), "open"
    return build_closed_record(image_id, image_path, question, question_type, vqa, min_conf), "closed"


class TrainingDataExporter:
    """
    训练数据导出器：把蒸馏/清洗结果目录转为统一两段式 JSONL。

    既支持流水线内调用（传入 config/logger），也支持 CLI 直接使用。
    """

    def __init__(self, config=None, logger=None):
        self.config = config
        self.logger = logger

    def _log(self, level: str, msg: str):
        if self.logger is not None:
            getattr(self.logger, level, self.logger.info)(msg)
        else:
            print(msg)

    def run(
        self,
        input_dir: str,
        output_path: str,
        split: bool = False,
        min_conf: float = 0.0,
    ) -> Dict[str, Any]:
        """
        生成 JSONL 训练数据。

        Args:
            input_dir: 输入目录（每张图一个 JSON）
            output_path: 输出 JSONL 路径（split=True 时作为前缀，生成 _open/_closed）
            split: 是否按 open/closed 拆分
            min_conf: 过滤硬标签置信度低于该值的闭合样本

        Returns:
            统计 dict（含 type_counts、各类型计数）
        """
        in_dir = Path(input_dir)
        if not in_dir.is_dir():
            raise FileNotFoundError(f"输入目录不存在: {in_dir}")

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if split:
            open_path = out_path.with_name(out_path.stem + "_open" + out_path.suffix)
            closed_path = out_path.with_name(out_path.stem + "_closed" + out_path.suffix)
            open_fp = open(open_path, "w", encoding="utf-8")
            closed_fp = open(closed_path, "w", encoding="utf-8")
            combined_fp = None
        else:
            open_path = closed_path = None
            open_fp = closed_fp = None
            combined_fp = open(out_path, "w", encoding="utf-8")

        stats = {
            "open": 0, "closed": 0,
            "closed_no_soft": 0, "open_no_answer": 0,
            "skipped": 0, "total_files": 0,
        }
        type_counts = {}

        for fp_in in sorted(in_dir.glob("*.json")):
            if fp_in.name == "merged_summary.json":
                continue
            stats["total_files"] += 1
            record, cat = process_file(fp_in, min_conf)
            if record is None or cat is None:
                stats["skipped"] += 1
                try:
                    with open(fp_in, "r", encoding="utf-8") as f:
                        d = json.load(f)
                    vqa = (d.get("tasks") or {}).get("vqa") or {}
                    qt = vqa.get("question_type")
                    if question_category(qt) == "open":
                        stats["open_no_answer"] += 1
                    else:
                        stats["closed_no_soft"] += 1
                except Exception:
                    pass
                continue

            line = json.dumps(record, ensure_ascii=False)
            if split:
                (open_fp if cat == "open" else closed_fp).write(line + "\n")
            else:
                combined_fp.write(line + "\n")

            stats[cat] += 1
            qt = record.get("question_type", "unknown")
            type_counts[qt] = type_counts.get(qt, 0) + 1

        for fp in (open_fp, closed_fp, combined_fp):
            if fp:
                fp.close()

        stats["type_counts"] = type_counts
        stats["output_path"] = str(out_path)
        if split:
            stats["open_path"] = str(open_path)
            stats["closed_path"] = str(closed_path)

        self._log("info", f"[TrainingDataExporter] 总文件 {stats['total_files']}, "
                           f"开放 {stats['open']}, 闭合 {stats['closed']}, 跳过 {stats['skipped']}")
        return stats
