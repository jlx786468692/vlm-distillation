"""
评估共享工具
============

答案抽取与归一化、文本相似度。
"""

import re
import string
from typing import Optional

from ..utils.answer_normalizer import normalize_answer


def normalize_for_match(answer: str) -> str:
    """VQA 风格归一化：小写、去标点/冠词、数字↔英文词。"""
    if not answer:
        return ""
    a = answer.strip().lower()
    # 去冠词
    a = re.sub(r"\b(a|an|the)\b", " ", a)
    # 去标点
    a = a.translate(str.maketrans("", "", string.punctuation))
    a = " ".join(a.split())
    # 数字/英文词统一（默认 word 形式）
    a = normalize_answer(a, "word")
    return a


def extract_student_answer(text: str) -> str:
    """
    从学生输出抽取简短答案。

    训练目标为两段式时形如 "[Reasoning] ...\\n[Answer] X"；
    闭集学生被训练成会吐 [Answer] 标记，取标记后整行（含多词答案）。
    无 [Answer] 标记时返回空串——开集输出是自由文本，没有可靠短答案，
    首词法（"The image..." -> "The"）是噪声，不再回退。
    """
    if not text:
        return ""
    if "[Answer]" in text:
        tail = text.rsplit("[Answer]", 1)[-1].strip()
        if tail:
            # 取 [Answer] 后第一行作为短答案（保留多词，如 "around fire hydrant"）
            return tail.splitlines()[0].strip()
        return ""
    return ""


def extract_student_reasoning(text: str) -> str:
    """
    从学生输出抽取 [Reasoning] 段。

    - 同时有 [Reasoning]/[Answer]：取两标签之间内容。
    - 仅有 [Reasoning]（开集格式泄漏，无 [Answer]）：取标签之后内容。
    - 仅有 [Answer]：无推理段，返回空。
    - 无任何标签：返回原文（开集裸推理段落）。
    """
    if not text:
        return ""
    if "[Reasoning]" in text:
        seg = text.split("[Reasoning]", 1)[-1]
        if "[Answer]" in seg:
            seg = seg.split("[Answer]")[0]
        return seg.strip()
    if "[Answer]" in text:
        return ""
    return text.strip()


def teacher_short_answer(rec: dict) -> Optional[str]:
    """记录里的教师短答案（闭集 hard_label.answer；开集 cot_reasoning.answer）。"""
    hl = rec.get("hard_label")
    if isinstance(hl, dict) and hl.get("answer"):
        return str(hl["answer"]).strip()
    cot = rec.get("cot_reasoning") or {}
    if isinstance(cot, dict) and cot.get("answer"):
        return str(cot["answer"]).strip()
    soft = rec.get("soft_label")
    if isinstance(soft, dict) and soft.get("primary_answer"):
        return str(soft["primary_answer"]).strip()
    return None


def teacher_reasoning(rec: dict) -> str:
    """记录里的教师推理段落。"""
    cot = rec.get("cot_reasoning") or {}
    if isinstance(cot, dict):
        return (cot.get("reasoning_paragraph") or "").strip()
    return ""


def token_overlap(a: str, b: str) -> float:
    """词级 Jaccard 相似度（用于 CoT/开集答案文本对比）。"""
    sa = set(normalize_for_match(a).split())
    sb = set(normalize_for_match(b).split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def score_match(output: str, gt: str, is_open: bool) -> bool:
    """
    学生输出 vs GT 答案的匹配判定。

    - 闭集：抽取 [Answer] 标记后整行，归一化后精确匹配 GT（VQA 风格归一
      已统一大小写/冠词/标点/数字↔英文词）。
    - 开集：优先用 [Answer] 抽取的短答案做精确比对（统一两段式目标后
      学生会吐 [Answer]）；抽不到（旧模型/格式泄漏）时退回 token 子集
      匹配——GT 的所有词是否作为词出现在学生输出里。这取代了原先取
      首词再精确比对的错误做法（首词几乎必为 "The/a/an" → 误判）。
    """
    g = normalize_for_match(gt)
    g_tokens = g.split()
    if not g_tokens:
        return False
    short = extract_student_answer(output)
    s = normalize_for_match(short)
    if is_open:
        if s:
            return s == g
        out_tokens = set(normalize_for_match(output).split())
        return all(t in out_tokens for t in g_tokens)
    return bool(s) and s == g
