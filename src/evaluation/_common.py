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
    否则取首个 token。
    """
    if not text:
        return ""
    if "[Answer]" in text:
        tail = text.rsplit("[Answer]", 1)[-1].strip()
        if tail:
            # 取第一行/第一个词作为短答案
            first_line = tail.splitlines()[0].strip()
            return first_line.split()[0] if first_line else first_line
    # 无格式标记：取第一个 token
    tokens = text.strip().split()
    return tokens[0] if tokens else ""


def extract_student_reasoning(text: str) -> str:
    """从学生输出抽取 [Reasoning] 段（若有）。"""
    if not text:
        return ""
    if "[Reasoning]" in text and "[Answer]" in text:
        seg = text.split("[Answer]")[0].split("[Reasoning]", 1)[-1].strip()
        return seg
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
