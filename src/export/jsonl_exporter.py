"""
JSONL Exporter（训练数据导出器）
================================

清洗完成后，将 clean_valid 合格样本导出为 JSON Lines 训练数据。

标签策略（与三层损失训练设计对齐：CE Loss=hard_label / KL Loss=soft_label / SFT Loss=CoT）：
- 闭合问题（closed_*）：同时生成 soft_label、hard_label、cot
- 开放问题（open）：只生成 hard_label、cot（开放问题无闭合候选集，不生成 soft_label）

输入：clean_valid 目录下的单图片 JSON（由 DataPartitioner 写入）
输出：单个 train.jsonl 文件，每行一条训练记录
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging

from ..utils.config import ConfigManager
from ..utils.logger import get_logger


# ───────────────────────────────────────────────────────────────
# 闭合/开放判定映射
# ───────────────────────────────────────────────────────────────
# 与 src/classification/question_classifier.py 中 QuestionType.to_major_category() 保持一致：
#   COUNT="counting" / COLOR="color" / BINARY="yes_no" / CHOICE="choice"
#     → CLOSED_ENUMERATE / CLOSED_YESNO / CLOSED_CHOICE → 闭合
#   OPEN="open_descriptive" / LOCATION="location" / OPEN_TYPE="open"
#     → OPEN_TYPE → 开放
# 这里用字符串映射而非导入 QuestionType，避免在导出路径上触发 torch / 分类器依赖。
_CLOSED_QUESTION_TYPES = {
    "counting",            # COUNT → CLOSED_ENUMERATE
    "color",               # COLOR → CLOSED_ENUMERATE
    "yes_no",              # BINARY → CLOSED_YESNO
    "choice",              # CHOICE → CLOSED_CHOICE
    "closed_choice",       # CLOSED_CHOICE
    "closed_yesno",        # CLOSED_YESNO
    "closed_enumerate",    # CLOSED_ENUMERATE
}

_OPEN_QUESTION_TYPES = {
    "open_descriptive",    # OPEN → OPEN_TYPE
    "open",                # OPEN_TYPE
    "location",            # LOCATION → OPEN_TYPE
}


class JSONLExporter:
    """
    训练数据 JSONL 导出器

    读取 clean_valid 目录中的合格样本，按闭合/开放策略生成 train.jsonl：
    - 闭合问题：保留 soft_label + hard_label + cot
    - 开放问题：仅保留 hard_label + cot（丢弃 soft_label）
    """

    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config or ConfigManager()
        self.logger = logger if logger else get_logger()

        # 输入：clean_valid 目录（合格样本）
        self.clean_valid_dir = Path(
            self.config.get("cleaning.output.clean_valid_dir", "./outputs/cleaned/clean_valid")
        )
        # 输出：train.jsonl 路径
        self.output_path = Path(
            self.config.get("cleaning.output.train_jsonl", "./outputs/cleaned/train.jsonl")
        )

    @staticmethod
    def is_closed_question(question_type: Optional[str]) -> bool:
        """
        判定问题是否为闭合问题。

        判定规则（对齐 QuestionType.to_major_category()）：
        - question_type 命中闭合集合 → True
        - question_type 命中开放集合 → False
        - 未知类型 → 兜底归为开放（与分类器 to_major_category 的默认兜底一致）
        """
        if not question_type:
            return False
        qt = question_type.strip().lower()
        if qt in _CLOSED_QUESTION_TYPES:
            return True
        if qt in _OPEN_QUESTION_TYPES:
            return False
        # 未知类型兜底为开放：绝不强行把无候选集的样本当作闭合
        return False

    def generate(
        self,
        clean_valid_dir: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        扫描 clean_valid 目录并生成 train.jsonl。

        Args:
            clean_valid_dir: 可选，覆盖配置中的 clean_valid 目录
            output_path: 可选，覆盖配置中的 train.jsonl 输出路径

        Returns:
            生成统计信息
        """
        input_dir = Path(clean_valid_dir) if clean_valid_dir else self.clean_valid_dir
        output = Path(output_path) if output_path else self.output_path

        if not input_dir.exists():
            self.logger.error(f"clean_valid 目录不存在: {input_dir}")
            return {"success": False, "error": f"目录不存在: {input_dir}"}

        sample_files = sorted(input_dir.glob("*.json"))
        # 排除分区报告等非样本文件
        sample_files = [f for f in sample_files if f.stem != "partition_report"]

        if not sample_files:
            self.logger.warning(f"clean_valid 目录中无样本: {input_dir}")
            return {
                "success": True,
                "total": 0,
                "closed": 0,
                "open": 0,
                "output_path": str(output),
            }

        self.logger.info(f"开始生成训练 JSONL：{len(sample_files)} 个合格样本")
        self.logger.info(f"  输入目录: {input_dir}")
        self.logger.info(f"  输出文件: {output}")

        output.parent.mkdir(parents=True, exist_ok=True)

        stats = {
            "success": True,
            "total": 0,
            "closed": 0,
            "open": 0,
            "skipped": 0,
            "output_path": str(output),
        }

        with open(output, "w", encoding="utf-8") as fout:
            for sample_file in sample_files:
                try:
                    with open(sample_file, "r", encoding="utf-8") as fin:
                        sample = json.load(fin)
                except Exception as e:
                    self.logger.warning(f"读取样本失败，跳过: {sample_file.name} ({e})")
                    stats["skipped"] += 1
                    continue

                record = self._build_record(sample)
                if record is None:
                    stats["skipped"] += 1
                    continue

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["total"] += 1
                if record["category"] == "closed":
                    stats["closed"] += 1
                else:
                    stats["open"] += 1

        self.logger.info(
            f"JSONL 生成完成: 共 {stats['total']} 条 "
            f"(闭合 {stats['closed']}, 开放 {stats['open']}, 跳过 {stats['skipped']})"
        )
        self.logger.info(f"  输出: {output}")
        return stats

    def _build_record(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        从单个 clean_valid 样本构建训练记录。

        闭合问题：保留 soft_label + hard_label + cot
        开放问题：仅保留 hard_label + cot（丢弃 soft_label）
        """
        vqa = sample.get("tasks", {}).get("vqa")
        if not vqa or not isinstance(vqa, dict):
            self.logger.warning(
                f"样本缺少 tasks.vqa，跳过: {sample.get('image_id', '?')}"
            )
            return None

        question_type = vqa.get("question_type")
        is_closed = self.is_closed_question(question_type)
        category = "closed" if is_closed else "open"

        # ── 必需字段校验：hard_label 与 cot 两类问题都需要 ──
        hard_label = vqa.get("hard_label")
        cot_reasoning = vqa.get("cot_reasoning")
        if not hard_label or not cot_reasoning:
            self.logger.warning(
                f"样本缺少 hard_label/cot_reasoning，跳过: {sample.get('image_id', '?')}"
            )
            return None

        record: Dict[str, Any] = {
            "image_id": sample.get("image_id"),
            "image_path": sample.get("image_path"),
            "question": vqa.get("question"),
            "question_type": question_type,
            "category": category,
            "ground_truth": vqa.get("ground_truth"),
            # 硬标签：闭合/开放均生成
            "hard_label": hard_label,
            # CoT：闭合/开放均生成
            "cot": cot_reasoning,
        }

        if is_closed:
            # 软标签：仅闭合问题生成（KL Loss 监督来源）
            # 若闭合样本缺失 soft_label，记录警告但仍输出 hard+cot
            soft_label = vqa.get("soft_label")
            if soft_label:
                record["soft_label"] = soft_label
            else:
                self.logger.warning(
                    f"闭合样本缺少 soft_label（仅输出 hard+cot）: "
                    f"{sample.get('image_id', '?')}"
                )

        # open 问题：不生成 soft_label（此处不写入该键）

        return record


if __name__ == "__main__":
    exporter = JSONLExporter()
    result = exporter.generate()
    print(json.dumps(result, indent=2, ensure_ascii=False))
