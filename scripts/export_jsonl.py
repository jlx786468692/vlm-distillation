#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导出学生模型训练数据 (JSONL) —— 薄 CLI
=====================================

实际逻辑在 src/export/training_data_exporter.py 的 TrainingDataExporter。

用法
----
    python scripts/export_jsonl.py --input outputs/merged \\
        --output outputs/training/train.jsonl --pretty

可选参数：
    --split        输出两个文件：train_open.jsonl / train_closed.jsonl
    --min-conf    过滤硬标签置信度低于该值的闭合样本（默认 0）
    --pretty       打印各类型统计
"""

import argparse
import sys
from pathlib import Path

# 让 src 成为可导入包
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.export.training_data_exporter import TrainingDataExporter  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="导出学生模型训练 JSONL")
    parser.add_argument("--input", "-i", default="outputs/merged",
                        help="蒸馏/清洗结果目录 (每张图一个 JSON)")
    parser.add_argument("--output", "-o", default="outputs/training/train.jsonl",
                        help="输出 JSONL 路径")
    parser.add_argument("--split", action="store_true",
                        help="按 open/closed 拆分为两个文件")
    parser.add_argument("--min-conf", type=float, default=0.0,
                        help="过滤硬标签置信度低于该值的闭合样本")
    parser.add_argument("--pretty", action="store_true",
                        help="打印统计信息")
    args = parser.parse_args()

    exporter = TrainingDataExporter()
    stats = exporter.run(
        input_dir=args.input,
        output_path=args.output,
        split=args.split,
        min_conf=args.min_conf,
    )

    if args.pretty:
        print("=" * 60)
        print(f"输入目录: {args.input}")
        print(f"总文件数: {stats['total_files']}")
        print(f"开放问题 (hard+cot):        {stats['open']}")
        print(f"闭合问题 (soft+hard+cot):   {stats['closed']}")
        print(f"跳过 (无效/缺字段):          {stats['skipped']}")
        print(f"  - 开放缺 answer:           {stats['open_no_answer']}")
        print(f"  - 闭合缺 soft_label:      {stats['closed_no_soft']}")
        print("-" * 60)
        print("细分类型分布:")
        for k, v in sorted(stats["type_counts"].items(), key=lambda x: -x[1]):
            print(f"  {k:20s}: {v}")
        print("=" * 60)
        if args.split:
            print(f"开放问题已写入: {stats['open_path']}")
            print(f"闭合问题已写入: {stats['closed_path']}")
        else:
            print(f"训练 JSONL 已写入: {stats['output_path']}")


if __name__ == "__main__":
    main()
