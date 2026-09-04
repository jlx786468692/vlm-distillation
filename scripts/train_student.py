"""
学生模型训练 CLI
================

用法:
    # 用 configs/train.yaml 默认配置 (LoRA)
    python scripts/train_student.py

    # 指定配置文件
    python scripts/train_student.py --config configs/train.yaml

    # 全量微调 + 调参
    python scripts/train_student.py --use_lora false --learning_rate 1e-5 --epochs 5

    # 小数据调试
    python scripts/train_student.py --max_samples 64 --epochs 1 --logging_steps 2
"""

import argparse
import sys
from pathlib import Path

# 让 src 成为可导入包
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from training.train import run_training  # noqa: E402
from utils.config import ConfigManager  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="VLM 蒸馏学生模型 SFT 训练")
    p.add_argument("--config", default=str(ROOT / "configs" / "train.yaml"),
                   help="训练配置文件路径")
    # ---- 常用覆盖项 ----
    p.add_argument("--train_data_path", default=None, help="训练 JSONL 路径")
    p.add_argument("--images_root", default=None, help="真实图像根目录")
    p.add_argument("--output_dir", default=None, help="训练输出目录")
    p.add_argument("--max_samples", type=int, default=None, help="仅用前 N 样本")
    p.add_argument("--target_mode", choices=["cot", "answer"], default=None)
    p.add_argument("--use_lora", type=str, default=None, help="true/false")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--grad_accum", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--max_length", type=int, default=None)
    p.add_argument("--save_steps", type=int, default=None)
    p.add_argument("--logging_steps", type=int, default=None)
    p.add_argument("--kl_weight", type=float, default=None, help="软标签 KL 权重 (0=关闭)")
    p.add_argument("--kl_temperature", type=float, default=None)
    p.add_argument("--resume", type=str, default=None,
                   help="断点续训：'true' 自动从最新 checkpoint 续，或直接给 checkpoint 路径")
    return p.parse_args()


def _to_bool(x):
    if x is None:
        return None
    return str(x).lower() in ("1", "true", "yes", "y")


def main():
    args = parse_args()

    # 通过 ConfigManager 加载 (支持 dot 记法 get/set)
    cm = ConfigManager(args.config)

    # CLI 覆盖
    def set_if(key, val):
        if val is not None:
            cm.set(key, val)

    set_if("train.train_data_path", args.train_data_path)
    set_if("train.images_root", args.images_root)
    set_if("train.output_dir", args.output_dir)
    set_if("train.max_samples", args.max_samples)
    set_if("train.target_mode", args.target_mode)
    set_if("train.use_lora", _to_bool(args.use_lora))
    set_if("train.num_train_epochs", args.epochs)
    set_if("train.per_device_train_batch_size", args.batch_size)
    set_if("train.gradient_accumulation_steps", args.grad_accum)
    set_if("train.learning_rate", args.learning_rate)
    set_if("train.max_length", args.max_length)
    set_if("train.save_steps", args.save_steps)
    set_if("train.logging_steps", args.logging_steps)
    set_if("train.kl_weight", args.kl_weight)
    set_if("train.kl_temperature", args.kl_temperature)
    # 断点续训：'true' -> True (自动找最新 checkpoint)；否则当作路径
    if args.resume is not None:
        resume_val = True if args.resume.lower() in ("1", "true", "yes", "y") else args.resume
        cm.set("train.resume_from_checkpoint", resume_val)

    print("=" * 60)
    print(f"[train_student] config: {args.config}")
    print(f"  student.model_name : {cm.get('student.model_name')}")
    print(f"  train_data_path     : {cm.get('train.train_data_path')}")
    print(f"  images_root         : {cm.get('train.images_root')}")
    print(f"  target_mode         : {cm.get('train.target_mode')}")
    print(f"  use_lora            : {cm.get('train.use_lora')}")
    print(f"  epochs              : {cm.get('train.num_train_epochs')}")
    print(f"  lr                  : {cm.get('train.learning_rate')}")
    print(f"  batch*accum         : {cm.get('train.per_device_train_batch_size')}"
          f" x {cm.get('train.gradient_accumulation_steps')}")
    print("=" * 60)

    output_dir = run_training(cm.config)
    print(f"\n✅ 训练完成。输出目录: {output_dir}")


if __name__ == "__main__":
    main()
