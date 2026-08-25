"""
Student Training Module
=======================

使用蒸馏标签 (outputs/merged/*.json) 对 Qwen2.5-VL-3B-Instruct 进行 SFT 蒸馏训练。

组件:
- DistillDataset : 解析标签、构建 chat、生成 prompt-mask 标签
- QwenVLDistillCollator : 处理 Qwen2.5-VL 可变尺寸图像的批量拼装
- run_training : 加载模型、构建数据、启动 HF Trainer
"""

from .distill_dataset import DistillDataset, build_target_text
from .collator import QwenVLDistillCollator
from .train import run_training, build_student_model, DistillTrainer

__all__ = [
    "DistillDataset",
    "QwenVLDistillCollator",
    "build_target_text",
    "build_student_model",
    "DistillTrainer",
    "run_training",
]
