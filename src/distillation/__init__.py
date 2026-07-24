"""
Distillation Core Module
========================

Core distillation logic for generating hard labels, soft labels, and CoT.
"""

from .hard_label_gen import HardLabelGenerator
from .vqa_soft_label_gen import VQASoftLabelGenerator, SoftLabelGenerator  # 兼容别名
from .cot_generator import CoTGenerator
from .distiller import Distiller

__all__ = [
    "HardLabelGenerator",
    "VQASoftLabelGenerator",
    "SoftLabelGenerator",  # 兼容别名
    "CoTGenerator",
    "Distiller",
]