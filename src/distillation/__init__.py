"""
Distillation Core Module
========================

Core distillation logic for generating hard labels, soft labels, and CoT.
"""

from .hard_label_gen import HardLabelGenerator
from .soft_label_gen import SoftLabelGenerator
from .cot_generator import CoTGenerator
from .distiller import Distiller

__all__ = [
    "HardLabelGenerator",
    "SoftLabelGenerator",
    "CoTGenerator",
    "Distiller",
]
