"""
Distillation Core Module
========================

Core distillation logic for generating hard labels, soft labels, and CoT.

Modules:
- HardLabelGenerator: Hard label generation
- VQASoftLabelGenerator: Soft label generation for VQA
- CoTGenerator: Chain-of-thought generation
- Distiller: Main distillation pipeline
- ReadingNumberCandidateGenerator: Candidate generator for reading number tasks (OCR)
- ReadingNumberExtractor: Extractor for precise number reading tasks
"""

from .hard_label_gen import HardLabelGenerator
from .vqa_soft_label_gen import VQASoftLabelGenerator, SoftLabelGenerator  # 兼容别名
from .cot_generator import CoTGenerator
from .distiller import Distiller

from .reading_number_candidate_generator import (
    ReadingNumberCandidateGenerator,
    ReadingNumberExtractor,
    CandidatePoolConfig
)

__all__ = [
    "HardLabelGenerator",
    "VQASoftLabelGenerator",
    "SoftLabelGenerator",  # 兼容别名
    "CoTGenerator",
    "Distiller",
    "ReadingNumberCandidateGenerator",
    "ReadingNumberExtractor",
    "CandidatePoolConfig",
]