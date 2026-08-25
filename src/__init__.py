"""
VLM Data Distillation Package
=============================

A comprehensive data distillation pipeline for Vision-Language Models.

This package provides tools for distilling knowledge from large VLM teacher models
(like Qwen2.5-VL-7B-Instruct) into smaller student models, generating:
- Hard labels (final predictions)
- Soft labels (probability distributions)
- Chain-of-Thought reasoning

Supported tasks:
- Visual Question Answering (VQA)
- Image Captioning
- Object Detection

Data cleaning features:
- Reward model scoring (rule-based + model-based)
- Quality validation and filtering
- Data partitioning and storage
- Confidence control

Author: VLM-Distillation Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "VLM-Distillation Team"

from .data import COCODataLoader, ImageProcessor
from .models import TeacherModel
from .distillation import Distiller
from .utils import ConfigManager, setup_logger
from .export import JSONExporter, TrainingDataExporter
from .cleaning import (
    RewardModelScorer,
    RewardModelJudge,
    ClosedSampleValidator,
    DataPartitioner,
    ConfidenceController
)

# PipelineRunner 依赖 torch/transformers 等，按需导入（失败时不阻断顶层导入）
try:
    from .pipeline import PipelineRunner
except Exception:  # pragma: no cover - 仅在缺少重型依赖时发生
    PipelineRunner = None  # type: ignore

__all__ = [
    "COCODataLoader",
    "ImageProcessor",
    "TeacherModel",
    "Distiller",
    "ConfigManager",
    "setup_logger",
    "JSONExporter",
    "TrainingDataExporter",
    "RewardModelScorer",
    "RewardModelJudge",
    "ClosedSampleValidator",
    "DataPartitioner",
    "ConfidenceController",
    "PipelineRunner",
]