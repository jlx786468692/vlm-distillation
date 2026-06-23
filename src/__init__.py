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

Author: VLM-Distillation Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "VLM-Distillation Team"

from .data import COCODataLoader, ImageProcessor
from .models import TeacherModel
from .distillation import Distiller
from .utils import ConfigManager, setup_logger
from .export import JSONExporter

__all__ = [
    "COCODataLoader",
    "ImageProcessor",
    "TeacherModel",
    "Distiller",
    "ConfigManager",
    "setup_logger",
    "JSONExporter",
]
