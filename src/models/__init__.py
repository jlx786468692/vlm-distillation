"""
Model Interfaces Module
=======================

Provides wrappers for teacher and student VLM models.
"""

from .teacher_model import TeacherModel
from .student_model import StudentModel
from .model_utils import load_model, get_device_info

__all__ = [
    "TeacherModel",
    "StudentModel",
    "load_model",
    "get_device_info",
]