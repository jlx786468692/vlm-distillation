"""
Data Loading and Processing Module
==================================

Handles COCO dataset loading, image preprocessing, and data management.
"""

from .coco_loader import COCODataLoader
from .image_processor import ImageProcessor
from .data_manager import DataManager

__all__ = [
    "COCODataLoader",
    "ImageProcessor",
    "DataManager",
]