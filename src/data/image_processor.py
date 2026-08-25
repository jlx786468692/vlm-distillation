"""
Image Processor
===============

Handles image preprocessing for VLM input.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np

from PIL import Image
import torch
from transformers import AutoProcessor

from ..utils.config import ConfigManager
from ..utils.logger import get_logger


class ImageProcessor:
    """
    Processes images for VLM model input.

    Handles:
    - Image resizing and normalization
    - Format conversion
    - Pixel normalization for Qwen-VL
    """

    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        processor_name: Optional[str] = None
    ):
        """
        Initialize ImageProcessor.

        Args:
            config: Configuration manager instance
            processor_name: HuggingFace processor name (default: from config)
        """
        self.config = config or ConfigManager()
        self.logger = get_logger()

        # Processor settings
        self.processor_name = processor_name or self.config.get("teacher.model_name")
        self.max_pixels = self.config.get("model.vision_config.max_pixels", 1003520)
        self.min_pixels = self.config.get("model.vision_config.min_pixels", 3136)
        self.image_size = self.config.get("data.image_size", 224)

        # Load HuggingFace processor
        self.processor = None
        self._load_processor()

    def _load_processor(self) -> None:
        """Load HuggingFace AutoProcessor for the model."""
        try:
            self.logger.info(f"Loading processor: {self.processor_name}")
            self.processor = AutoProcessor.from_pretrained(
                self.processor_name,
                trust_remote_code=True
            )
            self.logger.info("Processor loaded successfully")
        except Exception as e:
            self.logger.warning(f"Could not load AutoProcessor: {e}")
            self.logger.info("Using default PIL-based processing")
            self.processor = None

    def process_image(
        self,
        image: Union[Image.Image, str, Path],
        task: str = "vqa"
    ) -> Dict[str, Any]:
        """
        Process single image for model input.

        Args:
            image: PIL Image or image path
            task: Task type (vqa, captioning, detection)

        Returns:
            Dictionary with processed image data
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = self._load_image_from_path(image)

        if image is None:
            self.logger.error("Invalid image input")
            return {}

        # Get original dimensions
        original_width, original_height = image.size

        # Resize if needed (respecting max_pixels constraint)
        resized_image = self._resize_image(image)

        # Process based on available processor
        if self.processor:
            processed_data = self._process_with_processor(resized_image, task)
        else:
            processed_data = self._process_with_pil(resized_image, task)

        # Add metadata
        processed_data['original_size'] = (original_width, original_height)
        processed_data['processed_size'] = resized_image.size
        processed_data['task'] = task

        return processed_data

    def _load_image_from_path(self, image_path: Union[str, Path]) -> Optional[Image.Image]:
        """
        Load image from file path.

        Args:
            image_path: Path to image file

        Returns:
            PIL Image or None if failed
        """
        try:
            path = Path(image_path)
            if not path.exists():
                self.logger.error(f"Image path does not exist: {path}")
                return None

            image = Image.open(path).convert('RGB')
            return image
        except Exception as e:
            self.logger.error(f"Error loading image from {image_path}: {e}")
            return None

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """
        Resize image respecting pixel constraints.

        Args:
            image: PIL Image

        Returns:
            Resized PIL Image
        """
        width, height = image.size
        current_pixels = width * height

        # Check if resizing needed
        if current_pixels <= self.max_pixels:
            return image

        # Calculate new size maintaining aspect ratio
        ratio = (self.max_pixels / current_pixels) ** 0.5
        new_width = int(width * ratio)
        new_height = int(height * ratio)

        # Ensure minimum size
        if new_width * new_height < self.min_pixels:
            ratio = (self.min_pixels / current_pixels) ** 0.5
            new_width = int(width * ratio)
            new_height = int(height * ratio)

        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        self.logger.debug(f"Resized image from {(width, height)} to {(new_width, new_height)}")

        return resized_image

    def _process_with_processor(
        self,
        image: Image.Image,
        task: str
    ) -> Dict[str, Any]:
        """
        Process image using HuggingFace processor.

        Args:
            image: PIL Image
            task: Task type

        Returns:
            Processed data dictionary
        """
        try:
            # Use processor's image processor
            processed = self.processor.image_processor(
                images=image,
                return_tensors="pt"
            )

            return {
                'pixel_values': processed.get('pixel_values', None),
                'image_grid_thw': processed.get('image_grid_thw', None),
                'processor_used': True,
            }
        except Exception as e:
            self.logger.warning(f"Processor failed: {e}, falling back to PIL")
            return self._process_with_pil(image, task)

    def _process_with_pil(
        self,
        image: Image.Image,
        task: str
    ) -> Dict[str, Any]:
        """
        Process image using PIL (fallback method).

        Args:
            image: PIL Image
            task: Task type

        Returns:
            Processed data dictionary
        """
        # Resize to standard size
        processed_image = image.resize((self.image_size, self.image_size), Image.LANCZOS)

        # Convert to numpy array
        image_array = np.array(processed_image)

        # Normalize to [0, 1] range
        image_array = image_array.astype(np.float32) / 255.0

        # Convert to tensor (C, H, W)
        pixel_values = torch.from_numpy(image_array).permute(2, 0, 1)

        return {
            'pixel_values': pixel_values,
            'processor_used': False,
        }

    def process_batch(
        self,
        images: list,
        task: str = "vqa"
    ) -> Dict[str, Any]:
        """
        Process batch of images.

        Args:
            images: List of PIL Images or paths
            task: Task type

        Returns:
            Dictionary with batched processed data
        """
        processed_list = []

        for img in images:
            processed = self.process_image(img, task)
            if processed:
                processed_list.append(processed)

        if not processed_list:
            return {}

        # Stack pixel values if available
        if 'pixel_values' in processed_list[0]:
            pixel_values = torch.stack([p['pixel_values'] for p in processed_list])
            return {
                'pixel_values': pixel_values,
                'batch_size': len(processed_list),
                'task': task,
            }

        return {'processed_list': processed_list, 'batch_size': len(processed_list)}

    def prepare_image_for_qwen_vl(
        self,
        image: Image.Image,
        prompt: str
    ) -> Dict[str, Any]:
        """
        Prepare image and text for Qwen-VL model specifically.

        Args:
            image: PIL Image
            prompt: Text prompt/question

        Returns:
            Dictionary with model-ready inputs
        """
        # Process image
        processed_image = self._resize_image(image)

        # Use processor if available
        if self.processor:
            try:
                # Qwen-VL specific processing
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": processed_image},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]

                # Apply chat template
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                # Process image and text
                inputs = self.processor(
                    text=[text],
                    images=[processed_image],
                    padding=True,
                    return_tensors="pt"
                )

                return inputs
            except Exception as e:
                self.logger.warning(f"Qwen-VL specific processing failed: {e}")

        # Fallback to standard processing
        return self.process_image(processed_image, task="vqa")

    def save_processed_image(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        save_path: Union[str, Path],
        format: str = "PNG"
    ) -> bool:
        """
        Save processed image to file.

        Args:
            image: Image to save
            save_path: Destination path
            format: Image format (PNG, JPEG)

        Returns:
            True if successful, False otherwise
        """
        try:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Convert tensor/numpy to PIL if needed
            if isinstance(image, torch.Tensor):
                # Assume (C, H, W) format, convert to (H, W, C)
                image = image.permute(1, 2, 0).numpy()
                image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            # Save
            image.save(path, format=format)
            self.logger.info(f"Saved processed image to {path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving image: {e}")
            return False

    def get_image_info(self, image: Image.Image) -> Dict[str, Any]:
        """
        Get image metadata and statistics.

        Args:
            image: PIL Image

        Returns:
            Dictionary with image information
        """
        return {
            'size': image.size,
            'mode': image.mode,
            'format': image.format if hasattr(image, 'format') else None,
            'pixels': image.size[0] * image.size[1],
            'channels': len(image.getbands()),
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"ImageProcessor(processor={self.processor_name}, max_pixels={self.max_pixels})"