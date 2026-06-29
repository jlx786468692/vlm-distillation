"""
Model Utilities
===============

Shared utilities for model loading and management.
"""

import torch
from typing import Dict, Any, Optional, List
from pathlib import Path
import os

from ..utils.logger import get_logger


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available devices.

    Returns:
        Dictionary with device information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': None,
        'devices': [],
    }

    if torch.cuda.is_available():
        info['current_device'] = torch.cuda.current_device()

        for i in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(i)
            device_info = {
                'index': i,
                'name': device_props.name,
                'total_memory': device_props.total_memory,
                'memory_allocated': torch.cuda.memory_allocated(i),
                'memory_cached': torch.cuda.memory_reserved(i),
            }
            info['devices'].append(device_info)

    return info


def check_gpu_memory(device: Optional[int] = None) -> Dict[str, float]:
    """
    Check GPU memory usage.

    Args:
        device: GPU device index (default: current device)

    Returns:
        Dictionary with memory stats in GB
    """
    if not torch.cuda.is_available():
        return {'error': 'CUDA not available'}

    device = device or torch.cuda.current_device()

    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # GB
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)    # GB
    total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)  # GB

    return {
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'total_gb': total,
        'available_gb': total - allocated,
        'usage_percent': (allocated / total) * 100,
    }


def load_model(
    model_name: str,
    device: str = "cuda",
    precision: str = "bf16",
    load_tokenizer: bool = True,
    load_processor: bool = True,
    trust_remote_code: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Load model with components.

    Args:
        model_name: Model name or path
        device: Device to load on
        precision: Model precision (fp32/fp16/bf16)
        load_tokenizer: Whether to load tokenizer
        load_processor: Whether to load processor
        trust_remote_code: Trust remote code flag
        **kwargs: Additional arguments for model loading

    Returns:
        Dictionary with model and components
    """
    logger = get_logger()
    logger.info(f"Loading model: {model_name}")

    components = {}

    # Determine dtype
    dtype_map = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
    }
    torch_dtype = dtype_map.get(precision, torch.bfloat16)

    try:
        # Load processor if needed
        if load_processor:
            from transformers import AutoProcessor
            logger.info("Loading processor...")
            components['processor'] = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code
            )

        # Load tokenizer if needed
        if load_tokenizer:
            from transformers import AutoTokenizer
            logger.info("Loading tokenizer...")
            components['tokenizer'] = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code
            )

        # Load model
        from transformers import AutoModelForVision2Seq
        logger.info(f"Loading model on {device}...")
        components['model'] = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=trust_remote_code,
            **kwargs
        )

        logger.info("Model loaded successfully")
        components['loaded'] = True

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        components['error'] = str(e)
        components['loaded'] = False

    return components


def optimize_model_for_inference(model: Any) -> Any:
    """
    Optimize model for inference.

    Args:
        model: PyTorch model

    Returns:
        Optimized model
    """
    logger = get_logger()

    # Set to eval mode
    model.eval()

    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad = False

    # Enable optimizations
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()

    logger.info("Model optimized for inference")
    return model


def move_model_to_device(model: Any, device: str) -> Any:
    """
    Move model to specified device.

    Args:
        model: PyTorch model
        device: Target device

    Returns:
        Model on target device
    """
    logger = get_logger()
    logger.info(f"Moving model to {device}")

    if device == "cuda" and torch.cuda.is_available():
        model = model.to('cuda')
    elif device == "cpu":
        model = model.to('cpu')
    else:
        logger.warning(f"Device {device} not available, keeping on current device")

    return model


def clear_gpu_cache() -> None:
    """Clear GPU cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        get_logger().info("GPU cache cleared")


def get_model_size(model: Any) -> Dict[str, Any]:
    """
    Get model size information.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with size info
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate memory size (assuming float32)
    param_memory = total_params * 4  # bytes for fp32

    size_info = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'parameter_memory_bytes': param_memory,
        'parameter_memory_mb': param_memory / (1024 ** 2),
        'parameter_memory_gb': param_memory / (1024 ** 3),
    }

    return size_info


def check_model_loaded(model: Any) -> bool:
    """
    Check if model is properly loaded.

    Args:
        model: Model object

    Returns:
        True if model is loaded
    """
    if model is None:
        return False

    # Check if model has parameters
    try:
        params = list(model.parameters())
        return len(params) > 0
    except:
        return False


def save_model_checkpoint(
    model: Any,
    save_path: str,
    include_optimizer: bool = False,
    optimizer: Optional[Any] = None
) -> bool:
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        save_path: Path to save checkpoint
        include_optimizer: Whether to include optimizer state
        optimizer: Optimizer object

    Returns:
        True if successful
    """
    logger = get_logger()

    try:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': model.state_dict(),
        }

        if include_optimizer and optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        return False


def load_model_checkpoint(
    model: Any,
    checkpoint_path: str,
    load_optimizer: bool = False,
    optimizer: Optional[Any] = None
) -> bool:
    """
    Load model checkpoint.

    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint
        load_optimizer: Whether to load optimizer state
        optimizer: Optimizer object

    Returns:
        True if successful
    """
    logger = get_logger()

    try:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        if load_optimizer and optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return False


def estimate_inference_memory(
    model: Any,
    batch_size: int = 1,
    input_size: tuple = (224, 224),
    precision: str = "bf16"
) -> Dict[str, float]:
    """
    Estimate memory needed for inference.

    Args:
        model: PyTorch model
        batch_size: Batch size
        input_size: Input image size
        precision: Model precision

    Returns:
        Dictionary with memory estimates
    """
    model_size = get_model_size(model)

    # Adjust for precision
    precision_factor = {
        'fp32': 4,
        'fp16': 2,
        'bf16': 2,
    }.get(precision, 2)

    # Estimate activation memory (rough)
    # Assuming input image: batch_size * channels * height * width
    activation_memory = batch_size * 3 * input_size[0] * input_size[1] * precision_factor

    total_memory = (
        model_size['parameter_memory_bytes'] * precision_factor +
        activation_memory +
        model_size['parameter_memory_bytes']  # gradients (disabled in eval, but allocate)
    )

    return {
        'model_memory_gb': model_size['parameter_memory_gb'] * precision_factor / 4,
        'activation_memory_mb': activation_memory / (1024 ** 2),
        'total_memory_gb': total_memory / (1024 ** 3),
    }
