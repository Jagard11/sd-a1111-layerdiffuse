"""
A1111 compatibility layer for Forge's memory_management module.
Provides similar functionality using A1111's existing infrastructure.
"""

import torch
from modules import devices, shared

# Track loaded models for compatibility
current_loaded_models = []


def get_torch_device():
    """Get the primary torch device."""
    return devices.get_optimal_device()


def unet_offload_device():
    """Get the device to offload models to (usually CPU)."""
    return devices.cpu


def should_use_fp16(device=None):
    """Check if fp16 should be used based on device capabilities."""
    if device is None:
        device = get_torch_device()
    
    # Check if we're on CPU
    if device.type == 'cpu':
        return False
    
    # Check command line args
    if hasattr(shared.cmd_opts, 'no_half') and shared.cmd_opts.no_half:
        return False
    
    # Default to fp16 on CUDA
    if torch.cuda.is_available():
        return True
    
    return False


def load_model_gpu(model_patcher):
    """Load a model to GPU. Compatible with A1111's device management."""
    device = get_torch_device()
    dtype = torch.float16 if should_use_fp16(device) else torch.float32
    
    if hasattr(model_patcher, 'model'):
        model_patcher.model.to(device=device, dtype=dtype)
    elif hasattr(model_patcher, 'to'):
        model_patcher.to(device=device, dtype=dtype)
    
    return model_patcher


def unload_model_clones(model):
    """Unload model clones - stub for compatibility."""
    # A1111 doesn't have the same clone system as Forge
    pass


class LoadedModel:
    """Wrapper class to track loaded models."""
    def __init__(self, model):
        self.model = model
