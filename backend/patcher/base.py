"""
A1111 compatibility layer for Forge's ModelPatcher.
Provides a simplified patcher that works with A1111's model structure.
"""

import torch


class ModelPatcher:
    """
    Simplified model patcher compatible with A1111.
    Wraps a model and provides device management.
    """
    
    def __init__(self, model, load_device=None, offload_device=None):
        self.model = model
        self.load_device = load_device
        self.offload_device = offload_device
        self.patches = {}
        self.object_patches = {}
    
    def clone(self):
        """Create a clone of this patcher."""
        new_patcher = ModelPatcher(
            self.model,
            load_device=self.load_device,
            offload_device=self.offload_device
        )
        new_patcher.patches = self.patches.copy()
        new_patcher.object_patches = self.object_patches.copy()
        return new_patcher
    
    def to(self, device=None, dtype=None):
        """Move model to device/dtype."""
        if device is not None or dtype is not None:
            self.model.to(device=device, dtype=dtype)
        return self
    
    def add_object_patch(self, key, value):
        """Add an object patch."""
        self.object_patches[key] = value
    
    def get_model_object(self, key):
        """Get a model object by key."""
        if key in self.object_patches:
            return self.object_patches[key]
        
        from backend.utils import get_attr
        return get_attr(self.model, key)
