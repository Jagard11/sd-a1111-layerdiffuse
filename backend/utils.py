"""
A1111 compatibility layer for Forge's utils module.
"""

import torch
import safetensors.torch


def load_torch_file(filename, safe_load=True):
    """Load a torch file (safetensors or pickle)."""
    if filename.endswith('.safetensors'):
        return safetensors.torch.load_file(filename, device='cpu')
    else:
        if safe_load:
            return torch.load(filename, map_location='cpu', weights_only=True)
        else:
            return torch.load(filename, map_location='cpu')


def get_attr(obj, attr_path):
    """Get a nested attribute by path (e.g., 'input_blocks.1.1.transformer_blocks.0.attn1')."""
    attrs = attr_path.split('.')
    for attr in attrs:
        if attr.isdigit():
            obj = obj[int(attr)]
        else:
            obj = getattr(obj, attr)
    return obj


def set_attr(obj, attr_path, value):
    """Set a nested attribute by path."""
    attrs = attr_path.split('.')
    for attr in attrs[:-1]:
        if attr.isdigit():
            obj = obj[int(attr)]
        else:
            obj = getattr(obj, attr)
    
    final_attr = attrs[-1]
    if final_attr.isdigit():
        obj[int(final_attr)] = value
    else:
        setattr(obj, final_attr, value)
