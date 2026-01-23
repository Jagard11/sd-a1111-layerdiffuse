"""
A1111 compatibility layer for Forge's attention module.
"""

import torch
import math


def attention_function(q, k, v, heads, mask=None):
    """
    Standard scaled dot-product attention implementation.
    Compatible with A1111's attention mechanisms.
    """
    b, n, d = q.shape
    dim_head = d // heads
    
    # Reshape for multi-head attention
    q = q.view(b, n, heads, dim_head).transpose(1, 2)  # (b, heads, n, dim_head)
    k = k.view(b, -1, heads, dim_head).transpose(1, 2)  # (b, heads, seq_k, dim_head)
    v = v.view(b, -1, heads, dim_head).transpose(1, 2)  # (b, heads, seq_v, dim_head)
    
    # Scaled dot-product attention
    scale = 1.0 / math.sqrt(dim_head)
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    if mask is not None:
        attn = attn.masked_fill(mask == 0, float('-inf'))
    
    attn = torch.softmax(attn, dim=-1)
    
    # Apply attention to values
    out = torch.matmul(attn, v)
    
    # Reshape back
    out = out.transpose(1, 2).contiguous().view(b, n, d)
    
    return out
