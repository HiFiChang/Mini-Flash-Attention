import torch
import math

def naive_attention(q, k, v, scale=None):
    """
    Standard PyTorch implementation of attention.
    Shape: (B, H, L, D) - Batch, Heads, Sequence Length, Dimension
    
    Formula: Softmax(Q @ K.T / sqrt(dim)) @ V
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.size(-1))
    
    # q: [B, H, L, D]
    # k: [B, H, S, D] -> transpose -> [B, H, D, S]
    # scores: [B, H, L, S]
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    attn_weights = torch.softmax(scores, dim=-1)
    
    # v: [B, H, S, D]
    # output: [B, H, L, D]
    output = torch.matmul(attn_weights, v)
    
    return output

def pytorch_sdpa_attention(q, k, v):
    """
    Wrapper for PyTorch 2.0+ Flash Attention (Scaled Dot Product Attention).
    """
    # This automatically selects the best implementation (FlashAttention, MemEfficient, or C++)
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)
