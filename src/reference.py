import torch
import math

def naive_attention(q, k, v, scale=None):
    """
    标准的PyTorch Attention实现（朴素版本，用于性能对比）
    
    实现了标准的Scaled Dot-Product Attention算法。
    这是一个未经优化的版本，会产生O(N^2)的显存开销，主要用作性能基准。
    
    算法流程:
        1. 计算注意力分数: scores = (Q @ K^T) / sqrt(d)
        2. 应用Softmax归一化: attn_weights = softmax(scores)
        3. 加权求和: output = attn_weights @ V
    
    Args:
        q (torch.Tensor): Query张量, shape: (B, H, L, D)
            - B: Batch size (批次大小)
            - H: Number of heads (注意力头数)
            - L: Sequence length (序列长度)
            - D: Head dimension (每个头的维度)
        k (torch.Tensor): Key张量, shape: (B, H, S, D)
            - S: Source sequence length (源序列长度，通常等于L)
        v (torch.Tensor): Value张量, shape: (B, H, S, D)
        scale (float, optional): 缩放因子。如果为None，默认为1/sqrt(D)
    
    Returns:
        torch.Tensor: 注意力输出, shape: (B, H, L, D)
    
    注意:
        - 时间复杂度: O(L^2 * D)
        - 空间复杂度: O(L^2) - 需要存储完整的attention矩阵
        - 该实现在长序列时会占用大量显存
    
    示例:
        >>> q = torch.randn(2, 8, 512, 64, device='cuda')
        >>> k = torch.randn(2, 8, 512, 64, device='cuda')
        >>> v = torch.randn(2, 8, 512, 64, device='cuda')
        >>> output = naive_attention(q, k, v)
        >>> output.shape
        torch.Size([2, 8, 512, 64])
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
    PyTorch 2.0+ 官方优化的Scaled Dot-Product Attention（Flash Attention封装）
    
    这是PyTorch官方提供的高性能注意力实现，会自动选择最优的后端:
        - Flash Attention v2 (CUDA kernel, 首选)
        - Memory-Efficient Attention (次选)
        - 标准C++实现 (后备方案)
    
    该实现使用了以下优化技术:
        - Tiling (分块计算)
        - Kernel Fusion (算子融合)
        - Recomputation (重计算，减少显存)
    
    Args:
        q (torch.Tensor): Query张量, shape: (B, H, L, D)
        k (torch.Tensor): Key张量, shape: (B, H, S, D)
        v (torch.Tensor): Value张量, shape: (B, H, S, D)
    
    Returns:
        torch.Tensor: 注意力输出, shape: (B, H, L, D)
    
    注意:
        - 需要PyTorch >= 2.0
        - 在支持的GPU上会自动使用Flash Attention v2 (需要CUDA >= 11.6)
        - 性能通常优于手动实现的Triton版本（因为是高度优化的CUDA代码）
    
    性能特点:
        - 时间复杂度: O(L^2 * D) (与朴素版相同)
        - 空间复杂度: O(L) (相比朴素版的O(L^2)有巨大优势)
    
    参考:
        https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    """
    # This automatically selects the best implementation (FlashAttention, MemEfficient, or C++)
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)
