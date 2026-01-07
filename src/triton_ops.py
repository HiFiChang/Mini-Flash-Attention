import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=4, num_warps=4),
    ],
    key=['N_CTX', 'HEAD_DIM'],
)
@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    L, M,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr
):
    """
    Flash Attention的前向传播Kernel (Forward Pass)
    
    参数说明:
    Q, K, V: 输入的Query, Key, Value矩阵指针
    sm_scale: Softmax的缩放因子 (1/sqrt(d))
    Out: 输出矩阵指针
    stride_*: 各维度的步长，用于计算内存地址
    Z, H, N_CTX: Batch Size, Number of Heads, Sequence Length
    BLOCK_M, BLOCK_N: Triton分块大小 (Tiling Config)
    HEAD_DIM: 注意力头的维度
    """
    
    # 确保 BLOCK_N 小于或等于 HEAD_DIM，否则可能需要修改矩阵乘法逻辑 (仅作为简单的断言)
    # tl.static_assert(BLOCK_N <= HEAD_DIM)
    
    # 1. 确定当前Program ID (Grid中的位置)
    # Triton会将计算网格分为多个Program，每个Program负责计算Q的一个Block
    start_m = tl.program_id(0)  # Q 维度的Block索引 (沿着SeqLen M方向)
    off_hz = tl.program_id(1)   # Batch和Head的混合索引 (Z * H)
    
    # 2. 计算当前Batch和Head的内存偏移量
    # Q, K, V, Out 都是 (Batch, Head, SeqLen, Dim) 或者是类似的布局
    # off_hz 直接对应某个特定的 batch 和 head
    q_offset = off_hz * stride_qh
    k_offset = off_hz * stride_kh
    v_offset = off_hz * stride_vh
    o_offset = off_hz * stride_oh
    
    # 3. 初始化 Q 的 Block 指针
    # 我们现在的任务是计算 Out[start_m * BLOCK_M : (start_m+1) * BLOCK_M, :]
    # 这需要加载 Q 的对应行
    
    # m_range: 当前Block覆盖的行索引 [0, 1, ..., BLOCK_M-1] + offset
    m_range = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # d_range: 维度索引 [0, 1, ..., HEAD_DIM-1]
    d_range = tl.arange(0, HEAD_DIM)
    
    # 构建 Q 的指针矩阵 [BLOCK_M, HEAD_DIM]
    # 地址 = Base + (行索引 * 行步长) + (列索引 * 列步长)
    Q_block_ptr = Q + q_offset + m_range[:, None] * stride_qm + d_range[None, :] * stride_qk
    
    # 4. 初始化累加器 (Accumulators)
    # Online Softmax 技巧需要维护:
    # m_i: 当前行的最大值 (用于数值稳定性,防止exp溢出)
    # l_i: 当前行的 exp sum (分母)
    # acc:当前的 Attention Output (分子的一部分)
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    # 加载 Q 块到 SRAM (因为它在外循环中是不变的)
    # mask: 防止越界 (如果 Sequence Length 不是 BLOCK_M 的倍数)
    q = tl.load(Q_block_ptr, mask=m_range[:, None] < N_CTX, other=0.0)
    
    # 5. KV 循环 (Inner Loop)
    # 我们遍历 K 和 V 的所有块 (沿着 SeqLen N 方向)
    # 每次处理一个 BLOCK_N 大小的块
    for start_n in range(0, N_CTX, BLOCK_N):
        n_range = start_n + tl.arange(0, BLOCK_N)
        
        # --- 加载 K ---
        # 注意: 我们需要 Q @ K.T (Dot Product)
        # Triton 的 tl.dot(A, B) 执行标准的矩阵乘法。
        # Q 是 [BLOCK_M, HEAD_DIM]。
        # 为了得到 [BLOCK_M, BLOCK_N] 的 Score 矩阵，我们需要 K 为 [HEAD_DIM, BLOCK_N]。
        # 在内存中 K 是 [B, H, N, D]。
        # 我们这里的 _k_ptr 构造是通过让:
        #   行索引 = d_range (D维度)
        #   列索引 = n_range (N维度)
        # 从而实现了在加载时"隐式转置"，或者说加载出来就是我们需要计算乘法的形状。
        
        _k_ptr = K + k_offset + d_range[:, None] * stride_kk + n_range[None, :] * stride_kn
        k = tl.load(_k_ptr, mask=n_range[None, :] < N_CTX, other=0.0)
        
        # --- 计算 Attention Scores (QK^T) ---
        # qk shape: [BLOCK_M, BLOCK_N]
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        
        # 修复：Mask 掉 Padding 部分
        # 确保不会 attend 到序列长度之外的区域
        mask_n = n_range[None, :] < N_CTX
        qk = tl.where(mask_n, qk, float("-inf"))
        
        # --- Online Softmax 更新逻辑 (FlashAttention-2 Optimize) ---
        # 优化策略: 
        # 1. 直接基于更新后的全局最大值 m_i_new 计算 P，避免计算 beta 和额外的乘法
        # 2. 减少 exp 计算次数
        
        # 1. 计算当前块每行的最大值 m_ij
        m_ij = tl.max(qk, 1) # [BLOCK_M]
        
        # 2. 更新全局最大值 m_i_new
        m_i_new = tl.maximum(m_i, m_ij)
        
        # 3. 计算缩放系数 alpha (用于修正旧的 acc 和 l_i)
        # 如果 m_i_new == m_i (即当前块没有更大的值), 则 alpha = 1.0 (无需缩放)
        alpha = tl.exp(m_i - m_i_new)
        
        # 4. 计算当前块的 exp 值 P_ij
        # 直接减去新的全局最大值，这样得到的 P 已经是"Correctly Scaled"的
        # 不需要再乘以 beta
        p = tl.exp(qk - m_i_new[:, None])
        
        # --- 加载 V ---
        # V 需要是 [BLOCK_N, HEAD_DIM] 才能和 P [BLOCK_M, BLOCK_N] 相乘
        # 这里直接正常加载 V: 行是 n_range, 列是 d_range
        _v_ptr = V + v_offset + n_range[:, None] * stride_vn + d_range[None, :] * stride_vk
        v = tl.load(_v_ptr, mask=n_range[:, None] < N_CTX, other=0.0)
        
        # --- 更新 Accumulator ---
        
        # 更新累加器: acc_new = acc_old * alpha + P_new @ V
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(tl.float16), v) # 使用 fp16 计算 MatMul 加速
        
        # 更新全局 l_i (分母)
        # l_new = l_old * alpha + sum(P_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        
        # 更新 m_i 为下一轮做准备
        m_i = m_i_new

    # 6. 计算最终输出
    # Out = Acc / Denominator
    acc = acc / l_i[:, None]
    
    # 7. 写入结果到 HBM
    # 构造输出指针
    O_block_ptr = Out + o_offset + (m_range[:, None] * stride_om) + (d_range[None, :] * stride_on)
    tl.store(O_block_ptr, acc.to(tl.float16), mask=m_range[:, None] < N_CTX)


def triton_flash_attention(q, k, v, scale=None):
    """
    使用Triton实现的Flash Attention（教学版本）
    
    这是Flash Attention算法的Triton实现，使用了Online Softmax和分块计算技术。
    相比朴素实现，通过分块处理和重计算策略，将显存复杂度从O(N^2)降低到O(N)。
    
    核心优化技术:
        1. Tiling (分块): 将Q、K、V分块加载到SRAM，减少HBM访问
        2. Online Softmax: 逐块更新Softmax结果，避免存储完整attention矩阵
        3. Kernel Fusion: 将QK^T、Softmax、@V融合在一个kernel中
    
    Args:
        q (torch.Tensor): Query张量, shape: (B, H, L, D)
            - B: Batch size
            - H: Number of attention heads
            - L: Sequence length
            - D: Head dimension
        k (torch.Tensor): Key张量, shape: (B, H, L, D)
        v (torch.Tensor): Value张量, shape: (B, H, L, D)
        scale (float, optional): 注意力分数的缩放因子，默认为1/sqrt(D)
    
    Returns:
        torch.Tensor: 注意力输出, shape: (B, H, L, D)
    
    性能特点:
        - 时间复杂度: O(L^2 * D) (与标准实现相同)
        - 空间复杂度: O(L) (相比标准实现的O(L^2)大幅降低)
        - 显存节省: 在长序列(L>2048)时可节省数倍显存
    
    注意事项:
        - 该实现为教学目的，性能可能不如PyTorch官方的CUDA实现
        - 仅实现了前向传播(Forward Pass)，未实现反向传播
        - 数值精度使用FP16计算，FP32累加，精度略低于FP32全程计算
    
    Block Size选择说明:
        - BLOCK_M=128, BLOCK_N=64 是经验值，基于以下考虑:
          1. GPU SRAM大小限制 (通常为48-192KB)
          2. 需要同时容纳Q_block、K_block、V_block和累加器
          3. 对于D=64: (128*64 + 64*64 + 64*64)*2bytes ≈ 24KB < SRAM
        - 不同GPU架构和head_dim可能需要不同的block size
        - 更大的block可以减少循环次数，但需要更多SRAM
    
    参考论文:
        FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
        https://arxiv.org/abs/2205.14135
    
    示例:
        >>> q = torch.randn(2, 8, 1024, 64, device='cuda', dtype=torch.float16)
        >>> k = torch.randn(2, 8, 1024, 64, device='cuda', dtype=torch.float16)
        >>> v = torch.randn(2, 8, 1024, 64, device='cuda', dtype=torch.float16)
        >>> output = triton_flash_attention(q, k, v)
        >>> output.shape
        torch.Size([2, 8, 1024, 64])
    """
    # Shape checks
    BATCH, HEADS, N_CTX, D_HEAD = q.shape
    
    # ==================== Block Size Configuration ====================
    # BLOCK_M: Query方向的分块大小 (处理多少行Q)
    # BLOCK_N: Key/Value方向的分块大小 (处理多少列K和V)
    # 
    # 注意: 现在使用 Triton Autotune 自动选择最佳 Block Size 和 Warps 配置
    # ================================================================
    
    # Output buffer
    o = torch.empty_like(q)
    
    # Grid
    # we launch grid along M axis (rows of Q)
    # and Batch*Heads axis
    # The autotuner passes META so we can access BLOCK_M
    grid = lambda META: (triton.cdiv(N_CTX, META['BLOCK_M']), BATCH * HEADS)
    
    if scale is None:
        scale = 1.0 / (D_HEAD ** 0.5)
        
    _fwd_kernel[grid](
        q, k, v, scale,
        None, None, # L, M not used
        o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        BATCH, HEADS, N_CTX,
        HEAD_DIM=D_HEAD
    )
    
    return o

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=4, num_warps=4),
    ],
    key=['N_CTX', 'HEAD_DIM'],
)
@triton.jit
def _fwd_kernel_v2(
    Q, K, V, sm_scale,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    # Offsets for memory pointers
    q_offset = off_hz * stride_qh
    k_offset = off_hz * stride_kh
    v_offset = off_hz * stride_vh
    o_offset = off_hz * stride_oh

    # -----------------------------------------------------------
    # Block Pointers (Triton 2.0+)
    # -----------------------------------------------------------
    
    # Q Block Ptr
    # Shape: (N_CTX, HEAD_DIM)
    # Block: (BLOCK_M, HEAD_DIM)
    # Start: (start_m * BLOCK_M, 0)
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0)
    )

    # K Block Ptr
    # Shape: (HEAD_DIM, N_CTX) -- Transposed View for easier dot
    # Actually, keep (N_CTX, HEAD_DIM) and load, then transpose in registers
    # Because make_block_ptr order refers to memory layout
    # Memory: (N_CTX, HEAD_DIM) with stride (stride_kn, stride_kk)
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1)
    )
    
    # V Block Ptr
    # Shape: (N_CTX, HEAD_DIM)
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0)
    )
    
    # O Block Ptr
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0)
    )

    # Initialize accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    # Load Q
    # boundary_check=(0,): only check dim 0 (M)
    q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    
    # Loop over K, V blocks
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # Load K, V
        # K shape logic: 
        # We constructed K_ptr as (HEAD_DIM, N_CTX)
        # block_shape=(HEAD_DIM, BLOCK_N)
        # so k is (HEAD_DIM, BLOCK_N) -> ready for dot(q, k)
        k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
        
        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
        
        # QK^T
        qk = tl.dot(q, k)
        qk *= sm_scale
        
        # Masking (General/Padding checks if needed)
        # With block pointers, if padding_option="zero", k/v are 0 outside.
        # But for Softmax, we need -inf for masked positions.
        # The block pointer handles reading 0s. 
        # But we computed qk with zeros. exp(0) = 1. This affects softmax!
        # So we explicitly need a geometric mask for the columns.
        
        # Reconstruct ranges for masking
        # It's slightly redundant but needed for correctness with make_block_ptr
        # if implicit masking is zero-padding.
        # (Though if K was 0, dot product is 0. masked_fill needs -inf)
        
        # Explicit mask construction
        # (Alternatively, create a mask tensor)
        # For simplicity, we can do:
        if start_n + BLOCK_N > N_CTX:
             # Handle edge case for last block
             # (Only needed if N_CTX is not multiple of BLOCK_N)
             n_range = start_n + tl.arange(0, BLOCK_N)
             mask_n = n_range[None, :] < N_CTX
             qk = tl.where(mask_n, qk, float("-inf"))
        
        # Online Softmax updates - same as V1
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        
        acc = acc * alpha[:, None]
        p_scaled = p * beta[:, None]
        acc += tl.dot(p_scaled.to(tl.float16), v)
        
        l_i = l_i * alpha + l_ij * beta
        m_i = m_i_new
        
        # Advance pointers
        # K: advance along N_CTX (dim 1)
        # V: advance along N_CTX (dim 0)
        tl.advance(K_block_ptr, (0, BLOCK_N))
        tl.advance(V_block_ptr, (BLOCK_N, 0))

    # Epilogue
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(tl.float16), boundary_check=(0,))

def triton_flash_attention_v2(q, k, v, scale=None):
    """
    Flash Attention V2 (Triton Block Pointers)
    """
    BATCH, HEADS, N_CTX, D_HEAD = q.shape
    
    # Grid
    grid = lambda META: (triton.cdiv(N_CTX, META['BLOCK_M']), BATCH * HEADS)
    
    if scale is None:
        scale = 1.0 / (D_HEAD ** 0.5)
        
    o = torch.empty_like(q)
    
    _fwd_kernel_v2[grid](
        q, k, v, scale,
        o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        BATCH, HEADS, N_CTX,
        HEAD_DIM=D_HEAD
    )
    return o
