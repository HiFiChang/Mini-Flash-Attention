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
    
    # 8. 写入 LogSumExp (L) 用于反向传播
    # L = m_i + log(l_i)
    # L shape: (B, H, N_CTX) -> Pointer arithmetic assuming contiguous in last dim or similar structure
    # 这里我们假设 L 是 (B, H, N_CTX) 的布局
    # 指针地址: L_base + (off_hz * N_CTX) + m_range
    # 注意：为了简化，我们这里假设 L 是连续的 (B*H, N_CTX)
    L_block_ptr = L + off_hz * N_CTX + m_range
    l_i_log = m_i + tl.log(l_i)
    tl.store(L_block_ptr, l_i_log, mask=m_range < N_CTX)


@triton.jit
def _bwd_preprocess(
    Out, DO,
    Delta,
    BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr,
):
    """
    计算 Delta = sum(Out * DO, axis=-1)
    这是 FlashAttention 反向传播公式中的修正项
    dS_ij = P_ij * (dP_ij - Delta_i)
    """
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    # Load Out and DO
    # Assume contiguous layout for simplicity or use strides if passed (here assuming generic pointer math handles it roughly)
    # Ideally should pass strides, but for preprocessing we can assume simple layout or just pointer arithmetic
    # For robust implementation, we strictly follow grid
    
    # Grid: (N_CTX // BLOCK_M, B * H)
    # But for simplicity, let's treat pointers as flattened (B*H*N_CTX, D)
    
    # 实际上由于 Delta 需要与 Forward 的 loop 对应，我们让 grid 为 (N_CTX // BLOCK_M * B * H)
    # 所以 off_m 就是绝对的 row index
    
    o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    
    delta = tl.sum(o * do, axis=1)
    
    tl.store(Delta + off_m, delta)

@triton.jit
def _bwd_kernel(
    Q, K, V, sm_scale,
    Out, DO,
    DQ, DK, DV,
    L,
    D,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    num_head, seq_len_ctx,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr
):
    """
    Flash Attention Backward Kernel
    
    策略:
    - Parallelize over K/V blocks (Axis 0) to avoid atomic locks on DK, DV
    - Loop over Q blocks (Inner loop) and use atomic_add for DQ
    """
    # Grid: (N_CTX // BLOCK_N, B * H)
    off_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N) # K, V range
    off_hz = tl.program_id(1) # Batch * Head
    
    # Offsets for current batch/head
    q_offset = off_hz * stride_qh
    k_offset = off_hz * stride_kh
    v_offset = off_hz * stride_vh
    o_offset = off_hz * seq_len_ctx * HEAD_DIM # Assuming contiguous OUT
    do_offset = o_offset
    dq_offset = q_offset
    dk_offset = k_offset
    dv_offset = v_offset
    
    # Pointers to K, V (fixed for this program)
    k_ptrs = K + k_offset + (off_n[:, None] * stride_kn) + (tl.arange(0, HEAD_DIM)[None, :] * stride_kk)
    v_ptrs = V + v_offset + (off_n[:, None] * stride_vn) + (tl.arange(0, HEAD_DIM)[None, :] * stride_vk)
    
    # Accumulators for DK, DV
    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    
    # Load K and V
    # mask checks
    k = tl.load(k_ptrs, mask=off_n[:, None] < seq_len_ctx, other=0.0)
    v = tl.load(v_ptrs, mask=off_n[:, None] < seq_len_ctx, other=0.0)
    
    # Loop over Q blocks (M dimension)
    # We iterate through all Q blocks that attend to these K/V blocks
    # Attention is fully causal or non-causal. Here we assume non-causal (standard)
    # For M loop:
    for start_m in range(0, seq_len_ctx, BLOCK_M):
        off_m = start_m + tl.arange(0, BLOCK_M)
        
        # Load Q
        q_ptrs = Q + q_offset + (off_m[:, None] * stride_qm) + (tl.arange(0, HEAD_DIM)[None, :] * stride_qk)
        q = tl.load(q_ptrs, mask=off_m[:, None] < seq_len_ctx, other=0.0)
        
        # Compute QK^T (Attention Scores)
        # q: [BLOCK_M, HEAD_DIM], k: [BLOCK_N, HEAD_DIM] -> qk: [BLOCK_M, BLOCK_N]
        # Transpose k for dot product
        qk = tl.dot(q, tl.trans(k))
        qk *= sm_scale
        
        # Load L (LogSumExp)
        # L ptr: off_hz * seq_len + off_m
        l = tl.load(L + off_hz * seq_len_ctx + off_m, mask=off_m < seq_len_ctx, other=0.0)
        
        # Recompute P (Softmax)
        # p = exp(qk - l_i)
        p = tl.exp(qk - l[:, None])
        
        # Mask out padding/invalid (if any)
        # Note: In forward we masked qk. Here we mask p.
        p = tl.where((off_m[:, None] < seq_len_ctx) & (off_n[None, :] < seq_len_ctx), p, 0.0)
        
        # Load DO
        do_ptrs = DO + do_offset + (off_m[:, None] * HEAD_DIM) + (tl.arange(0, HEAD_DIM)[None, :])
        do = tl.load(do_ptrs, mask=off_m[:, None] < seq_len_ctx, other=0.0)
        
        # Load Delta
        # Delta ptr: off_hz * seq_len + off_m
        delta = tl.load(D + off_hz * seq_len_ctx + off_m, mask=off_m < seq_len_ctx, other=0.0)
        
        # Compute dP
        # dP = dot(DO, V^T)
        # DO: [BLOCK_M, HEAD_DIM], V: [BLOCK_N, HEAD_DIM] -> dp: [BLOCK_M, BLOCK_N]
        dp = tl.dot(do, tl.trans(v))
        
        # Compute dS (Gradient w.r.t Attention Scores)
        # dS = P * (dP - Delta)
        # dS = P * dP - P * Delta
        ds = p * (dp - delta[:, None])
        ds = ds.to(tl.float16) * sm_scale # Scale gradient
        
        # Accumulate DV
        # dV += P^T @ DO
        # P^T: [BLOCK_N, BLOCK_M], DO: [BLOCK_M, HEAD_DIM]
        dv += tl.dot(tl.trans(p.to(tl.float16)), do)
        
        # Accumulate DK
        # dK += dS^T @ Q
        # dS^T: [BLOCK_N, BLOCK_M], Q: [BLOCK_M, HEAD_DIM]
        dk += tl.dot(tl.trans(ds.to(tl.float16)), q)
        
        # Accumulate DQ (Atomic Add necessary as multiple K blocks contribute to same Q)
        # dQ += dS @ K
        # dS: [BLOCK_M, BLOCK_N], K: [BLOCK_N, HEAD_DIM]
        # Wait, inside this loop we only compute contribution from current K block to current Q block.
        # We need to atomic_add this contribution to global DQ memory.
        dq = tl.dot(ds.to(tl.float16), k)
        
        dq_ptrs = DQ + dq_offset + (off_m[:, None] * stride_qm) + (tl.arange(0, HEAD_DIM)[None, :] * stride_qk)
        tl.atomic_add(dq_ptrs, dq, mask=off_m[:, None] < seq_len_ctx)
        
    # Store DK, DV
    # These are exclusive to this program/block, so no atomics needed
    dk_ptrs = DK + dk_offset + (off_n[:, None] * stride_kn) + (tl.arange(0, HEAD_DIM)[None, :] * stride_kk)
    tl.store(dk_ptrs, dk.to(tl.float16), mask=off_n[:, None] < seq_len_ctx)
    
    dv_ptrs = DV + dv_offset + (off_n[:, None] * stride_vn) + (tl.arange(0, HEAD_DIM)[None, :] * stride_vk)
    tl.store(dv_ptrs, dv.to(tl.float16), mask=off_n[:, None] < seq_len_ctx)


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale=None):
        # Shape checks
        BATCH, HEADS, N_CTX, D_HEAD = q.shape
    
        # Default scale
        if scale is None:
            scale = 1.0 / (D_HEAD ** 0.5)
            
        # 1. Output buffers
        o = torch.empty_like(q)
        # LogSumExp buffer for backward pass
        # Shape: (B, H, N_CTX)
        # We ensure it's contiguous for simple pointer math
        L = torch.empty((BATCH, HEADS, N_CTX), device=q.device, dtype=torch.float32)
        
        # 2. Kernel launch parameters
        # Autotune handles BLOCK_M/BLOCK_N
        
        grid = lambda META: (triton.cdiv(N_CTX, META['BLOCK_M']), BATCH * HEADS)
        
        _fwd_kernel[grid](
            q, k, v, scale,
            L, None, # L passed here
            o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            BATCH, HEADS, N_CTX,
            HEAD_DIM=D_HEAD
        )
        
        # Save for backward
        ctx.save_for_backward(q, k, v, o, L)
        ctx.scale = scale
        ctx.BLOCK_M = 128 # Default or from autotune? Best to hardcode consistent block size for bwd or use heuristics
        # Note: Backward often works better with specific block sizes (e.g. 128)
        # For simplicity in this demo, we assume consistent configs or basic configs for backward
        
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, L = ctx.saved_tensors
        scale = ctx.scale
        
        BATCH, HEADS, N_CTX, D_HEAD = q.shape
        
        # 1. Preprocess: Compute Delta = sum(do * o, dim=-1)
        delta = torch.empty_like(L)
        
        # Use simple heuristics for preprocess grid
        # We can reuse similar block size or just 128
        PRE_BLOCK_M = 128
        grid_pre = (triton.cdiv(N_CTX, PRE_BLOCK_M) * BATCH * HEADS, )
        
        _bwd_preprocess[grid_pre](
            o, do,
            delta,
            BLOCK_M=PRE_BLOCK_M, D_HEAD=D_HEAD
        )
        
        # 2. Backward Kernel
        dq = torch.zeros_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        
        # Config for bwd kernel
        # We process K/V in blocks of BLOCK_N (grid dimension)
        # We loop over Q inside
        # Empirically, BLOCK_N=64, BLOCK_M=64 works robustly
        BLOCK_M = 64
        BLOCK_N = 64 
        num_stages = 3
        num_warps = 4
        
        grid_bwd = (triton.cdiv(N_CTX, BLOCK_N), BATCH * HEADS)
        
        _bwd_kernel[grid_bwd](
            q, k, v, scale,
            o, do,
            dq, dk, dv,
            L, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            HEADS, N_CTX,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D_HEAD,
            num_warps=num_warps,
            num_stages=num_stages
        )
        
        return dq, dk, dv, None

def triton_flash_attention(q, k, v, scale=None):
    """
    使用Triton实现的Flash Attention（Forward + Backward）
    """
    return FlashAttention.apply(q, k, v, scale)

# Legacy / Unused code removed
# def _old_triton_flash_attention(q, k, v, scale=None):
#     pass

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
