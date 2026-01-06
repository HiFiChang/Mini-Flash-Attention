import torch
import triton
import triton.language as tl

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
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    
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
        
        # --- Online Softmax 更新逻辑 ---
        # 这部分是 FlashAttention 的精髓：在不完全计算出整个 Softmax 矩阵的情况下，
        # 逐步更新输出，大大节省显存。
        
        # 1. 计算当前块每行的最大值 m_ij
        m_ij = tl.max(qk, 1) # [BLOCK_M]
        
        # 2. 计算当前块的 exp 值 (减去最大值防止溢出) P_ij
        p = tl.exp(qk - m_ij[:, None]) # [BLOCK_M, BLOCK_N]
        
        # 3. 计算当前块的 row sum (l_ij)
        l_ij = tl.sum(p, 1) # [BLOCK_M]
        
        # 4. 更新全局最大值 m_i_new
        m_i_new = tl.maximum(m_i, m_ij)
        
        # 5. 计算缩放系数 alpha 和 beta
        # alpha用于修正之前的累加结果 acc 和 l_i (因为最大值 m_i 变了)
        # beta用于修正当前的 exp 值 (因为我们要统一到新的全局最大值)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        
        # --- 加载 V ---
        # V 需要是 [BLOCK_N, HEAD_DIM] 才能和 P [BLOCK_M, BLOCK_N] 相乘
        # 这里直接正常加载 V: 行是 n_range, 列是 d_range
        _v_ptr = V + v_offset + n_range[:, None] * stride_vn + d_range[None, :] * stride_vk
        v = tl.load(_v_ptr, mask=n_range[:, None] < N_CTX, other=0.0)
        
        # --- 更新 Accumulator ---
        
        # 修正当前的 P，使其基于新的全局最大值
        p_scaled = p * beta[:, None]
        
        # 更新累加器
        # acc_new = acc_old * alpha + P_new @ V
        acc = acc * alpha[:, None]
        acc += tl.dot(p_scaled.to(tl.float16), v) # 使用 fp16 计算 MatMul 加速，结果累加到 fp32
        
        # 更新全局 l_i (分母)
        l_i = l_i * alpha + l_ij * beta
        
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
    # Shape checks
    BATCH, HEADS, N_CTX, D_HEAD = q.shape
    
    # Define block sizes
    BLOCK_M = 128
    BLOCK_N = 64
    
    # Output buffer
    o = torch.empty_like(q)
    
    # Grid
    # we launch grid along M axis (rows of Q)
    # and Batch*Heads axis
    grid = (triton.cdiv(N_CTX, BLOCK_M), BATCH * HEADS)
    
    if scale is None:
        scale = 1.0 / (D_HEAD ** 0.5)
        
    num_stages = 4 if torch.cuda.get_device_properties(0).major >= 8 else 3
    num_warps = 4
    
    _fwd_kernel[grid](
        q, k, v, scale,
        None, None, # L, M not used
        o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        BATCH, HEADS, N_CTX,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D_HEAD,
        num_warps=num_warps,
        num_stages=num_stages
    )
    
    return o
