import torch
import time
import pandas as pd
import triton.testing
from src.reference import naive_attention, pytorch_sdpa_attention
from src.triton_ops import triton_flash_attention

# ==================== Configuration ====================
# Model Configuration (两组配置供选择)
# Config 1: 较小配置，适合显存受限的GPU
BATCH_SIZE = 4
NUM_HEADS = 8
HEAD_DIM = 64

# Config 2: 更大配置，更接近实际LLM场景 (如GPT-3: 96 heads, 128 dim)
# 如需测试，请注释掉上面的配置，取消注释下面的配置
# BATCH_SIZE = 8
# NUM_HEADS = 16
# HEAD_DIM = 128

# 序列长度范围：从128到16384，覆盖不同应用场景
# 短序列(128-512): 适合编码器模型
# 中序列(1024-4096): 适合常规对话
# 长序列(8192-16384): 适合长文本处理
SEQ_LENS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]

# Benchmark参数
WARMUP = 5      # 预热次数，确保GPU达到稳定状态
REPEATS = 100   # 重复测试次数，提高测量精度

# Compiled Naive Attention
naive_attention_compiled = torch.compile(naive_attention)

def benchmark_op(op, q, k, v):
    """
    Benchmark using Triton's standardized testing utility.
    """
    return triton.testing.do_bench(lambda: op(q, k, v), warmup=WARMUP, rep=REPEATS)

def benchmark_memory(op, q, k, v):
    """测量操作的峰值显存占用"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        op(q, k, v)
        mem = torch.cuda.max_memory_allocated() / 1024 / 1024 # MB
    except torch.cuda.OutOfMemoryError:
        mem = float('nan')
    torch.cuda.empty_cache()
    return mem

def calculate_flops(batch, heads, seq_len, head_dim):
    """
    计算标准Attention的理论FLOPS数
    
    Attention计算流程:
    1. Q @ K^T: (B*H*L*D) @ (B*H*D*L) = B*H*L*L*D 次乘加 = 2*B*H*L^2*D FLOPs
    2. Softmax: 主要是exp和sum，忽略不计
    3. Attn @ V: (B*H*L*L) @ (B*H*L*D) = 2*B*H*L^2*D FLOPs
    总计: 4*B*H*L^2*D FLOPs
    """
    return 4 * batch * heads * seq_len * seq_len * head_dim

def calculate_throughput(flops, time_ms):
    """
    计算吞吐量指标
    
    Args:
        flops: 浮点操作数
        time_ms: 执行时间(毫秒)
    
    Returns:
        TFLOPS: 每秒万亿次浮点运算
    """
    if time_ms != time_ms or time_ms == 0:  # NaN or zero check
        return float('nan')
    time_s = time_ms / 1000.0
    tflops = (flops / time_s) / 1e12
    return tflops


def main():
    if not torch.cuda.is_available():
        print("CUDA not available! Benchmarking on CPU is not recommended for this task.")
        return

    device = torch.device("cuda")
    print(f"Benchmarking on: {torch.cuda.get_device_name(0)}")
    
    results = []

    for seq_len in SEQ_LENS:
        print(f"Running for Sequence Length: {seq_len}...")
        
        # Prepare inputs (random data)
        # Shape: (B, H, L, D)
        q = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM, device=device, dtype=torch.float16)
        k = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM, device=device, dtype=torch.float16)
        v = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM, device=device, dtype=torch.float16)
        
        try:
            # 1. Naive PyTorch
            try:
                # 显存测试 (跑一次) - 优先运行，不容易 OOM
                mem_naive = benchmark_memory(naive_attention, q, k, v)
            except torch.cuda.OutOfMemoryError:
                mem_naive = float('nan')
                torch.cuda.empty_cache()

            try:
                # 速度测试 (跑多次) - 容易因碎片化 OOM
                time_naive = benchmark_op(naive_attention, q, k, v)
            except torch.cuda.OutOfMemoryError:
                time_naive = float('nan')
                torch.cuda.empty_cache()

            # 1.5 Naive PyTorch (Compiled)
            # 预热编译
            try:
                mem_compiled = benchmark_memory(naive_attention_compiled, q, k, v)
                time_compiled = benchmark_op(naive_attention_compiled, q, k, v)
            except torch.cuda.OutOfMemoryError:
                mem_compiled = float('nan')
                time_compiled = float('nan')
                torch.cuda.empty_cache()

            # 2. PyTorch SDPA (Flash/MemEfficient)
            time_sdpa = benchmark_op(pytorch_sdpa_attention, q, k, v)
            mem_sdpa = benchmark_memory(pytorch_sdpa_attention, q, k, v)

            # 3. Triton Flash Attention
            time_triton = benchmark_op(triton_flash_attention, q, k, v)
            mem_triton = benchmark_memory(triton_flash_attention, q, k, v)
            
            # Verify correctness (使用更严格的相对误差检查)
            if seq_len <= 1024:
                ref_out = pytorch_sdpa_attention(q, k, v)
                triton_out = triton_flash_attention(q, k, v)
                
                # 使用torch.allclose进行正确性验证
                # atol=1e-2: 绝对误差容忍度 (考虑到FP16精度)
                # rtol=1e-3: 相对误差容忍度
                is_correct = torch.allclose(ref_out, triton_out, atol=1e-2, rtol=1e-3)
                max_diff = torch.abs(ref_out - triton_out).max().item()
                mean_diff = torch.abs(ref_out - triton_out).mean().item()
                
                print(f"Correctness Check for L={seq_len}:")
                print(f"  Max Diff: {max_diff:.6f}, Mean Diff: {mean_diff:.6f}")
                print(f"  Result: {'✓ PASS' if is_correct else '✗ FAIL'}")
                
                if not is_correct:
                    print(f"  WARNING: Numerical accuracy issue detected!")
                    print(f"  This may be due to FP16 precision or implementation differences.")

            # 计算吞吐量指标
            total_flops = calculate_flops(BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM)
            throughput_naive = calculate_throughput(total_flops, time_naive)
            throughput_compiled = calculate_throughput(total_flops, time_compiled)
            throughput_sdpa = calculate_throughput(total_flops, time_sdpa)
            throughput_triton = calculate_throughput(total_flops, time_triton)
            
            results.append({
                "SeqLen": seq_len,
                "Naive (ms)": time_naive,
                "Compiled (ms)": time_compiled,
                "SDPA (ms)": time_sdpa,
                "Triton (ms)": time_triton,
                "Naive (MB)": mem_naive,
                # "Compiled (MB)": mem_compiled, # 暂不显示，为了表格整洁
                "SDPA (MB)": mem_sdpa,
                "Triton (MB)": mem_triton,
                "Naive (TFLOPS)": throughput_naive,
                "Compiled (TFLOPS)": throughput_compiled,
                "SDPA (TFLOPS)": throughput_sdpa,
                "Triton (TFLOPS)": throughput_triton,
                "Speedup vs Naive": time_naive / time_triton if time_naive == time_naive else float('inf'),
                "Speedup vs SDPA": time_sdpa / time_triton if time_sdpa == time_sdpa else float('nan')
            })
            
        except Exception as e:
            print(f"Error at len {seq_len}: {e}")

    # Display results
    df = pd.DataFrame(results)
    
    # Save to CSV
    csv_filename = "benchmark_results.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nResults saved to {csv_filename}")

    print("\nBenchmark Results:")
    try:
        print(df.to_markdown(index=False))
    except ImportError:
        print(df)


if __name__ == "__main__":
    main()
