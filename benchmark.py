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

def benchmark_op(op, q, k, v, requires_grad=False):
    """
    Benchmark using Triton's standardized testing utility.
    """
    if requires_grad:
        # Benchmark Forward + Backward
        def fwd_bwd_op():
            # Need to clone or recreate inputs if they are modified inplace, but attention usually isn't
            # We need to reset grads
            if q.grad is not None:
                q.grad = None
            if k.grad is not None:
                k.grad = None
            if v.grad is not None:
                v.grad = None
                
            out = op(q, k, v)
            loss = out.mean() # Hacky scalar loss
            loss.backward()
            
        return triton.testing.do_bench(fwd_bwd_op, warmup=WARMUP, rep=REPEATS)
    else:
        # Benchmark Forward Only
        return triton.testing.do_bench(lambda: op(q, k, v), warmup=WARMUP, rep=REPEATS)

def benchmark_memory(op, q, k, v, requires_grad=False):
    """测量操作的峰值显存占用"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        if requires_grad:
            out = op(q, k, v)
            loss = out.mean()
            loss.backward()
        else:
            op(q, k, v)
        mem = torch.cuda.max_memory_allocated() / 1024 / 1024 # MB
    except torch.cuda.OutOfMemoryError:
        mem = float('nan')
    torch.cuda.empty_cache()
    return mem

def calculate_flops(batch, heads, seq_len, head_dim, backward=False):
    """
    计算Attention的理论FLOPS数
    
    精确计算:
    Forward GEMMs:
      1. Q @ K^T: 2 * B * H * L^2 * D
      2. P @ V:   2 * B * H * L^2 * D
      Total Forward = 4 * B * H * L^2 * D
      
    Backward GEMMs (Standard Backprop):
      1. dV = P^T @ dO:  2 * B * H * L^2 * D
      2. dP = dO @ V^T:  2 * B * H * L^2 * D
      3. dQ = dS @ K:    2 * B * H * L^2 * D
      4. dK = dS^T @ Q:  2 * B * H * L^2 * D
      Total Backward = 8 * B * H * L^2 * D
    
    Total Training (Fwd + Bwd) = 12 * B * H * L^2 * D
    
    注意: FlashAttention引入了重计算(Recomputation)，实际上会多做一次Forward (4*...),
    也就是实际计算量为 16 * ..., 但为了公平对比Throughput(有效吞吐),
    我们通常使用理论最小计算量 (12x) 或 包含检查点开销的标准量.
    这里我们采用理论值: Forward=4N^2d, Training=12N^2d (1:3比例).
    """
    fwd_flops = 4 * batch * heads * seq_len * seq_len * head_dim
    if backward:
        return 3.0 * fwd_flops # Fwd(1) + Bwd(2) = 3x total
    return fwd_flops

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
        # For training benchmark, we need gradients
        q = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM, device=device, dtype=torch.float16, requires_grad=True)
        k = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM, device=device, dtype=torch.float16, requires_grad=True)
        v = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM, device=device, dtype=torch.float16, requires_grad=True)
        
        try:
            # 1. Forward Pass Benchmarks
            # --------------------------------------------------------------------------------
            # Naive PyTorch (Compiled)
            try:
                if seq_len <= 4096:
                    time_compiled_fwd = benchmark_op(naive_attention_compiled, q, k, v, requires_grad=False)
                else:
                    time_compiled_fwd = float('nan')
            except Exception:
                time_compiled_fwd = float('nan')

            # PyTorch SDPA
            time_sdpa_fwd = benchmark_op(pytorch_sdpa_attention, q, k, v, requires_grad=False)

            # Triton Flash Attention
            time_triton_fwd = benchmark_op(triton_flash_attention, q, k, v, requires_grad=False)
            
            
            # 2. Combined (Forward + Backward) Benchmarks
            # --------------------------------------------------------------------------------
            
            # PyTorch SDPA (Backward)
            time_sdpa_bwd = benchmark_op(pytorch_sdpa_attention, q, k, v, requires_grad=True)
            mem_sdpa_bwd = benchmark_memory(pytorch_sdpa_attention, q, k, v, requires_grad=True)
            
            # Triton Flash Attention (Backward)
            time_triton_bwd = benchmark_op(triton_flash_attention, q, k, v, requires_grad=True)
            mem_triton_bwd = benchmark_memory(triton_flash_attention, q, k, v, requires_grad=True)
            
            
            # 3. Naive & Compiled Combined
            # --------------------------------------------------------------------------------
            # Naive Eager Backward
            if seq_len <= 1024:
                try:
                    time_naive_bwd = benchmark_op(naive_attention, q, k, v, requires_grad=True)
                except Exception:
                    time_naive_bwd = float('nan')
            else:
                time_naive_bwd = float('nan')

            # Naive Compiled Backward
            if seq_len <= 1024:
                try:
                    time_compiled_bwd = benchmark_op(naive_attention_compiled, q, k, v, requires_grad=True)
                except Exception:
                    time_compiled_bwd = float('nan')
            else:
                time_compiled_bwd = float('nan')

            # Verify correctness (Forward only for simplicity in this loop, backward verified separately)
            if seq_len <= 1024:
                with torch.no_grad():
                     ref_out = pytorch_sdpa_attention(q, k, v)
                     triton_out = triton_flash_attention(q, k, v)
                     is_correct = torch.allclose(ref_out, triton_out, atol=1e-2, rtol=1e-3)
                     if not is_correct:
                        print(f"  WARNING: Numerical accuracy issue detected!")

            # 计算吞吐量指标 (Using Fwd+Bwd FLOPs for Combined times)
            flop_fwd = calculate_flops(BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM, backward=False)
            flop_bwd = calculate_flops(BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM, backward=True)
            
            # Fwd TFLOPS
            tflops_compiled = calculate_throughput(flop_fwd, time_compiled_fwd)
            tflops_sdpa = calculate_throughput(flop_fwd, time_sdpa_fwd)
            tflops_triton = calculate_throughput(flop_fwd, time_triton_fwd)
            
            # Combined TFLOPS
            tflops_naive_train = calculate_throughput(flop_bwd, time_naive_bwd)
            tflops_compiled_train = calculate_throughput(flop_bwd, time_compiled_bwd)
            tflops_sdpa_train = calculate_throughput(flop_bwd, time_sdpa_bwd)
            tflops_triton_train = calculate_throughput(flop_bwd, time_triton_bwd)
            
            results.append({
                "SeqLen": seq_len,
                # Forward
                "Comp Fwd(ms)": time_compiled_fwd,
                "SDPA Fwd(ms)": time_sdpa_fwd,
                "Tri Fwd(ms)": time_triton_fwd,
                # Combined (Train)
                "Naive Train(ms)": time_naive_bwd,
                "Comp Train(ms)": time_compiled_bwd,
                "SDPA Train(ms)": time_sdpa_bwd,
                "Tri Train(ms)": time_triton_bwd,
                
                # Training Memory
                "SDPA Mem(MB)": mem_sdpa_bwd,
                "Tri Mem(MB)": mem_triton_bwd,
                
                # Training TFLOPS
                "Naive TFLOPS": tflops_naive_train,
                "SDPA TFLOPS": tflops_sdpa_train,
                "Tri TFLOPS": tflops_triton_train,
                
                "Speedup (Train)": time_sdpa_bwd / time_triton_bwd if time_sdpa_bwd == time_sdpa_bwd else float('nan')
            })
            
        except Exception as e:
            print(f"Error at len {seq_len}: {e}")
            import traceback
            traceback.print_exc()

    # Display results
    df = pd.DataFrame(results)
    
    # Save to CSV
    csv_filename = "benchmark_train_results.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nResults saved to {csv_filename}")
    
    print("\nBenchmark Training Results:")
    try:
        print(df.to_markdown(index=False))
    except ImportError:
        print(df)


if __name__ == "__main__":
    main()
