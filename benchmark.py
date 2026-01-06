import torch
import time
import pandas as pd
import triton.testing
from src.reference import naive_attention, pytorch_sdpa_attention
from src.triton_ops import triton_flash_attention

# Configuration
BATCH_SIZE = 4
NUM_HEADS = 8
HEAD_DIM = 64
SEQ_LENS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
WARMUP = 5
REPEATS = 100

def benchmark_op(op, q, k, v):
    """
    Benchmark using Triton's standardized testing utility.
    """
    return triton.testing.do_bench(lambda: op(q, k, v), warmup=WARMUP, rep=REPEATS)

def benchmark_memory(op, q, k, v):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        op(q, k, v)
        mem = torch.cuda.max_memory_allocated() / 1024 / 1024 # MB
    except torch.cuda.OutOfMemoryError:
        mem = float('nan')
    torch.cuda.empty_cache()
    return mem


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

            # 2. PyTorch SDPA (Flash/MemEfficient)
            time_sdpa = benchmark_op(pytorch_sdpa_attention, q, k, v)
            mem_sdpa = benchmark_memory(pytorch_sdpa_attention, q, k, v)

            # 3. Triton Flash Attention
            time_triton = benchmark_op(triton_flash_attention, q, k, v)
            mem_triton = benchmark_memory(triton_flash_attention, q, k, v)
            
            # Verify correctness
            if seq_len <= 1024:
                ref_out = pytorch_sdpa_attention(q, k, v)
                triton_out = triton_flash_attention(q, k, v)
                diff = torch.abs(ref_out - triton_out).max().item()
                print(f"Max Diff for L={seq_len}: {diff:.6f}")
                if diff > 0.1:
                   print(f"WARNING: Result mismatch significantly at L={seq_len}!")

            results.append({
                "SeqLen": seq_len,
                "Naive (ms)": time_naive,
                "SDPA (ms)": time_sdpa,
                "Triton (ms)": time_triton,
                "Naive (MB)": mem_naive,
                "SDPA (MB)": mem_sdpa,
                "Triton (MB)": mem_triton,
                "Speedup": time_naive / time_triton if time_naive == time_naive else float('inf')
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
