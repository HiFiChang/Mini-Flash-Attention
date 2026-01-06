import torch
from src.reference import naive_attention

def debug_oom():
    device = torch.device("cuda")
    seq_len = 16384
    h_dim = 64
    heads = 8
    bs = 4
    
    print(f"Allocating inputs for L={seq_len}...")
    q = torch.randn(bs, heads, seq_len, h_dim, device=device, dtype=torch.float16)
    k = torch.randn(bs, heads, seq_len, h_dim, device=device, dtype=torch.float16)
    v = torch.randn(bs, heads, seq_len, h_dim, device=device, dtype=torch.float16)
    
    torch.cuda.reset_peak_memory_stats()
    try:
        print("Running naive_attention...")
        naive_attention(q, k, v)
    except torch.cuda.OutOfMemoryError as e:
        print("\nCAUGHT OOM ERROR:")
        print(e)
        print(f"\nPeak Memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    
    print("Debug script finished.")

if __name__ == "__main__":
    debug_oom()
