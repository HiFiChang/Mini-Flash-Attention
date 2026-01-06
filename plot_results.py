import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def setup_plot_style():
    """Configures matplotlib for publication-quality plots."""
    # Use a style similar to seaborn-paper or similar if available, else standard
    try:
        plt.style.use('seaborn-v0_8-paper')
    except:
        pass
    
    # Update individual parameters for "Paper Quality"
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 300,
        'lines.linewidth': 2,
        'lines.markersize': 8,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05
    })

def plot_latency(df):
    plt.figure(figsize=(7, 5))
    
    plt.plot(df['SeqLen'], df['Naive (ms)'], marker='o', label='PyTorch Naive', color='tab:red', linestyle='--')
    
    if 'SDPA (ms)' in df.columns:
        plt.plot(df['SeqLen'], df['SDPA (ms)'], marker='s', label='PyTorch SDPA', color='tab:orange', linestyle='-.')

    plt.plot(df['SeqLen'], df['Triton (ms)'], marker='^', label='Triton Flash', color='tab:blue')

    plt.xscale('log', base=2)
    plt.yscale('log')
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Latency (ms)')
    plt.title('Attention Latency vs Sequence Length')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.savefig('plot_latency.pdf')
    print("Saved plot_latency.pdf")
    plt.close()

def plot_memory(df):
    plt.figure(figsize=(7, 5))
    
    plt.plot(df['SeqLen'], df['Naive (MB)'], marker='o', label='PyTorch Naive', color='tab:red', linestyle='--')
    
    if 'SDPA (MB)' in df.columns:
        plt.plot(df['SeqLen'], df['SDPA (MB)'], marker='s', label='PyTorch SDPA', color='tab:orange', linestyle='-.')
        
    plt.plot(df['SeqLen'], df['Triton (MB)'], marker='^', label='Triton Flash', color='tab:blue')

    plt.xscale('log', base=2)
    plt.yscale('log')
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Peak Memory (MB)')
    plt.title('Memory Usage vs Sequence Length')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.savefig('plot_memory.pdf')
    print("Saved plot_memory.pdf")
    plt.close()

def plot_speedup(df):
    plt.figure(figsize=(7, 5))
    
    # Handle Inf/Nan in Speedup
    # Replace inf with a large number or drop? Or just plot valid ones.
    # If speedup is Inf (meaning Naive was Nan/OOM but Triton ran), we can't plot infinity.
    # We will plot available points.
    
    valid_speedup = df[np.isfinite(df['Speedup'])]
    
    plt.plot(valid_speedup['SeqLen'], valid_speedup['Speedup'], marker='*', label='Speedup (Triton vs Naive)', color='tab:green')
    
    plt.axhline(y=1.0, color='gray', linestyle=':', label='Baseline (1x)')
    
    plt.xscale('log', base=2)
    # plt.yscale('log') # Speedup might not need log scale unless it varies wildly 1 to 100.
    # It goes from 1.5 to 4. Linear might be better to show the saturation.
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Speedup Factor')
    plt.title('Speedup over Naive Attention')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.savefig('plot_speedup.pdf')
    print("Saved plot_speedup.pdf")
    plt.close()

def main():
    if not os.path.exists("benchmark_results.csv"):
        print("benchmark_results.csv not found. Please run benchmark.py first.")
        return
        
    df = pd.read_csv("benchmark_results.csv")
    setup_plot_style()
    
    plot_latency(df)
    plot_memory(df)
    plot_speedup(df)

if __name__ == "__main__":
    main()
