import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Standalone implementation to avoid importing torch/roofline_analysis if environment is missing torch
def get_gpu_specs_hardcoded():
    # Hardcoded for L20 based on report text
    return {
        "name": "NVIDIA L20",
        "fp16_tflops": 119.5,
        "bandwidth_gb": 864
    }

def calculate_ai(batch, heads, seq_len, head_dim):
    B, H, L, D = batch, heads, seq_len, head_dim
    dtype_bytes = 2 # FP16
    
    # FLOPs
    qk_flops = 2 * B * H * L * L * D
    av_flops = 2 * B * H * L * L * D
    total_flops = qk_flops + av_flops
    
    # Memory Access
    read_qkv = 3 * B * H * L * D * dtype_bytes
    write_out = B * H * L * D * dtype_bytes
    
    # Naive Intermediate
    intermediate_matrix_size = B * H * L * L * dtype_bytes
    naive_intermediate_access = 4 * intermediate_matrix_size
    naive_bytes = read_qkv + write_out + naive_intermediate_access
    
    # Flash Bytes
    flash_bytes = read_qkv + write_out
    
    return {
        "ai_naive": total_flops / naive_bytes,
        "ai_flash": total_flops / flash_bytes
    }

def setup_plot_style():
    """Configures matplotlib for publication-quality plots."""
    plt.rcdefaults()
    try:
        plt.style.use('seaborn-v0_8-paper')
    except:
        pass
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
        'font.size': 14,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 300,
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    })

def load_data(filename='benchmark_results.csv'):
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return None
    return pd.read_csv(filename)

def plot_latency(df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        save = True
    else:
        save = False

    ax.plot(df['SeqLen'], df['Naive (ms)'], marker='o', label='PyTorch Naive', color='tab:red', linestyle='--')
    if 'SDPA (ms)' in df.columns:
        ax.plot(df['SeqLen'], df['SDPA (ms)'], marker='s', label='PyTorch SDPA', color='tab:orange', linestyle='-.')
    ax.plot(df['SeqLen'], df['Triton (ms)'], marker='^', label='Ours (Triton)', color='tab:blue')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log') # Log scale enabled
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Latency (ms)')
    
    if save:
        ax.legend()
        plt.savefig('report/latency.pdf')
        print("Saved report/latency.pdf")
        plt.close()

def plot_memory(df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        save = True
    else:
        save = False
        
    ax.plot(df['SeqLen'], df['Naive (MB)'], marker='o', label='PyTorch Naive', color='tab:red', linestyle='--')
    if 'SDPA (MB)' in df.columns:
        ax.plot(df['SeqLen'], df['SDPA (MB)'], marker='s', label='PyTorch SDPA', color='tab:orange', linestyle='-.')
    ax.plot(df['SeqLen'], df['Triton (MB)'], marker='^', label='Ours (Triton)', color='tab:blue')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log') # Log scale enabled
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Peak Memory (MB)')
    if save:
        ax.legend()
        plt.savefig('report/memory.pdf')
        print("Saved report/memory.pdf")
        plt.close()

def plot_flops(df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        save = True
    else:
        save = False

    df_naive = df.dropna(subset=['Naive (TFLOPS)'])
    ax.plot(df_naive['SeqLen'], df_naive['Naive (TFLOPS)'], marker='o', label='PyTorch Naive', color='tab:red', linestyle='--')
    if 'SDPA (TFLOPS)' in df.columns:
        df_sdpa = df.dropna(subset=['SDPA (TFLOPS)'])
        ax.plot(df_sdpa['SeqLen'], df_sdpa['SDPA (TFLOPS)'], marker='s', label='PyTorch SDPA', color='tab:orange', linestyle='-.')
    if 'Triton (TFLOPS)' in df.columns:
        df_triton = df.dropna(subset=['Triton (TFLOPS)'])
        ax.plot(df_triton['SeqLen'], df_triton['Triton (TFLOPS)'], marker='^', label='Ours (Triton)', color='tab:blue')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Throughput (TFLOPS)')
    if save:
        ax.legend(loc='lower right')
        plt.savefig('report/flops.pdf')
        print("Saved report/flops.pdf")
        plt.close()

def plot_roofline_content(ax, df, batch=4, heads=8, head_dim=64):
    gpu_specs = get_gpu_specs_hardcoded()
    peak_flops_fp16 = gpu_specs["fp16_tflops"]
    bandwidth = gpu_specs["bandwidth_gb"]
    peak_gflops = peak_flops_fp16 * 1000  
    
    ai_min, ai_max = 1e1, 5e3
    ai_range = np.logspace(np.log10(ai_min), np.log10(ai_max), 100)
    bandwidth_roof = bandwidth * ai_range
    compute_roof = np.ones_like(ai_range) * peak_gflops
    roofline = np.minimum(bandwidth_roof, compute_roof)
    
    ax.plot(ai_range, roofline, 'k-', linewidth=1.5, label='Roofline')
    ax.fill_between(ai_range, 0, roofline, alpha=0.05, color='gray')
    ax.axhline(y=peak_gflops, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(ai_min * 1.1, peak_gflops * 1.1, f'Peak: {peak_flops_fp16:.0f} TFLOPS', fontsize=8, color='gray')
    ridge_ai = peak_gflops / bandwidth
    ax.axvline(x=ridge_ai, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(ridge_ai * 1.1, 110, f'Ridge: {ridge_ai:.0f}', fontsize=8, color='gray', rotation=90)
    
    impl_configs = [
        {'name': 'Naive',  'col': 'Naive (TFLOPS)',  'color': 'tab:red',    'marker': 'o'},
        {'name': 'SDPA',   'col': 'SDPA (TFLOPS)',   'color': 'tab:orange', 'marker': 's'},
        {'name': 'Ours', 'col': 'Triton (TFLOPS)', 'color': 'tab:blue',   'marker': '^'}
    ]
    for config in impl_configs:
        perf_col = config['col']
        if perf_col not in df.columns: continue
        ai_points, perf_points = [], []
        labels = []
        for _, row in df.iterrows():
            if pd.isna(row[perf_col]): continue
            seq_len = int(row['SeqLen'])
            tflops = row[perf_col]
            
            ai_info = calculate_ai(batch, heads, seq_len, head_dim)
            ai = ai_info['ai_naive'] if config['name'] == 'Naive' else ai_info['ai_flash']
            
            ai_points.append(ai)
            perf_points.append(tflops * 1000)
            labels.append(seq_len)
            
        if ai_points:
            ax.scatter(ai_points, perf_points, c=config['color'], marker=config['marker'], s=40, label=config['name'], edgecolors='white', linewidth=0.5, zorder=10)
            
            # Annotate "Ours" and "SDPA" points with SeqLen
            if config['name'] in ['Ours', 'SDPA']:
                for i, txt in enumerate(labels):
                    is_sdpa = config['name'] == 'SDPA'
                    va_val = 'top' if is_sdpa else 'bottom'
                    xy_offset = (0, -8) if is_sdpa else (0, 5)
                    ax.annotate(str(txt), (ai_points[i], perf_points[i]), 
                                xytext=xy_offset, textcoords='offset points', 
                                ha='center', va=va_val, fontsize=9, color=config['color'])

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Arithmetic Intensity (FLOPs/Byte)')
    ax.set_ylabel('Performance (GFLOPS)')
    ax.grid(True, which="major", ls="-", alpha=0.3)
    ax.set_xlim(ai_min, ai_max)
    ax.set_ylim(100, peak_gflops * 2)

def plot_roofline_paper(df, batch=4, heads=8, head_dim=64):
    plt.figure(figsize=(6, 4.5))
    ax = plt.gca()
    plot_roofline_content(ax, df, batch, heads, head_dim)
    plt.legend(loc='lower right')
    plt.savefig('report/roofline.pdf')
    print("Saved report/roofline.pdf")
    plt.close()

def plot_combined_figure(df):
    """Creates a combined figure with 4 subplots arranged horizontally."""
    # Create figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 3.5)) # Slightly reduced height
    
    # 1. Latency
    plot_latency(df, ax=axes[0])
    axes[0].set_title('(a) Latency')
    
    # 2. Peak Memory
    plot_memory(df, ax=axes[1])
    axes[1].set_title('(b) Peak Memory')
    
    # 3. Throughput (FLOPS)
    plot_flops(df, ax=axes[2])
    axes[2].set_title('(c) Throughput')
    
    # 4. Roofline
    plot_roofline_content(axes[3], df)
    axes[3].set_title('(d) Roofline Model')
    
    # Add a global legend
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=3)
    
    plt.tight_layout()
    plt.savefig('report/combined_plots.pdf', bbox_inches='tight')
    print("Saved report/combined_plots.pdf")
    plt.close()

if __name__ == "__main__":
    setup_plot_style()
    df = load_data()
    if df is not None:
        if not os.path.exists('report'): os.makedirs('report')
        plot_latency(df)
        plot_memory(df)
        plot_flops(df)
        plot_roofline_paper(df)
        plot_combined_figure(df)
