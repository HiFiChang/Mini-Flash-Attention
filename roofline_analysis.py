"""
Roofline分析模块
用于分析Attention操作的性能瓶颈：计算受限 vs 带宽受限
"""

import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def get_gpu_specs():
    """
    获取当前GPU的硬件规格
    
    Returns:
        dict: 包含峰值FLOPS和内存带宽的字典
    """
    if not torch.cuda.is_available():
        return None
    
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    gpu_name = props.name
    
    # GPU规格数据库 (理论峰值 FP16 Tensor Cores, Bandwidth GB/s)
    # 数据来源: NVIDIA Whitepapers
    gpu_specs_db = {
        "H100": {"fp16_tflops": 989.0, "bandwidth_gb": 3350}, # SXM5
        "A100": {"fp16_tflops": 312.0, "bandwidth_gb": 1935}, # 80GB
        "L20":  {"fp16_tflops": 119.5, "bandwidth_gb": 864},
        "L40":  {"fp16_tflops": 181.0, "bandwidth_gb": 864},
        "A10":  {"fp16_tflops": 125.0, "bandwidth_gb": 600},
        "T4":   {"fp16_tflops": 65.0,  "bandwidth_gb": 320},
        "3090": {"fp16_tflops": 71.0,  "bandwidth_gb": 936},  # Consumer
        "4090": {"fp16_tflops": 82.6,  "bandwidth_gb": 1008}, # Consumer (Tensor Cores approx)
        "V100": {"fp16_tflops": 125.0, "bandwidth_gb": 900},
    }
    
    # 尝试匹配GPU型号
    matched_spec = None
    
    # 1. 精确/模糊匹配
    for key, spec in gpu_specs_db.items():
        if key.lower() in gpu_name.lower():
            matched_spec = spec.copy()
            matched_spec["name"] = gpu_name
            break
            
    # 2. 如果未匹配，使用默认值或根据显存猜测 (这里为了安全返回一个保守的通用值或None)
    if matched_spec is None:
        print(f"Warning: GPU {gpu_name} not found in database. Using generic specs.")
        matched_spec = {
            "name": gpu_name + " (Generic)",
            "fp16_tflops": 100.0,
            "bandwidth_gb": 500,
        }

    return matched_spec


def calculate_attention_arithmetic_intensity(batch, heads, seq_len, head_dim, dtype_bytes=2):
    """
    计算Attention操作的算术强度 (Arithmetic Intensity)
    
    AI = FLOPs / Bytes Accessed
    
    Args:
        batch: Batch size
        heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Head dimension
        dtype_bytes: 数据类型字节数 (FP16=2, FP32=4)
    
    Returns:
        dict: 包含详细计算过程的字典
    """
    B, H, L, D = batch, heads, seq_len, head_dim
    
    # ==================== FLOPs 计算 ====================
    # Attention公式: Softmax(Q @ K^T / sqrt(d)) @ V
    # 忽略Softmax和Scale的微小FLOPs
    
    # 1. Q @ K^T: [B,H,L,D] @ [B,H,D,L] -> [B,H,L,L]
    #    2 * B * H * L * L * D
    qk_flops = 2 * B * H * L * L * D
    
    # 2. Attn @ V: [B,H,L,L] @ [B,H,L,D] -> [B,H,L,D]
    #    2 * B * H * L * L * D
    av_flops = 2 * B * H * L * L * D
    
    # 总FLOPs (近似 4 * B * H * L^2 * D)
    total_flops = qk_flops + av_flops
    
    # ==================== Memory Access 计算 ====================
    # 基础读写 (输入输出)
    # Read Q, K, V
    read_qkv = 3 * B * H * L * D * dtype_bytes
    # Write Output
    write_out = B * H * L * D * dtype_bytes
    
    # --- Naive Implementation (Standard PyTorch) ---
    # 必须显式存储中间的 Attention Matrix (Scores 和 Probabilities)
    # 流程:
    # 1. Read Q, K -> Compute Scores -> Write Scores (B,H,L,L)
    # 2. Read Scores -> Softmax -> Write Probs (B,H,L,L)
    # 3. Read Probs, V -> Compute Output -> Write Output
    #
    # 中间矩阵内存访问 = Write Scores + Read Scores + Write Probs + Read Probs
    #                 = 4 * (B * H * L * L * dtype_bytes)
    
    intermediate_matrix_size = B * H * L * L * dtype_bytes
    naive_intermediate_access = 4 * intermediate_matrix_size
    
    naive_bytes = read_qkv + write_out + naive_intermediate_access
    
    # --- Flash Attention Implementation ---
    # Tiling技术：在SRAM中计算，不将L*L矩阵写回HBM
    # 理想情况下只读Q,K,V一次，写Output一次
    flash_bytes = read_qkv + write_out
    
    # ==================== 算术强度 ====================
    ai_naive = total_flops / naive_bytes
    ai_flash = total_flops / flash_bytes
    
    return {
        "flops": total_flops,
        "naive_bytes": naive_bytes,
        "flash_bytes": flash_bytes,
        "ai_naive": ai_naive,
        "ai_flash": ai_flash,
        "details": {
            "B": B, "H": H, "L": L, "D": D,
            "read_qkv_mb": read_qkv / 1024**2,
            "write_output_mb": write_out / 1024**2,
            "naive_intermediate_mb": naive_intermediate_access / 1024**2,
        }
    }


def plot_roofline(benchmark_results_csv, gpu_specs, batch, heads, head_dim, output_file="roofline.pdf"):
    """
    绘制Roofline图
    
    Args:
        benchmark_results_csv: benchmark结果CSV文件路径
        gpu_specs: GPU硬件规格字典
        batch, heads, head_dim: 用于计算AI的模型配置
        output_file: 输出PDF文件名
    """
    # 读取benchmark结果
    try:
        df = pd.read_csv(benchmark_results_csv)
    except FileNotFoundError:
        print(f"Error: File {benchmark_results_csv} not found.")
        return

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # GPU硬件参数
    peak_flops_fp16 = gpu_specs["fp16_tflops"]  # TFLOPS
    bandwidth = gpu_specs["bandwidth_gb"]  # GB/s
    
    # 转换为统一单位 (GFLOPS 和 GB/s)
    # y轴使用 GFLOPS
    peak_gflops = peak_flops_fp16 * 1000  
    
    # ==================== 绘制Roofline边界 ====================
    # 算术强度范围 (扩展范围以适应不同模型)
    ai_min = 1e1
    ai_max = 1e4
    ai_range = np.logspace(np.log10(ai_min), np.log10(ai_max), 100)
    
    # 带宽上限线 (斜线): Performance = Bandwidth * AI
    # Bandwidth (GB/s) * AI (FLOPs/Byte) = GFLOPS
    bandwidth_roof = bandwidth * ai_range
    
    # 计算上限线 (水平线)
    compute_roof = np.ones_like(ai_range) * peak_gflops
    
    # 实际Roofline
    roofline = np.minimum(bandwidth_roof, compute_roof)
    
    # 绘制
    ax.plot(ai_range, roofline, 'k-', linewidth=2, label='Roofline Model')
    ax.fill_between(ai_range, 0, roofline, alpha=0.1, color='gray')
    
    # 辅助线
    ax.axhline(y=peak_gflops, color='r', linestyle='--', alpha=0.5, label=f'Peak Compute ({peak_flops_fp16:.1f} TFLOPS)')
    ridge_ai = peak_gflops / bandwidth
    ax.axvline(x=ridge_ai, color='orange', linestyle='--', alpha=0.5, label=f'Ridge Point ({ridge_ai:.1f} FLOPs/Byte)')
    
    # ==================== 绘制实验数据点 ====================
    impl_configs = [
        {'name': 'Naive',  'color': 'red',    'marker': 'o'},
        {'name': 'SDPA',   'color': 'orange', 'marker': 's'},
        {'name': 'Triton', 'color': 'blue',   'marker': '^'}
    ]
    
    for config in impl_configs:
        impl = config['name']
        perf_col = f"{impl} (TFLOPS)"
        
        # 检查数据是否存在
        if perf_col not in df.columns:
            continue
            
        ai_points = []
        perf_points = []
        labels = []
        
        for _, row in df.iterrows():
            seq_len = int(row['SeqLen'])
            tflops = row[perf_col]
            
            if pd.isna(tflops):
                continue
                
            # 计算AI
            ai_info = calculate_attention_arithmetic_intensity(
                batch, heads, seq_len, head_dim
            )
            
            # 确定使用的AI类型
            # Naive使用naive_ai，SDPA和Triton实际上使用了Flash/优化算法，内存访问接近Flash AI
            if impl == 'Naive':
                ai = ai_info['ai_naive']
            else:
                ai = ai_info['ai_flash']
            
            ai_points.append(ai)
            perf_points.append(tflops * 1000) # Convert TFLOPS to GFLOPS
            labels.append(seq_len)
            
        if ai_points:
            ax.scatter(ai_points, perf_points, 
                      c=config['color'], marker=config['marker'], s=100, 
                      label=impl, edgecolors='k', zorder=10)
            
            # 标注序列长度 (仅标注首尾以避免拥挤，或者针对所有点)
            for i, txt in enumerate(labels):
                 # 简单的防重叠逻辑：只在特定点或所有点标注
                 ax.annotate(f"{txt}", (ai_points[i], perf_points[i]), 
                             xytext=(0, 10), textcoords='offset points', 
                             ha='center', fontsize=8)

    # ==================== 装饰图表 ====================
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance (GFLOPS)', fontsize=12, fontweight='bold')
    ax.set_title(f'Roofline Model - {gpu_specs["name"]}\n(B={batch}, H={heads}, D={head_dim})', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)
    
    # 设置坐标轴范围
    ax.set_xlim(ai_range[0], ai_range[-1])
    ax.set_ylim(1, peak_gflops * 1.5)
    
    # 添加性能区域标注
    ax.text(ridge_ai / 4, peak_gflops * 0.1, 'Memory\nBound', 
            fontsize=12, ha='center', va='center', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(ridge_ai * 4, peak_gflops * 0.5, 'Compute\nBound', 
            fontsize=12, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Roofline图已保存到: {output_file}")
    plt.close()


def analyze_roofline(benchmark_results_csv, batch=4, heads=8, head_dim=64):
    """
    分析Attention操作的Roofline特性
    
    Args:
        benchmark_results_csv: benchmark结果文件
        batch, heads, head_dim: 模型配置
    """
    # 获取GPU规格
    gpu_specs = get_gpu_specs()
    if gpu_specs is None:
        print("错误: 未检测到CUDA GPU")
        return
    
    print("=" * 70)
    print(f"GPU Roofline 分析")
    print("=" * 70)
    print(f"GPU型号: {gpu_specs['name']}")
    print(f"峰值性能 (FP16): {gpu_specs['fp16_tflops']:.1f} TFLOPS")
    print(f"内存带宽: {gpu_specs['bandwidth_gb']:.0f} GB/s")
    
    ridge_point = (gpu_specs['fp16_tflops'] * 1000) / gpu_specs['bandwidth_gb']
    print(f"Ridge Point (临界算术强度): {ridge_point:.2f} FLOPs/Byte")
    print()
    
    # 读取benchmark结果
    try:
        df = pd.read_csv(benchmark_results_csv)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {benchmark_results_csv}")
        return
    
    print("各序列长度的算术强度分析:")
    print("-" * 70)
    print(f"{'SeqLen':<8} {'AI(Naive)':<12} {'AI(Flash)':<12} {'瓶颈(Naive)':<12} {'瓶颈(Flash)':<12}")
    print("-" * 70)
    
    for _, row in df.iterrows():
        seq_len = int(row['SeqLen'])
        ai_info = calculate_attention_arithmetic_intensity(
            batch, heads, seq_len, head_dim, dtype_bytes=2
        )
        
        ai_naive = ai_info['ai_naive']
        ai_flash = ai_info['ai_flash']
        
        # 判断瓶颈
        bottleneck_naive = "Compute" if ai_naive > ridge_point else "Memory"
        bottleneck_flash = "Compute" if ai_flash > ridge_point else "Memory"
        
        print(f"{seq_len:<8} {ai_naive:<12.2f} {ai_flash:<12.2f} "
              f"{bottleneck_naive:<12} {bottleneck_flash:<12}")
    
    print()
    print("关键观察:")
    print("  • AI < Ridge Point → Memory Bound (受带宽限制)")
    print("  • AI > Ridge Point → Compute Bound (受计算能力限制)")
    print("  • Flash Attention通过减少HBM访问提高了AI，使其更容易达到Compute Bound")
    print()
    
    # 绘制Roofline图
    plot_roofline(benchmark_results_csv, gpu_specs, batch, heads, head_dim)
    
    # 计算效率
    print("\n性能效率分析:")
    print("-" * 70)
    print(f"{'实现':<10} {'平均效率':<15} {'峰值效率':<15}")
    print("-" * 70)
    
    peak_tflops = gpu_specs['fp16_tflops']
    for impl, col in [('Naive', 'Naive (TFLOPS)'), 
                      ('SDPA', 'SDPA (TFLOPS)'), 
                      ('Triton', 'Triton (TFLOPS)')]:
        if col in df.columns:
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                avg_tflops = valid_data.mean()
                max_tflops = valid_data.max()
                avg_efficiency = (avg_tflops / peak_tflops) * 100
                max_efficiency = (max_tflops / peak_tflops) * 100
                print(f"{impl:<10} {avg_efficiency:>6.2f}%        {max_efficiency:>6.2f}%")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # 运行Roofline分析
    analyze_roofline("benchmark_results.csv")
