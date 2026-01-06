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
    
    # GPU规格数据库 (理论峰值)
    gpu_specs = {
        # NVIDIA Data Center GPUs
        "L20": {
            "fp32_tflops": 59.8,    # FP32 TFLOPS
            "fp16_tflops": 119.5,     # FP16 with Tensor Cores
            "bandwidth_gb": 864,   # HBM2 bandwidth GB/s
        }
    }
    
    # 尝试匹配GPU型号
    matched_spec = None
    for key, spec in gpu_specs.items():
        if key.lower() in gpu_name.lower():
            matched_spec = spec.copy()
            matched_spec["name"] = gpu_name
            break

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
    
    # 1. Q @ K^T: [B,H,L,D] @ [B,H,D,L] = [B,H,L,L]
    #    每个输出元素需要D次乘加 = 2D FLOPs
    #    总共 B*H*L*L 个输出元素
    qk_flops = 2 * B * H * L * L * D
    
    # 2. Softmax: 主要是exp和sum，相对FLOPs较少，可以忽略
    #    或简单估计为 ~5 * B*H*L*L (exp, sub, sum, div等)
    softmax_flops = 5 * B * H * L * L
    
    # 3. Attn @ V: [B,H,L,L] @ [B,H,L,D] = [B,H,L,D]
    #    每个输出元素需要L次乘加 = 2L FLOPs
    #    总共 B*H*L*D 个输出元素
    av_flops = 2 * B * H * L * L * D
    
    total_flops = qk_flops + softmax_flops + av_flops
    
    # ==================== Memory Access 计算 ====================
    # 标准实现 (Naive Attention):
    # 需要从HBM读取: Q, K, V
    # 需要写入HBM: QK^T矩阵, 最终输出
    
    # 读取:
    read_q = B * H * L * D * dtype_bytes
    read_k = B * H * L * D * dtype_bytes
    read_v = B * H * L * D * dtype_bytes
    
    # 写入:
    write_qk = B * H * L * L * dtype_bytes  # 注意力分数矩阵
    write_output = B * H * L * D * dtype_bytes
    
    naive_bytes = read_q + read_k + read_v + write_qk + write_output
    
    # Flash Attention优化后的内存访问:
    # 通过tiling避免写入完整的QK^T矩阵
    # 主要访问: 读Q,K,V + 写Output
    flash_bytes = read_q + read_k + read_v + write_output
    
    # ==================== 算术强度 ====================
    ai_naive = total_flops / naive_bytes
    ai_flash = total_flops / flash_bytes
    
    return {
        "flops": total_flops,
        "naive_bytes": naive_bytes,
        "flash_bytes": flash_bytes,
        "ai_naive": ai_naive,
        "ai_flash": ai_flash,
        "qk_flops": qk_flops,
        "av_flops": av_flops,
        "details": {
            "B": B, "H": H, "L": L, "D": D,
            "read_qkv_mb": (read_q + read_k + read_v) / 1024 / 1024,
            "write_attention_matrix_mb": write_qk / 1024 / 1024,
            "write_output_mb": write_output / 1024 / 1024,
        }
    }


def plot_roofline(benchmark_results_csv, gpu_specs, output_file="roofline.pdf"):
    """
    绘制Roofline图
    
    Args:
        benchmark_results_csv: benchmark结果CSV文件路径
        gpu_specs: GPU硬件规格字典
        output_file: 输出PDF文件名
    """
    # 读取benchmark结果
    df = pd.read_csv(benchmark_results_csv)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # GPU硬件参数
    peak_flops_fp16 = gpu_specs["fp16_tflops"]  # TFLOPS
    bandwidth = gpu_specs["bandwidth_gb"]  # GB/s
    
    # 转换为统一单位 (GFLOPS 和 GB/s)
    peak_flops = peak_flops_fp16 * 1000  # GFLOPS
    
    # ==================== 绘制Roofline边界 ====================
    # 算术强度范围
    ai_range = np.logspace(-2, 3, 100)  # 0.01 to 1000 FLOPs/Byte
    
    # 带宽上限线: Performance = Bandwidth × AI
    bandwidth_roof = bandwidth * ai_range
    
    # 计算上限线: 常数
    compute_roof = np.ones_like(ai_range) * peak_flops
    
    # 实际Roofline是两者的最小值
    roofline = np.minimum(bandwidth_roof, compute_roof)
    
    # 绘制Roofline
    ax.plot(ai_range, roofline, 'k-', linewidth=3, label='Roofline', zorder=10)
    ax.fill_between(ai_range, 0, roofline, alpha=0.1, color='gray')
    
    # 绘制边界线
    ax.axhline(y=peak_flops, color='red', linestyle='--', 
               linewidth=2, alpha=0.7, label=f'Peak Compute ({peak_flops_fp16} TFLOPS)')
    
    # 找到临界点 (ridge point)
    ridge_ai = peak_flops / bandwidth
    ax.axvline(x=ridge_ai, color='orange', linestyle='--', 
               linewidth=2, alpha=0.7, label=f'Ridge Point (AI={ridge_ai:.2f})')
    
    # ==================== 绘制实验数据点 ====================
    # 从benchmark结果中提取数据
    # 需要计算每个测试点的AI和实际性能
    
    colors = {'Naive': 'red', 'SDPA': 'orange', 'Triton': 'blue'}
    markers = {'Naive': 'o', 'SDPA': 's', 'Triton': '^'}
    
    # 假设配置 (应该从benchmark.py读取)
    BATCH_SIZE = 4
    NUM_HEADS = 8
    HEAD_DIM = 64
    
    for impl in ['Naive', 'SDPA', 'Triton']:
        ai_list = []
        perf_list = []
        labels = []
        
        for _, row in df.iterrows():
            seq_len = row['SeqLen']
            
            # 计算算术强度
            ai_info = calculate_attention_arithmetic_intensity(
                BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM, dtype_bytes=2
            )
            
            # 根据实现选择AI
            if impl == 'Naive':
                ai = ai_info['ai_naive']
                time_ms = row.get('Naive (ms)', float('nan'))
                tflops = row.get('Naive (TFLOPS)', float('nan'))
            elif impl == 'SDPA':
                ai = ai_info['ai_flash']  # SDPA使用Flash优化
                time_ms = row.get('SDPA (ms)', float('nan'))
                tflops = row.get('SDPA (TFLOPS)', float('nan'))
            else:  # Triton
                ai = ai_info['ai_flash']
                time_ms = row.get('Triton (ms)', float('nan'))
                tflops = row.get('Triton (TFLOPS)', float('nan'))
            
            # 跳过无效数据
            if np.isnan(tflops) or np.isnan(ai):
                continue
            
            ai_list.append(ai)
            perf_list.append(tflops * 1000)  # 转换为GFLOPS
            labels.append(f"L={seq_len}")
        
        # 绘制数据点
        if ai_list:
            ax.scatter(ai_list, perf_list, 
                      c=colors[impl], marker=markers[impl], 
                      s=100, alpha=0.7, label=impl, zorder=5,
                      edgecolors='black', linewidths=1)
            
            # 为每个点添加标签
            for ai, perf, label in zip(ai_list, perf_list, labels):
                ax.annotate(label, (ai, perf), 
                           textcoords="offset points", 
                           xytext=(5, 5), fontsize=8, alpha=0.7)
    
    # ==================== 图表设置 ====================
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance (GFLOPS)', fontsize=14, fontweight='bold')
    ax.set_title(f'Roofline Model - {gpu_specs["name"]}', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)
    
    # 设置坐标轴范围
    ax.set_xlim([ai_range[0], ai_range[-1]])
    ax.set_ylim([1, peak_flops * 1.5])
    
    # 添加性能区域标注
    ax.text(0.1, peak_flops * 0.3, 'Memory\nBound', 
            fontsize=12, ha='center', va='center', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(ridge_ai * 10, peak_flops * 0.8, 'Compute\nBound', 
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
    df = pd.read_csv(benchmark_results_csv)
    
    print("各序列长度的算术强度分析:")
    print("-" * 70)
    print(f"{'SeqLen':<8} {'AI(Naive)':<12} {'AI(Flash)':<12} {'瓶颈(Naive)':<15} {'瓶颈(Flash)':<15}")
    print("-" * 70)
    
    for _, row in df.iterrows():
        seq_len = int(row['SeqLen'])
        ai_info = calculate_attention_arithmetic_intensity(
            batch, heads, seq_len, head_dim, dtype_bytes=2
        )
        
        ai_naive = ai_info['ai_naive']
        ai_flash = ai_info['ai_flash']
        
        # 判断瓶颈
        bottleneck_naive = "Compute Bound" if ai_naive > ridge_point else "Memory Bound"
        bottleneck_flash = "Compute Bound" if ai_flash > ridge_point else "Memory Bound"
        
        print(f"{seq_len:<8} {ai_naive:<12.2f} {ai_flash:<12.2f} "
              f"{bottleneck_naive:<15} {bottleneck_flash:<15}")
    
    print()
    print("关键观察:")
    print("  • AI < Ridge Point → Memory Bound (受带宽限制)")
    print("  • AI > Ridge Point → Compute Bound (受计算能力限制)")
    print("  • Flash Attention提高了AI，更容易达到Compute Bound")
    print()
    
    # 绘制Roofline图
    plot_roofline(benchmark_results_csv, gpu_specs)
    
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
