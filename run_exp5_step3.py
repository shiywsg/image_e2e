import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def plot_metric_vs_avg_snr(df, metric_suffix, title, ylabel, filename_suffix, ylim_top=None, lower_is_better=False):
    """通用的快衰落 Average SNR 扫描绘图函数"""
    plt.figure(figsize=(10, 7))
    
    # 确保 X 轴按平均物理 SNR 从小到大排序
    df_sorted = df.sort_values(by='physical_snr_db')
    x_data = df_sorted['physical_snr_db']

    # 传统基线
    plt.plot(x_data, df_sorted[f'jpeg_{metric_suffix}'], 
             marker='s', linestyle='--', color='gray', linewidth=2, markersize=8, 
             label='Traditional SSCC (JPEG + CC)')

    # DL-SSCC 强基线
    plt.plot(x_data, df_sorted[f'dlsscc_{metric_suffix}'], 
             marker='^', linestyle='--', color='blue', linewidth=2, markersize=8, 
             label='DL-SSCC (CompressAI + CC)')

    # [新增] 模拟 JSCC 基线
    plt.plot(x_data, df_sorted[f'analog_jscc_{metric_suffix}'], 
             marker='v', linestyle='-.', color='green', linewidth=2, markersize=8, 
             label='Analog Deep JSCC (Continuous)')
    
    # 我们的系统 (Ours)
    plt.plot(x_data, df_sorted[f'ours_{metric_suffix}'], 
             marker='o', linestyle='-', color='red', linewidth=3, markersize=9, 
             label='Ours (End-to-End Graceful Degradation)')

    plt.xlabel('Average Physical Signal-to-Noise Ratio (SNR) [dB]', fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    
    plt.title(f'{title}', fontsize=15, fontweight='bold')
    plt.xticks(x_data)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # ==========================================
    # 自适应留白与图例防遮挡逻辑
    # ==========================================
    ymin, ymax = plt.ylim()
    if ylim_top is not None:
        plt.ylim(ymin, ylim_top)
    else:
        if lower_is_better:
            # 对于越低越好的指标（LPIPS/FID），信道差(左侧)数值极大，信道好(右侧)数值趋近0
            # 曲线呈现 "\" 走势，顶部需要留出足够空间给图例
            plt.ylim(max(0, ymin - (ymax - ymin) * 0.05), ymax + (ymax - ymin) * 0.25)
        else:
            # 对于越高越好的指标，曲线呈现 "/" 走势
            plt.ylim(ymin, ymax + (ymax - ymin) * 0.18)

    # LPIPS/FID 曲线从高到底走，左上方拥挤，图例放右上；PSNR/SSIM 等放左上
    loc = 'upper right' if lower_is_better else 'upper left'
    plt.legend(fontsize=12, loc=loc, framealpha=0.9, edgecolor='lightgray')
    plt.tight_layout()
    
    filename = f"exp5_fast_rayleigh_chart_{filename_suffix}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✅ 成功生成图表: {filename}")
    plt.close()

def main():
    print("🚀 [Step 3] 启动实验 5：瑞利快衰落信道可视化绘图 (6指标全覆盖)")
    
    # 扫描实验 5 的专属结果文件
    csv_files = glob.glob("EXP5_FAST_RAYLEIGH_METRICS_*.csv")
    if not csv_files:
        print("❌ 找不到 EXP5_FAST_RAYLEIGH_METRICS_*.csv！请先运行 Step 2 评估脚本。")
        return

    # 跨平台稳定的基于文件名的字典序排序
    latest_csv = sorted(csv_files)[-1]
    print(f"📦 正在加载最新汇总数据: {latest_csv}\n")
    df = pd.read_csv(latest_csv)

    # 1. 绘制 PSNR 对比图
    plot_metric_vs_avg_snr(
        df=df, metric_suffix='psnr', 
        title='Reconstruction Quality vs. Average SNR (Fast Rayleigh Fading)', 
        ylabel='PSNR [dB]', filename_suffix='PSNR'
    )

    # 2. 绘制 SSIM 对比图 (应用上限锁定防遮挡)
    plot_metric_vs_avg_snr(
        df=df, metric_suffix='ssim', 
        title='Structural Similarity vs. Average SNR (Fast Rayleigh Fading)', 
        ylabel='SSIM', filename_suffix='SSIM',
        ylim_top=1.05  # SSIM 理论上限为 1.0
    )

    # 3. 绘制 S-SNR 对比图
    plot_metric_vs_avg_snr(
        df=df, metric_suffix='ssnr', 
        title='Semantic Signal-to-Noise Ratio vs. Average SNR (Fast Rayleigh)', 
        ylabel='S-SNR [dB]', filename_suffix='SSNR'
    )
    
    # 4. 绘制 MS-SSIM 对比图
    plot_metric_vs_avg_snr(
        df=df, metric_suffix='ms_ssim', 
        title='Multi-Scale SSIM vs. Average SNR (Fast Rayleigh Fading)', 
        ylabel='MS-SSIM', filename_suffix='MSSSIM',
        ylim_top=1.05  # MS-SSIM 理论上限同样为 1.0
    )
    
    # 5. 绘制 LPIPS 对比图
    plot_metric_vs_avg_snr(
        df=df, metric_suffix='lpips', 
        title='LPIPS vs. Average SNR (Fast Rayleigh Fading)', 
        ylabel='LPIPS (Lower is Better)', filename_suffix='LPIPS',
        lower_is_better=True
    )
    
    # 6. 绘制 FID 对比图
    plot_metric_vs_avg_snr(
        df=df, metric_suffix='fid', 
        title='Frechet Inception Distance vs. Average SNR (Fast Rayleigh)', 
        ylabel='FID (Lower is Better)', filename_suffix='FID',
        lower_is_better=True
    )

    print("\n🎉 全部绘图完成！共生成 6 张快衰落对比图表。请查看当前目录下的 exp5_fast_rayleigh_chart_*.png。")

if __name__ == "__main__":
    main()