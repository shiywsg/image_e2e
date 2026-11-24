import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def plot_metric_vs_impulse(df, metric_suffix, title, ylabel, filename_suffix, ylim_top=None, lower_is_better=False):
    """通用的脉冲概率扫描绘图函数 (含 Analog JSCC 全指标防遮挡优化版)"""
    plt.figure(figsize=(10, 7))
    
    # 确保 X 轴按概率从小到大排序
    df_sorted = df.sort_values(by='impulse_prob_percent')
    x_data = df_sorted['impulse_prob_percent']

    # 传统基线
    plt.plot(x_data, df_sorted[f'jpeg_{metric_suffix}'], 
             marker='s', linestyle='--', color='gray', linewidth=2, markersize=8, 
             label='Traditional SSCC (JPEG + CC)')

    # DL-SSCC 基线
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
             label='Ours (CNN Spatial Denoising)')

    plt.xlabel('Burst Impulse Probability (%)', fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    
    fixed_snr = df['fixed_snr_db'].iloc[0]
    plt.title(f'{title}\nFixed Background SNR = {fixed_snr}dB', fontsize=15, fontweight='bold')
    
    plt.xticks(x_data)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # ==========================================
    # 🌟 图例防遮挡核心逻辑 (Headroom Adjustment)
    # ==========================================
    ymin, ymax = plt.ylim()
    if ylim_top is not None:
        # 如果强制指定了上限 (例如 SSIM 指定为 1.05)
        plt.ylim(ymin, ylim_top)
    else:
        if lower_is_better:
            # 越低越好的指标（如LPIPS/FID），曲线从左下升至右上
            # 顶部需要给图例留出更充裕的空间 (25%)，并且避免 Y 轴变成负数
            plt.ylim(max(0, ymin - (ymax - ymin) * 0.05), ymax + (ymax - ymin) * 0.25)
        else:
            # 越高越好的指标（如PSNR），曲线从左上降至右下，自动给顶部增加 18% 留白
            plt.ylim(ymin, ymax + (ymax - ymin) * 0.18)

    # 图例位置动态指派
    # 曲线下降(\)放右上，曲线上升(/)放左上
    loc = 'upper left' if lower_is_better else 'upper right'

    # 绘制图例 (添加半透明白色底板，防止干扰网格)
    plt.legend(fontsize=12, loc=loc, framealpha=0.9, edgecolor='lightgray')
    plt.tight_layout()
    
    filename = f"exp2_impulse_chart_{filename_suffix}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✅ 成功生成图表: {filename}")
    plt.close()

def main():
    print("🚀 [Step 3] 启动新实验2：突发脉冲干扰可视化绘图 (集成 Analog JSCC 6指标全覆盖)")
    
    csv_files = glob.glob("EXP2_IMPULSE_METRICS_*.csv")
    if not csv_files:
        print("❌ 找不到 EXP2_IMPULSE_METRICS_*.csv！请先运行 Step 2 评估脚本。")
        return

    # ==========================================
    # 核心修复：抛弃 getctime，直接使用基于文件名的字典序排序
    # ==========================================
    latest_csv = sorted(csv_files)[-1]
    
    print(f"📦 正在加载最新汇总数据: {latest_csv}\n")
    df = pd.read_csv(latest_csv)

    # 1. 绘制 PSNR 对比图
    plot_metric_vs_impulse(
        df=df, metric_suffix='psnr', 
        title='Reconstruction Quality vs. Burst Impulse Noise', 
        ylabel='PSNR [dB]', filename_suffix='PSNR'
    )

    # 2. 绘制 SSIM 对比图 (应用专属修正)
    plot_metric_vs_impulse(
        df=df, metric_suffix='ssim', 
        title='Structural Similarity vs. Burst Impulse Noise', 
        ylabel='SSIM', filename_suffix='SSIM',
        ylim_top=1.05  # 强制最高点为 1.05
    )

    # 3. 绘制 S-SNR 对比图
    plot_metric_vs_impulse(
        df=df, metric_suffix='ssnr', 
        title='Semantic Signal-to-Noise Ratio vs. Burst Impulse Noise', 
        ylabel='S-SNR [dB]', filename_suffix='SSNR'
    )

    # 4. 绘制 MS-SSIM 对比图
    plot_metric_vs_impulse(
        df=df, metric_suffix='ms_ssim', 
        title='Multi-Scale SSIM vs. Burst Impulse Noise', 
        ylabel='MS-SSIM', filename_suffix='MSSSIM',
        ylim_top=1.05  # MS-SSIM 理论上限同样为 1.0
    )
    
    # 5. 绘制 LPIPS 对比图
    plot_metric_vs_impulse(
        df=df, metric_suffix='lpips', 
        title='Learned Perceptual Image Patch Similarity vs. Burst Impulse Noise', 
        ylabel='LPIPS (Lower is Better)', filename_suffix='LPIPS',
        lower_is_better=True
    )
    
    # 6. 绘制 FID 对比图
    plot_metric_vs_impulse(
        df=df, metric_suffix='fid', 
        title='Frechet Inception Distance vs. Burst Impulse Noise', 
        ylabel='FID (Lower is Better)', filename_suffix='FID',
        lower_is_better=True
    )

    print("\n🎉 全部绘图完成！共生成 6 张图表，请查看更新后的 exp2_impulse_chart_*.png。")

if __name__ == "__main__":
    main()
