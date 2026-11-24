import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def plot_metric_vs_sir(df_subset, i_type, fixed_snr, metric_suffix, title, ylabel, filename_suffix, ylim_top=None, lower_is_better=False):
    """通用的 SIR 扫描绘图函数 (含 Analog JSCC 全指标防遮挡优化版)"""
    plt.figure(figsize=(10, 7))
    
    # 确保 X 轴 (SIR) 按从小到大排序 (左边是干扰极大，右边是干扰极小)
    df_sorted = df_subset.sort_values(by='collision_sir')
    x_data = df_sorted['collision_sir']

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
             label='Ours (Semantic Orthogonality)')

    plt.xlabel('Signal-to-Interference Ratio (SIR) [dB]', fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    
    # 生成人性化的标题 (标注当前的固定环境)
    type_name_map = {
        'same_domain': 'Same-Domain (Images)',
        'out_of_domain': 'Out-of-Domain (Medical/Text)',
        'digital_legacy': 'Legacy Digital (16QAM)'
    }
    pretty_type = type_name_map.get(i_type, i_type)
    plt.title(f'{title}\nInterference: {pretty_type} | Fixed SNR = {fixed_snr}dB', fontsize=15, fontweight='bold')
    
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
            # 越低越好的指标（LPIPS/FID）：SIR低时(左)数值大，SIR高时(右)数值低。曲线从左上走向右下。
            # 图例适合放在右上角，顶部需留出额外空间防止压线。
            plt.ylim(max(0, ymin - (ymax - ymin) * 0.05), ymax + (ymax - ymin) * 0.25)
        else:
            # 越高越好的指标（PSNR）：SIR低时(左)数值小，SIR高时(右)数值大。曲线从左下走向右上。
            # 图例适合放在左上角，顶部自动增加 18% 留白。
            plt.ylim(ymin, ymax + (ymax - ymin) * 0.18)

    # 动态分配图例位置
    loc = 'upper right' if lower_is_better else 'upper left'
    plt.legend(fontsize=12, loc=loc, framealpha=0.9, edgecolor='lightgray')
    plt.tight_layout()
    
    filename = f"exp3_chart_{i_type}_{filename_suffix}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✅ 成功生成图表: {filename}")
    plt.close()

def main():
    print("🚀 [Step 3] 启动实验3：多维场景可视化生成 (集成 Analog JSCC 6指标全覆盖)")
    
    csv_files = glob.glob("EXP3_ALL_METRICS_*.csv")
    if not csv_files:
        print("❌ 找不到 EXP3_ALL_METRICS_*.csv！请先运行 Step 2 评估脚本。")
        return

    # ==========================================
    # 核心修复：抛弃 getctime，直接使用基于文件名的字典序排序
    # ==========================================
    latest_csv = sorted(csv_files)[-1]
    
    print(f"📦 正在加载最新汇总数据: {latest_csv}\n")
    df = pd.read_csv(latest_csv)

    # 我们选择一个固定的物理 SNR 来展示 (通常选取最高信道 10.0dB)
    # 以凸显完全是"并发碰撞"造成的降级，而非高斯底噪
    TARGET_SNR = 10.0
    
    # 获取所有的干扰类型
    i_types = df['interference_type'].unique()

    for i_type in i_types:
        print(f"\n🎨 正在绘制干扰场景: {i_type} (Fixed SNR={TARGET_SNR}dB)")
        
        # 过滤出当前场景的数据
        df_subset = df[(df['interference_type'] == i_type) & (df['physical_snr'] == TARGET_SNR)]
        
        if df_subset.empty:
            print(f"⚠️ 未找到 {i_type} 在 SNR={TARGET_SNR} 下的数据，跳过。")
            continue

        # 1. 绘制 PSNR 对比图
        plot_metric_vs_sir(
            df_subset=df_subset, i_type=i_type, fixed_snr=TARGET_SNR,
            metric_suffix='psnr', title='Reconstruction Quality (PSNR) vs. Collision', 
            ylabel='PSNR [dB]', filename_suffix='PSNR'
        )

        # 2. 绘制 SSIM 对比图
        plot_metric_vs_sir(
            df_subset=df_subset, i_type=i_type, fixed_snr=TARGET_SNR,
            metric_suffix='ssim', title='Structural Similarity (SSIM) vs. Collision', 
            ylabel='SSIM', filename_suffix='SSIM', ylim_top=1.05
        )

        # 3. 绘制 S-SNR 对比图
        plot_metric_vs_sir(
            df_subset=df_subset, i_type=i_type, fixed_snr=TARGET_SNR,
            metric_suffix='ssnr', title='Semantic Signal-to-Noise Ratio (S-SNR) vs. Collision', 
            ylabel='S-SNR [dB]', filename_suffix='SSNR'
        )
        
        # 4. 绘制 MS-SSIM 对比图
        plot_metric_vs_sir(
            df_subset=df_subset, i_type=i_type, fixed_snr=TARGET_SNR,
            metric_suffix='ms_ssim', title='Multi-Scale SSIM vs. Collision', 
            ylabel='MS-SSIM', filename_suffix='MSSSIM', ylim_top=1.05
        )
        
        # 5. 绘制 LPIPS 对比图
        plot_metric_vs_sir(
            df_subset=df_subset, i_type=i_type, fixed_snr=TARGET_SNR,
            metric_suffix='lpips', title='Learned Perceptual Image Patch Similarity vs. Collision', 
            ylabel='LPIPS (Lower is Better)', filename_suffix='LPIPS', lower_is_better=True
        )
        
        # 6. 绘制 FID 对比图
        plot_metric_vs_sir(
            df_subset=df_subset, i_type=i_type, fixed_snr=TARGET_SNR,
            metric_suffix='fid', title='Frechet Inception Distance vs. Collision', 
            ylabel='FID (Lower is Better)', filename_suffix='FID', lower_is_better=True
        )

    print("\n🎉 全部绘图完成！请在当前目录下查看所有 exp3_chart_*.png 图片。")

if __name__ == "__main__":
    main()
