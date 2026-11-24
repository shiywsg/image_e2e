import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import json

# ==========================================
# 1. 样式字典：为消融实验的 4 个版本定义统一的可视化风格
# ==========================================
ABLATION_STYLES = {
    "full": {
        "label": "Group A (Full): Ternary + Dynamic T",
        "color": "#d62728", # 红色
        "marker": "o",
        "linestyle": "-",
        "linewidth": 3,
        "markersize": 9
    },
    "no_dynamic": {
        "label": "Group B: Ternary + Fixed T=1",
        "color": "#1f77b4", # 蓝色
        "marker": "s",
        "linestyle": "--",
        "linewidth": 2.5,
        "markersize": 8
    },
    "no_ternary": {
        "label": "Group C: Hard Decision + Dynamic T",
        "color": "#2ca02c", # 绿色
        "marker": "^",
        "linestyle": "-.",
        "linewidth": 2.5,
        "markersize": 8
    },
    "base": {
        "label": "Group D (Base): Hard Decision + Fixed T=1",
        "color": "#ff7f0e", # 橙色
        "marker": "D",
        "linestyle": ":",
        "linewidth": 2.5,
        "markersize": 8
    }
}

def load_ablation_data(config_file="ablation_config.json"):
    """读取配置文件并加载 4 个版本的数据"""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"❌ 找不到配置文件 {config_file}！请检查路径。")
        
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
        
    dfs = {}
    for version_key, dir_path in config["ablation_dirs"].items():
        # 兼容跨平台路径查找
        search_pattern = os.path.join(dir_path, "EXP1_AWGN_METRICS_*.csv")
        csv_files = glob.glob(search_pattern)
        
        if not csv_files:
            print(f"⚠️ [警告] 在 {dir_path} 下找不到 EXP1_AWGN_METRICS_*.csv 文件，跳过 {version_key}。")
            continue
            
        # 核心修复：使用字典序排序代替 getctime，确保跨平台绝对稳定地获取最新文件
        latest_csv = sorted(csv_files)[-1]
        print(f"📦 加载 [{version_key}] 数据: {latest_csv}")
        
        # 读取并按 SNR 从小到大排序
        df = pd.read_csv(latest_csv).sort_values(by='physical_snr_db')
        dfs[version_key] = df
        
    return dfs

def export_aggregated_data(dfs_dict, output_csv="ablation_exp1_aggregated_data.csv"):
    """将4个版本的绘图数据提取并合并为一个CSV文件保存"""
    if not dfs_dict:
        return
        
    # 以第一个加载的版本为基准获取 SNR 序列
    first_key = list(dfs_dict.keys())[0]
    agg_df = pd.DataFrame({'physical_snr_db': dfs_dict[first_key]['physical_snr_db'].reset_index(drop=True)})
    
    # 全面覆盖 6 项核心指标
    metrics = ['psnr', 'ssim', 'ms_ssim', 'lpips', 'fid', 'ssnr']
    
    # 遍历所有版本，提取核心列并合并
    for version_key, df in dfs_dict.items():
        if version_key not in ABLATION_STYLES:
            continue
            
        for metric in metrics:
            col_name = f'ours_{metric}'
            if col_name in df.columns:
                # 重命名列名以区分不同版本 (例如: full_psnr, no_dynamic_ssim)
                new_col_name = f"{version_key}_{metric}"
                temp_df = df[['physical_snr_db', col_name]].rename(columns={col_name: new_col_name})
                # 以 physical_snr_db 为基准进行左连接，确保数据对齐
                agg_df = pd.merge(agg_df, temp_df, on='physical_snr_db', how='left')
                
    agg_df.to_csv(output_csv, index=False)
    print(f"\n💾 已成功将合并后的绘图数据 (6 项指标) 导出至: {output_csv}")

def plot_ablation_metric(dfs_dict, metric_suffix, title, ylabel, filename_suffix, ylim_top=None, lower_is_better=False):
    """消融实验多曲线绘制函数 (全指标防遮挡优化版)"""
    plt.figure(figsize=(10, 7))
    
    # 遍历绘制 4 条曲线（消融实验只关注 ours 各个分支的数据对比）
    for version_key, df in dfs_dict.items():
        if version_key not in ABLATION_STYLES:
            continue
            
        style = ABLATION_STYLES[version_key]
        x_data = df['physical_snr_db']
        y_data = df[f'ours_{metric_suffix}']  # 【核心】：只取本系统的数据，忽略基线
        
        plt.plot(x_data, y_data,
                 label=style["label"],
                 color=style["color"],
                 marker=style["marker"],
                 linestyle=style["linestyle"],
                 linewidth=style["linewidth"],
                 markersize=style["markersize"])

    plt.xlabel('Physical Signal-to-Noise Ratio (SNR) [dB]', fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    plt.title(f'{title}', fontsize=15, fontweight='bold')
    
    # 确保 X 轴刻度对齐
    if dfs_dict:
        first_key = list(dfs_dict.keys())[0]
        plt.xticks(dfs_dict[first_key]['physical_snr_db'])
        
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # ==========================================
    # 自适应留白与图例排版逻辑
    # ==========================================
    ymin, ymax = plt.ylim()
    if ylim_top is not None:
        plt.ylim(ymin, ylim_top)
    else:
        if lower_is_better:
            # LPIPS/FID: 信道变好(SNR变大)时数值下降，呈 "\" 趋势
            # 顶部需要留出足够空间给图例，并防止负值
            plt.ylim(max(0, ymin - (ymax - ymin) * 0.05), ymax + (ymax - ymin) * 0.25)
        else:
            # PSNR/SSNR: 信道变好(SNR变大)时数值上升，呈 "/" 趋势
            # 自适应留白：将顶部拉高 25%
            plt.ylim(ymin, ymax + (ymax - ymin) * 0.25)

    # LPIPS/FID 曲线从高到底走，左上方拥挤，图例放右上；PSNR等放左上
    loc = 'upper right' if lower_is_better else 'upper left'
    plt.legend(fontsize=12, loc=loc, framealpha=0.9, edgecolor='lightgray')
    plt.tight_layout()
    
    filename = f"ablation_exp1_{filename_suffix}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✅ 成功生成消融实验图表: {filename}")
    plt.close()

def main():
    print("🚀 启动消融实验 1 (Ablation Study) 可视化绘图程序 (6指标全覆盖)\n")
    
    # 1. 加载 4 个版本的数据
    try:
        dfs_dict = load_ablation_data("ablation_config.json")
    except Exception as e:
        print(e)
        return
        
    if not dfs_dict:
        print("❌ 未加载到任何有效数据，程序退出。")
        return

    # 2. 将数据提取并汇总为单独的 CSV 文件落盘
    export_aggregated_data(dfs_dict)
    print("\n📊 开始绘制对比图表...")

    # 3. 绘制 PSNR 对比图
    plot_ablation_metric(
        dfs_dict=dfs_dict,
        metric_suffix='psnr',
        title='Ablation Study: Reconstruction Quality (PSNR) vs. Physical SNR',
        ylabel='PSNR [dB]',
        filename_suffix='PSNR'
    )

    # 4. 绘制 SSIM 对比图
    plot_ablation_metric(
        dfs_dict=dfs_dict,
        metric_suffix='ssim',
        title='Ablation Study: Structural Similarity (SSIM) vs. Physical SNR',
        ylabel='SSIM',
        filename_suffix='SSIM',
        ylim_top=1.05  # SSIM 理论上限为 1.0
    )

    # 5. 绘制 S-SNR 对比图
    plot_ablation_metric(
        dfs_dict=dfs_dict,
        metric_suffix='ssnr',
        title='Ablation Study: Semantic SNR vs. Physical SNR',
        ylabel='S-SNR [dB]',
        filename_suffix='SSNR'
    )
    
    # 6. 绘制 MS-SSIM 对比图
    plot_ablation_metric(
        dfs_dict=dfs_dict,
        metric_suffix='ms_ssim',
        title='Ablation Study: Multi-Scale SSIM vs. Physical SNR',
        ylabel='MS-SSIM',
        filename_suffix='MSSSIM',
        ylim_top=1.05  # MS-SSIM 理论上限为 1.0
    )
    
    # 7. 绘制 LPIPS 对比图
    plot_ablation_metric(
        dfs_dict=dfs_dict,
        metric_suffix='lpips',
        title='Ablation Study: LPIPS vs. Physical SNR',
        ylabel='LPIPS (Lower is Better)',
        filename_suffix='LPIPS',
        lower_is_better=True
    )
    
    # 8. 绘制 FID 对比图
    plot_ablation_metric(
        dfs_dict=dfs_dict,
        metric_suffix='fid',
        title='Ablation Study: FID vs. Physical SNR',
        ylabel='FID (Lower is Better)',
        filename_suffix='FID',
        lower_is_better=True
    )

    print("\n🎉 全部消融实验图表绘图完成！共生成 6 张图表。请查看生成的 ablation_exp1_*.png 文件和合并的 CSV 文件。")

if __name__ == "__main__":
    main()