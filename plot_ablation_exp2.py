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
    """读取配置文件并加载实验 2 的 4 个版本数据"""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"❌ 找不到配置文件 {config_file}！")
        
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
        
    dfs = {}
    for version_key, dir_path in config["ablation_dirs"].items():
        search_pattern = os.path.join(dir_path, "EXP2_IMPULSE_METRICS_*.csv")
        csv_files = glob.glob(search_pattern)
        
        if not csv_files:
            print(f"⚠️ [警告] 在 {dir_path} 下找不到 EXP2_IMPULSE_METRICS_*.csv 文件，跳过 {version_key}。")
            continue
            
        latest_csv = max(csv_files, key=os.path.getctime)
        print(f"📦 加载 [{version_key}] 数据: {latest_csv}")
        
        df = pd.read_csv(latest_csv).sort_values(by='impulse_prob_percent')
        dfs[version_key] = df
        
    return dfs

def export_aggregated_data(dfs_dict, output_csv="ablation_exp2_aggregated_data.csv"):
    """将实验 2 的 4 个版本数据提取并合并为一个 6 指标宽表 CSV"""
    if not dfs_dict:
        return
        
    first_key = list(dfs_dict.keys())[0]
    agg_df = pd.DataFrame({'impulse_prob_percent': dfs_dict[first_key]['impulse_prob_percent'].reset_index(drop=True)})
    
    metrics = ['psnr', 'ssim', 'ms_ssim', 'lpips', 'fid', 'ssnr']
    
    for version_key, df in dfs_dict.items():
        if version_key not in ABLATION_STYLES:
            continue
            
        for metric in metrics:
            col_name = f'ours_{metric}'
            if col_name in df.columns:
                new_col_name = f"{version_key}_{metric}"
                temp_df = df[['impulse_prob_percent', col_name]].rename(columns={col_name: new_col_name})
                agg_df = pd.merge(agg_df, temp_df, on='impulse_prob_percent', how='left')
                
    agg_df.to_csv(output_csv, index=False)
    print(f"\n💾 已成功将合并后的绘图数据 (6 项指标) 导出至: {output_csv}")

def plot_ablation_metric(dfs_dict, metric_suffix, title, ylabel, filename_suffix, ylim_top=None, lower_is_better=False):
    """实验 2 消融多曲线绘制函数 (全指标优化版)"""
    plt.figure(figsize=(10, 7))
    
    for version_key, df in dfs_dict.items():
        if version_key not in ABLATION_STYLES:
            continue
            
        style = ABLATION_STYLES[version_key]
        x_data = df['impulse_prob_percent']
        y_data = df[f'ours_{metric_suffix}']
        
        plt.plot(x_data, y_data, 
                 label=style["label"], 
                 color=style["color"], 
                 marker=style["marker"], 
                 linestyle=style["linestyle"], 
                 linewidth=style["linewidth"], 
                 markersize=style["markersize"])

    plt.xlabel('Impulse Occurrence Probability (%)', fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    plt.title(f'{title}', fontsize=15, fontweight='bold')
    
    if dfs_dict:
        first_key = list(dfs_dict.keys())[0]
        plt.xticks(dfs_dict[first_key]['impulse_prob_percent'])
        
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 动态留白逻辑
    ymin, ymax = plt.ylim()
    if ylim_top is not None:
        plt.ylim(ymin, ylim_top)
    else:
        if lower_is_better:
            # 越低越好（如LPIPS/FID）：曲线随脉冲概率增加而上升，左上为空旷区
            plt.ylim(max(0, ymin - (ymax - ymin) * 0.05), ymax + (ymax - ymin) * 0.25)
        else:
            # 越高越好（如PSNR/SSIM）：曲线随脉冲概率增加而下降，右上为空旷区
            plt.ylim(ymin, ymax + (ymax - ymin) * 0.18)

    # 图例位置自动切换
    loc = 'upper left' if lower_is_better else 'upper right'
    plt.legend(fontsize=12, loc=loc, framealpha=0.9, edgecolor='lightgray')
    plt.tight_layout()
    
    filename = f"ablation_exp2_{filename_suffix}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✅ 成功生成图表: {filename}")
    plt.close()

def main():
    print("🚀 启动消融实验 2 (突发脉冲干扰) 可视化绘图程序 (6指标全覆盖)\n")
    
    try:
        dfs_dict = load_ablation_data("ablation_config.json")
    except Exception as e:
        print(e)
        return
        
    if not dfs_dict:
        return

    export_aggregated_data(dfs_dict)
    print("\n📊 开始绘制对比图表...")

    # 1. PSNR
    plot_ablation_metric(dfs_dict, 'psnr', 'Quality (PSNR) vs. Impulse Prob', 'PSNR [dB]', 'PSNR')
    # 2. SSIM
    plot_ablation_metric(dfs_dict, 'ssim', 'Structural Similarity vs. Impulse Prob', 'SSIM', 'SSIM', ylim_top=1.05)
    # 3. SSNR
    plot_ablation_metric(dfs_dict, 'ssnr', 'Semantic SNR vs. Impulse Prob', 'S-SNR [dB]', 'SSNR')
    # 4. MS-SSIM
    plot_ablation_metric(dfs_dict, 'ms_ssim', 'Multi-Scale SSIM vs. Impulse Prob', 'MS-SSIM', 'MSSSIM', ylim_top=1.05)
    # 5. LPIPS (越低越好)
    plot_ablation_metric(dfs_dict, 'lpips', 'LPIPS vs. Impulse Prob', 'LPIPS (Lower is Better)', 'LPIPS', lower_is_better=True)
    # 6. FID (越低越好)
    plot_ablation_metric(dfs_dict, 'fid', 'FID vs. Impulse Prob', 'FID (Lower is Better)', 'FID', lower_is_better=True)

    print("\n🎉 全部图表绘图完成！请查看 exp2 目录下生成的图片。")

if __name__ == "__main__":
    main()