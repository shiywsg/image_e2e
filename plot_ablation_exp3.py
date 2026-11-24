import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import json

# ==========================================
# 1. 样式字典
# ==========================================
ABLATION_STYLES = {
    "full": {"label": "Group A (Full): Ternary + Dynamic T", "color": "#d62728", "marker": "o", "linestyle": "-", "linewidth": 3, "markersize": 9},
    "no_dynamic": {"label": "Group B: Ternary + Fixed T=1", "color": "#1f77b4", "marker": "s", "linestyle": "--", "linewidth": 2.5, "markersize": 8},
    "no_ternary": {"label": "Group C: Hard Decision + Dynamic T", "color": "#2ca02c", "marker": "^", "linestyle": "-.", "linewidth": 2.5, "markersize": 8},
    "base": {"label": "Group D (Base): Hard Decision + Fixed T=1", "color": "#ff7f0e", "marker": "D", "linestyle": ":", "linewidth": 2.5, "markersize": 8}
}

def load_ablation_data(config_file="ablation_config.json"):
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"❌ 找不到配置文件 {config_file}！")
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    dfs = {}
    for version_key, dir_path in config["ablation_dirs"].items():
        search_pattern = os.path.join(dir_path, "EXP3_ALL_METRICS_*.csv")
        csv_files = glob.glob(search_pattern)
        if not csv_files: continue
        latest_csv = sorted(csv_files)[-1]
        print(f"📦 加载 [{version_key}] 数据: {latest_csv}")
        dfs[version_key] = pd.read_csv(latest_csv)
    return dfs

def export_aggregated_data(dfs_dict, output_csv="ablation_exp3_aggregated_data.csv"):
    if not dfs_dict: return
    base_cols = ['interference_type', 'physical_snr', 'collision_sir']
    first_key = list(dfs_dict.keys())[0]
    agg_df = dfs_dict[first_key][base_cols].drop_duplicates().reset_index(drop=True)
    
    metrics = ['psnr', 'ssim', 'ms_ssim', 'lpips', 'fid', 'ssnr']
    for version_key, df in dfs_dict.items():
        if version_key not in ABLATION_STYLES: continue
        for metric in metrics:
            col_name = f'ours_{metric}'
            if col_name in df.columns:
                new_col_name = f"{version_key}_{metric}"
                temp_df = df[base_cols + [col_name]].rename(columns={col_name: new_col_name})
                agg_df = pd.merge(agg_df, temp_df, on=base_cols, how='left')
    agg_df = agg_df.sort_values(by=['interference_type', 'physical_snr', 'collision_sir'])
    agg_df.to_csv(output_csv, index=False)
    print(f"\n💾 已导出实验 3 聚合数据 (6 项指标) 至: {output_csv}")

def plot_ablation_metric_vs_sir(dfs_dict, i_type, fixed_snr, metric_suffix, title, ylabel, filename_suffix, ylim_top=None, lower_is_better=False):
    plt.figure(figsize=(10, 7))
    valid_plot_count = 0
    for version_key, df in dfs_dict.items():
        if version_key not in ABLATION_STYLES: continue
        df_subset = df[(df['interference_type'] == i_type) & (df['physical_snr'] == fixed_snr)]
        if df_subset.empty: continue
        
        df_sorted = df_subset.sort_values(by='collision_sir')
        style = ABLATION_STYLES[version_key]
        plt.plot(df_sorted['collision_sir'], df_sorted[f'ours_{metric_suffix}'], 
                 label=style["label"], color=style["color"], marker=style["marker"], 
                 linestyle=style["linestyle"], linewidth=style["linewidth"], markersize=style["markersize"])
        valid_plot_count += 1

    if valid_plot_count == 0: return

    plt.xlabel('Signal-to-Interference Ratio (SIR) [dB]', fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    
    type_name_map = {'same_domain': 'Same-Domain', 'out_of_domain': 'Out-of-Domain', 'digital_legacy': 'Legacy Digital'}
    plt.title(f'{title}\nInterference: {type_name_map.get(i_type, i_type)} | SNR={fixed_snr}dB', fontsize=15, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 动态防遮挡
    ymin, ymax = plt.ylim()
    if ylim_top is not None: plt.ylim(ymin, ylim_top)
    else:
        if lower_is_better: plt.ylim(max(0, ymin - (ymax - ymin) * 0.05), ymax + (ymax - ymin) * 0.25)
        else: plt.ylim(ymin, ymax + (ymax - ymin) * 0.25)

    loc = 'upper right' if lower_is_better else 'upper left'
    plt.legend(fontsize=12, loc=loc, framealpha=0.9, edgecolor='lightgray')
    plt.tight_layout()
    plt.savefig(f"ablation_exp3_{i_type}_{filename_suffix}.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("🚀 启动消融实验 3 可视化程序 (6指标全覆盖)\n")
    dfs_dict = load_ablation_data("ablation_config.json")
    if not dfs_dict: return
    export_aggregated_data(dfs_dict)
    
    TARGET_SNR = 10.0
    i_types = dfs_dict["full"]['interference_type'].unique() if "full" in dfs_dict else ['same_domain']

    for i_type in i_types:
        args = [(dfs_dict, i_type, TARGET_SNR, 'psnr', 'PSNR', 'PSNR [dB]', 'PSNR'),
                (dfs_dict, i_type, TARGET_SNR, 'ssim', 'SSIM', 'SSIM', 'SSIM', 1.05),
                (dfs_dict, i_type, TARGET_SNR, 'ssnr', 'S-SNR', 'S-SNR [dB]', 'SSNR'),
                (dfs_dict, i_type, TARGET_SNR, 'ms_ssim', 'MS-SSIM', 'MS-SSIM', 'MSSSIM', 1.05),
                (dfs_dict, i_type, TARGET_SNR, 'lpips', 'LPIPS', 'LPIPS (Lower is Better)', 'LPIPS', None, True),
                (dfs_dict, i_type, TARGET_SNR, 'fid', 'FID', 'FID (Lower is Better)', 'FID', None, True)]
        
        for arg in args: plot_ablation_metric_vs_sir(*arg)
    print("\n🎉 全部绘图完成！")

if __name__ == "__main__":
    main()