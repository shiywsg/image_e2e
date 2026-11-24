import os
import subprocess
import re
import pandas as pd
import json
from datetime import datetime

def run_command_and_extract_metrics(cmd_list):
    """执行指令并从终端输出抓取 6 大核心指标 (终极防弹防鲁棒版)"""
    result = subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='ignore')
    output = result.stdout
    
    # 1. 核心修复：用正则洗掉所有的 ANSI 颜色控制字符 (防终端污染)
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    clean_output = ansi_escape.sub('', output)
    
    # 2. 扩展正则，精准抓取 6 项指标 (增强对 inf / nan / 科学计数法的健壮匹配)
    psnr_match = re.search(r"平均 PSNR:\s+([0-9.a-zA-Z-]+)\s+dB", clean_output)
    ssim_match = re.search(r"平均 SSIM:\s+([0-9.a-zA-Z-]+)", clean_output)
    ms_ssim_match = re.search(r"平均 MS-SSIM:\s+([0-9.a-zA-Z-]+)", clean_output)
    lpips_match = re.search(r"平均 LPIPS:\s+([0-9.a-zA-Z-]+)", clean_output)
    fid_match = re.search(r"全局 FID:\s+([0-9.a-zA-Z-]+)", clean_output)
    ssnr_match = re.search(r"平均 S-SNR:\s+([0-9.a-zA-Z-]+)\s+dB", clean_output)
    
    # 增加告警打印，如果有没抓到的情况，终端会直接标黄提示
    if not (psnr_match and ssim_match and ms_ssim_match and lpips_match and fid_match and ssnr_match):
        print(f"    \033[93m[警告] 数据抓取可能存在遗漏！请检查 evaluator.py 的单次输出。\033[0m")
    
    def safe_float(match_obj):
        if not match_obj: 
            return 0.0
        val_str = match_obj.group(1).lower()
        if 'inf' in val_str: 
            return float('inf')
        if 'nan' in val_str: 
            return float('nan')
        try:
            return float(match_obj.group(1))
        except ValueError:
            return 0.0

    return {
        "psnr": safe_float(psnr_match),
        "ssim": safe_float(ssim_match),
        "ms_ssim": safe_float(ms_ssim_match),
        "lpips": safe_float(lpips_match),
        "fid": safe_float(fid_match),
        "ssnr": safe_float(ssnr_match)
    }

def main():
    print("🚀 [Step 2] 启动重构实验 1：AWGN 物理干扰评估与数据萃取 (已集成 Analog Deep JSCC 基线)")
    
    # 1. 加载配置
    if not os.path.exists('config_exp1.json'):
        print("❌ [错误] 未能在当前目录下找到 config_exp1.json，请检查执行路径。")
        return
        
    with open('config_exp1.json', 'r', encoding='utf-8') as f:
        EXP1_CFG = json.load(f)["experiment1_awgn_params"]

    CLEAN_REF_DIR = EXP1_CFG["tx_source_dir"]
    OUTPUT_BASE = EXP1_CFG["output_base_dir"]
    snr_levels = EXP1_CFG["snr_levels"]

    summary_results = []
    total_snr = len(snr_levels)

    # 支持的四大系统：jpeg, dl_sscc, ours, analog_jscc
    target_systems = ['jpeg', 'dl_sscc', 'ours', 'analog_jscc']

    for i, snr in enumerate(snr_levels, 1):
        print(f"\n{'='*50}\n📊 进度 [{i}/{total_snr}] | 正在评估 物理信噪比: {snr}dB")
        
        dst_base = os.path.join(OUTPUT_BASE, f"snr_{snr}")
        metrics = {}

        for sys_name in target_systems:
            sys_dir = os.path.join(dst_base, sys_name)
            print(f"  -> 评估 {sys_name.upper()} 系统...")
            
            if not os.path.exists(sys_dir) or not os.listdir(sys_dir):
                print(f"    \033[93m[警告] 未找到 {sys_name} 的重建图片数据，此项指标归零。\033[0m")
                metrics[sys_name] = {
                    "psnr": 0.0, "ssim": 0.0, "ms_ssim": 0.0, 
                    "lpips": 0.0, "fid": 0.0, "ssnr": 0.0
                }
                continue

            # 调用评测脚本 evaluator.py 
            cmd = [
                "python", "evaluator.py", 
                "--src", CLEAN_REF_DIR, 
                "--dst", sys_dir, 
                "--out", f"tmp_eval_exp1_{sys_name}"
            ]
            metrics[sys_name] = run_command_and_extract_metrics(cmd)

        # 整理当前 SNR 等级下的所有系统数据
        row_data = {"physical_snr_db": snr}
        for sys_name in target_systems:
            # 兼容处理：将 dl_sscc 转换为最终表格要求的 dlsscc_ 前缀
            prefix = "dlsscc_" if sys_name == "dl_sscc" else f"{sys_name}_"
            
            row_data[f"{prefix}psnr"] = metrics[sys_name]["psnr"]
            row_data[f"{prefix}ssim"] = metrics[sys_name]["ssim"]
            row_data[f"{prefix}ms_ssim"] = metrics[sys_name]["ms_ssim"]
            row_data[f"{prefix}lpips"] = metrics[sys_name]["lpips"]
            row_data[f"{prefix}fid"] = metrics[sys_name]["fid"]
            row_data[f"{prefix}ssnr"] = metrics[sys_name]["ssnr"]

        summary_results.append(row_data)

    # 输出汇总表
    df_summary = pd.DataFrame(summary_results)
    
    # 严格控制列的输出顺序：物理SNR -> JPEG 6指标 -> DLSSCC 6指标 -> OURS 6指标 -> Analog_JSCC 6指标
    columns_order = ["physical_snr_db"]
    for prefix in ["jpeg_", "dlsscc_", "ours_", "analog_jscc_"]:
        for metric in ["psnr", "ssim", "ms_ssim", "lpips", "fid", "ssnr"]:
            columns_order.append(f"{prefix}{metric}")
            
    # 容错处理：确保所有列都在 DataFrame 中
    for col in columns_order:
        if col not in df_summary.columns:
            df_summary[col] = 0.0
            
    df_summary = df_summary[columns_order]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    final_csv = f"EXP1_AWGN_METRICS_{timestamp}.csv"
    df_summary.to_csv(final_csv, index=False)
    
    print("\n🎉 评估全部完成！AWGN 干扰数据 (四大系统 × 6项核心指标) 已成功萃取。")
    print(f"✅ 汇总数据已成功落盘至: {final_csv}")

if __name__ == "__main__":
    main()