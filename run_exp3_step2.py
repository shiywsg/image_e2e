import os
import subprocess
import re
import pandas as pd
import json
from datetime import datetime

def run_command_and_extract_metrics(cmd_list):
    """执行指令并从屏幕输出抓取 6 大核心指标，包含防崩溃机制 (防弹版)"""
    # 忽略终端颜色代码等引起的编码错误
    result = subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='ignore')
    output = result.stdout
    
    # 1. 核心修复：用正则洗掉所有的 ANSI 颜色控制字符
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\\[0-?]*[ -/]*[@-~])')
    clean_output = ansi_escape.sub('', output)
    
    # 2. 核心增强：使用合并了英文字母(支持 inf/nan) 的宽松正则
    psnr_match = re.search(r"平均\s*PSNR:.*?([0-9.a-zA-Z-]+)\s*dB", clean_output, re.IGNORECASE)
    ssim_match = re.search(r"平均\s*SSIM:.*?([0-9.a-zA-Z-]+)", clean_output, re.IGNORECASE)
    ms_ssim_match = re.search(r"平均\s*MS-SSIM:.*?([0-9.a-zA-Z-]+)", clean_output, re.IGNORECASE)
    lpips_match = re.search(r"平均\s*LPIPS:.*?([0-9.a-zA-Z-]+)", clean_output, re.IGNORECASE)
    fid_match = re.search(r"全局\s*FID:.*?([0-9.a-zA-Z-]+)", clean_output, re.IGNORECASE)
    ssnr_match = re.search(r"平均\s*S-SNR:.*?([0-9.a-zA-Z-]+)\s*dB", clean_output, re.IGNORECASE)
    
    # 抓取失败警告
    if not (psnr_match and ssim_match and ms_ssim_match and lpips_match and fid_match and ssnr_match):
        print(f"    \033[93m[警告] 数据抓取可能存在遗漏，请检查子程序输出格式。\033[0m")
    
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
    print("🚀 [Step 2] 启动实验3：多维质量评估与数据萃取 (包含 Analog JSCC，四大系统 6 指标全覆盖)")
    
    # 读取配置
    if not os.path.exists('config_exp3.json'):
        print("❌ [错误] 未能在当前目录下找到 config_exp3.json")
        return
        
    with open('config_exp3.json', 'r', encoding='utf-8') as f:
        EXP3_CFG = json.load(f)["experiment3_params"]

    CLEAN_REF_DIR = EXP3_CFG["tx_source_dir"]
    OUTPUT_BASE = EXP3_CFG["output_base_dir"]
    snr_levels = EXP3_CFG["snr_levels"]
    sir_levels = EXP3_CFG["sir_levels"]
    i_types = EXP3_CFG["interference_types"]

    summary_results = []
    total_combinations = len(i_types) * len(snr_levels) * len(sir_levels)
    current_count = 0

    target_systems = ['jpeg', 'dl_sscc', 'ours', 'analog_jscc']

    for i_type in i_types:
        for snr in snr_levels:
            for sir in sir_levels:
                current_count += 1
                print(f"\n{'='*50}")
                print(f"📊 进度 [{current_count}/{total_combinations}] | 干扰: {i_type} | SNR: {snr}dB | SIR: {sir}dB")
                
                dst_base = os.path.join(OUTPUT_BASE, i_type, f"snr_{snr}_sir_{sir}")
                metrics = {}

                for sys_name in target_systems:
                    sys_dir = os.path.join(dst_base, sys_name)
                    print(f"  -> 正在评估 {sys_name.upper()} 系统...")
                    
                    if not os.path.exists(sys_dir) or not os.listdir(sys_dir):
                        print(f"    \033[93m[警告] 未找到 {sys_name} 数据，记录为 0。\033[0m")
                        metrics[sys_name] = {
                            "psnr": 0.0, "ssim": 0.0, "ms_ssim": 0.0,
                            "lpips": 0.0, "fid": 0.0, "ssnr": 0.0
                        }
                        continue

                    # 调用你现有的高质量 evaluator.py
                    cmd = [
                        "python", "evaluator.py", 
                        "--src", CLEAN_REF_DIR, 
                        "--dst", sys_dir, 
                        "--out", f"tmp_eval_exp3_{sys_name}"
                    ]
                    metrics[sys_name] = run_command_and_extract_metrics(cmd)

                # 组装本概率下的完整行数据 (动态适配所有系统)
                row_data = {
                    "interference_type": i_type,
                    "physical_snr": snr,
                    "collision_sir": sir
                }
                for sys_name in target_systems:
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
    
    # 严格按照规范重排列名，加入 analog_jscc
    columns_order = ["interference_type", "physical_snr", "collision_sir"]
    for prefix in ["jpeg_", "dlsscc_", "ours_", "analog_jscc_"]:
        for metric in ["psnr", "ssim", "ms_ssim", "lpips", "fid", "ssnr"]:
            columns_order.append(f"{prefix}{metric}")
            
    # 容错处理：确保所有列都在 DataFrame 中
    for col in columns_order:
        if col not in df_summary.columns:
            df_summary[col] = 0.0
            
    df_summary = df_summary[columns_order]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    final_csv = f"EXP3_ALL_METRICS_{timestamp}.csv"
    df_summary.to_csv(final_csv, index=False)
    
    print("\n🎉 评估全部完成！多维数据矩阵 (四大系统 × 6 项指标) 已生成。")
    print(f"✅ 文件保存至: {final_csv}")
    print("\n下一步：请运行 run_exp3_step3.py 进行可视化绘图！")

if __name__ == "__main__":
    main()
