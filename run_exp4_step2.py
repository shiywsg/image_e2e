import os
import subprocess
import re
import pandas as pd
import json
from datetime import datetime

def run_command_and_extract_metrics(cmd_list):
    """执行指令并从终端输出抓取 6 大核心指标 (终极防弹版)"""
    result = subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='ignore')
    output = result.stdout
    
    # 1. 核心修复：用正则洗掉所有的 ANSI 颜色控制字符 (防终端污染)
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\\[0-?]*[ -/]*[@-~])')
    clean_output = ansi_escape.sub('', output)
    
    # 2. 扩展正则，精准抓取 6 项指标 (支持 inf/nan 边缘值)
    psnr_match = re.search(r"平均\s*PSNR:.*?([0-9.a-zA-Z-]+)\s*dB", clean_output, re.IGNORECASE)
    ssim_match = re.search(r"平均\s*SSIM:.*?([0-9.a-zA-Z-]+)", clean_output, re.IGNORECASE)
    ms_ssim_match = re.search(r"平均\s*MS-SSIM:.*?([0-9.a-zA-Z-]+)", clean_output, re.IGNORECASE)
    lpips_match = re.search(r"平均\s*LPIPS:.*?([0-9.a-zA-Z-]+)", clean_output, re.IGNORECASE)
    fid_match = re.search(r"全局\s*FID:.*?([0-9.a-zA-Z-]+)", clean_output, re.IGNORECASE)
    ssnr_match = re.search(r"平均\s*S-SNR:.*?([0-9.a-zA-Z-]+)\s*dB", clean_output, re.IGNORECASE)
    
    # 增加告警打印
    if not (psnr_match and ssim_match and ms_ssim_match and lpips_match and fid_match and ssnr_match):
        print(f"    \033[93m[警告] 数据抓取遗漏！请检查 evaluator.py 的输出格式。\033[0m")
    
    def safe_float(match_obj):
        if not match_obj: return 0.0
        val_str = match_obj.group(1).lower()
        if 'inf' in val_str: return float('inf')
        if 'nan' in val_str: return float('nan')
        try: return float(match_obj.group(1))
        except ValueError: return 0.0

    return {
        "psnr": safe_float(psnr_match),
        "ssim": safe_float(ssim_match),
        "ms_ssim": safe_float(ms_ssim_match),
        "lpips": safe_float(lpips_match),
        "fid": safe_float(fid_match),
        "ssnr": safe_float(ssnr_match)
    }

def main():
    print("🚀 [Step 2] 启动实验 4：瑞利慢衰落信道 (集成 Analog JSCC，四大系统 6 指标全覆盖)")
    
    # 1. 加载实验 4 专属配置
    try:
        with open('config_exp4.json', 'r', encoding='utf-8') as f:
            EXP4_CFG = json.load(f)["experiment4_rayleigh_params"]
    except FileNotFoundError:
        print("\033[91m[错误] 找不到 config_exp4.json，请确保已运行第一步传输程序。\033[0m")
        return

    CLEAN_REF_DIR = EXP4_CFG["tx_source_dir"]
    OUTPUT_BASE = EXP4_CFG["output_base_dir"]
    avg_snr_levels = EXP4_CFG["snr_levels"]

    summary_results = []
    total_snr = len(avg_snr_levels)
    
    target_systems = ['jpeg', 'dl_sscc', 'ours', 'analog_jscc']

    for i, snr in enumerate(avg_snr_levels, 1):
        print(f"\n{'='*50}\n📊 进度 [{i}/{total_snr}] | 正在评估 平均物理信噪比: {snr}dB")
        
        # 对应第一步生成的存储目录结构
        dst_base = os.path.join(OUTPUT_BASE, f"snr_{snr}")
        metrics = {}

        for sys_name in target_systems:
            sys_dir = os.path.join(dst_base, sys_name)
            print(f"  -> 评估 {sys_name.upper()} 系统...")
            
            if not os.path.exists(sys_dir) or not os.listdir(sys_dir):
                print(f"    \033[93m[警告] 未找到 {sys_name} 数据，记录为 0。\033[0m")
                metrics[sys_name] = {
                    "psnr": 0.0, "ssim": 0.0, "ms_ssim": 0.0, 
                    "lpips": 0.0, "fid": 0.0, "ssnr": 0.0
                }
                continue

            # 调用中立裁判 evaluator.py
            cmd = [
                "python", "evaluator.py", 
                "--src", CLEAN_REF_DIR, 
                "--dst", sys_dir, 
                "--out", f"tmp_eval_exp4_{sys_name}"
            ]
            metrics[sys_name] = run_command_and_extract_metrics(cmd)

        # 动态组装行数据
        row_data = {"physical_snr_db": snr}
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
    
    # 严格控制列的输出顺序
    columns_order = ["physical_snr_db"] 
    for prefix in ["jpeg_", "dlsscc_", "ours_", "analog_jscc_"]:
        for metric in ["psnr", "ssim", "ms_ssim", "lpips", "fid", "ssnr"]:
            columns_order.append(f"{prefix}{metric}")
            
    for col in columns_order:
        if col not in df_summary.columns:
            df_summary[col] = 0.0
            
    df_summary = df_summary[columns_order]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    final_csv = f"EXP4_RAYLEIGH_METRICS_{timestamp}.csv"
    df_summary.to_csv(final_csv, index=False)
    
    print("\n🎉 评估全部完成！瑞利慢衰落信道干扰数据 (四大系统 × 6 项指标) 已成功汇总。")
    print(f"✅ 文件保存至: {final_csv}")

if __name__ == "__main__":
    main()
