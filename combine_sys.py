import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_input_num(prompt, valid_range, allow_zero=True):
    """通用输入校验函数：只接受数字，且在指定范围内。0为返回上一级"""
    while True:
        try:
            val = int(input(prompt))
            if val == 0 and allow_zero:
                return 0
            if val in valid_range:
                return val
            print(f"⚠️ 无效输入，请输入 {min(valid_range)} 到 {max(valid_range)} 之间的数字" + (" (或输入 0 返回)" if allow_zero else ""))
        except ValueError:
            print("⚠️ 格式错误，请输入纯数字！")

def load_json_safe(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 无法读取配置文件 {filepath}: {e}")
        return None

def find_dict_key_containing(d, substring):
    for k in d.keys():
        if substring in k:
            return d[k]
    return None

def main():
    # 读取主配置以获取全局路径
    main_cfg = load_json_safe("config.json")
    if not main_cfg:
        print("❌ 无法读取主配置文件 config.json")
        sys.exit(1)
    
    global_tx_source_dir = main_cfg.get("tx_source_dir", "D:/dataset/test_images")

    # 读取总的消融配置文件
    ablation_cfg = load_json_safe("ablation_config.json")
    if not ablation_cfg or "ablation_dirs" not in ablation_cfg:
        print("❌ 缺失 ablation_config.json 或格式不正确。程序中止。")
        sys.exit(1)
        
    ablation_dirs = ablation_cfg["ablation_dirs"]
    
    state = 1
    exp_num = None
    exp3_type = None
    exp3_snr = None
    main_param = None
    img_num = None
    include_orig = False
    include_interf = False
    
    config_data = None
    exp_params = None

    while True:
        if state == 1:
            clear_screen()
            print("========================================")
            print(" 🌟 多系统基线视觉对比图交互生成器 🌟")
            print("========================================")
            for i in range(1, 7): print(f"{i}. 实验 {i}")
            print("0. 退出程序")
            print("========================================")
            
            exp_num = get_input_num("👉 请选择要可视化的实验 (0-6): ", range(1, 7))
            if exp_num == 0: sys.exit(0)
                
            config_data = load_json_safe(f"config_exp{exp_num}.json")
            if not config_data:
                input("按回车键重新选择...")
                continue
                
            exp_params = find_dict_key_containing(config_data, f"experiment{exp_num}")
            state = 2 if exp_num == 3 else 3
            
        elif state == 2:
            clear_screen()
            print("========================================")
            print(" 📡 实验 3: 多用户碰撞干扰类型选择")
            print("========================================")
            print("1. 同类图片碰撞 | 2. 异类图片碰撞 | 3. 传统数字信号 | 0. 返回")
            type_choice = get_input_num("👉 请选择干扰类型: ", [1, 2, 3])
            if type_choice == 0: state -= 1; continue
            
            exp3_type = {1: "same_domain", 2: "out_of_domain", 3: "digital_legacy"}[type_choice]
            snr_list = exp_params.get("snr_levels", [10.0, 5.0, 0.0])
            print("\n👉 请选择 SNR:")
            for i, snr in enumerate(snr_list, 1): print(f"  {i}. {snr} dB")
            snr_choice = get_input_num("选择 SNR 序号: ", range(1, len(snr_list) + 1))
            if snr_choice == 0: continue
            exp3_snr = snr_list[snr_choice - 1]
            state = 3

        elif state == 3:
            clear_screen()
            param_list = exp_params.get("snr_levels" if exp_num in [1, 4, 5, 6] else "burst_probabilities" if exp_num == 2 else "sir_levels", [])
            for i, val in enumerate(param_list, 1):
                label = f"{val} dB" if exp_num != 2 else f"脉冲占比 {int(round(val * 100))}%"
                print(f"{i}. {label}")
            param_choice_idx = get_input_num("👉 选择参数序号: ", range(1, len(param_list) + 1))
            if param_choice_idx == 0: state = 2 if exp_num == 3 else 1; continue
            main_param = param_list[param_choice_idx - 1]
            state = 4

        elif state == 4:
            img_num = get_input_num("👉 图片序号 (1-16, 0返回): ", range(1, 17))
            if img_num == 0: state = 3; continue
            state = 5

        elif state == 5:
            include_orig = (get_input_num("👉 显示【原始发送图】? (1:是, 2:否): ", [1, 2]) == 1)
            include_interf = (exp_num == 3 and exp3_type in ["same_domain", "out_of_domain"] and 
                             get_input_num("👉 显示【多址干扰原图】? (1:是, 2:否): ", [1, 2]) == 1)
            state = 6

        elif state == 6:
            img_name = f"{img_num:03d}.png"
            final_images = {}
            
            # 【修复逻辑】：统一使用 global_tx_source_dir
            if include_orig:
                final_images["Reference (Clean)"] = os.path.join(global_tx_source_dir, img_name).replace('\\', '/')
            
            if include_interf:
                interf_dir = exp_params.get("same_domain_dir" if exp3_type == "same_domain" else "out_of_domain_dir", "")
                if interf_dir:
                    final_images["Interference"] = os.path.join(interf_dir, img_name).replace('\\', '/')
                
            folder_suffix = f"snr_{main_param}" if exp_num in [1, 4, 5, 6] else (f"prob_{int(round(main_param * 100))}" if exp_num == 2 else os.path.join(exp3_type, f"snr_{exp3_snr}_sir_{main_param}"))
            output_base = exp_params.get("output_base_dir", f"output_exp{exp_num}").replace("./", "").replace(".\\", "")

            for sys_dir_name, title in [("jpeg", "JPEG (Legacy)"), ("dl_sscc", "DL-SSCC (Digital)"), ("analog_jscc", "Analog JSCC"), ("ours", "Ours (Semantic)")]:
                final_images[title] = os.path.join(output_base, folder_suffix, sys_dir_name, img_name).replace('\\', '/')

            fig, axes = plt.subplots(2, len(final_images), figsize=(4 * len(final_images), 8))
            if len(final_images) == 1: axes = np.expand_dims(axes, axis=1)

            for idx, (title, path) in enumerate(final_images.items()):
                is_corrupted = False
                img = cv2.imread(path)
                if img is None or not np.any(img):
                    img = np.zeros((256, 256, 3), dtype=np.uint8)
                    is_corrupted = True
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                if is_corrupted:
                    cv2.putText(img, "transmission corrupted", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
                
                ax_f, ax_z = axes[0, idx], axes[1, idx]
                ax_f.imshow(img); ax_f.set_title(title); ax_f.axis('off')
                ax_f.add_patch(Rectangle((80, 80), 60, 60, edgecolor='red', facecolor='none'))
                ax_z.imshow(cv2.resize(img[80:140, 80:140], (256, 256), interpolation=cv2.INTER_NEAREST))
                ax_z.set_xticks([]); ax_z.set_yticks([])
                for s in ax_z.spines.values(): s.set_edgecolor('red'); s.set_linewidth(3)

            plt.tight_layout()
            save_name = f"system_visual_exp{exp_num}_img{img_num:03d}.png"
            plt.savefig(save_name, dpi=300, bbox_inches='tight')
            print(f"\n🎉 成功生成对比图: {save_name}")
            input("\n按回车键返回..."); state = 3 

if __name__ == "__main__":
    main()
