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
    """在字典中模糊寻找包含特定字符串的键"""
    for k in d.keys():
        if substring in k:
            return d[k]
    return None

def main():
    # 1. 优先读取主配置文件以获得全局参数
    main_cfg = load_json_safe("config.json")
    if not main_cfg:
        print("❌ 无法读取主配置文件 config.json。程序中止。")
        sys.exit(1)
        
    global_tx_source_dir = main_cfg.get("tx_source_dir", "D:/dataset/test_images")

    # 2. 读取总的消融配置文件
    ablation_cfg = load_json_safe("ablation_config.json")
    if not ablation_cfg or "ablation_dirs" not in ablation_cfg:
        print("❌ 缺失 ablation_config.json 或格式不正确。程序中止。")
        sys.exit(1)
        
    ablation_dirs = ablation_cfg["ablation_dirs"]
    
    # 状态机变量
    state = 1
    exp_num = None
    exp3_type = None
    exp3_snr = None
    main_param = None  # SNR, Prob, 或 SIR
    img_num = None
    include_orig = False
    include_interf = False
    
    config_data = None
    exp_params = None

    while True:
        # ==========================================
        # State 1: 选择实验
        # ==========================================
        if state == 1:
            clear_screen()
            print("========================================")
            print(" 🌟 消融实验视觉对比图交互生成器 (终极修复版) 🌟")
            print("========================================")
            print("1. 实验 1: AWGN 物理白噪声信道")
            print("2. 实验 2: 突发脉冲干扰信道")
            print("3. 实验 3: MAC 多址同频干扰信道")
            print("4. 实验 4: 瑞利慢衰落信道 (Slow Rayleigh)")
            print("5. 实验 5: 瑞利快衰落信道 (Fast Rayleigh)")
            print("6. 实验 6: 莱斯衰落信道 (Rician Fading)")
            print("0. 退出程序")
            print("========================================")
            
            exp_num = get_input_num("👉 请选择要可视化的实验 (0-6): ", range(1, 7))
            if exp_num == 0:
                print("👋 退出程序。")
                sys.exit(0)
                
            config_data = load_json_safe(f"config_exp{exp_num}.json")
            if not config_data:
                input("按回车键重新选择...")
                continue
                
            exp_params = find_dict_key_containing(config_data, f"experiment{exp_num}")
            if not exp_params:
                print(f"❌ 配置文件内找不到实验 {exp_num} 的参数段落。")
                input("按回车键重新选择...")
                continue
            
            state = 2 if exp_num == 3 else 3
            
        # ==========================================
        # State 2: 实验 3 专属 - 干扰类型选择
        # ==========================================
        elif state == 2:
            clear_screen()
            print("========================================")
            print(" 📡 实验 3: 多用户碰撞干扰类型选择")
            print("========================================")
            print("1. 同类图片碰撞 (Same Domain)")
            print("2. 异类图片碰撞 (Out of Domain)")
            print("3. 传统数字信号碰撞 (Digital Legacy)")
            print("0. 返回上一级")
            
            type_choice = get_input_num("👉 请选择干扰类型: ", [1, 2, 3])
            if type_choice == 0:
                state -= 1
                continue
            
            type_map = {1: "same_domain", 2: "out_of_domain", 3: "digital_legacy"}
            exp3_type = type_map[type_choice]
            
            snr_list = exp_params.get("snr_levels", [10.0, 5.0, 0.0])
            print("\n👉 请选择基础物理信噪比 (SNR):")
            for i, snr in enumerate(snr_list, 1):
                print(f"  {i}. {snr} dB")
            print("  0. 返回上一级")
            
            snr_choice = get_input_num("选择 SNR 序号: ", range(1, len(snr_list) + 1))
            if snr_choice == 0:
                continue 
            exp3_snr = snr_list[snr_choice - 1]
            state = 3

        # ==========================================
        # State 3: 选择核心物理参数 (SNR / Prob / SIR)
        # ==========================================
        elif state == 3:
            clear_screen()
            print("========================================")
            print(f" 🎛️ 实验 {exp_num}: 核心物理参数选择")
            print("========================================")
            
            if exp_num in [1, 4, 5, 6]:
                param_name = "物理信噪比 (SNR)"
                param_list = exp_params.get("snr_levels", [])
            elif exp_num == 2:
                param_name = "脉冲发生概率 (Prob)"
                # 【修复核心1】：精确读取 burst_probabilities 字段
                param_list = exp_params.get("burst_probabilities", [])
            elif exp_num == 3:
                param_name = "信号干扰比 (SIR)"
                param_list = exp_params.get("sir_levels", [])
            
            if not param_list:
                print(f"❌ 配置文件中找不到可选的 {param_name} 列表。")
                input("按回车键返回...")
                state = 1
                continue
                
            for i, val in enumerate(param_list, 1):
                # 【修复核心2】：针对实验2将其转化为百分数并加上文本
                if exp_num == 2:
                    percent_val = int(round(val * 100))
                    print(f"{i}. 脉冲占比 {percent_val}%")
                else:
                    print(f"{i}. {val} dB")
                    
            print("0. 返回上一级")
            
            param_choice_idx = get_input_num(f"👉 选择 {param_name} 序号: ", range(1, len(param_list) + 1))
            if param_choice_idx == 0:
                state = 2 if exp_num == 3 else 1
                continue
                
            main_param = param_list[param_choice_idx - 1]
            state = 4

        # ==========================================
        # State 4: 选择测试图片序号
        # ==========================================
        elif state == 4:
            clear_screen()
            print("========================================")
            print(" 🖼️ 图片序号选择")
            print("========================================")
            img_num = get_input_num("👉 请输入要查看的图片序号 (1-16, 输入0返回): ", range(1, 17))
            if img_num == 0:
                state = 3
                continue
            state = 5

        # ==========================================
        # State 5: 选择附加参考图像
        # ==========================================
        elif state == 5:
            clear_screen()
            print("========================================")
            print(" ⚙️ 附加参考图设置")
            print("========================================")
            
            orig_choice = get_input_num("👉 是否在对比图最左侧显示【原始发送图】? (1: 是, 2: 否, 0: 返回): ", [1, 2])
            if orig_choice == 0:
                state -= 1
                continue
            include_orig = (orig_choice == 1)
            
            if exp_num == 3 and exp3_type in ["same_domain", "out_of_domain"]:
                interf_choice = get_input_num("👉 是否显示【多址干扰原图】? (1: 是, 2: 否, 0: 返回): ", [1, 2])
                if interf_choice == 0:
                    continue
                include_interf = (interf_choice == 1)
            else:
                include_interf = False
                
            state = 6 # 准备生成

        # ==========================================
        # State 6: 生成图表
        # ==========================================
        elif state == 6:
            print("\n🚀 开始收集图片与拼接画板...")
            
            img_name = f"{img_num:03d}.png"
            final_images = {} # 格式: { "标题": "图片路径" }
            
            # --- 1. 尝试添加原始参考图 ---
            if include_orig:
                # 【修复核心3】：这里使用从 config.json 读到的全局发送目录
                orig_path = os.path.join(global_tx_source_dir, img_name).replace('\\', '/')
                final_images["Reference (Clean)"] = orig_path
            
            # --- 2. 尝试添加干扰参考图 (仅限实验3特定情况) ---
            if include_interf:
                if exp3_type == "same_domain":
                    interf_dir_key = "same_domain_dir"
                elif exp3_type == "out_of_domain":
                    interf_dir_key = "out_of_domain_dir"
                else:
                    interf_dir_key = None

                if interf_dir_key:
                    interf_dir = exp_params.get(interf_dir_key, "")
                    if interf_dir:
                        interf_path = os.path.join(interf_dir, img_name).replace('\\', '/')
                        final_images["Interference"] = interf_path
                
            # --- 3. 收集消融组图片 ---
            # 【修复核心4】：确保实验2使用 prob_百分比整数 作为子目录
            if exp_num in [1, 4, 5, 6]:
                folder_suffix = f"snr_{main_param}"
            elif exp_num == 2:
                percent_val = int(round(main_param * 100))
                folder_suffix = f"prob_{percent_val}"
            elif exp_num == 3:
                folder_suffix = os.path.join(exp3_type, f"snr_{exp3_snr}_sir_{main_param}")

            # 获取当前实验输出目录并清理相对路径前缀
            output_base = exp_params.get("output_base_dir", f"output_exp{exp_num}")
            if output_base.startswith("./") or output_base.startswith(".\\"):
                output_base = output_base[2:]

            # 对应字典配置项，顺序固定
            group_keys = [("full", "Group A (Full)"), 
                          ("no_dynamic", "Group B (No Mask)"), 
                          ("no_ternary", "Group C (No Ternary)"), 
                          ("base", "Group D (Base)")]
                          
            for cfg_key, title in group_keys:
                if cfg_key in ablation_dirs:
                    base_dir = ablation_dirs[cfg_key]
                    img_path = os.path.join(base_dir, output_base, folder_suffix, "ours", img_name)
                    img_path = img_path.replace('\\', '/')
                    final_images[title] = img_path

            # --- 4. Matplotlib 绘图 ---
            num_cols = len(final_images)
            fig, axes = plt.subplots(2, num_cols, figsize=(4 * num_cols, 8))
            
            # 如果是一维数组做安全转换
            if num_cols == 1: axes = np.expand_dims(axes, axis=1)

            # 根据实验类型动态生成标题
            if exp_num == 2:
                param_str = f"Prob={int(round(main_param*100))}%"
            elif exp_num == 3:
                param_str = f"SNR={exp3_snr}dB, SIR={main_param}dB, {exp3_type}"
            else:
                param_str = f"SNR={main_param}dB"
            
            exp_titles = ["", "Exp 1: AWGN", "Exp 2: Impulse", "Exp 3: MAC Collision", "Exp 4: Slow Rayleigh", "Exp 5: Fast Rayleigh", "Exp 6: Rician Fading"]
            
            fig.suptitle(f"Ablation Visual Comparison - {exp_titles[exp_num]} ({param_str}) | Image: {img_name}", fontsize=18, fontweight='bold')

            zoom_box = (80, 80, 60, 60) # 默认局部放大坐标
            x, y, w, h = zoom_box

            for idx, (title, path) in enumerate(final_images.items()):
                try:
                    img = cv2.imread(path)
                    if img is None: raise ValueError("Empty Image")
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except:
                    # 图片读取失败，生成黑色占位图
                    img = np.zeros((256, 256, 3), dtype=np.uint8)
                    print(f"  ⚠️ 警告: 读取不到图片 {path}，使用黑色占位图。")
                
                # 第一排：全图
                ax_full = axes[0, idx]
                ax_full.imshow(img)
                ax_full.set_title(title, fontsize=14, pad=10)
                ax_full.axis('off')
                
                # 为所有列的图片画红框，保持第一行视觉效果统一
                rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
                ax_full.add_patch(rect)
                
                # 第二排：局部放大 (Zoom-in) 统一应用于包括参考图在内的所有图
                ax_zoom = axes[1, idx]
                zoom_crop = img[y:y+h, x:x+w]
                zoom_resized = cv2.resize(zoom_crop, (256, 256), interpolation=cv2.INTER_NEAREST)
                ax_zoom.imshow(zoom_resized)
                for spine in ax_zoom.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)
                    
                ax_zoom.set_xticks([])
                ax_zoom.set_yticks([])

            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 为 suptitle 留出空间
            
            # 命名策略
            safe_param_str = param_str.replace(" ", "_").replace("=", "").replace(",", "").replace("%", "pct")
            save_name = f"ablation_visual_exp{exp_num}_{safe_param_str}_img{img_num:03d}.png"
            
            plt.savefig(save_name, dpi=300, bbox_inches='tight')
            print(f"\n🎉 成功生成学术对比拼图：\033[92m{save_name}\033[0m")
            
            # 生成完毕后，回到选择参数的状态，方便看下一组
            input("\n按回车键返回参数选择菜单...")
            state = 3 

if __name__ == "__main__":
    main()
