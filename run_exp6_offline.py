import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import json
import glob

# 导入系统核心组件
from image_e2e_model import ImageSemanticEncoder, ImageSemanticDecoder

MOD_POWER = {'4QAM': 1.0, 'BPSK': 1.0}

def get_semantic_extractors(device):
    print("\033[90m[系统] 正在加载两套 ResNet50 语义裁判...\033[0m")
    weights = models.ResNet50_Weights.DEFAULT
    model_kl = models.resnet50(weights=weights).to(device).eval()
    model_feat = nn.Sequential(*list(model_kl.children())[:-1], nn.Flatten()).eval()
    return model_feat, model_kl

def calculate_normalized_ssnr_db(tensor_src, tensor_dst, semantic_model_feat, device):
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    src_input = preprocess(tensor_src).unsqueeze(0).to(device)
    dst_input = preprocess(tensor_dst).unsqueeze(0).to(device)

    with torch.no_grad():
        V_orig = semantic_model_feat(src_input).squeeze()
        V_dec = semantic_model_feat(dst_input).squeeze()

    V_orig_norm = V_orig / (torch.norm(V_orig, p=2) + 1e-9)
    V_dec_norm = V_dec / (torch.norm(V_dec, p=2) + 1e-9)

    power_s = 1.0 
    noise_vec = V_dec_norm - V_orig_norm
    power_n = max(torch.sum(noise_vec ** 2).item(), 1e-9)
    return 10 * np.log10(power_s / power_n)

def calculate_kl_ssnr_db(tensor_src, tensor_dst, semantic_model_kl, device):
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    src_input = preprocess(tensor_src).unsqueeze(0).to(device)
    dst_input = preprocess(tensor_dst).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_orig = semantic_model_kl(src_input).squeeze()
        logits_dec = semantic_model_kl(dst_input).squeeze()

    P_orig = F.softmax(logits_orig, dim=-1)
    P_dec = F.softmax(logits_dec, dim=-1)
    epsilon = 1e-6
    kl_div = torch.sum(P_orig * torch.log((P_orig + epsilon) / (P_dec + epsilon))).item()
    kl_div = max(kl_div, 0.0)
    return 10 * np.log10(1.0 / (kl_div + epsilon))

def calculate_total_variation(img_tensor):
    tv_h = torch.mean(torch.abs(img_tensor[:, :, 1:, :] - img_tensor[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(img_tensor[:, :, :, 1:] - img_tensor[:, :, :, :-1]))
    return (tv_h + tv_w).item()

def main():
    print("\033[96m初始化离线实验 6 数据采集 (莱斯慢衰落 Rician Channel)...\033[0m")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        with open('config.json', 'r', encoding='utf-8') as f: MAIN_CONFIG = json.load(f)
        with open('config_offline.json', 'r', encoding='utf-8') as f: OFF_CONFIG = json.load(f)
    except FileNotFoundError as e:
        return print(f"\033[91m[错误] 配置文件读取失败: {e}\033[0m")

    COMMON_CFG = OFF_CONFIG["offline_common"]
    EXP6_CFG = OFF_CONFIG["exp6_rician_params"]
    
    E2E_SYMBOL_LENGTH = 16384
    if "exp1_awgn_params" in OFF_CONFIG and "e2e_symbol_length" in OFF_CONFIG["exp1_awgn_params"]:
        E2E_SYMBOL_LENGTH = OFF_CONFIG["exp1_awgn_params"]["e2e_symbol_length"]

    tx_model = ImageSemanticEncoder().to(device).eval()
    rx_model = ImageSemanticDecoder().to(device).eval()
    try:
        tx_model.load_state_dict(torch.load(MAIN_CONFIG["paths"]["tx_weights"], map_location=device))
        rx_model.load_state_dict(torch.load(MAIN_CONFIG["paths"]["rx_weights"], map_location=device))
        print("[加载] 端到端网络权重加载成功。")
    except Exception as e:
        print(f"\033[93m[警告] 模型加载失败: {e}\n使用随机初始化参数。\033[0m")

    semantic_model_feat, semantic_model_kl = get_semantic_extractors(device)

    src_dir = COMMON_CFG["tx_source_dir"]
    image_files = glob.glob(os.path.join(src_dir, "**/*.*"), recursive=True)
    valid_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not valid_files:
        print(f"\033[91m[错误] 目录 {src_dir} 中未找到图片。\033[0m")
        return

    MAX_IMAGES = COMMON_CFG["max_images_per_condition"]
    test_files = valid_files[:MAX_IMAGES]
    res = COMMON_CFG["target_resolution"]
    
    avg_snr_list = EXP6_CFG["snr_levels"]
    k_factor_db = EXP6_CFG.get("k_factor_db", 2.0)
    
    # 将 K 因子转化为线性值，并计算 LOS 与 NLOS 的功率占比
    k_linear = 10 ** (k_factor_db / 10.0)
    p_los = k_linear / (k_linear + 1.0)
    p_nlos = 1.0 / (k_linear + 1.0)

    print(f"[准备] 测试集 {len(test_files)} 张 | 莱斯信道 K 因子 = {k_factor_db}dB")
    print(f"🚀 开始扫描 (平均物理 SNR: {avg_snr_list[0]}dB to {avg_snr_list[-1]}dB)...")
    
    results = []

    for avg_snr in avg_snr_list:
        mod_type = '4QAM' if avg_snr >= MAIN_CONFIG["amc_thresholds"]["4QAM"] else 'BPSK'
        fixed_tau = MAIN_CONFIG["tau_config"][mod_type]
        avg_snr_linear = 10 ** (avg_snr / 10.0)
        avg_noise_power = MOD_POWER.get(mod_type, 1.0) / avg_snr_linear
        
        print(f"正在仿真 平均SNR: {avg_snr:>5.1f}dB | 调制: {mod_type} | 导频基准Tau: {fixed_tau}")

        for img_path in test_files:
            img_name = os.path.basename(img_path)
            
            # =========================================================
            # 【莱斯衰落物理建模】
            # 直达波 (LOS): 幅度为 sqrt(P_los)，相位假定为 0
            # 散射波 (NLOS): 幅度方差为 P_nlos 的复高斯随机变量
            # =========================================================
            h_los = np.sqrt(p_los)
            h_nlos_re = np.random.randn() * np.sqrt(p_nlos / 2.0)
            h_nlos_im = np.random.randn() * np.sqrt(p_nlos / 2.0)
            
            h_re = h_los + h_nlos_re
            h_im = h_nlos_im
            h_mag = np.sqrt(h_re**2 + h_im**2)
            
            # 迫零均衡后的等效瞬时放大噪声功率
            inst_noise_power = avg_noise_power / (h_mag**2)
            theoretical_tau = (inst_noise_power * np.log(3)) / 2.0
            decode_tau = min(theoretical_tau, 0.7)

            img = Image.open(img_path).convert('RGB')
            transform = transforms.Compose([transforms.Resize((res, res)), transforms.ToTensor()])
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                tx_symbols = tx_model(img_tensor, mod_type=mod_type, temperature=1.0)
                
                noise = torch.sqrt(torch.tensor(inst_noise_power, device=device)) * torch.randn_like(tx_symbols)
                noisy_symbols = tx_symbols + noise
                
                is_ambiguous = torch.abs(noisy_symbols) < decode_tau
                ternary_symbols = torch.where(is_ambiguous, torch.zeros_like(noisy_symbols), torch.sign(noisy_symbols))
                
                reconstructed_img, sparsity_val = rx_model(ternary_symbols, return_sparsity=True)
                
                # 控制面导频
                pilot_tx = np.random.choice([-1.0, 1.0], E2E_SYMBOL_LENGTH).astype(np.float32)
                pilot_noise = np.sqrt(inst_noise_power) * np.random.randn(E2E_SYMBOL_LENGTH).astype(np.float32)
                pilot_rx = pilot_tx + pilot_noise
                
                pilot_evm_val = np.mean(np.abs(pilot_rx - np.sign(pilot_rx)))
                pilot_tar_val = np.mean(np.abs(pilot_rx) < fixed_tau) * 100.0
                
                tv_val = calculate_total_variation(reconstructed_img)
                re_encoded_symbols = tx_model(reconstructed_img, mod_type=mod_type, temperature=1.0)
                cyclic_error_val = torch.nn.functional.mse_loss(re_encoded_symbols, ternary_symbols).item()
                
                norm_ssnr_val = calculate_normalized_ssnr_db(img_tensor.cpu().squeeze(0), reconstructed_img.cpu().squeeze(0), semantic_model_feat, device)
                kl_ssnr_val = calculate_kl_ssnr_db(img_tensor.cpu().squeeze(0), reconstructed_img.cpu().squeeze(0), semantic_model_kl, device)

            results.append({
                "image_name": img_name,
                "mod_type": mod_type,
                "avg_physical_snr": avg_snr,
                "h_magnitude": round(h_mag, 4),               
                "inst_noise_power": round(inst_noise_power, 4),
                "tar_percent": round(pilot_tar_val, 2),        
                "soft_evm": round(pilot_evm_val, 4),           
                "feature_sparsity": round(sparsity_val, 4),
                "total_variation": round(tv_val, 4),          
                "cyclic_error": round(cyclic_error_val, 4),   
                "norm_ssnr_db": round(norm_ssnr_val, 4),  
                "kl_ssnr_db": round(kl_ssnr_val, 4)       
            })

    df_results = pd.DataFrame(results)
    timestamp = datetime.now().strftime('%Y%m%d%H%M')
    csv_filename = f"offline_exp6_rician_features_{timestamp}.csv"
    df_results.to_csv(csv_filename, index=False)
    print(f"\n✅ 图像语义实验 6 (莱斯衰落 K={k_factor_db}dB) 离线采集完成！共采集 {len(results)} 条记录。")

if __name__ == "__main__":
    main()