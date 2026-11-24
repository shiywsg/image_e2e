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

from image_e2e_model import ImageSemanticEncoder, ImageSemanticDecoder

MOD_POWER = {'4QAM': 1.0, 'BPSK': 1.0}

def get_semantic_extractors(device):
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
    print("\033[96m初始化离线实验 5 数据采集 (瑞利快衰落 M=64)...\033[0m")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        with open('config.json', 'r', encoding='utf-8') as f: MAIN_CONFIG = json.load(f)
        with open('config_offline.json', 'r', encoding='utf-8') as f: OFF_CONFIG = json.load(f)
    except FileNotFoundError as e:
        return print(f"\033[91m[错误] 配置文件读取失败: {e}\033[0m")

    COMMON_CFG = OFF_CONFIG["offline_common"]
    EXP5_CFG = OFF_CONFIG["exp5_fast_rayleigh_params"]
    NUM_CHUNKS = EXP5_CFG.get("num_chunks", 64)
    E2E_SYMBOL_LENGTH = 16384
    if "exp1_awgn_params" in OFF_CONFIG and "e2e_symbol_length" in OFF_CONFIG["exp1_awgn_params"]:
        E2E_SYMBOL_LENGTH = OFF_CONFIG["exp1_awgn_params"]["e2e_symbol_length"]

    tx_model = ImageSemanticEncoder().to(device).eval()
    rx_model = ImageSemanticDecoder().to(device).eval()
    try:
        tx_model.load_state_dict(torch.load(MAIN_CONFIG["paths"]["tx_weights"], map_location=device))
        rx_model.load_state_dict(torch.load(MAIN_CONFIG["paths"]["rx_weights"], map_location=device))
    except Exception as e:
        print(f"\033[93m[警告] 模型加载失败: {e}\n使用随机初始化参数进行测试。\033[0m")

    semantic_model_feat, semantic_model_kl = get_semantic_extractors(device)

    src_dir = COMMON_CFG["tx_source_dir"]
    image_files = glob.glob(os.path.join(src_dir, "**/*.*"), recursive=True)
    valid_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    MAX_IMAGES = COMMON_CFG["max_images_per_condition"]
    test_files = valid_files[:MAX_IMAGES]
    res = COMMON_CFG["target_resolution"]

    results = []
    avg_snr_list = EXP5_CFG["snr_levels"]

    print(f"\n🚀 开始瑞利快衰落信道扫描 (平均物理 SNR: {avg_snr_list[0]}dB to {avg_snr_list[-1]}dB | M={NUM_CHUNKS})...")
    
    for avg_snr in avg_snr_list:
        mod_type = '4QAM' if avg_snr >= MAIN_CONFIG["amc_thresholds"]["4QAM"] else 'BPSK'
        fixed_tau = MAIN_CONFIG["tau_config"][mod_type]
        avg_snr_linear = 10 ** (avg_snr / 10.0)
        avg_noise_power = MOD_POWER.get(mod_type, 1.0) / avg_snr_linear
        
        print(f"正在仿真 平均SNR: {avg_snr:>5.1f}dB | 调制: {mod_type} | 激进Tau上限: 0.85")

        for img_path in test_files:
            img_name = os.path.basename(img_path)
            
            img = Image.open(img_path).convert('RGB')
            transform = transforms.Compose([transforms.Resize((res, res)), transforms.ToTensor()])
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                tx_symbols = tx_model(img_tensor, mod_type=mod_type, temperature=1.0)
                tx_flat = tx_symbols.view(-1)
                sym_len = tx_flat.shape[0]
                chunk_size = sym_len // NUM_CHUNKS
                
                rx_flat = torch.zeros_like(tx_flat)
                is_ambiguous_flat = torch.zeros_like(tx_flat, dtype=torch.bool)
                
                # 导频配置
                pilot_tx = np.random.choice([-1.0, 1.0], E2E_SYMBOL_LENGTH).astype(np.float32)
                pilot_rx = np.zeros_like(pilot_tx)
                pilot_chunk_size = E2E_SYMBOL_LENGTH // NUM_CHUNKS
                
                # =========================================================
                # 核心机制：块状快衰落 (Block-Fast Fading)
                # =========================================================
                for m in range(NUM_CHUNKS):
                    h_re = np.random.randn() / np.sqrt(2.0)
                    h_im = np.random.randn() / np.sqrt(2.0)
                    h_mag = np.sqrt(h_re**2 + h_im**2)
                    inst_noise_power = avg_noise_power / (h_mag**2)
                    
                    # 采用激进的上限 0.85
                    theoretical_tau = (inst_noise_power * np.log(3)) / 2.0
                    decode_tau = min(theoretical_tau, 0.85)
                    
                    # 处理业务数据块
                    s_idx = m * chunk_size
                    e_idx = (m + 1) * chunk_size if m < NUM_CHUNKS - 1 else sym_len
                    chunk_sym = tx_flat[s_idx:e_idx]
                    chunk_noise = torch.sqrt(torch.tensor(inst_noise_power, device=device)) * torch.randn_like(chunk_sym)
                    chunk_rx = chunk_sym + chunk_noise
                    
                    rx_flat[s_idx:e_idx] = chunk_rx
                    is_ambiguous_flat[s_idx:e_idx] = torch.abs(chunk_rx) < decode_tau
                    
                    # 处理导频块
                    p_s_idx = m * pilot_chunk_size
                    p_e_idx = (m + 1) * pilot_chunk_size if m < NUM_CHUNKS - 1 else E2E_SYMBOL_LENGTH
                    p_chunk = pilot_tx[p_s_idx:p_e_idx]
                    p_noise = np.sqrt(inst_noise_power) * np.random.randn(len(p_chunk)).astype(np.float32)
                    pilot_rx[p_s_idx:p_e_idx] = p_chunk + p_noise

                noisy_symbols = rx_flat.view_as(tx_symbols)
                is_ambiguous = is_ambiguous_flat.view_as(tx_symbols)
                ternary_symbols = torch.where(is_ambiguous, torch.zeros_like(noisy_symbols), torch.sign(noisy_symbols))
                
                reconstructed_img, sparsity_val = rx_model(ternary_symbols, return_sparsity=True)
                
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
    csv_filename = f"offline_exp5_fast_rayleigh_features_{timestamp}.csv"
    df_results.to_csv(csv_filename, index=False)
    print(f"\n✅ 图像语义实验 5 (快衰落) 盲估特征采集完成！共采集 {len(results)} 条记录。")

if __name__ == "__main__":
    main()