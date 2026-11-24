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
import random

# 导入系统核心组件 (底层已集成 Instance Normalization)
from image_e2e_model import ImageSemanticEncoder, ImageSemanticDecoder

def get_semantic_extractors(device):
    """加载两套 ResNet50 作为真实 S-SNR 的物理外裁判"""
    print("\033[90m[系统] 正在加载两套 ResNet50 语义裁判...\033[0m")
    weights = models.ResNet50_Weights.DEFAULT
    model_kl = models.resnet50(weights=weights).to(device).eval()
    model_feat = nn.Sequential(*list(model_kl.children())[:-1], nn.Flatten()).eval()
    return model_feat, model_kl

def calculate_normalized_ssnr_db(tensor_src, tensor_dst, semantic_model_feat, device):
    """计算归一化 L2 范数 S-SNR"""
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

    noise_vec = V_dec_norm - V_orig_norm
    power_n = max(torch.sum(noise_vec ** 2).item(), 1e-9)
    return 10 * np.log10(1.0 / power_n)

def calculate_kl_ssnr_db(tensor_src, tensor_dst, semantic_model_kl, device):
    """计算基于 KL 散度倒数的 S-SNR"""
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
    kl_div = max(torch.sum(P_orig * torch.log((P_orig + epsilon) / (P_dec + epsilon))).item(), 0.0)
    return 10 * np.log10(1.0 / (kl_div + epsilon))

def generate_burst_impulse_noise(shape, prob, burst_len, sigma, device):
    """
    物理层突发脉冲干扰生成器 (Bernoulli-Gaussian Model)
    """
    flat_len = np.prod(shape)
    impulse_noise = torch.zeros(flat_len, device=device)
    
    if prob > 0.0:
        # 计算应当发生的突发块数量
        num_bursts = int((flat_len * prob) / burst_len)
        for _ in range(num_bursts):
            start_idx = random.randint(0, flat_len - burst_len - 1)
            # 生成高振幅的高斯脉冲
            burst = torch.randn(burst_len, device=device) * sigma
            impulse_noise[start_idx : start_idx + burst_len] = burst
            
    return impulse_noise.view(shape)

def main():
    print("\033[96m初始化离线实验 2 数据采集 (物理突发脉冲干扰)...\033[0m")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 联合加载主配置与离线配置
    try:
        with open('config.json', 'r', encoding='utf-8') as f: MAIN_CONFIG = json.load(f)
        with open('config_offline.json', 'r', encoding='utf-8') as f: OFF_CONFIG = json.load(f)
    except FileNotFoundError as e:
        return print(f"\033[91m[错误] 配置文件读取失败: {e}\033[0m")

    COMMON_CFG = OFF_CONFIG["offline_common"]
    EXP2_CFG = OFF_CONFIG["exp2_impulse_params"]
    
    # 提取导频长度
    E2E_SYMBOL_LENGTH = EXP2_CFG.get("e2e_symbol_length", 16384)

    tx_model = ImageSemanticEncoder().to(device).eval()
    rx_model = ImageSemanticDecoder().to(device).eval()
    try:
        tx_model.load_state_dict(torch.load(MAIN_CONFIG["paths"]["tx_weights"], map_location=device))
        rx_model.load_state_dict(torch.load(MAIN_CONFIG["paths"]["rx_weights"], map_location=device))
        print("[加载] 端到端网络权重加载成功。")
    except Exception as e:
        print(f"\033[93m[警告] 模型加载失败: {e}。使用随机初始化进行兜底。\033[0m")

    semantic_model_feat, semantic_model_kl = get_semantic_extractors(device)

    # 2. 准备数据集
    image_files = glob.glob(os.path.join(COMMON_CFG["tx_source_dir"], "**/*.*"), recursive=True)
    valid_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    test_files = valid_files[:COMMON_CFG["max_images_per_condition"]]

    # 3. 提取实验 2 专属参数
    bg_snr_db = EXP2_CFG["fixed_bg_snr_db"]
    mod_type = '4QAM' if bg_snr_db >= MAIN_CONFIG["amc_thresholds"]["4QAM"] else 'BPSK'
    
    # 【核心架构】：分离控制面(固定)与数据面(动态)阈值
    fixed_tau = MAIN_CONFIG["tau_config"][mod_type]
    bg_noise_power = 1.0 / (10 ** (bg_snr_db / 10.0))
    
    theoretical_tau = (bg_noise_power * np.log(3)) / 2.0
    decode_tau = min(theoretical_tau, 0.7)
    
    burst_len = EXP2_CFG["burst_length"]
    impulse_sigma = EXP2_CFG["impulse_sigma"]
    
    prob_list = EXP2_CFG["impulse_probs"]
    results = []
    res = COMMON_CFG["target_resolution"]

    print(f"\n🚀 开始物理突发脉冲参数扫描 (背景 SNR: {bg_snr_db}dB | 调制: {mod_type})")
    print(f"👉 导频测算基准 Tau: {fixed_tau} | 图像数据动态解调 Tau: {decode_tau:.3f}")
    
    for p in prob_list:
        print(f"正在仿真脉冲发生概率 P: {p*100:>3.0f}%")

        for img_path in test_files:
            img_name = os.path.basename(img_path)
            
            img = Image.open(img_path).convert('RGB')
            transform = transforms.Compose([transforms.Resize((res, res)), transforms.ToTensor()])
            clean_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                # --- 1. 发送端映射 (业务图像, 满载温度 T=1.0) ---
                tx_symbols = tx_model(clean_tensor, mod_type=mod_type, temperature=1.0)
                
                # --- 2. 物理信道叠加 (图像载荷：背景 AWGN + 突发脉冲) ---
                awgn = torch.sqrt(torch.tensor(bg_noise_power, device=device)) * torch.randn_like(tx_symbols)
                impulse = generate_burst_impulse_noise(tx_symbols.shape, p, burst_len, impulse_sigma, device)
                noisy_symbols = tx_symbols + awgn + impulse
                
                # --- 3. 【数据面隔离】: 接收端业务数据使用动态 decode_tau 解调 ---
                is_ambiguous = torch.abs(noisy_symbols) < decode_tau
                ternary_symbols = torch.where(is_ambiguous, torch.zeros_like(noisy_symbols), torch.sign(noisy_symbols))
                
                # --- 4. 特征盲估与图像重构 ---
                reconstructed_img, sparsity_val = rx_model(ternary_symbols, return_sparsity=True)
                
                # =========================================================
                # 5. 【控制面隔离】：独立生成导频并注入【双重干扰】，用固定 tau 测算
                # =========================================================
                pilot_tx = np.random.choice([-1.0, 1.0], E2E_SYMBOL_LENGTH).astype(np.float32)
                
                # 导频叠加 AWGN
                pilot_awgn = np.sqrt(bg_noise_power) * np.random.randn(E2E_SYMBOL_LENGTH).astype(np.float32)
                
                # 导频叠加 突发脉冲 (与图像承受同等级别概率)
                pilot_impulse = np.zeros(E2E_SYMBOL_LENGTH, dtype=np.float32)
                if p > 0.0:
                    num_bursts_pilot = int((E2E_SYMBOL_LENGTH * p) / burst_len)
                    for _ in range(num_bursts_pilot):
                        start_idx = random.randint(0, max(0, E2E_SYMBOL_LENGTH - burst_len - 1))
                        pilot_impulse[start_idx : start_idx + burst_len] = np.random.normal(0, impulse_sigma, burst_len).astype(np.float32)
                        
                pilot_rx = pilot_tx + pilot_awgn + pilot_impulse
                
                # 严格使用固定基准尺(fixed_tau)提取特征给 XGBoost
                pilot_evm_val = np.mean(np.abs(pilot_rx - np.sign(pilot_rx)))
                pilot_tar_val = np.mean(np.abs(pilot_rx) < fixed_tau) * 100.0
                # =========================================================

                # 重构回环误差 (记录系统内部一致性)
                re_encoded_symbols = tx_model(reconstructed_img, mod_type=mod_type, temperature=1.0)
                cyclic_error_val = torch.nn.functional.mse_loss(re_encoded_symbols, ternary_symbols).item()
                
                # --- [外环监督] 计算物理真值 S-SNR ---
                norm_ssnr_val = calculate_normalized_ssnr_db(clean_tensor.cpu().squeeze(0), reconstructed_img.cpu().squeeze(0), semantic_model_feat, device)
                kl_ssnr_val = calculate_kl_ssnr_db(clean_tensor.cpu().squeeze(0), reconstructed_img.cpu().squeeze(0), semantic_model_kl, device)

            results.append({
                "image_name": img_name,
                "impulse_p": p,                  
                "physical_snr": bg_snr_db,
                "tar_percent": round(pilot_tar_val, 2),        # XGBoost 特征
                "feature_sparsity": round(sparsity_val, 4),
                "soft_evm": round(pilot_evm_val, 4),           # XGBoost 特征
                "cyclic_error": round(cyclic_error_val, 4),   
                "norm_ssnr_db": round(norm_ssnr_val, 4),   
                "kl_ssnr_db": round(kl_ssnr_val, 4)        
            })

    df_results = pd.DataFrame(results)
    timestamp = datetime.now().strftime('%Y%m%d%H%M')
    csv_filename = f"offline_exp2_impulse_features_{timestamp}.csv"
    df_results.to_csv(csv_filename, index=False)
    print(f"\n✅ 突发脉冲双轨评估数据收集完成！共采集 {len(results)} 条记录。")
    print(f"数据已保存至: {csv_filename}")

if __name__ == "__main__":
    main()