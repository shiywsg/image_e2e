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
    """加载两套 ResNet50 裁判 (KL 散度和归一化 L2 范数)"""
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

    power_s = 1.0 
    noise_vec = V_dec_norm - V_orig_norm
    power_n = max(torch.sum(noise_vec ** 2).item(), 1e-9)

    return 10 * np.log10(power_s / power_n)

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
    kl_div = torch.sum(P_orig * torch.log((P_orig + epsilon) / (P_dec + epsilon))).item()
    kl_div = max(kl_div, 0.0)

    return 10 * np.log10(1.0 / (kl_div + epsilon))

def calculate_total_variation(img_tensor):
    """计算图像总变差 (TV)"""
    tv_h = torch.mean(torch.abs(img_tensor[:, :, 1:, :] - img_tensor[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(img_tensor[:, :, :, 1:] - img_tensor[:, :, :, :-1]))
    return (tv_h + tv_w).item()

def main():
    print("\033[96m初始化离线实验 4 数据采集 (瑞利慢衰落信道)...\033[0m")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 【修复】：强制使用 utf-8 读取，彻底解决跨平台与中文解码报错
    try:
        with open('config.json', 'r', encoding='utf-8') as f: MAIN_CONFIG = json.load(f)
        with open('config_offline.json', 'r', encoding='utf-8') as f: OFF_CONFIG = json.load(f)
    except FileNotFoundError as e:
        return print(f"\033[91m[错误] 配置文件读取失败: {e}\033[0m")

    COMMON_CFG = OFF_CONFIG["offline_common"]
    EXP4_CFG = OFF_CONFIG["exp4_rayleigh_params"]
    
    # 兼容配置项读取
    E2E_SYMBOL_LENGTH = 16384
    if "exp1_awgn_params" in OFF_CONFIG and "e2e_symbol_length" in OFF_CONFIG["exp1_awgn_params"]:
        E2E_SYMBOL_LENGTH = OFF_CONFIG["exp1_awgn_params"]["e2e_symbol_length"]

    # 初始化 E2E 模型
    tx_model = ImageSemanticEncoder().to(device).eval()
    rx_model = ImageSemanticDecoder().to(device).eval()
    try:
        tx_model.load_state_dict(torch.load(MAIN_CONFIG["paths"]["tx_weights"], map_location=device))
        rx_model.load_state_dict(torch.load(MAIN_CONFIG["paths"]["rx_weights"], map_location=device))
        print("[加载] 端到端网络权重加载成功。")
    except Exception as e:
        print(f"\033[93m[警告] 模型加载失败: {e}\n使用随机初始化参数进行测试。\033[0m")

    semantic_model_feat, semantic_model_kl = get_semantic_extractors(device)

    src_dir = COMMON_CFG["tx_source_dir"]
    image_files = glob.glob(os.path.join(src_dir, "**/*.*"), recursive=True)
    valid_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not valid_files:
        print(f"\033[91m[错误] 目录 {src_dir} 中没有找到图片。\033[0m")
        return

    MAX_IMAGES = COMMON_CFG["max_images_per_condition"]
    test_files = valid_files[:MAX_IMAGES]
    res = COMMON_CFG["target_resolution"]
    print(f"[准备] 共选择 {len(test_files)} 张测试图片参与衰落扫描。")

    results = []
    avg_snr_list = EXP4_CFG["snr_levels"]

    print(f"\n🚀 开始瑞利衰落信道扫描 (平均物理信噪比: {avg_snr_list[0]}dB to {avg_snr_list[-1]}dB)...")
    
    for avg_snr in avg_snr_list:
        if avg_snr >= MAIN_CONFIG["amc_thresholds"]["4QAM"]:
            mod_type = '4QAM'
        else:
            mod_type = 'BPSK'
            
        fixed_tau = MAIN_CONFIG["tau_config"][mod_type]
        avg_snr_linear = 10 ** (avg_snr / 10.0)
        avg_noise_power = MOD_POWER.get(mod_type, 1.0) / avg_snr_linear
        
        print(f"正在仿真 平均SNR: {avg_snr:>5.1f}dB | 调制: {mod_type} | 导频基准Tau: {fixed_tau}")

        for img_path in test_files:
            img_name = os.path.basename(img_path)
            
            # =========================================================
            # 【物理层重构】：生成慢衰落 (块衰落) 系数
            # 瑞利分布幅度 |h|，由两个独立的零均值高斯分布组成
            # 保证平均信道功率 E[|h|^2] = 1
            # =========================================================
            h_re = np.random.randn() / np.sqrt(2.0)
            h_im = np.random.randn() / np.sqrt(2.0)
            h_mag = np.sqrt(h_re**2 + h_im**2)
            
            # 迫零均衡后的等效瞬时噪声功率 (放大效应)
            inst_noise_power = avg_noise_power / (h_mag**2)
            
            # 动态门限 tau (接收机通过导频/前导码获知瞬时噪声后计算得出)
            theoretical_tau = (inst_noise_power * np.log(3)) / 2.0
            decode_tau = min(theoretical_tau, 0.7)

            img = Image.open(img_path).convert('RGB')
            transform = transforms.Compose([transforms.Resize((res, res)), transforms.ToTensor()])
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                # 1. 物理发射 (满载温度 T=1.0)
                tx_symbols = tx_model(img_tensor, mod_type=mod_type, temperature=1.0)
                
                # 2. 物理信道加噪 (使用等效瞬时噪声)
                noise = torch.sqrt(torch.tensor(inst_noise_power, device=device)) * torch.randn_like(tx_symbols)
                noisy_symbols = tx_symbols + noise
                
                # 3. 接收端业务数据解调
                is_ambiguous = torch.abs(noisy_symbols) < decode_tau
                ternary_symbols = torch.where(is_ambiguous, torch.zeros_like(noisy_symbols), torch.sign(noisy_symbols))
                
                # 4. 接收端重构图像
                reconstructed_img, sparsity_val = rx_model(ternary_symbols, return_sparsity=True)
                
                # 5. 控制面导频 (同样经历该时刻的块衰落)
                pilot_tx = np.random.choice([-1.0, 1.0], E2E_SYMBOL_LENGTH).astype(np.float32)
                pilot_noise = np.sqrt(inst_noise_power) * np.random.randn(E2E_SYMBOL_LENGTH).astype(np.float32)
                pilot_rx = pilot_tx + pilot_noise
                
                pilot_evm_val = np.mean(np.abs(pilot_rx - np.sign(pilot_rx)))
                pilot_tar_val = np.mean(np.abs(pilot_rx) < fixed_tau) * 100.0  # 使用固定的基准尺
                
                # 其他指标计算
                tv_val = calculate_total_variation(reconstructed_img)
                re_encoded_symbols = tx_model(reconstructed_img, mod_type=mod_type, temperature=1.0)
                cyclic_error_val = torch.nn.functional.mse_loss(re_encoded_symbols, ternary_symbols).item()
                
                # 6. 双轨 S-SNR 评估
                norm_ssnr_val = calculate_normalized_ssnr_db(img_tensor.cpu().squeeze(0), reconstructed_img.cpu().squeeze(0), semantic_model_feat, device)
                kl_ssnr_val = calculate_kl_ssnr_db(img_tensor.cpu().squeeze(0), reconstructed_img.cpu().squeeze(0), semantic_model_kl, device)

            results.append({
                "image_name": img_name,
                "mod_type": mod_type,
                "avg_physical_snr": avg_snr,
                "h_magnitude": round(h_mag, 4),               # 记录本帧的实际衰落幅度 (重要参考)
                "inst_noise_power": round(inst_noise_power, 4), # 记录迫零均衡后的实际噪声功率
                "tar_percent": round(pilot_tar_val, 2),        # XGBoost 特征 1
                "soft_evm": round(pilot_evm_val, 4),           # XGBoost 特征 2
                "feature_sparsity": round(sparsity_val, 4),
                "total_variation": round(tv_val, 4),          
                "cyclic_error": round(cyclic_error_val, 4),   
                "norm_ssnr_db": round(norm_ssnr_val, 4),  
                "kl_ssnr_db": round(kl_ssnr_val, 4)       
            })

    df_results = pd.DataFrame(results)
    timestamp = datetime.now().strftime('%Y%m%d%H%M')
    csv_filename = f"offline_exp4_rayleigh_features_{timestamp}.csv"
    df_results.to_csv(csv_filename, index=False)
    print(f"\n✅ 图像语义实验 4 (瑞利衰落) 盲估特征离线采集完成！共采集 {len(results)} 条记录。")
    print(f"数据已保存至: {csv_filename}")

if __name__ == "__main__":
    main()