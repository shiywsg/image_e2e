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
import itertools

# 导入系统核心组件 (底层已集成 Instance Normalization)
from image_e2e_model import ImageSemanticEncoder, ImageSemanticDecoder

MOD_POWER = {'4QAM': 1.0, 'BPSK': 1.0}

def get_semantic_extractors(device):
    """
    加载两套 ResNet50 语义裁判：
    一套输出特征向量 (用于归一化 L2 范数)，一套输出分类概率 (用于 KL 散度)
    """
    print("\033[90m[系统] 正在加载两套 ResNet50 语义裁判...\033[0m")
    weights = models.ResNet50_Weights.DEFAULT
    
    # 1. 完整版 (用于 KL 散度)
    model_kl = models.resnet50(weights=weights).to(device).eval()
    
    # 2. 截断版 (用于归一化 L2 范数)
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

def load_image_tensor(path, device, size=(256, 256)):
    """安全加载图片并调整到统一大小，返回张量"""
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
    return transform(img).unsqueeze(0).to(device)

def get_random_image_tensor(img_dir, device, size=(256, 256)):
    """从指定目录随机抽取一张图片并转换为张量"""
    files = glob.glob(os.path.join(img_dir, "**/*.*"), recursive=True)
    valid_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not valid_files:
        # 降级兜底：如果找不到图片，返回全零全黑张量
        return torch.zeros((1, 3, size[0], size[1])).to(device)
    chosen_path = np.random.choice(valid_files)
    return load_image_tensor(chosen_path, device, size)

def generate_legacy_16qam_interference(shape, device):
    """模拟传统数字通信系统的满载 16QAM 盲发干扰"""
    levels = torch.tensor([-3.0, -1.0, 1.0, 3.0], device=device)
    indices = torch.randint(0, 4, shape, device=device)
    return levels[indices]

def calculate_total_variation(img_tensor):
    """计算总变差 TV"""
    tv_h = torch.mean(torch.abs(img_tensor[:, :, 1:, :] - img_tensor[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(img_tensor[:, :, :, 1:] - img_tensor[:, :, :, :-1]))
    return (tv_h + tv_w).item()

def main():
    print("\033[96m初始化离线实验 3 数据采集 (多用户免调度碰撞干扰)...\033[0m")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 联合加载主配置与离线配置
    try:
        with open('config.json', 'r', encoding='utf-8') as f: MAIN_CONFIG = json.load(f)
        with open('config_offline.json', 'r', encoding='utf-8') as f: OFF_CONFIG = json.load(f)
    except FileNotFoundError as e:
        return print(f"\033[91m[错误] 配置文件读取失败: {e}\033[0m")

    COMMON_CFG = OFF_CONFIG["offline_common"]
    EXP3_CFG = OFF_CONFIG["exp3_mac_params"]
    
    # 提取导频长度
    E2E_SYMBOL_LENGTH = EXP3_CFG.get("e2e_symbol_length", 16384)

    # 2. 初始化收发端语义网络 (已应用 Instance Normalization)
    tx_model = ImageSemanticEncoder().to(device).eval()
    rx_model = ImageSemanticDecoder().to(device).eval()
    try:
        tx_model.load_state_dict(torch.load(MAIN_CONFIG["paths"]["tx_weights"], map_location=device))
        rx_model.load_state_dict(torch.load(MAIN_CONFIG["paths"]["rx_weights"], map_location=device))
        print("[加载] 端到端网络权重加载成功。")
    except Exception as e:
        print(f"\033[93m[警告] 模型加载失败: {e}，使用随机初始化进行兜底测试。\033[0m")

    # 加载裁判模型
    semantic_model_feat, semantic_model_kl = get_semantic_extractors(device)

    # 3. 路径适配与容错处理
    main_dir = COMMON_CFG["tx_source_dir"]
    same_domain_dir = EXP3_CFG.get("same_domain_dir", "./dataset/test_images_same")
    ood_dir = EXP3_CFG.get("out_of_domain_dir", "./dataset/test_images_ood")
    
    main_files = glob.glob(os.path.join(main_dir, "**/*.*"), recursive=True)
    main_files = [f for f in main_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    test_files = main_files[:COMMON_CFG["max_images_per_condition"]]
    res = COMMON_CFG["target_resolution"]
    print(f"[准备] 主用户图片集大小: {len(test_files)} 张 (已应用统一离线截断上限)")

    # 4. 从大一统离线配置文件中提取扫描范围
    fixed_snr = EXP3_CFG["fixed_bg_snr_db"]
    sir_range = EXP3_CFG["sir_levels"]
    interference_types = ['same_domain', 'out_of_domain', 'digital_legacy']
    
    if fixed_snr >= MAIN_CONFIG["amc_thresholds"]["4QAM"]:
        mod_type = '4QAM'
    else:
        mod_type = 'BPSK'
        
    noise_power = MOD_POWER.get(mod_type, 1.0) / (10 ** (fixed_snr / 10.0))
    
    # =========================================================
    # 【核心架构】：分离控制面(固定)与数据面(动态)阈值
    # =========================================================
    fixed_tau = MAIN_CONFIG["tau_config"][mod_type]
    theoretical_tau = (noise_power * np.log(3)) / 2.0
    decode_tau = min(theoretical_tau, 0.7)
    
    results = []

    print(f"\n🚀 开始执行多址碰撞参数扫描 (固定背景 SNR: {fixed_snr}dB | 对应调制: {mod_type})")
    print(f"👉 导频测算基准 Tau: {fixed_tau} | 图像数据动态解调 Tau: {decode_tau:.3f}")
    
    for i_type, sir in itertools.product(interference_types, sir_range):
        print(f"正在仿真 => 干扰类型: {i_type:<15} | SIR: {sir:>5.1f}dB")
        alpha = 10 ** (-sir / 20.0) # 根据信号干扰比计算强干扰振幅比例系数

        for img_path in test_files:
            img_name = os.path.basename(img_path)
            
            tensor_A = load_image_tensor(img_path, device, size=(res, res))

            with torch.no_grad():
                # --- 1. 主用户符号发射 (满载温度 T=1.0) ---
                XA = tx_model(tensor_A, mod_type=mod_type, temperature=1.0)
                
                # --- 2. 非协作干扰源信号 User B 生成 ---
                if i_type == 'same_domain':
                    tensor_B = get_random_image_tensor(same_domain_dir, device, size=(res, res))
                    XB = tx_model(tensor_B, mod_type=mod_type, temperature=1.0)
                elif i_type == 'out_of_domain':
                    tensor_B = get_random_image_tensor(ood_dir, device, size=(res, res))
                    XB = tx_model(tensor_B, mod_type=mod_type, temperature=1.0)
                elif i_type == 'digital_legacy':
                    XB = generate_legacy_16qam_interference(XA.shape, device)

                # --- 3. 物理空中信道线性混叠碰撞 ---
                W = torch.sqrt(torch.tensor(noise_power, device=device)) * torch.randn_like(XA)
                Y_rx = XA + (alpha * XB) + W
                
                # --- 4. 【数据面隔离】: 接收端业务数据使用动态 decode_tau 解调 ---
                is_ambiguous = torch.abs(Y_rx) < decode_tau
                ternary_symbols = torch.where(is_ambiguous, torch.zeros_like(Y_rx), torch.sign(Y_rx))
                
                # --- 5. 接收端重构 ---
                reconstructed_img, sparsity_val = rx_model(ternary_symbols, return_sparsity=True)
                
                # =========================================================
                # 6. 【控制面隔离】：独立生成导频，注入干扰并用 fixed_tau 提取特征
                # =========================================================
                pilot_tx_np = np.random.choice([-1.0, 1.0], E2E_SYMBOL_LENGTH).astype(np.float32)
                pilot_tx = torch.tensor(pilot_tx_np, device=device)
                
                if i_type == 'digital_legacy':
                    pilot_XB = generate_legacy_16qam_interference((E2E_SYMBOL_LENGTH,), device)
                else:
                    pilot_XB_np = np.random.choice([-1.0, 1.0], E2E_SYMBOL_LENGTH).astype(np.float32)
                    pilot_XB = torch.tensor(pilot_XB_np, device=device)
                
                pilot_W = torch.sqrt(torch.tensor(noise_power, device=device)) * torch.randn_like(pilot_tx)
                
                pilot_rx = pilot_tx + (alpha * pilot_XB) + pilot_W
                
                # 严格对齐 XGBoost 的特征提取 (使用 fixed_tau 作为量尺)
                pilot_rx_np = pilot_rx.cpu().numpy()
                pilot_evm_val = np.mean(np.abs(pilot_rx_np - np.sign(pilot_rx_np)))
                pilot_tar_val = np.mean(np.abs(pilot_rx_np) < fixed_tau) * 100.0
                # =========================================================

                # 重构特征参数计算
                tv_val = calculate_total_variation(reconstructed_img)
                re_encoded_symbols = tx_model(reconstructed_img, mod_type=mod_type, temperature=1.0)
                cyclic_error_val = F.mse_loss(re_encoded_symbols, ternary_symbols).item()
                
                # --- [外环真值监督] 计算两种客观 S-SNR ---
                norm_ssnr_val = calculate_normalized_ssnr_db(tensor_A.cpu().squeeze(0), reconstructed_img.cpu().squeeze(0), semantic_model_feat, device)
                kl_ssnr_val = calculate_kl_ssnr_db(tensor_A.cpu().squeeze(0), reconstructed_img.cpu().squeeze(0), semantic_model_kl, device)

            results.append({
                "image_name": img_name,
                "interference_type": i_type,
                "physical_snr": fixed_snr,
                "collision_sir": sir,
                "tar_percent": round(pilot_tar_val, 2),        # XGBoost 特征
                "feature_sparsity": round(sparsity_val, 4),
                "soft_evm": round(pilot_evm_val, 4),           # XGBoost 特征
                "total_variation": round(tv_val, 4),
                "cyclic_error": round(cyclic_error_val, 4),
                "norm_ssnr_db": round(norm_ssnr_val, 4),
                "kl_ssnr_db": round(kl_ssnr_val, 4)
            })

    df_results = pd.DataFrame(results)
    timestamp = datetime.now().strftime('%Y%m%d%H%M')
    csv_filename = f"offline_exp3_mac_features_{timestamp}.csv"
    df_results.to_csv(csv_filename, index=False)
    print(f"\n✅ 离线实验三多址碰撞双轨特征采集完成！共采集 {len(results)} 条记录。")
    print(f"数据已保存至: {csv_filename}")

if __name__ == "__main__":
    main()