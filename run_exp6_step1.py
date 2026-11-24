import os
import cv2
import json
import torch
import numpy as np
import glob
import random
import joblib
import xgboost as xgb
import csv
import commpy.channelcoding.convcode as cc
import torchvision.transforms as T
from compressai.zoo import bmshj2018_factorized
import warnings

# 导入端到端系统组件
from image_e2e_model import ImageSemanticEncoder, ImageSemanticDecoder

warnings.filterwarnings("ignore")

# ==========================================
# 1. 辅助函数：基线编解码与【硬核防崩溃】保护 (同步实验5终极版)
# ==========================================
def compress_jpeg_to_budget(img_array, budget_bytes, k_bits):
    low, high, best_encoded = 0, 100, None
    for _ in range(7): 
        mid = (low + high) // 2
        _, encimg = cv2.imencode('.jpg', img_array, [int(cv2.IMWRITE_JPEG_QUALITY), mid])
        if len(encimg) <= budget_bytes:
            best_encoded, low = encimg, mid + 1
        else:
            high = mid - 1
            
    if best_encoded is None:
        _, best_encoded = cv2.imencode('.jpg', img_array, [int(cv2.IMWRITE_JPEG_QUALITY), 0])
        
    is_over_budget = len(best_encoded) > budget_bytes
    actual_bytes = min(len(best_encoded), budget_bytes)
    
    best_encoded_tx = best_encoded[:actual_bytes]
    bits = np.unpackbits(best_encoded_tx)
    
    if len(bits) < k_bits:
        bits = np.concatenate((bits, np.zeros(k_bits - len(bits), dtype=np.uint8)))
    else:
        bits = bits[:k_bits]
    return bits, actual_bytes, is_over_budget

def decompress_jpeg(bits, actual_bytes):
    try:
        valid_bits = bits[:actual_bytes * 8]
        byte_data = np.packbits(valid_bits).tobytes()
        return cv2.imdecode(np.frombuffer(byte_data, np.uint8), cv2.IMREAD_COLOR)
    except: return None 

def compress_ai_to_budget(img_tensor, budget_bytes, k_bits, ai_models):
    best_q, best_out = 0, None
    for q in range(8, 0, -1):
        with torch.no_grad():
            out_enc = ai_models[q].compress(img_tensor)
            string_bytes = out_enc['strings'][0][0]
            if len(string_bytes) <= budget_bytes:
                best_q, best_out = q, out_enc
                break
                
    if best_out is None:
        with torch.no_grad(): best_out = ai_models[1].compress(img_tensor)
        best_q = 1
        
    byte_data = best_out['strings'][0][0]
    
    is_over_budget = len(byte_data) > budget_bytes
    actual_bytes = min(len(byte_data), budget_bytes)
    
    byte_data_tx = byte_data[:actual_bytes]
    bits = np.unpackbits(np.frombuffer(byte_data_tx, dtype=np.uint8))
    
    if len(bits) < k_bits:
        bits = np.concatenate((bits, np.zeros(k_bits - len(bits), dtype=np.uint8)))
    else:
        bits = bits[:k_bits]
    return bits, actual_bytes, best_out['shape'], best_q, is_over_budget

def decompress_ai(bits, actual_bytes, shape, q_used, ai_models):
    try:
        valid_bits = bits[:actual_bytes * 8]
        byte_data = np.packbits(valid_bits).tobytes()
        out_enc = {'strings': [[byte_data]], 'shape': shape}
        with torch.no_grad():
            out_dec = ai_models[q_used].decompress(out_enc['strings'], out_enc['shape'])
        rec_img = out_dec['x_hat'].clamp(0, 1)[0]
        rec_img_np = (rec_img.cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
        return cv2.cvtColor(rec_img_np, cv2.COLOR_RGB2BGR)
    except: return None

def safe_imread_and_resize(path, size):
    try:
        img_data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img is not None: return cv2.resize(img, (size, size))
    except: pass
    return None

def bpsk_modulate(bits):
    return np.where(bits == 0, -1.0, 1.0).astype(np.float32)

def hard_demodulate(noisy_symbols):
    return np.where(noisy_symbols > 0, 1, 0)

# ==========================================
# 2. 主执行流水线：实验 6 莱斯慢衰落传输程序
# ==========================================
def main():
    print("\n🚀 启动实验 6 传输程序: 莱斯慢衰落信道 (集成 Analog Deep JSCC 基线 + 防崩溃完美版)")
    
    DEBUG_MODE = True  
    
    with open('config_exp6.json', 'r', encoding='utf-8') as f:
        EXP6_CFG = json.load(f)["experiment6_rician_params"]
    with open('config.json', 'r', encoding='utf-8') as f:
        MAIN_CFG = json.load(f)

    # 固定全局随机种子
    SEED = EXP6_CFG["random_seed"]
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 初始化传统卷积码
    memory = np.array(EXP6_CFG["cc_memory"])
    g_matrix = np.array(EXP6_CFG["cc_g_matrix"])
    trellis = cc.Trellis(memory, g_matrix)
    E2E_SYMBOL_LENGTH = EXP6_CFG["e2e_symbol_length"]
    K_BITS = (E2E_SYMBOL_LENGTH // 2) - memory[0]
    BUDGET_BYTES = K_BITS // 8

    print("🤖 正在加载 DL-SSCC (CompressAI) 模型...")
    ai_models = {q: bmshj2018_factorized(quality=q, pretrained=True).eval().to(device) for q in range(1, 9)}
    
    print("🧠 正在加载 Ours (E2E Semantic) 收发模型...")
    tx_model = ImageSemanticEncoder().to(device).eval()
    rx_model = ImageSemanticDecoder().to(device).eval()
    tx_model.load_state_dict(torch.load(MAIN_CFG["paths"]["tx_weights"], map_location=device))
    rx_model.load_state_dict(torch.load(MAIN_CFG["paths"]["rx_weights"], map_location=device))

    print("🌊 正在加载 Analog Deep JSCC (纯模拟连续传输) 基线模型...")
    analog_tx_model = ImageSemanticEncoder().to(device).eval()
    analog_rx_model = ImageSemanticDecoder().to(device).eval()
    try:
        analog_tx_model.load_state_dict(torch.load(MAIN_CFG["paths"]["analog_tx_weights"], map_location=device))
        analog_rx_model.load_state_dict(torch.load(MAIN_CFG["paths"]["analog_rx_weights"], map_location=device))
        print("  ✅ Analog 权重加载成功！")
    except Exception as e:
        print(f"  ⚠️ 警告: 无法加载 Analog 权重，将使用随机初始化作为占位。({e})")

    print("📊 正在加载 S-CSI 物理特征盲估器 (XGBoost 4分类全信道版)...")
    scsi_scaler = joblib.load('scsi_scaler.pkl')
    scsi_xgb = xgb.XGBClassifier()
    scsi_xgb.load_model('scsi_xgboost.json')
    TEMP_MAPPING = {3: 1.0, 2: 0.8, 1: 0.3, 0: 0.1}

    test_files = glob.glob(os.path.join(EXP6_CFG["tx_source_dir"], "*.*"))
    test_files = [f for f in test_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    res = EXP6_CFG["target_resolution"]
    avg_snr_levels = EXP6_CFG["snr_levels"]
    
    # 提取并计算莱斯信道核心参数
    k_factor_db = EXP6_CFG.get("k_factor_db", 2.0)
    k_linear = 10 ** (k_factor_db / 10.0)
    p_los = k_linear / (k_linear + 1.0)
    p_nlos = 1.0 / (k_linear + 1.0)

    print(f"\n📦 共计测试图片 {len(test_files)} 张，遍历平均物理 SNR 个数: {len(avg_snr_levels)}")
    print(f"📡 莱斯信道配置: K-Factor = {k_factor_db}dB | 直达波占比: {p_los*100:.1f}%")

    # 准备 Debug 日志
    debug_log_path = os.path.join(EXP6_CFG["output_base_dir"], "debug_temperature_log_exp6.csv")
    if DEBUG_MODE:
        os.makedirs(EXP6_CFG["output_base_dir"], exist_ok=True)
        with open(debug_log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Avg_SNR(dB)", "Image_Name", "h_mag", "Inst_Noise", "Pilot_EVM", "Pilot_TAR(%)", "Predicted_State", "Assigned_Temp", "Decode_Tau"])
        print(f"\n🛠️  [调试模式] 动态温度及莱斯衰落诊断日志写入: {debug_log_path}")

    for avg_snr in avg_snr_levels:
        out_dir_base = os.path.join(EXP6_CFG["output_base_dir"], f"snr_{avg_snr}")
        for sys_name in ['jpeg', 'dl_sscc', 'ours', 'analog_jscc']: 
            os.makedirs(os.path.join(out_dir_base, sys_name), exist_ok=True)
        
        print(f"\n{'='*50}\n▶ 当前信道环境: 莱斯平均 SNR = {avg_snr}dB")
        avg_snr_linear = 10 ** (avg_snr / 10.0)
        
        # 调制映射与基础参数计算 (基于平均 SNR)
        mod_type = '4QAM' if avg_snr >= MAIN_CFG["amc_thresholds"].get("4QAM", 5.0) else 'BPSK'
        fixed_tau = MAIN_CFG["tau_config"].get(mod_type, 0.4)
        avg_noise_power = 1.0 / avg_snr_linear

        for img_path in test_files:
            img_name = os.path.basename(img_path)
            name_no_ext, _ = os.path.splitext(img_name)
            png_name = f"{name_no_ext}.png"
            
            img_bgr = safe_imread_and_resize(img_path, res)
            if img_bgr is None: continue
                
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            tensor_img = T.ToTensor()(img_rgb).unsqueeze(0).to(device)

            # =================================================================
            # 【莱斯衰落核心】：生成带 LOS 分量的全局块衰落系数 h
            # =================================================================
            h_los = np.sqrt(p_los)
            h_nlos_re = np.random.randn() * np.sqrt(p_nlos / 2.0)
            h_nlos_im = np.random.randn() * np.sqrt(p_nlos / 2.0)
            
            h_re = h_los + h_nlos_re
            h_im = h_nlos_im
            h_mag = np.sqrt(h_re**2 + h_im**2)
            
            # 迫零均衡后的瞬时等效噪声
            inst_noise_power = avg_noise_power / (h_mag**2 + 1e-9)
            
            # 业务数据面动态阈值
            theoretical_tau = (inst_noise_power * np.log(3)) / 2.0
            decode_tau = min(theoretical_tau, 0.7)

            # =================================================================
            # 【控制面】：物理导频 (Pilot) 与 XGBoost 动态控温
            # =================================================================
            pilot_tx = np.random.choice([-1.0, 1.0], E2E_SYMBOL_LENGTH).astype(np.float32)
            pilot_noise = np.sqrt(inst_noise_power) * np.random.randn(E2E_SYMBOL_LENGTH).astype(np.float32)
            pilot_rx = pilot_tx + pilot_noise 
            
            # 提取真实物理特征
            pilot_evm = np.mean(np.abs(pilot_rx - np.sign(pilot_rx)))
            pilot_tar = np.mean(np.abs(pilot_rx) < fixed_tau) * 100.0
            
            features = np.array([[pilot_evm, avg_snr, pilot_tar]])
            features_scaled = scsi_scaler.transform(features)
            scsi_state = int(scsi_xgb.predict(features_scaled)[0])
            
            # =================================================================
            # 【莱斯信道专属兜底】：由于有直射波，惩罚设为 -2dB (区别于瑞利的 -5dB)
            # =================================================================
            effective_snr = avg_snr - 2.0
            if scsi_state == 2 and effective_snr >= 10.0:
                scsi_state = 3
                
            dynamic_temp = TEMP_MAPPING.get(scsi_state, 1.0)

            if DEBUG_MODE:
                print(f"  └─ [诊断] {png_name} | |h|:{h_mag:.3f} | EVM:{pilot_evm:.3f} | 状态:{scsi_state} | T={dynamic_temp} | 解调τ={decode_tau:.3f}")
                with open(debug_log_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([avg_snr, png_name, f"{h_mag:.4f}", f"{inst_noise_power:.4f}", f"{pilot_evm:.4f}", f"{pilot_tar:.2f}", scsi_state, dynamic_temp, f"{decode_tau:.4f}"])

            # ================== 1. 信号调制与映射 ==================
            tx_bits_jpg, act_bytes_jpg, over_budget_jpg = compress_jpeg_to_budget(img_bgr, BUDGET_BYTES, K_BITS)
            sym_jpg = bpsk_modulate(cc.conv_encode(tx_bits_jpg, trellis))
            
            tx_bits_ai, act_bytes_ai, shp_ai, q_ai, over_budget_ai = compress_ai_to_budget(tensor_img, BUDGET_BYTES, K_BITS, ai_models)
            sym_ai = bpsk_modulate(cc.conv_encode(tx_bits_ai, trellis))
            
            with torch.no_grad():
                # Ours: 带有动态掩码的离散特征发送
                sym_ours_tensor = tx_model(tensor_img, mod_type=mod_type, temperature=dynamic_temp)
                sym_ours = sym_ours_tensor.cpu().numpy().flatten().astype(np.float32)

                # Analog JSCC: 连续特征发送与功率归一化
                z_analog = analog_tx_model(tensor_img, mod_type='BPSK', temperature=1.0)
                norm_val = torch.norm(z_analog, p=2)
                x_analog = np.sqrt(z_analog.numel()) * (z_analog / (norm_val + 1e-9))
                sym_analog = x_analog.cpu().numpy().flatten().astype(np.float32)

            # ================== 2. 注入全局统一物理信道干扰 ==================
            awgn_base = np.sqrt(inst_noise_power) * np.random.randn(E2E_SYMBOL_LENGTH).astype(np.float32)
            rx_jpg  = sym_jpg  + awgn_base
            rx_ai   = sym_ai   + awgn_base
            
            awgn_ours = np.sqrt(inst_noise_power) * np.random.randn(len(sym_ours)).astype(np.float32)
            rx_ours = sym_ours + awgn_ours

            # Analog JSCC 共享统一的物理噪声底分布
            awgn_analog = np.sqrt(inst_noise_power) * np.random.randn(len(sym_analog)).astype(np.float32)
            rx_analog = sym_analog + awgn_analog

            # ================== 3. 信号解码与恢复 ==================
            
            # --- JPEG ---
            dec_bits_jpg = cc.viterbi_decode(hard_demodulate(rx_jpg), trellis, decoding_type='hard')[:K_BITS]
            valid_len_jpg = act_bytes_jpg * 8
            ber_jpg = np.mean(dec_bits_jpg[:valid_len_jpg] != tx_bits_jpg[:valid_len_jpg])
            
            if ber_jpg > 0.05 or over_budget_jpg: 
                rec_jpg = np.zeros_like(img_bgr)
            else:
                rec_jpg = decompress_jpeg(dec_bits_jpg, act_bytes_jpg)
                if rec_jpg is None: rec_jpg = np.zeros_like(img_bgr)

            # --- DL-SSCC ---
            dec_bits_ai = cc.viterbi_decode(hard_demodulate(rx_ai), trellis, decoding_type='hard')[:K_BITS]
            valid_len_ai = act_bytes_ai * 8
            ber_ai = np.mean(dec_bits_ai[:valid_len_ai] != tx_bits_ai[:valid_len_ai])
            
            if ber_ai > 0.0 or over_budget_ai:
                rec_ai = np.zeros_like(img_bgr)
            else:
                rec_ai = decompress_ai(dec_bits_ai, act_bytes_ai, shp_ai, q_ai, ai_models)
                if rec_ai is None: rec_ai = np.zeros_like(img_bgr)

            # --- Ours ---
            with torch.no_grad():
                rx_ours_tensor = torch.tensor(rx_ours, device=device).view(sym_ours_tensor.shape)
                is_ambiguous = torch.abs(rx_ours_tensor) < decode_tau
                ternary_symbols = torch.where(is_ambiguous, torch.zeros_like(rx_ours_tensor), torch.sign(rx_ours_tensor))
                
                rec_tensor_ours, _ = rx_model(ternary_symbols, return_sparsity=True)
                rec_ours_np = (rec_tensor_ours.clamp(0,1).cpu().squeeze(0).numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
                rec_ours = cv2.cvtColor(rec_ours_np, cv2.COLOR_RGB2BGR)

            # --- Analog JSCC ---
            with torch.no_grad():
                # 无需三值擦除，直接用 4D 结构喂入解码器
                rx_analog_tensor = torch.tensor(rx_analog, device=device).view(z_analog.shape)
                rec_tensor_analog, _ = analog_rx_model(rx_analog_tensor, return_sparsity=True)
                
                rec_analog_np = (rec_tensor_analog.clamp(0,1).cpu().squeeze(0).numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
                rec_analog = cv2.cvtColor(rec_analog_np, cv2.COLOR_RGB2BGR)

            # --- 保存结果 ---
            cv2.imwrite(os.path.join(out_dir_base, 'jpeg', png_name), rec_jpg)
            cv2.imwrite(os.path.join(out_dir_base, 'dl_sscc', png_name), rec_ai)
            cv2.imwrite(os.path.join(out_dir_base, 'ours', png_name), rec_ours)
            cv2.imwrite(os.path.join(out_dir_base, 'analog_jscc', png_name), rec_analog)

        print(f"✅ 完成当前平均 SNR 组合: 图像已落盘至 {out_dir_base}")

if __name__ == "__main__":
    main()
