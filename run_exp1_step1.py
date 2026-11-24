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
# 1. 辅助函数：基线编解码与防崩溃读取
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
        if len(best_encoded) > budget_bytes: best_encoded = best_encoded[:budget_bytes]
             
    bits = np.unpackbits(best_encoded)
    if len(bits) < k_bits:
        bits = np.concatenate((bits, np.zeros(k_bits - len(bits), dtype=np.uint8)))
    else:
        bits = bits[:k_bits]
    return bits, len(best_encoded)

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
        string_bytes = best_out['strings'][0][0][:budget_bytes]
        best_out['strings'][0] = [string_bytes]
        
    byte_data = best_out['strings'][0][0]
    bits = np.unpackbits(np.frombuffer(byte_data, dtype=np.uint8))
    
    if len(bits) < k_bits:
        bits = np.concatenate((bits, np.zeros(k_bits - len(bits), dtype=np.uint8)))
    else:
        bits = bits[:k_bits]
    return bits, len(byte_data), best_out['shape'], best_q

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
# 2. 主执行流水线：实验 1 白噪声传输 (含 Analog JSCC)
# ==========================================
def main():
    print("\n🚀 启动实验 1 传输程序: AWGN 白噪声信道 (包含 Analog Deep JSCC 基线)")
    
    with open('config_exp1.json', 'r', encoding='utf-8') as f:
        EXP1_CFG = json.load(f)["experiment1_awgn_params"]
    with open('config.json', 'r', encoding='utf-8') as f:
        MAIN_CFG = json.load(f)

    # 固定全局随机种子
    SEED = EXP1_CFG.get("random_seed", 1089)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 初始化传统卷积码
    memory = np.array(EXP1_CFG.get("cc_memory", [4]))
    g_matrix = np.array(EXP1_CFG.get("cc_g_matrix", [[19, 29]]))
    trellis = cc.Trellis(memory, g_matrix)
    E2E_SYMBOL_LENGTH = EXP1_CFG.get("e2e_symbol_length", 131072)
    K_BITS = (E2E_SYMBOL_LENGTH // 2) - memory[0]
    BUDGET_BYTES = K_BITS // 8

    print("🤖 正在加载 DL-SSCC (CompressAI) 模型...")
    ai_models = {q: bmshj2018_factorized(quality=q, pretrained=True).eval().to(device) for q in range(1, 9)}
    
    print("🧠 正在加载 Ours (E2E Semantic) 离散三值收发模型...")
    tx_model = ImageSemanticEncoder().to(device).eval()
    rx_model = ImageSemanticDecoder().to(device).eval()
    tx_model.load_state_dict(torch.load(MAIN_CFG["paths"]["tx_weights"], map_location=device))
    rx_model.load_state_dict(torch.load(MAIN_CFG["paths"]["rx_weights"], map_location=device))

    # =========================================================
    # [新增] 加载 Analog Deep JSCC 基线模型
    # =========================================================
    print("🌊 正在加载 Analog Deep JSCC (纯模拟连续传输) 基线模型...")
    analog_tx_model = ImageSemanticEncoder().to(device).eval()
    analog_rx_model = ImageSemanticDecoder().to(device).eval()
    try:
        analog_tx_model.load_state_dict(torch.load(MAIN_CFG["paths"]["analog_tx_weights"], map_location=device))
        analog_rx_model.load_state_dict(torch.load(MAIN_CFG["paths"]["analog_rx_weights"], map_location=device))
        print("  ✅ Analog 权重加载成功！")
    except Exception as e:
        print(f"  ⚠️ 警告: 无法加载 Analog 权重，将使用随机初始化作为占位。({e})")

    print("📊 正在加载 S-CSI 物理特征盲估器 (XGBoost)...")
    scsi_scaler = joblib.load('scsi_scaler.pkl')
    scsi_xgb = xgb.XGBClassifier()
    scsi_xgb.load_model('scsi_xgboost.json')
    TEMP_MAPPING = {3: 1.0, 2: 0.8, 1: 0.3, 0: 0.1}

    test_files = glob.glob(os.path.join(EXP1_CFG["tx_source_dir"], "*.*"))
    test_files = [f for f in test_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    res = EXP1_CFG["target_resolution"]
    
    snr_levels = EXP1_CFG["snr_levels"]
    print(f"\n📦 共计测试图片 {len(test_files)} 张，遍历物理 SNR 个数: {len(snr_levels)}")

    for snr in snr_levels:
        out_dir_base = os.path.join(EXP1_CFG["output_base_dir"], f"snr_{snr}")
        # [新增] 在输出目录中创建 analog_jscc 文件夹
        for sys_name in ['jpeg', 'dl_sscc', 'ours', 'analog_jscc']: 
            os.makedirs(os.path.join(out_dir_base, sys_name), exist_ok=True)
        
        print(f"\n{'='*50}\n▶ 当前信道环境: AWGN SNR = {snr}dB")
        snr_linear = 10 ** (snr / 10.0)
        
        # 调制映射与基础参数计算
        mod_type = '4QAM' if snr >= MAIN_CFG["amc_thresholds"].get("4QAM", 5.0) else 'BPSK'
        fixed_tau = MAIN_CFG["tau_config"].get(mod_type, 0.4)
        noise_power = 1.0 / snr_linear
        
        # 业务数据面动态截断阈值
        theoretical_tau = (noise_power * np.log(3)) / 2.0
        decode_tau = min(theoretical_tau, 0.7)

        for img_path in test_files:
            img_name = os.path.basename(img_path)
            name_no_ext, _ = os.path.splitext(img_name)
            png_name = f"{name_no_ext}.png"
            
            img_bgr = safe_imread_and_resize(img_path, res)
            if img_bgr is None: continue
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            tensor_img = T.ToTensor()(img_rgb).unsqueeze(0).to(device)

            # =================================================================
            # 【控制面】：物理导频 (Pilot) 与 XGBoost 动态控温
            # =================================================================
            pilot_tx = np.random.choice([-1.0, 1.0], E2E_SYMBOL_LENGTH).astype(np.float32)
            pilot_noise = np.sqrt(noise_power) * np.random.randn(E2E_SYMBOL_LENGTH).astype(np.float32)
            pilot_rx = pilot_tx + pilot_noise 
            
            pilot_evm = np.mean(np.abs(pilot_rx - np.sign(pilot_rx)))
            pilot_tar = np.mean(np.abs(pilot_rx) < fixed_tau) * 100.0
            
            features = np.array([[pilot_evm, snr, pilot_tar]])
            features_scaled = scsi_scaler.transform(features)
            scsi_state = int(scsi_xgb.predict(features_scaled)[0])
            
            # AWGN 兜底策略
            effective_snr = snr - 2.0
            if scsi_state == 2 and effective_snr >= 10.0:
                scsi_state = 3
                
            dynamic_temp = TEMP_MAPPING.get(scsi_state, 1.0)

            # ================== 1. 信号发送与调制 ==================
            tx_bits_jpg, act_bytes_jpg = compress_jpeg_to_budget(img_bgr, BUDGET_BYTES, K_BITS)
            sym_jpg = bpsk_modulate(cc.conv_encode(tx_bits_jpg, trellis))
            
            tx_bits_ai, act_bytes_ai, shp_ai, q_ai = compress_ai_to_budget(tensor_img, BUDGET_BYTES, K_BITS, ai_models)
            sym_ai = bpsk_modulate(cc.conv_encode(tx_bits_ai, trellis))
            
            with torch.no_grad():
                # Ours: 离散特征发送 (带动态温度掩码)
                sym_ours_tensor = tx_model(tensor_img, mod_type=mod_type, temperature=dynamic_temp)
                sym_ours = sym_ours_tensor.cpu().numpy().flatten().astype(np.float32)

                # [新增] Analog JSCC: 连续特征发送与功率归一化
                # 使用固定的温度 1.0，提取纯特征，不执行硬判决量化
                z_analog = analog_tx_model(tensor_img, mod_type='BPSK', temperature=1.0)
                
                # 【核心公式】：L2 功率归一化，使得每个符号的平均功率为 1
                N_symbols = z_analog.numel() # 131072
                norm_val = torch.norm(z_analog, p=2)
                x_analog = np.sqrt(N_symbols) * (z_analog / (norm_val + 1e-9))
                
                sym_analog = x_analog.cpu().numpy().flatten().astype(np.float32)

            # ================== 2. 物理信道加噪 ==================
            awgn_base = np.sqrt(noise_power) * np.random.randn(E2E_SYMBOL_LENGTH).astype(np.float32)
            rx_jpg  = sym_jpg  + awgn_base
            rx_ai   = sym_ai   + awgn_base
            
            awgn_ours = np.sqrt(noise_power) * np.random.randn(len(sym_ours)).astype(np.float32)
            rx_ours = sym_ours + awgn_ours
            
            # [新增] Analog JSCC 共享统一的物理噪声底分布
            awgn_analog = np.sqrt(noise_power) * np.random.randn(len(sym_analog)).astype(np.float32)
            rx_analog = sym_analog + awgn_analog 

            # ================== 3. 信号解码与恢复 ==================
            
            # --- JPEG ---
            dec_bits_jpg = cc.viterbi_decode(hard_demodulate(rx_jpg), trellis, decoding_type='hard')[:K_BITS]
            valid_len_jpg = act_bytes_jpg * 8
            ber_jpg = np.mean(dec_bits_jpg[:valid_len_jpg] != tx_bits_jpg[:valid_len_jpg])
            if ber_jpg > 0.05: 
                rec_jpg = np.zeros_like(img_bgr)
            else:
                rec_jpg = decompress_jpeg(dec_bits_jpg, act_bytes_jpg)
                if rec_jpg is None: rec_jpg = np.zeros_like(img_bgr)

            # --- DL-SSCC ---
            dec_bits_ai = cc.viterbi_decode(hard_demodulate(rx_ai), trellis, decoding_type='hard')[:K_BITS]
            valid_len_ai = act_bytes_ai * 8
            ber_ai = np.mean(dec_bits_ai[:valid_len_ai] != tx_bits_ai[:valid_len_ai])
            if ber_ai > 0.01:
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

            # --- [新增] Analog JSCC 解码 ---
            with torch.no_grad():
                # 无需三值阈值门限，将带有连续模拟噪声的信号直接喂给解码器
                rx_analog_tensor = torch.tensor(rx_analog, device=device).view(z_analog.shape)
                rec_tensor_analog, _ = analog_rx_model(rx_analog_tensor, return_sparsity=True)
                
                rec_analog_np = (rec_tensor_analog.clamp(0,1).cpu().squeeze(0).numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
                rec_analog = cv2.cvtColor(rec_analog_np, cv2.COLOR_RGB2BGR)

            # --- 保存结果 ---
            cv2.imwrite(os.path.join(out_dir_base, 'jpeg', png_name), rec_jpg)
            cv2.imwrite(os.path.join(out_dir_base, 'dl_sscc', png_name), rec_ai)
            cv2.imwrite(os.path.join(out_dir_base, 'ours', png_name), rec_ours)
            # 保存 Analog JSCC 图片
            cv2.imwrite(os.path.join(out_dir_base, 'analog_jscc', png_name), rec_analog)

        print(f"✅ 完成当前 SNR 组合: 图像已落盘至 {out_dir_base}")

if __name__ == "__main__":
    main()