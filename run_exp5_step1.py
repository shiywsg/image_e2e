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

from image_e2e_model import ImageSemanticEncoder, ImageSemanticDecoder

warnings.filterwarnings("ignore")

# ==========================================
# 1. 辅助函数：基线编解码与防崩溃保护
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
# 2. 主执行流水线：实验 5 快衰落传输
# ==========================================
def main():
    print("\n🚀 启动实验 5 传输程序: 瑞利快衰落信道 (M=64 块衰落 | Analog JSCC 防崩溃完美版)")
    
    with open('config_exp5.json', 'r', encoding='utf-8') as f:
        EXP5_CFG = json.load(f)["experiment5_fast_rayleigh_params"]
    with open('config.json', 'r', encoding='utf-8') as f:
        MAIN_CFG = json.load(f)

    SEED = EXP5_CFG["random_seed"]
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    memory = np.array(EXP5_CFG["cc_memory"])
    g_matrix = np.array(EXP5_CFG["cc_g_matrix"])
    trellis = cc.Trellis(memory, g_matrix)
    E2E_SYMBOL_LENGTH = EXP5_CFG["e2e_symbol_length"]
    K_BITS = (E2E_SYMBOL_LENGTH // 2) - memory[0]
    BUDGET_BYTES = K_BITS // 8
    NUM_CHUNKS = EXP5_CFG["num_chunks"]

    print("🤖 正在加载传统编解码基线...")
    ai_models = {q: bmshj2018_factorized(quality=q, pretrained=True).eval().to(device) for q in range(1, 9)}
    
    tx_model = ImageSemanticEncoder().to(device).eval()
    rx_model = ImageSemanticDecoder().to(device).eval()
    tx_model.load_state_dict(torch.load(MAIN_CFG["paths"]["tx_weights"], map_location=device))
    rx_model.load_state_dict(torch.load(MAIN_CFG["paths"]["rx_weights"], map_location=device))

    print("🌊 正在加载 Analog Deep JSCC...")
    analog_tx_model = ImageSemanticEncoder().to(device).eval()
    analog_rx_model = ImageSemanticDecoder().to(device).eval()
    try:
        analog_tx_model.load_state_dict(torch.load(MAIN_CFG["paths"]["analog_tx_weights"], map_location=device))
        analog_rx_model.load_state_dict(torch.load(MAIN_CFG["paths"]["analog_rx_weights"], map_location=device))
    except Exception as e:
        print(f"  ⚠️ 警告: 无法加载 Analog 权重 ({e})")

    print("📊 正在加载 S-CSI XGBoost...")
    scsi_scaler = joblib.load('scsi_scaler.pkl')
    scsi_xgb = xgb.XGBClassifier()
    scsi_xgb.load_model('scsi_xgboost.json')
    TEMP_MAPPING = {3: 1.0, 2: 0.8, 1: 0.3, 0: 0.1}

    test_files = glob.glob(os.path.join(EXP5_CFG["tx_source_dir"], "*.*"))
    test_files = [f for f in test_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    res = EXP5_CFG["target_resolution"]
    avg_snr_levels = EXP5_CFG["snr_levels"]

    for avg_snr in avg_snr_levels:
        out_dir_base = os.path.join(EXP5_CFG["output_base_dir"], f"snr_{avg_snr}")
        for sys_name in ['jpeg', 'dl_sscc', 'ours', 'analog_jscc']: 
            os.makedirs(os.path.join(out_dir_base, sys_name), exist_ok=True)
        
        print(f"\n{'='*50}\n▶ 当前信道环境: 瑞利快衰落 (M={NUM_CHUNKS}) | 平均 SNR = {avg_snr}dB")
        avg_snr_linear = 10 ** (avg_snr / 10.0)
        mod_type = '4QAM' if avg_snr >= MAIN_CFG["amc_thresholds"].get("4QAM", 5.0) else 'BPSK'
        fixed_tau = MAIN_CFG["tau_config"].get(mod_type, 0.4)
        avg_noise_power = 1.0 / avg_snr_linear

        for img_path in test_files:
            img_name = os.path.basename(img_path)
            png_name = f"{os.path.splitext(img_name)[0]}.png"
            
            img_bgr = safe_imread_and_resize(img_path, res)
            if img_bgr is None: continue
            tensor_img = T.ToTensor()(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)

            # --- 模拟导频快衰落以提取 XGBoost 特征 ---
            pilot_tx = np.random.choice([-1.0, 1.0], E2E_SYMBOL_LENGTH).astype(np.float32)
            pilot_rx = np.zeros_like(pilot_tx)
            pilot_chunk_size = E2E_SYMBOL_LENGTH // NUM_CHUNKS
            
            for m in range(NUM_CHUNKS):
                h_mag = np.sqrt((np.random.randn()/np.sqrt(2))**2 + (np.random.randn()/np.sqrt(2))**2)
                inst_n_power = avg_noise_power / (h_mag**2 + 1e-9)
                p_s = m * pilot_chunk_size
                p_e = (m + 1) * pilot_chunk_size if m < NUM_CHUNKS - 1 else E2E_SYMBOL_LENGTH
                pilot_rx[p_s:p_e] = pilot_tx[p_s:p_e] + np.sqrt(inst_n_power) * np.random.randn(p_e - p_s).astype(np.float32)
                
            pilot_evm = np.mean(np.abs(pilot_rx - np.sign(pilot_rx)))
            pilot_tar = np.mean(np.abs(pilot_rx) < fixed_tau) * 100.0
            
            features_scaled = scsi_scaler.transform(np.array([[pilot_evm, avg_snr, pilot_tar]]))
            scsi_state = int(scsi_xgb.predict(features_scaled)[0])
            
            if scsi_state == 2 and (avg_snr - 2.0) >= 10.0: scsi_state = 3
            dynamic_temp = TEMP_MAPPING.get(scsi_state, 1.0)

            # ================== 信号调制 ==================
            tx_bits_jpg, act_bytes_jpg, over_budget_jpg = compress_jpeg_to_budget(img_bgr, BUDGET_BYTES, K_BITS)
            sym_jpg = bpsk_modulate(cc.conv_encode(tx_bits_jpg, trellis))
            
            tx_bits_ai, act_bytes_ai, shp_ai, q_ai, over_budget_ai = compress_ai_to_budget(tensor_img, BUDGET_BYTES, K_BITS, ai_models)
            sym_ai = bpsk_modulate(cc.conv_encode(tx_bits_ai, trellis))
            
            with torch.no_grad():
                # Ours
                sym_ours_tensor = tx_model(tensor_img, mod_type=mod_type, temperature=dynamic_temp)
                sym_ours = sym_ours_tensor.cpu().numpy().flatten().astype(np.float32)
                
                # Analog JSCC
                z_analog = analog_tx_model(tensor_img, mod_type='BPSK', temperature=1.0)
                norm_val = torch.norm(z_analog, p=2)
                x_analog = np.sqrt(z_analog.numel()) * (z_analog / (norm_val + 1e-9))
                sym_analog = x_analog.cpu().numpy().flatten().astype(np.float32)

            # ================== 分块快衰落加噪与动态擦除 ==================
            rx_jpg = np.zeros_like(sym_jpg)
            rx_ai = np.zeros_like(sym_ai)
            rx_ours = np.zeros_like(sym_ours)
            rx_ana = np.zeros_like(sym_analog)
            is_ambiguous_flat = np.zeros_like(sym_ours, dtype=bool)
            
            base_chunk_size = E2E_SYMBOL_LENGTH // NUM_CHUNKS
            ours_chunk_size = len(sym_ours) // NUM_CHUNKS

            for m in range(NUM_CHUNKS):
                h_mag = np.sqrt((np.random.randn()/np.sqrt(2))**2 + (np.random.randn()/np.sqrt(2))**2)
                inst_noise_power = avg_noise_power / (h_mag**2 + 1e-9)
                
                decode_tau = min((inst_noise_power * np.log(3)) / 2.0, 0.85)
                
                # 处理基线 (定长)
                b_s = m * base_chunk_size
                b_e = (m + 1) * base_chunk_size if m < NUM_CHUNKS - 1 else E2E_SYMBOL_LENGTH
                awgn_base = np.sqrt(inst_noise_power) * np.random.randn(b_e - b_s).astype(np.float32)
                rx_jpg[b_s:b_e] = sym_jpg[b_s:b_e] + awgn_base
                rx_ai[b_s:b_e] = sym_ai[b_s:b_e] + awgn_base
                
                # 处理 Ours & Analog (变长)
                o_s = m * ours_chunk_size
                o_e = (m + 1) * ours_chunk_size if m < NUM_CHUNKS - 1 else len(sym_ours)
                n_sem = np.sqrt(inst_noise_power) * np.random.randn(o_e - o_s).astype(np.float32)
                
                chunk_rx_ours = sym_ours[o_s:o_e] + n_sem
                rx_ours[o_s:o_e] = chunk_rx_ours
                is_ambiguous_flat[o_s:o_e] = np.abs(chunk_rx_ours) < decode_tau
                
                rx_ana[o_s:o_e] = sym_analog[o_s:o_e] + n_sem

            # ================== 信号解码与恢复 ==================
            
            # --- JPEG ---
            dec_bits_jpg = cc.viterbi_decode(hard_demodulate(rx_jpg), trellis, decoding_type='hard')[:K_BITS]
            if np.mean(dec_bits_jpg[:act_bytes_jpg * 8] != tx_bits_jpg[:act_bytes_jpg * 8]) > 0.05 or over_budget_jpg: 
                rec_jpg = np.zeros_like(img_bgr)
            else:
                rec_jpg = decompress_jpeg(dec_bits_jpg, act_bytes_jpg)
                if rec_jpg is None: rec_jpg = np.zeros_like(img_bgr)

            # --- DL-SSCC ---
            dec_bits_ai = cc.viterbi_decode(hard_demodulate(rx_ai), trellis, decoding_type='hard')[:K_BITS]
            if np.mean(dec_bits_ai[:act_bytes_ai * 8] != tx_bits_ai[:act_bytes_ai * 8]) > 0.01 or over_budget_ai:
                rec_ai = np.zeros_like(img_bgr)
            else:
                rec_ai = decompress_ai(dec_bits_ai, act_bytes_ai, shp_ai, q_ai, ai_models)
                if rec_ai is None: rec_ai = np.zeros_like(img_bgr)

            # --- Ours ---
            with torch.no_grad():
                rx_ours_tensor = torch.tensor(rx_ours, device=device).view(sym_ours_tensor.shape)
                amb_tensor = torch.tensor(is_ambiguous_flat, device=device).view(sym_ours_tensor.shape)
                ternary_symbols = torch.where(amb_tensor, torch.zeros_like(rx_ours_tensor), torch.sign(rx_ours_tensor))
                
                rec_tensor_ours, _ = rx_model(ternary_symbols, return_sparsity=True)
                rec_ours_np = (rec_tensor_ours.clamp(0,1).cpu().squeeze(0).numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
                rec_ours = cv2.cvtColor(rec_ours_np, cv2.COLOR_RGB2BGR)

            # --- Analog JSCC ---
            with torch.no_grad():
                rx_ana_tensor = torch.tensor(rx_ana, device=device).view(z_analog.shape)
                rec_tensor_ana, _ = analog_rx_model(rx_ana_tensor, return_sparsity=True)
                rec_ana_np = (rec_tensor_ana.clamp(0,1).cpu().squeeze(0).numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
                rec_ana = cv2.cvtColor(rec_ana_np, cv2.COLOR_RGB2BGR)

            # --- 保存 ---
            cv2.imwrite(os.path.join(out_dir_base, 'jpeg', png_name), rec_jpg)
            cv2.imwrite(os.path.join(out_dir_base, 'dl_sscc', png_name), rec_ai)
            cv2.imwrite(os.path.join(out_dir_base, 'ours', png_name), rec_ours)
            cv2.imwrite(os.path.join(out_dir_base, 'analog_jscc', png_name), rec_ana)

        print(f"✅ 完成当前平均 SNR 组合: 图像已落盘至 {out_dir_base}")

if __name__ == "__main__":
    main()
