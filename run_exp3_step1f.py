import os
import cv2
import json
import torch
import numpy as np
import glob
import random
import itertools
import joblib
import xgboost as xgb
import commpy.channelcoding.convcode as cc
import torchvision.transforms as T
from compressai.zoo import bmshj2018_factorized
import warnings

# 导入端到端系统组件 (底层已集成 Instance Normalization)
from image_e2e_model import ImageSemanticEncoder, ImageSemanticDecoder

warnings.filterwarnings("ignore")

# ==========================================
# 1. 辅助函数：基线编解码与【硬核防崩溃】保护
# ==========================================
def compress_jpeg(img_array, budget_bytes, k_bits):
    low, high, best_encoded = 0, 100, None
    for _ in range(7): 
        mid = (low + high) // 2
        _, encimg = cv2.imencode('.jpg', img_array, [int(cv2.IMWRITE_JPEG_QUALITY), mid])
        if len(encimg) <= budget_bytes: best_encoded, low = encimg, mid + 1
        else: high = mid - 1
        
    if best_encoded is None:
        _, best_encoded = cv2.imencode('.jpg', img_array, [int(cv2.IMWRITE_JPEG_QUALITY), 0])
        
    # 【修复核心1】：标记是否超载，不盲目截断导致底层解码器崩溃
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
        byte_data = np.packbits(bits[:actual_bytes * 8]).tobytes()
        return cv2.imdecode(np.frombuffer(byte_data, np.uint8), cv2.IMREAD_COLOR)
    except Exception as e: 
        return None 

def compress_ai(img_tensor, budget_bytes, k_bits, ai_models):
    best_q, best_out = 0, None
    for q in range(8, 0, -1):
        with torch.no_grad():
            out_enc = ai_models[q].compress(img_tensor)
            if len(out_enc['strings'][0][0]) <= budget_bytes:
                best_q, best_out = q, out_enc
                break
                
    if best_out is None:
        with torch.no_grad(): 
            best_out = ai_models[1].compress(img_tensor)
        best_q = 1
        
    byte_data = best_out['strings'][0][0]
    
    # 【修复核心2】：记录超载状态，切断向 C++ 算术解码器输送被截断乱码的可能
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
        byte_data = np.packbits(bits[:actual_bytes * 8]).tobytes()
        out_enc = {'strings': [[byte_data]], 'shape': shape}
        with torch.no_grad(): 
            out_dec = ai_models[q_used].decompress(out_enc['strings'], out_enc['shape'])
        rec_img = out_dec['x_hat'].clamp(0, 1)[0]
        rec_img_np = (rec_img.cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
        return cv2.cvtColor(rec_img_np, cv2.COLOR_RGB2BGR)
    except Exception as e: 
        return None

def generate_legacy_16qam_numpy(length):
    """供基线定长导频使用的 Numpy 版 16QAM 发生器"""
    levels = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float32) / np.sqrt(5.0)
    return np.random.choice(levels, length).astype(np.float32)

def generate_legacy_16qam_torch(shape, device):
    """供 CNN 使用的 Torch 版 16QAM 发生器 (严格匹配目标张量维度)"""
    levels = torch.tensor([-3.0, -1.0, 1.0, 3.0], device=device) / np.sqrt(5.0)
    indices = torch.randint(0, 4, shape, device=device)
    return levels[indices]

def get_random_image_path(img_dir):
    files = glob.glob(os.path.join(img_dir, "**/*.*"), recursive=True)
    valid_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    return random.choice(valid_files) if valid_files else None

def safe_imread_and_resize(path, size):
    try:
        img_data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img is not None:
            return cv2.resize(img, (size, size))
    except Exception as e:
        pass
    return None

# ==========================================
# 2. 主执行流水线：完全体实验 3 传输程序 (集成 Analog JSCC & 防崩溃机制)
# ==========================================
def main():
    print("\n🚀 启动完全体实验 3: 多用户碰撞传输程序 (修复版: 彻底解决 C++ Segfault 崩溃)")
    
    with open('config_exp3.json', 'r', encoding='utf-8') as f:
        EXP3_CFG = json.load(f)["experiment3_params"]
    
    with open('config.json', 'r', encoding='utf-8') as f:
        MAIN_CFG = json.load(f)

    SEED = EXP3_CFG["random_seed"]
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    memory = np.array(EXP3_CFG["cc_memory"])
    g_matrix = np.array(EXP3_CFG["cc_g_matrix"])
    trellis = cc.Trellis(memory, g_matrix)
    E2E_SYMBOL_LENGTH = EXP3_CFG["e2e_symbol_length"]
    K_BITS = (E2E_SYMBOL_LENGTH // 2) - memory[0]
    BUDGET_BYTES = K_BITS // 8

    print("🤖 加载 DL-SSCC (CompressAI) 模型...")
    ai_models = {q: bmshj2018_factorized(quality=q, pretrained=True).eval().to(device) for q in range(1, 9)}
    
    print("🧠 加载 Ours (E2E Semantic) 模型...")
    tx_model = ImageSemanticEncoder().to(device).eval()
    rx_model = ImageSemanticDecoder().to(device).eval()
    tx_model.load_state_dict(torch.load(MAIN_CFG["paths"]["tx_weights"], map_location=device))
    rx_model.load_state_dict(torch.load(MAIN_CFG["paths"]["rx_weights"], map_location=device))

    print("🌊 加载 Analog Deep JSCC 基线模型...")
    analog_tx_model = ImageSemanticEncoder().to(device).eval()
    analog_rx_model = ImageSemanticDecoder().to(device).eval()
    try:
        analog_tx_model.load_state_dict(torch.load(MAIN_CFG["paths"]["analog_tx_weights"], map_location=device))
        analog_rx_model.load_state_dict(torch.load(MAIN_CFG["paths"]["analog_rx_weights"], map_location=device))
        print("  ✅ Analog 权重加载成功！")
    except Exception as e:
        print(f"  ⚠️ 警告: 无法加载 Analog 权重，将使用随机初始化 ({e})")

    print("📊 正在加载 S-CSI 物理特征盲估器...")
    scsi_scaler = joblib.load('scsi_scaler.pkl')
    scsi_xgb = xgb.XGBClassifier()
    scsi_xgb.load_model('scsi_xgboost.json')
    TEMP_MAPPING = {3: 1.0, 2: 0.8, 1: 0.3, 0: 0.1}

    test_files = glob.glob(os.path.join(EXP3_CFG["tx_source_dir"], "*.*"))
    test_files = [f for f in test_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:20] 
    res = EXP3_CFG["target_resolution"]

    snr_levels = EXP3_CFG["snr_levels"]
    sir_levels = EXP3_CFG["sir_levels"]
    i_types = EXP3_CFG["interference_types"]

    print(f"\n📦 共计测试图片 {len(test_files)} 张，参数组合数: {len(snr_levels)*len(sir_levels)*len(i_types)}")

    for i_type, snr, sir in itertools.product(i_types, snr_levels, sir_levels):
        out_dir_base = os.path.join(EXP3_CFG["output_base_dir"], i_type, f"snr_{snr}_sir_{sir}")
        for sys_name in ['jpeg', 'dl_sscc', 'ours', 'analog_jscc']: 
            os.makedirs(os.path.join(out_dir_base, sys_name), exist_ok=True)
        
        snr_linear = 10 ** (snr / 10.0)
        noise_power = 1.0 / snr_linear
        alpha_sir = 10 ** (-sir / 20.0)
        
        mod_type = '4QAM' if snr >= MAIN_CFG["amc_thresholds"].get("4QAM", 5.0) else 'BPSK'
        fixed_tau = MAIN_CFG["tau_config"].get(mod_type, 0.4)
        theoretical_tau = (noise_power * np.log(3)) / 2.0
        decode_tau = min(theoretical_tau, 0.7) 
        
        print(f"\n{'='*50}\n▶ 状态: 干扰={i_type} | SNR={snr}dB | SIR={sir}dB | 解调Tau={decode_tau:.3f}")

        for img_path in test_files:
            img_name = os.path.basename(img_path)
            name_no_ext, _ = os.path.splitext(img_name)
            png_name = f"{name_no_ext}.png"
            
            img_bgr_A = safe_imread_and_resize(img_path, res)
            if img_bgr_A is None: continue
                
            img_rgb_A = cv2.cvtColor(img_bgr_A, cv2.COLOR_BGR2RGB)
            tensor_A = T.ToTensor()(img_rgb_A).unsqueeze(0).to(device)

            img_bgr_B = None
            if i_type == 'same_domain':
                path_B = get_random_image_path(EXP3_CFG.get("same_domain_dir", "./dataset"))
                if path_B: img_bgr_B = safe_imread_and_resize(path_B, res)
            elif i_type == 'out_of_domain':
                path_B = get_random_image_path(EXP3_CFG.get("out_of_domain_dir", "./dataset"))
                if path_B: img_bgr_B = safe_imread_and_resize(path_B, res)
            
            if i_type in ['same_domain', 'out_of_domain'] and img_bgr_B is None:
                img_bgr_B = np.zeros((res, res, 3), dtype=np.uint8)

            tensor_B = None
            if img_bgr_B is not None:
                img_rgb_B = cv2.cvtColor(img_bgr_B, cv2.COLOR_BGR2RGB)
                tensor_B = T.ToTensor()(img_rgb_B).unsqueeze(0).to(device)

            # =================================================================
            # 【控制面】
            # =================================================================
            pilot_awgn = np.sqrt(noise_power) * np.random.randn(E2E_SYMBOL_LENGTH).astype(np.float32)
            pilot_tx = np.random.choice([-1.0, 1.0], E2E_SYMBOL_LENGTH).astype(np.float32)
            
            if i_type == 'digital_legacy':
                pilot_XB = generate_legacy_16qam_numpy(E2E_SYMBOL_LENGTH)
            else:
                pilot_XB = np.random.choice([-1.0, 1.0], E2E_SYMBOL_LENGTH).astype(np.float32)
                
            pilot_rx = pilot_tx + (alpha_sir * pilot_XB) + pilot_awgn
            
            pilot_evm = np.mean(np.abs(pilot_rx - np.sign(pilot_rx)))
            pilot_tar = np.mean(np.abs(pilot_rx) < fixed_tau) * 100.0
            
            features = np.array([[pilot_evm, snr, pilot_tar]])
            features_scaled = scsi_scaler.transform(features)
            scsi_state = int(scsi_xgb.predict(features_scaled)[0])
            
            if scsi_state == 2 and snr >= 10.0: scsi_state = 3
            dynamic_temp = TEMP_MAPPING.get(scsi_state, 1.0) 

            awgn_base = np.sqrt(noise_power) * np.random.randn(E2E_SYMBOL_LENGTH).astype(np.float32)

            # ================== A. JPEG 基线 ==================
            tx_bits_A_jpg, act_bytes_A_jpg, over_budget_A_jpg = compress_jpeg(img_bgr_A, BUDGET_BYTES, K_BITS)
            sym_A_jpg = np.where(cc.conv_encode(tx_bits_A_jpg, trellis) == 0, -1.0, 1.0).astype(np.float32)
            
            if i_type == 'digital_legacy': 
                sym_B_jpg = generate_legacy_16qam_numpy(E2E_SYMBOL_LENGTH)
            else:
                tx_bits_B_jpg, _, _ = compress_jpeg(img_bgr_B, BUDGET_BYTES, K_BITS)
                sym_B_jpg = np.where(cc.conv_encode(tx_bits_B_jpg, trellis) == 0, -1.0, 1.0).astype(np.float32)
            
            rx_jpg = sym_A_jpg + (alpha_sir * sym_B_jpg) + awgn_base
            dec_bits_jpg = cc.viterbi_decode(np.where(rx_jpg > 0, 1, 0), trellis, decoding_type='hard')[:K_BITS]
            
            valid_len_jpg = act_bytes_A_jpg * 8
            ber_jpg = np.mean(dec_bits_jpg[:valid_len_jpg] != tx_bits_A_jpg[:valid_len_jpg])
            
            # 【完美避障】: 若超载或误码过高，直接当做丢包(黑图)，防止 OpenCV 输出脏数据
            if ber_jpg > 0.05 or over_budget_A_jpg:
                rec_jpg = np.zeros_like(img_bgr_A)
            else:
                rec_jpg = decompress_jpeg(dec_bits_jpg, act_bytes_A_jpg)
                if rec_jpg is None: rec_jpg = np.zeros_like(img_bgr_A)

            # ================== B. DL-SSCC 基线 ==================
            tx_bits_A_ai, act_bytes_A_ai, shp_A, q_A, over_budget_A_ai = compress_ai(tensor_A, BUDGET_BYTES, K_BITS, ai_models)
            sym_A_ai = np.where(cc.conv_encode(tx_bits_A_ai, trellis) == 0, -1.0, 1.0).astype(np.float32)
            
            if i_type == 'digital_legacy': 
                sym_B_ai = generate_legacy_16qam_numpy(E2E_SYMBOL_LENGTH)
            else:
                tx_bits_B_ai, _, _, _, _ = compress_ai(tensor_B, BUDGET_BYTES, K_BITS, ai_models)
                sym_B_ai = np.where(cc.conv_encode(tx_bits_B_ai, trellis) == 0, -1.0, 1.0).astype(np.float32)
            
            rx_ai = sym_A_ai + (alpha_sir * sym_B_ai) + awgn_base
            dec_bits_ai = cc.viterbi_decode(np.where(rx_ai > 0, 1, 0), trellis, decoding_type='hard')[:K_BITS]
            
            valid_len_ai = act_bytes_A_ai * 8
            ber_ai = np.mean(dec_bits_ai[:valid_len_ai] != tx_bits_A_ai[:valid_len_ai])
            
            # 【绝对核心防御】：算术编码(ANS)对任意比特翻转和截断极度敏感。
            # 如果 ber_ai > 0 (信道没纠回来) 或 over_budget (超出物理带宽被截断)，
            # 直接丢包(黑图)。强行喂给 C++ decompress 会引发底层段错误(Segfault)导致进程暴毙！
            if ber_ai > 0.0 or over_budget_A_ai:
                rec_ai = np.zeros_like(img_bgr_A)
            else:
                rec_ai = decompress_ai(dec_bits_ai, act_bytes_A_ai, shp_A, q_A, ai_models)
                if rec_ai is None: rec_ai = np.zeros_like(img_bgr_A)

            # ================== C. Ours 端到端链路 ==================
            with torch.no_grad():
                sym_A_ours = tx_model(tensor_A, mod_type=mod_type, temperature=dynamic_temp)
                
                if i_type == 'digital_legacy':
                    sym_B_ours = generate_legacy_16qam_torch(sym_A_ours.shape, device)
                else:
                    sym_B_ours = tx_model(tensor_B, mod_type=mod_type, temperature=1.0)
                
                noise_torch = torch.sqrt(torch.tensor(noise_power, device=device)) * torch.randn_like(sym_A_ours)
                rx_ours = sym_A_ours + (alpha_sir * sym_B_ours) + noise_torch
                
                is_ambiguous = torch.abs(rx_ours) < decode_tau
                ternary_symbols = torch.where(is_ambiguous, torch.zeros_like(rx_ours), torch.sign(rx_ours))
                
                rec_tensor_ours, _ = rx_model(ternary_symbols, return_sparsity=True)
                rec_ours_np = (rec_tensor_ours.clamp(0,1).cpu().squeeze(0).numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
                rec_ours = cv2.cvtColor(rec_ours_np, cv2.COLOR_RGB2BGR)

            # ================== D. Analog JSCC 基线 ==================
            with torch.no_grad():
                z_analog_A = analog_tx_model(tensor_A, mod_type='BPSK', temperature=1.0)
                norm_val_A = torch.norm(z_analog_A, p=2)
                sym_A_analog = np.sqrt(z_analog_A.numel()) * (z_analog_A / (norm_val_A + 1e-9))
                
                if i_type == 'digital_legacy':
                    sym_B_analog = generate_legacy_16qam_torch(sym_A_analog.shape, device)
                else:
                    z_analog_B = analog_tx_model(tensor_B, mod_type='BPSK', temperature=1.0)
                    norm_val_B = torch.norm(z_analog_B, p=2)
                    sym_B_analog = np.sqrt(z_analog_B.numel()) * (z_analog_B / (norm_val_B + 1e-9))
                
                noise_analog = torch.sqrt(torch.tensor(noise_power, device=device)) * torch.randn_like(sym_A_analog)
                rx_analog = sym_A_analog + (alpha_sir * sym_B_analog) + noise_analog
                
                rec_tensor_analog, _ = analog_rx_model(rx_analog, return_sparsity=True)
                rec_analog_np = (rec_tensor_analog.clamp(0,1).cpu().squeeze(0).numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
                rec_analog = cv2.cvtColor(rec_analog_np, cv2.COLOR_RGB2BGR)

            # --- 保存结果 ---
            cv2.imwrite(os.path.join(out_dir_base, 'jpeg', png_name), rec_jpg)
            cv2.imwrite(os.path.join(out_dir_base, 'dl_sscc', png_name), rec_ai)
            cv2.imwrite(os.path.join(out_dir_base, 'ours', png_name), rec_ours)
            cv2.imwrite(os.path.join(out_dir_base, 'analog_jscc', png_name), rec_analog)

        print(f"✅ 完成当前组合: 生成图片已落盘至 {out_dir_base}")

if __name__ == "__main__":
    main()
