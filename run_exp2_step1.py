import os
import cv2
import json
import torch
import numpy as np
import glob
import random
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
# 1. 辅助函数：基线编解码与安全读取
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
        best_encoded = best_encoded[:budget_bytes]
    bits = np.unpackbits(best_encoded)
    if len(bits) < k_bits: bits = np.concatenate((bits, np.zeros(k_bits - len(bits), dtype=np.uint8)))
    else: bits = bits[:k_bits]
    return bits, len(best_encoded)

def decompress_jpeg(bits, actual_bytes):
    try:
        byte_data = np.packbits(bits[:actual_bytes * 8]).tobytes()
        return cv2.imdecode(np.frombuffer(byte_data, np.uint8), cv2.IMREAD_COLOR)
    except: return None 

def compress_ai(img_tensor, budget_bytes, k_bits, ai_models):
    best_q, best_out = 0, None
    for q in range(8, 0, -1):
        with torch.no_grad():
            out_enc = ai_models[q].compress(img_tensor)
            if len(out_enc['strings'][0][0]) <= budget_bytes:
                best_q, best_out = q, out_enc
                break
    if best_out is None:
        with torch.no_grad(): best_out = ai_models[1].compress(img_tensor)
        best_q = 1
        best_out['strings'][0] = [best_out['strings'][0][0][:budget_bytes]]
    byte_data = best_out['strings'][0][0]
    bits = np.unpackbits(np.frombuffer(byte_data, dtype=np.uint8))
    if len(bits) < k_bits: bits = np.concatenate((bits, np.zeros(k_bits - len(bits), dtype=np.uint8)))
    else: bits = bits[:k_bits]
    return bits, len(byte_data), best_out['shape'], best_q

def decompress_ai(bits, actual_bytes, shape, q_used, ai_models):
    try:
        byte_data = np.packbits(bits[:actual_bytes * 8]).tobytes()
        out_enc = {'strings': [[byte_data]], 'shape': shape}
        with torch.no_grad(): out_dec = ai_models[q_used].decompress(out_enc['strings'], out_enc['shape'])
        rec_img = out_dec['x_hat'].clamp(0, 1)[0]
        rec_img_np = (rec_img.cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
        return cv2.cvtColor(rec_img_np, cv2.COLOR_RGB2BGR)
    except: return None

def safe_imread_and_resize(path, size):
    """支持中文路径的安全读取函数"""
    try:
        img_data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img is not None:
            return cv2.resize(img, (size, size))
    except Exception as e:
        pass
    return None

# ==========================================
# 2. 核心：突发脉冲干扰生成器
# ==========================================
def generate_burst_impulse_noise(total_length, prob, burst_length, impulse_std):
    """
    生成突发脉冲干扰数组 (支持动态长度)
    """
    noise = np.zeros(total_length, dtype=np.float32)
    if prob <= 0.0:
        return noise
        
    target_corrupted = int(total_length * prob)
    num_bursts = max(1, target_corrupted // burst_length)
    
    for _ in range(num_bursts):
        start = np.random.randint(0, max(1, total_length - burst_length))
        burst_noise = np.random.normal(0, impulse_std, burst_length).astype(np.float32)
        noise[start : start + burst_length] = burst_noise
        
    return noise

# ==========================================
# 3. 主执行流水线：完全体实验 2 传输程序 (集成 Analog JSCC)
# ==========================================
def main():
    print("\n🚀 启动完全体实验 2: 突发脉冲干扰传输程序 (集成 Analog Deep JSCC 基线)")
    
    with open('config_exp2.json', 'r', encoding='utf-8') as f:
        EXP2_CFG = json.load(f)["experiment2_impulse_params"]
    
    with open('config.json', 'r', encoding='utf-8') as f:
        MAIN_CFG = json.load(f)

    # 固定全局随机种子
    SEED = EXP2_CFG["random_seed"]
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    memory = np.array(EXP2_CFG["cc_memory"])
    g_matrix = np.array(EXP2_CFG["cc_g_matrix"])
    trellis = cc.Trellis(memory, g_matrix)
    E2E_SYMBOL_LENGTH = EXP2_CFG["e2e_symbol_length"]
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

    print("📊 正在加载 S-CSI 物理特征盲估器 (XGBoost 4分类)...")
    scsi_scaler = joblib.load('scsi_scaler.pkl')
    scsi_xgb = xgb.XGBClassifier()
    scsi_xgb.load_model('scsi_xgboost.json')
    TEMP_MAPPING = {3: 1.0, 2: 0.8, 1: 0.3, 0: 0.1}

    test_files = glob.glob(os.path.join(MAIN_CFG["tx_source_dir"], "*.*"))
    test_files = [f for f in test_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:30] # 截取前30张测试
    res = EXP2_CFG["target_resolution"]

    fixed_snr = EXP2_CFG["fixed_snr_db"]
    snr_linear = 10 ** (fixed_snr / 10.0)
    noise_power = 1.0 / snr_linear
    
    probs = EXP2_CFG["burst_probabilities"]
    burst_len = EXP2_CFG["burst_length"]
    impulse_std = EXP2_CFG["impulse_std"]
    
    # 【架构对齐】：预计算调制方式与解调双轨阈值
    mod_type = '4QAM' if fixed_snr >= MAIN_CFG["amc_thresholds"].get("4QAM", 5.0) else 'BPSK'
    fixed_tau = MAIN_CFG["tau_config"].get(mod_type, 0.4)
    
    theoretical_tau = (noise_power * np.log(3)) / 2.0
    decode_tau = min(theoretical_tau, 0.7)

    print(f"\n📦 共计测试图片 {len(test_files)} 张，扫描脉冲概率个数: {len(probs)}")

    for prob in probs:
        prob_percent = int(prob * 100)
        out_dir_base = os.path.join(EXP2_CFG["output_base_dir"], f"prob_{prob_percent}")
        
        # [新增] 在输出目录中创建 analog_jscc 文件夹
        for sys_name in ['jpeg', 'dl_sscc', 'ours', 'analog_jscc']: 
            os.makedirs(os.path.join(out_dir_base, sys_name), exist_ok=True)
        
        print(f"\n{'='*50}\n▶ 环境状态: SNR={fixed_snr}dB | 脉冲概率={prob_percent}% | 动态解调Tau={decode_tau:.3f}")

        for img_path in test_files:
            img_name = os.path.basename(img_path)
            # 强制统一保存为 PNG (无损格式)
            name_no_ext, _ = os.path.splitext(img_name)
            png_name = f"{name_no_ext}.png"
            
            img_bgr = safe_imread_and_resize(img_path, res)
            if img_bgr is None: continue
                
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            tensor_img = T.ToTensor()(img_rgb).unsqueeze(0).to(device)

            # =================================================================
            # 【控制面】：利用前置物理导频 (Pilot) 执行 XGBoost 掩码温度推断
            # =================================================================
            # 1. 产生当前信道的本帧 AWGN 和 突发脉冲干扰 (基于固定的导频长度)
            awgn_base = np.sqrt(noise_power) * np.random.randn(E2E_SYMBOL_LENGTH).astype(np.float32)
            impulse_base = generate_burst_impulse_noise(E2E_SYMBOL_LENGTH, prob, burst_len, impulse_std)
            
            # 2. 假设发送一段随机导频被混合噪声干扰
            pilot_tx = np.random.choice([-1.0, 1.0], E2E_SYMBOL_LENGTH).astype(np.float32)
            pilot_rx = pilot_tx + awgn_base + impulse_base 
            
            # 3. 接收端使用基准固定刻度 (fixed_tau) 提取特征
            pilot_evm = np.mean(np.abs(pilot_rx - np.sign(pilot_rx)))
            pilot_tar = np.mean(np.abs(pilot_rx) < fixed_tau) * 100.0
            
            # 4. XGBoost 进行特征降维与 S-CSI 状态推断
            features = np.array([[pilot_evm, fixed_snr, pilot_tar]])
            features_scaled = scsi_scaler.transform(features)
            scsi_state = int(scsi_xgb.predict(features_scaled)[0])
            
            # 5. 物理常识兜底覆盖并激活“动态掩码温度”
            if scsi_state == 2 and fixed_snr >= 10.0:
                scsi_state = 3
            dynamic_temp = TEMP_MAPPING.get(scsi_state, 1.0)

            # ================== 1. 信号生成 ==================
            tx_bits_jpg, act_bytes_jpg = compress_jpeg(img_bgr, BUDGET_BYTES, K_BITS)
            sym_jpg = np.where(cc.conv_encode(tx_bits_jpg, trellis) == 0, -1.0, 1.0).astype(np.float32)
            
            tx_bits_ai, act_bytes_ai, shp_ai, q_ai = compress_ai(tensor_img, BUDGET_BYTES, K_BITS, ai_models)
            sym_ai = np.where(cc.conv_encode(tx_bits_ai, trellis) == 0, -1.0, 1.0).astype(np.float32)
            
            with torch.no_grad():
                sym_ours_tensor = tx_model(tensor_img, mod_type=mod_type, temperature=dynamic_temp)
                sym_ours = sym_ours_tensor.cpu().numpy().flatten().astype(np.float32)

                # [新增] Analog JSCC: 连续特征发送与功率归一化
                # 使用固定的温度 1.0，提取纯特征，不执行硬判决量化
                z_analog = analog_tx_model(tensor_img, mod_type='BPSK', temperature=1.0)
                
                # 【核心公式】：L2 功率归一化，使得每个符号的平均功率为 1
                N_symbols = z_analog.numel()
                norm_val = torch.norm(z_analog, p=2)
                x_analog = np.sqrt(N_symbols) * (z_analog / (norm_val + 1e-9))
                
                sym_analog = x_analog.cpu().numpy().flatten().astype(np.float32)

            # ================== 2. 严苛信道注入 ==================
            # 传统方案长度等于 E2E_SYMBOL_LENGTH，直接使用 base 噪声
            rx_jpg  = sym_jpg  + awgn_base + impulse_base
            rx_ai   = sym_ai   + awgn_base + impulse_base
            
            # 【工程修复】：Ours 的输出维度可能不等于 E2E_SYMBOL_LENGTH，独立生成等长双干扰
            awgn_ours = np.sqrt(noise_power) * np.random.randn(len(sym_ours)).astype(np.float32)
            impulse_ours = generate_burst_impulse_noise(len(sym_ours), prob, burst_len, impulse_std)
            rx_ours = sym_ours + awgn_ours + impulse_ours

            # [新增] Analog JSCC 共享独立的等长双干扰机制
            awgn_analog = np.sqrt(noise_power) * np.random.randn(len(sym_analog)).astype(np.float32)
            impulse_analog = generate_burst_impulse_noise(len(sym_analog), prob, burst_len, impulse_std)
            rx_analog = sym_analog + awgn_analog + impulse_analog

            # ================== 3. 信号解码与恢复 ==================
            
            # --- JPEG ---
            dec_bits_jpg = cc.viterbi_decode(np.where(rx_jpg > 0, 1, 0), trellis, decoding_type='hard')[:K_BITS]
            valid_len_jpg = act_bytes_jpg * 8
            ber_jpg = np.mean(dec_bits_jpg[:valid_len_jpg] != tx_bits_jpg[:valid_len_jpg])
            
            if ber_jpg > 0.05: # JPEG 容忍 5% 的严重误码截断
                rec_jpg = np.zeros_like(img_bgr)
            else:
                rec_jpg = decompress_jpeg(dec_bits_jpg, act_bytes_jpg)
                if rec_jpg is None: rec_jpg = np.zeros_like(img_bgr)

            # --- DL-SSCC ---
            dec_bits_ai = cc.viterbi_decode(np.where(rx_ai > 0, 1, 0), trellis, decoding_type='hard')[:K_BITS]
            valid_len_ai = act_bytes_ai * 8
            ber_ai = np.mean(dec_bits_ai[:valid_len_ai] != tx_bits_ai[:valid_len_ai])
            
            if ber_ai > 0.01: # 保护算术解码器免于段错误
                rec_ai = np.zeros_like(img_bgr)
            else:
                rec_ai = decompress_ai(dec_bits_ai, act_bytes_ai, shp_ai, q_ai, ai_models)
                if rec_ai is None: rec_ai = np.zeros_like(img_bgr)

            # --- Ours (端到端 CNN) ---
            with torch.no_grad():
                rx_ours_tensor = torch.tensor(rx_ours, device=device).view(sym_ours_tensor.shape)
                
                # 【数据面】：使用严格的自适应边界 decode_tau 执行最优擦除
                is_ambiguous = torch.abs(rx_ours_tensor) < decode_tau
                ternary_symbols = torch.where(is_ambiguous, torch.zeros_like(rx_ours_tensor), torch.sign(rx_ours_tensor))
                
                rec_tensor_ours, _ = rx_model(ternary_symbols, return_sparsity=True)
                rec_ours_np = (rec_tensor_ours.clamp(0,1).cpu().squeeze(0).numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
                rec_ours = cv2.cvtColor(rec_ours_np, cv2.COLOR_RGB2BGR)

            # --- [新增] Analog JSCC 解码 ---
            with torch.no_grad():
                # 无需三值阈值门限擦除，将带有连续突发噪声和AWGN的混合信号直接喂给解码器
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

        print(f"✅ 完成当前脉冲概率 ({prob_percent}%) 图像落盘: {out_dir_base}")

if __name__ == "__main__":
    main()
