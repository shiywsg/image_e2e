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
from PIL import Image

try:
    from diffusers import AutoPipelineForImage2Image
except ImportError:
    print("\033[91m[错误] 请先安装 diffusers 库: pip install diffusers transformers accelerate\033[0m")
    exit(1)

from compressai.zoo import bmshj2018_factorized
import warnings

# 导入端到端系统组件
from image_e2e_model import ImageSemanticEncoder, ImageSemanticDecoder

warnings.filterwarnings("ignore")

# ==========================================
# 1. 辅助函数 (保持不变)
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
# 2. 主执行流水线：引入 Diffusion 增强验证
# ==========================================
def main():
    print("\n🚀 启动扩散模型增强 (Diffusion POC) 实验 1 传输程序")
    
    with open('config_exp1.json', 'r', encoding='utf-8') as f:
        EXP1_CFG = json.load(f)["experiment1_awgn_params"]
    with open('config.json', 'r', encoding='utf-8') as f:
        MAIN_CFG = json.load(f)

    SEED = EXP1_CFG["random_seed"]
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # =====================================================================
    # 🌟 核心：加载轻量级、低显存、低延迟的扩散模型 (SD-Turbo)
    # 耗时评估: T400 首次加载约30秒，后续单张推理约 0.8 ~ 1.5 秒
    # =====================================================================
    print("🪄 正在加载 SD-Turbo 扩散模型 (开启 4GB 显存极致优化)...")
    try:
        # 使用 fp16 半精度加载，极大降低显存占用
        pipe = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/sd-turbo", 
            torch_dtype=torch.float16, 
            variant="fp16",
            # 如果下载慢，可以加上 mirror="https://hf-mirror.com" 环境变量
        )
        
        # 【增强修复】：智能显存优化回退机制
        try:
            # 尝试最优解：CPU Offload (需要 accelerate 库支持)
            pipe.enable_model_cpu_offload()
            print("  └─ 启用模型 CPU Offload 极致优化")
        except Exception as e:
            # 如果未安装 accelerate，优雅回退至标准加载模式
            print(f"  └─ ⚠️ 缺少 accelerate 库 ({e.args[0] if e.args else e})")
            print("  └─ 自动回退至标准 GPU 加载模式...")
            pipe = pipe.to(device)
            
        # VAE 切片：极其节省显存，且不依赖 accelerate，也不会触发 batch=1 错误
        pipe.enable_vae_slicing()
        print("  └─ 启用 VAE 切片优化")
        
        # 禁用进度条以提高一点点终端输出速度
        pipe.set_progress_bar_config(disable=True)
        print("  ✅ 扩散模型就绪！")
    except Exception as e:
        print(f"\033[91m[错误] 扩散模型加载失败，请检查网络或配置: {e}\033[0m")
        return

    # 初始化传统卷积码
    memory = np.array(EXP1_CFG["cc_memory"])
    g_matrix = np.array(EXP1_CFG["cc_g_matrix"])
    trellis = cc.Trellis(memory, g_matrix)
    E2E_SYMBOL_LENGTH = EXP1_CFG["e2e_symbol_length"]
    K_BITS = (E2E_SYMBOL_LENGTH // 2) - memory[0]
    BUDGET_BYTES = K_BITS // 8

    print("🤖 正在加载 DL-SSCC (CompressAI) 模型...")
    ai_models = {q: bmshj2018_factorized(quality=q, pretrained=True).eval().to(device) for q in range(1, 9)}
    
    print("🧠 正在加载 Ours (E2E Semantic) 收发模型...")
    tx_model = ImageSemanticEncoder().to(device).eval()
    rx_model = ImageSemanticDecoder().to(device).eval()
    tx_model.load_state_dict(torch.load(MAIN_CFG["paths"]["tx_weights"], map_location=device))
    rx_model.load_state_dict(torch.load(MAIN_CFG["paths"]["rx_weights"], map_location=device))

    print("📊 正在加载 S-CSI 物理特征盲估器...")
    scsi_scaler = joblib.load('scsi_scaler.pkl')
    scsi_xgb = xgb.XGBClassifier()
    scsi_xgb.load_model('scsi_xgboost.json')
    TEMP_MAPPING = {3: 1.0, 2: 0.8, 1: 0.3, 0: 0.1}

    test_files = glob.glob(os.path.join(EXP1_CFG["tx_source_dir"], "*.*"))
    test_files = [f for f in test_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:20] # POC 阶段建议先跑20张测测速
    res = EXP1_CFG["target_resolution"]
    snr_levels = EXP1_CFG["snr_levels"]

    print(f"\n📦 [概念验证] 选择测试图片 {len(test_files)} 张。")

    for snr in snr_levels:
        out_dir_base = os.path.join(EXP1_CFG["output_base_dir"], f"snr_{snr}")
        # 增加 ours_diff 文件夹存放锐化后的图像
        for sys_name in ['jpeg', 'dl_sscc', 'ours', 'ours_diff']: 
            os.makedirs(os.path.join(out_dir_base, sys_name), exist_ok=True)
        
        print(f"\n{'='*50}\n▶ 当前信道环境: AWGN SNR = {snr}dB")
        snr_linear = 10 ** (snr / 10.0)
        
        mod_type = '4QAM' if snr >= MAIN_CFG["amc_thresholds"].get("4QAM", 5.0) else 'BPSK'
        fixed_tau = MAIN_CFG["tau_config"].get(mod_type, 0.4)
        noise_power = 1.0 / snr_linear
        
        theoretical_tau = (noise_power * np.log(3)) / 2.0
        decode_tau = min(theoretical_tau, 0.7)

        for img_path in test_files:
            img_name = os.path.basename(img_path)
            png_name = f"{os.path.splitext(img_name)[0]}.png"
            
            img_bgr = safe_imread_and_resize(img_path, res)
            if img_bgr is None: continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            tensor_img = T.ToTensor()(img_rgb).unsqueeze(0).to(device)

            # --- 控制面：XGBoost 动态控温 ---
            pilot_tx = np.random.choice([-1.0, 1.0], E2E_SYMBOL_LENGTH).astype(np.float32)
            pilot_noise = np.sqrt(noise_power) * np.random.randn(E2E_SYMBOL_LENGTH).astype(np.float32)
            pilot_rx = pilot_tx + pilot_noise 
            
            pilot_evm = np.mean(np.abs(pilot_rx - np.sign(pilot_rx)))
            pilot_tar = np.mean(np.abs(pilot_rx) < fixed_tau) * 100.0
            
            features_scaled = scsi_scaler.transform(np.array([[pilot_evm, snr, pilot_tar]]))
            scsi_state = int(scsi_xgb.predict(features_scaled)[0])
            
            if scsi_state == 2 and snr >= 10.0: scsi_state = 3
            dynamic_temp = TEMP_MAPPING.get(scsi_state, 1.0)

            # ================== 1. 信号调制 ==================
            tx_bits_jpg, act_bytes_jpg = compress_jpeg_to_budget(img_bgr, BUDGET_BYTES, K_BITS)
            sym_jpg = bpsk_modulate(cc.conv_encode(tx_bits_jpg, trellis))
            
            tx_bits_ai, act_bytes_ai, shp_ai, q_ai = compress_ai_to_budget(tensor_img, BUDGET_BYTES, K_BITS, ai_models)
            sym_ai = bpsk_modulate(cc.conv_encode(tx_bits_ai, trellis))
            
            with torch.no_grad():
                sym_ours_tensor = tx_model(tensor_img, mod_type=mod_type, temperature=dynamic_temp)
                sym_ours = sym_ours_tensor.cpu().numpy().flatten().astype(np.float32)

            # ================== 2. 加噪 ==================
            awgn_base = np.sqrt(noise_power) * np.random.randn(E2E_SYMBOL_LENGTH).astype(np.float32)
            rx_jpg  = sym_jpg  + awgn_base
            rx_ai   = sym_ai   + awgn_base
            rx_ours = sym_ours + np.sqrt(noise_power) * np.random.randn(len(sym_ours)).astype(np.float32)

            # ================== 3. 信号解码与 CNN 恢复 ==================
            # ... (JPEG 和 DL_SSCC 解码保持简写) ...
            dec_bits_jpg = cc.viterbi_decode(hard_demodulate(rx_jpg), trellis, decoding_type='hard')[:K_BITS]
            rec_jpg = decompress_jpeg(dec_bits_jpg, act_bytes_jpg) if np.mean(dec_bits_jpg[:act_bytes_jpg*8] != tx_bits_jpg[:act_bytes_jpg*8]) <= 0.05 else np.zeros_like(img_bgr)
            if rec_jpg is None: rec_jpg = np.zeros_like(img_bgr)

            dec_bits_ai = cc.viterbi_decode(hard_demodulate(rx_ai), trellis, decoding_type='hard')[:K_BITS]
            rec_ai = decompress_ai(dec_bits_ai, act_bytes_ai, shp_ai, q_ai, ai_models) if np.mean(dec_bits_ai[:act_bytes_ai*8] != tx_bits_ai[:act_bytes_ai*8]) <= 0.01 else np.zeros_like(img_bgr)
            if rec_ai is None: rec_ai = np.zeros_like(img_bgr)

            # Ours - CNN 解码
            with torch.no_grad():
                rx_ours_tensor = torch.tensor(rx_ours, device=device).view(sym_ours_tensor.shape)
                is_ambiguous = torch.abs(rx_ours_tensor) < decode_tau
                ternary_symbols = torch.where(is_ambiguous, torch.zeros_like(rx_ours_tensor), torch.sign(rx_ours_tensor))
                
                rec_tensor_ours, _ = rx_model(ternary_symbols, return_sparsity=True)
                rec_ours_rgb = (rec_tensor_ours.clamp(0,1).cpu().squeeze(0).numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
                rec_ours = cv2.cvtColor(rec_ours_rgb, cv2.COLOR_RGB2BGR)

            # =================================================================
            # 🌟 4. 接收端增强：扩散模型 (SD-Turbo) 锐化去噪
            # =================================================================
            # a. 准备 PIL Image，先升采样到 512x512 以匹配扩散模型的分布感受野
            init_image = Image.fromarray(rec_ours_rgb).resize((512, 512), Image.BICUBIC)
            
            # b. 执行图生图流水线
            #    num_inference_steps=4 和 strength=0.3 组合，确保实际执行步数 int(4*0.3) = 1 步
            #    完美避开 diffusers 的 0 步坍缩 Bug，同时保持极致的秒级生成速度！
            prompt = "highly detailed, sharp focus, 4k resolution, photorealistic" # 通用正向提示词
            with torch.no_grad():
                diff_result = pipe(
                    prompt=prompt,
                    image=init_image,
                    num_inference_steps=4, 
                    strength=0.3, 
                    guidance_scale=0.0
                ).images[0]
            
            # c. 将增强后的图像缩放回 256x256，与系统兼容
            rec_ours_diff_rgb = np.array(diff_result.resize((res, res), Image.BICUBIC))
            rec_ours_diff = cv2.cvtColor(rec_ours_diff_rgb, cv2.COLOR_RGB2BGR)

            # --- 保存结果 ---
            cv2.imwrite(os.path.join(out_dir_base, 'jpeg', png_name), rec_jpg)
            cv2.imwrite(os.path.join(out_dir_base, 'dl_sscc', png_name), rec_ai)
            cv2.imwrite(os.path.join(out_dir_base, 'ours', png_name), rec_ours)
            cv2.imwrite(os.path.join(out_dir_base, 'ours_diff', png_name), rec_ours_diff)

        print(f"✅ 完成当前 SNR 组合: 图像及 Diffusion 增强已落盘至 {out_dir_base}")

if __name__ == "__main__":
    main()