import socket
import numpy as np
import torch
import torchvision.utils as vutils
import torch.nn.functional as F
import os
import random
import argparse
import re
import string
import json
import joblib
import xgboost as xgb
import warnings
import traceback
from image_e2e_model import ImageSemanticEncoder, ImageSemanticDecoder

# 从支持工具中引入核心 Socket 传输及上行物理层编码函数
from comm_utils import send_msg, recv_msg, encode_feedback_to_waveform

warnings.filterwarnings("ignore")

MOD_POWER = {'4QAM': 1.0, 'BPSK': 1.0}

PATTERNS = {
    'EOT':     np.tile(np.array([3.0, -3.0, 3.0, -3.0], dtype=np.float32), 128),
    '4QAM':    np.tile(np.array([3.0, 3.0, -3.0, -3.0], dtype=np.float32), 128),
    'BPSK':    np.tile(np.array([3.0, -3.0, -3.0, 3.0], dtype=np.float32), 128)
}

VALID_EXTS = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.webp']

def detect_control_signal(noisy_payload, threshold=0.6):
    best_match, max_corr = None, 0
    for name, pattern in PATTERNS.items():
        if len(noisy_payload) == len(pattern):
            corr = np.sum(noisy_payload * pattern) / np.sum(pattern ** 2)
            if corr > max_corr:
                max_corr, best_match = corr, name
    if max_corr > threshold: return best_match
    return None

def bits_to_text(bits):
    if len(bits) % 8 != 0: bits = bits[:len(bits) - (len(bits)%8)]
    if len(bits) == 0: return "", False
    byte_vals = np.packbits(bits)
    data_bytes, checksum_received = byte_vals[:-1], byte_vals[-1]
    calculated_checksum = int(np.sum(data_bytes, dtype=int)) % 256
    if calculated_checksum == checksum_received:
        return bytes(data_bytes).decode('utf-8', errors='ignore'), True
    return bytes(data_bytes).decode('utf-8', errors='ignore'), False

def majority_vote_bits(history_bits_list):
    min_len = min(len(b) for b in history_bits_list)
    aligned_history = [b[:min_len] for b in history_bits_list]
    matrix = np.vstack(aligned_history)
    voted_bits = np.median(matrix, axis=0).astype(np.uint8)
    text, _ = bits_to_text(voted_bits) 
    return text

def sanitize_and_correct_path(filepath):
    if not filepath: return ""
    local_path = os.path.normpath(filepath).lstrip(os.sep).replace('..', '')
    parts = local_path.split(os.sep)
    safe_parts = []
    illegal_pattern = r'[<>:"/\\|?*\x00-\x1F]'
    for i, part in enumerate(parts):
        if i == len(parts) - 1:
            base, ext = os.path.splitext(part)
            base = re.sub(illegal_pattern, lambda m: random.choice(string.ascii_lowercase), base)
            if not base: return ""
            ext = ext.lower()
            if ext not in VALID_EXTS: ext = '.png'
            safe_parts.append(base + ext)
        else:
            safe_dir = re.sub(illegal_pattern, lambda m: random.choice(string.ascii_lowercase), part)
            if not safe_dir: safe_dir = "folder"
            safe_parts.append(safe_dir)
    return os.path.join(*safe_parts)

def save_image_to_disk(tensor_img, relative_path, target_base_dir):
    if tensor_img is None or not relative_path: return
    safe_rel_path = sanitize_and_correct_path(relative_path)
    if not safe_rel_path: return
    
    full_save_path = os.path.join(os.getcwd(), target_base_dir, safe_rel_path)
    target_dir = os.path.dirname(full_save_path)
    if target_dir: os.makedirs(target_dir, exist_ok=True)
    vutils.save_image(tensor_img, full_save_path)
    print(f"\033[92m[存盘] 落盘成功: {full_save_path}\033[0m")

def calculate_soft_evm(noisy_payload):
    ideal_symbols = np.sign(noisy_payload)
    ideal_symbols[ideal_symbols == 0] = 1.0 
    return np.mean(np.abs(noisy_payload - ideal_symbols))

# =========================================================================
# 【核心新增】：上行控制链路多环境物理干扰注入器 (与下行实验环境完全对齐)
# =========================================================================
def inject_uplink_channel_noise(fb_waveform, noise_power, CONFIG):
    """
    根据当前的测试场景，在反馈波形上叠加完全同等物理性质的干扰
    """
    total_len = len(fb_waveform)
    
    # 1. 恒定叠加加性高斯白噪声 (AWGN)
    uplink_noise = np.sqrt(noise_power) * np.random.randn(total_len).astype(np.float32)
    
    # 2. 联动检测：是否存在实验 2 (突发脉冲) 的物理配置
    try:
        if os.path.exists('config_exp2.json'):
            with open('config_exp2.json', 'r', encoding='utf-8') as f:
                exp2_cfg = json.load(f).get("experiment2_impulse_params", {})
            prob = exp2_cfg.get("burst_probabilities", [0.05])[0] 
            burst_len = exp2_cfg.get("burst_length", 100)
            impulse_std = exp2_cfg.get("impulse_std", 10.0)
            
            if prob > 0:
                num_bursts = max(1, int(total_len * prob) // burst_len)
                impulse_noise = np.zeros(total_len, dtype=np.float32)
                for _ in range(num_bursts):
                    start = np.random.randint(0, max(1, total_len - burst_len))
                    impulse_noise[start : start + burst_len] = np.random.normal(0, impulse_std, burst_len)
                uplink_noise += impulse_noise
    except Exception:
        pass

    # 3. 联动检测：是否存在实验 3 (多用户同频碰撞) 的物理配置
    try:
        if os.path.exists('config_exp3.json'):
            with open('config_exp3.json', 'r', encoding='utf-8') as f:
                exp3_cfg = json.load(f).get("experiment3_mac_params", {})
            sir_levels = exp3_cfg.get("sir_levels", [0.0])
            # 取当前可能处于的干扰强度 (例如矩阵中最严酷的 0dB 碰撞)
            sir = sir_levels[-1] if isinstance(sir_levels, list) else sir_levels
            alpha = 10 ** (-sir / 20.0)
            
            # 模拟上行反馈同样遭遇传统数字系统 (16QAM) 的同频强碰撞
            levels = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float32)
            mac_interference = np.random.choice(levels, size=total_len) * alpha
            uplink_noise += mac_interference
    except Exception:
        pass

    return fb_waveform + uplink_noise

def main():
    with open('config.json', 'r', encoding='utf-8') as f:
        CONFIG = json.load(f)

    parser = argparse.ArgumentParser(description="语义图片接收端")
    parser.add_argument("--snr", type=float, default=None, help="恒定物理信噪比 (默认: 20.0 dB)")
    parser.add_argument("--snrmin", type=float, default=None, help="随机波动下限")
    parser.add_argument("--snrmax", type=float, default=None, help="随机波动上限")
    parser.add_argument("--testseed", action="store_true", help="使用 config.json 中的随机种子保证基准一致")
    parser.add_argument("--outdir", type=str, default=None, help="重隔图片的输出落盘目录") 
    args = parser.parse_args()

    rx_outdir = args.outdir if args.outdir else CONFIG.get("rx_target_dir", "./output_images")

    if args.testseed:
        seed = CONFIG.get("experiment_params", {}).get("random_seed", 2026)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"\033[94m[参数配置] 已启用固定随机种子: {seed}\033[0m")

    is_random_snr = (args.snrmin is not None) or (args.snrmax is not None)
    if is_random_snr:
        snr_min = args.snrmin if args.snrmin is not None else -20.0
        snr_max = args.snrmax if args.snrmax is not None else 20.0
        print(f"\033[94m[参数配置] 启用随机信道波动: SNR ∈ [{snr_min:.1f} dB, {snr_max:.1f} dB]\033[0m")
    else:
        fixed_snr = args.snr if args.snr is not None else 20.0
        print(f"\033[94m[参数配置] 启用恒定信道环境: SNR = {fixed_snr:.1f} dB\033[0m")
        
    print(f"\033[94m[参数配置] 图像输出总目录: {rx_outdir}\033[0m")

    def get_current_snr():
        if is_random_snr: return random.uniform(snr_min, snr_max)
        return fixed_snr

    print("\033[96m初始化接收端 (动态AMC, 纯语义盲估 S-CSI, CNN重构)\033[0m")
    tx_model = ImageSemanticEncoder().eval()
    rx_model = ImageSemanticDecoder().eval()
    try:
        tx_model.load_state_dict(torch.load(CONFIG["paths"]["tx_weights"], map_location='cpu'))
        rx_model.load_state_dict(torch.load(CONFIG["paths"]["rx_weights"], map_location='cpu'))
        print("\033[92m[系统] 成功加载端到端网络权重！\033[0m")
    except Exception as e:
        print(f"\033[93m[警告] CNN权重加载异常: {e}\033[0m")

    try:
        scsi_scaler = joblib.load('scsi_scaler.pkl')
        scsi_xgb = xgb.XGBClassifier()
        scsi_xgb.load_model('scsi_xgboost.json')
        print("\033[92m[系统] 成功加载 S-CSI 4分类状态分类器！\033[0m")
    except Exception as e:
        print(f"\033[91m[致命错误] 无法加载机器学习模型: {e}\033[0m")
        return

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind(('0.0.0.0', 9999))
        server_socket.listen(1)
        print("\033[96m[系统] 正在监听端口 9999，等待发送端连接...\033[0m")
    except Exception as e:
        return print(f"\033[91m[错误] 端口绑定失败: {e}\033[0m")

    env_snr_db = get_current_snr() 
    current_mod = '4QAM' 

    while True:
        try:
            conn, addr = server_socket.accept()
            print(f"\n\033[92m[系统] 发送端已接入: {addr}\033[0m")
            conn.settimeout(15.0)
            
            image_counter = 0 
            last_img_tensor, confirmed_filename, corrupted_bits_history = None, "", [] 

            while True:
                try: 
                    msg = recv_msg(conn)
                    if not msg: break
                    
                    snr_linear = 10 ** (env_snr_db / 10.0)
                    noise_power = MOD_POWER.get(current_mod, 1.0) / snr_linear

                    if msg.get('type') == 'control_frame':
                        noisy_payload = msg['payload'] + np.sqrt(noise_power) * np.random.randn(*msg['payload'].shape)
                        detected_cmd = detect_control_signal(noisy_payload)
                        
                        if detected_cmd == 'EOT':
                            image_counter += 1
                            if not confirmed_filename and corrupted_bits_history:
                                confirmed_filename = majority_vote_bits(corrupted_bits_history)
                            
                            final_name = confirmed_filename.strip()
                            if not final_name or not sanitize_and_correct_path(final_name):
                                final_name = f"pic_{image_counter:05d}.png"
                                
                            save_image_to_disk(last_img_tensor, final_name, rx_outdir)
                            
                            env_snr_db = get_current_snr()
                            confirmed_filename, corrupted_bits_history = "", []
                            
                        elif detected_cmd in ['4QAM', 'BPSK']:
                            current_mod = detected_cmd
                            print(f"\033[93m[MAC层] 调制切换至: {current_mod}\033[0m")

                    elif msg.get('type') == 'metadata':
                        noisy_payload = msg['payload'] + np.sqrt(noise_power) * np.random.randn(*msg['payload'].shape)
                        bits = np.where(noisy_payload <= 0, 0, 1).astype(np.uint8)
                        text, is_valid = bits_to_text(bits)
                        if is_valid:
                            confirmed_filename = text
                            send_msg(conn, {'type': 'ack', 'valid': True})
                        else:
                            corrupted_bits_history.append(bits) 
                            send_msg(conn, {'type': 'ack', 'valid': False})

                    elif msg.get('type') == 'data':
                        # 1. 模拟下行信道接收物理符号并加噪
                        noisy_payload = msg['payload'] + np.sqrt(noise_power) * np.random.randn(*msg['payload'].shape)
                        tau = CONFIG.get("tau_config", {}).get(current_mod, 0.4)
                        
                        # 2. 执行三值擦除硬判决与零延迟物理特征计算
                        is_ambiguous = np.abs(noisy_payload) < tau
                        tar_val = np.mean(is_ambiguous) * 100 
                        ternary_payload = np.where(is_ambiguous, 0.0, np.sign(noisy_payload))
                        soft_evm_val = calculate_soft_evm(noisy_payload)

                        # 3. 图像空间去噪与语义解码 (用于落盘)
                        rx_tensor = torch.tensor(ternary_payload, dtype=torch.float32).view(msg['shape'])
                        with torch.no_grad():
                            reconstructed_img, sparsity_val = rx_model(rx_tensor, return_sparsity=True)
                            last_img_tensor = reconstructed_img

                        # 【核心改动】：严格对齐新模型的 Top 3 纯物理零延迟输入特征池
                        # 特征顺序必须为：['soft_evm', 'physical_snr', 'tar_percent']
                        features = np.array([[soft_evm_val, env_snr_db, tar_val]])
                        features_scaled = scsi_scaler.transform(features)
                        
                        # 4. 机器学习零延迟多状态推断
                        scsi_state = int(scsi_xgb.predict(features_scaled)[0])
                        state_bits_str = f"{scsi_state:02b}" # 4分类对应 2-bit

                        # 5. 反馈信令拼装与全物理层 BPSK 扩频调制
                        # 1-bit 调制指示位：1 代表建议采用 4QAM，0 代表建议采用 BPSK
                        amc_bit = 1 if env_snr_db >= CONFIG.get("amc_thresholds", {}).get("4QAM", 10.0) else 0
                        # 2-bit 语义状态位
                        scsi_bit1 = (scsi_state >> 1) & 1
                        scsi_bit0 = scsi_state & 1
                        
                        # 组装完整的 3-bit 原始控制帧
                        feedback_bits = [amc_bit, scsi_bit1, scsi_bit0]
                        
                        # 调用工具映射为物理 BPSK 实数序列，应用 31 倍强冗余扩频保护 (总长 93 符号)
                        fb_waveform = encode_feedback_to_waveform(feedback_bits, repeat=31)
                        
                        # 6. 【核心改动】：上行反馈链路物理干扰注入 (原汁原味还原当前实验恶劣场景)
                        noisy_fb_waveform = inject_uplink_channel_noise(fb_waveform, noise_power, CONFIG)

                        print(f"[AI 盲估] 物理SNR:{env_snr_db:.1f}dB | EVM:{soft_evm_val:.3f} | 擦除率:{tar_val:.1f}% -> 状态判决: {state_bits_str} (状态 {scsi_state})")
                        print(f"[上行物理层] 原始反馈:{feedback_bits} -> 已完成BPSK 31x扩频保护并注入混叠物理干扰。")
                        
                        # 将带噪声的实数波形数组封装进控制信道发给 Tx
                        send_msg(conn, {'type': 'feedback_waveform', 'payload': noisy_fb_waveform})

                except socket.timeout: 
                    break 
                except Exception as e:
                    print(f"\n\033[91m[内部严重崩溃] 接收端处理数据时发生错误:\033[0m")
                    traceback.print_exc() 
                    break 

        except ConnectionResetError: print("\033[93m[系统] 远端断开连接。\033[0m")
        except KeyboardInterrupt:
            print("\n[系统] 手动停止接收端。")
            break
        finally: conn.close()

if __name__ == "__main__":
    main()