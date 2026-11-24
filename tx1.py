import socket
import time
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import glob
import argparse
import json

from comm_utils import send_msg, recv_msg, decode_feedback_from_waveform
from image_e2e_model import ImageSemanticEncoder

PATTERNS = {
    'EOT':    np.tile(np.array([3.0, -3.0, 3.0, -3.0], dtype=np.float32), 128),
    '4QAM':   np.tile(np.array([3.0, 3.0, -3.0, -3.0], dtype=np.float32), 128),
    'BPSK':   np.tile(np.array([3.0, -3.0, -3.0, 3.0], dtype=np.float32), 128)
}

def str_to_physical_symbols(text):
    b_data = text.encode('utf-8')
    checksum = sum(b_data) % 256
    all_bytes = list(b_data) + [checksum]
    bits = np.unpackbits(np.array(all_bytes, dtype=np.uint8))
    return np.where(bits == 0, -1.0, 1.0).astype(np.float32)

def read_instantaneous_snr_from_hardware(default_snr=15.0):
    return default_snr 

def main():
    with open('config.json', 'r', encoding='utf-8') as f:
        CONFIG = json.load(f)

    parser = argparse.ArgumentParser(description="端到端语义发送端 (智能执行节点)")
    parser.add_argument("--dir", type=str, default=None, help="发送的源图片文件夹")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="接收端 IP")
    parser.add_argument("--initsnr", type=float, default=15.0, help="首帧启动时的硬件初始 SNR")
    args = parser.parse_args()
    
    tx_dir = args.dir if args.dir else CONFIG.get("tx_source_dir", "./dataset/test_images")
    tx_model = ImageSemanticEncoder().eval()
    try:
        tx_model.load_state_dict(torch.load(CONFIG["paths"]["tx_weights"], map_location='cpu'))
        print("\033[92m[系统] 成功加载发送端语义编码器权重！\033[0m")
    except Exception as e:
        print("\033[93m[警告] 未找到预训练权重，使用随机初始化。\033[0m")
        
    search_path = os.path.join(tx_dir, "**/*.*")
    all_files = glob.glob(search_path, recursive=True)
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    if not image_files: 
        print(f"\033[91m[错误] 在目录 {tx_dir} 下未找到任何图片！\033[0m")
        return

    print(f"\033[94m[参数配置] 数据源目录: {tx_dir} (共找到 {len(image_files)} 张图片)\033[0m")

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        print(f"[系统] 正在尝试连接接收端 {args.ip}:9999...")
        client_socket.connect((args.ip, 9999))
        print("\033[92m[系统] 成功连接到接收端！\033[0m")
    except Exception as e: 
        print(f"\033[91m[致命错误] 无法连接到接收端: {e}。请先启动接收端！\033[0m")
        return

    init_snr = read_instantaneous_snr_from_hardware(args.initsnr)
    MOD_LEVELS = ['4QAM', 'BPSK']
    
    target_mod = '4QAM' if init_snr >= CONFIG.get("amc_thresholds", {}).get("4QAM", 10.0) else 'BPSK'
    current_mod = target_mod
    
    if init_snr >= 12.0: current_scsi_state = 3
    elif init_snr >= 5.0: current_scsi_state = 2
    elif init_snr >= -2.0: current_scsi_state = 1
    else: current_scsi_state = 0
    
    print(f"\033[96m[冷启动] 探测到初始 SNR: {init_snr}dB, 映射起始调制: {current_mod}, S-CSI状态: {current_scsi_state}\033[0m")

    TEMP_MAPPING = {3: 1.0, 2: 0.8, 1: 0.3, 0: 0.1}

    try:
        for file_path in image_files:
            rel_path = os.path.relpath(file_path, tx_dir) if not os.path.isabs(tx_dir) else os.path.join(os.path.basename(os.path.normpath(tx_dir)), os.path.relpath(file_path, tx_dir))
            rel_path_unix = rel_path.replace('\\', '/') 
            print(f"\n>>> 开始任务: {rel_path_unix}")

            current_temp = TEMP_MAPPING.get(current_scsi_state, 1.0)
            img = Image.open(file_path).convert('RGB')
            w, h = img.size
            new_w, new_h = (w // 4) * 4, (h // 4) * 4
            transform = transforms.Compose([transforms.Resize((new_h, new_w)), transforms.ToTensor()])
            img_tensor = transform(img).unsqueeze(0)

            with torch.no_grad():
                tx_symbols = tx_model(img_tensor, mod_type=current_mod, temperature=current_temp)
            
            shape_str = f"{tx_symbols.shape[0]},{tx_symbols.shape[1]},{tx_symbols.shape[2]},{tx_symbols.shape[3]}"
            meta_text = f"{rel_path_unix}|{shape_str}"

            if target_mod != current_mod:
                send_msg(client_socket, {'type': 'control_frame', 'payload': PATTERNS[target_mod]})
                current_mod = target_mod
                time.sleep(0.2)

            metadata_success = False
            start_mod_idx = MOD_LEVELS.index(current_mod)
            for attempt_mod in MOD_LEVELS[start_mod_idx:]:
                if attempt_mod != current_mod:
                    send_msg(client_socket, {'type': 'control_frame', 'payload': PATTERNS[attempt_mod]})
                    current_mod = attempt_mod
                    time.sleep(0.2)
                for try_idx in range(3):
                    meta_syms = str_to_physical_symbols(meta_text)
                    send_msg(client_socket, {'type': 'metadata', 'payload': meta_syms})
                    msg = recv_msg(client_socket)
                    if msg and msg.get('type') == 'ack' and msg.get('valid'):
                        metadata_success = True
                        break
                if metadata_success: break

            payload = tx_symbols.cpu().numpy().flatten()
            state_bits = f"{current_scsi_state:02b}"
            print(f"[下行数据发送] 调制:{current_mod} | S-CSI状态:{state_bits} -> 执行掩码温度(Temp):{current_temp:.2f}")
            send_msg(client_socket, {'type': 'data', 'payload': payload})
            
            msg = recv_msg(client_socket)
            if msg and msg.get('type') == 'feedback_waveform':
                noisy_fb_waveform = msg.get('payload')
                reliable_bits = decode_feedback_from_waveform(noisy_fb_waveform, num_bits=3, repeat=31)
                
                amc_bit = reliable_bits[0]
                target_mod = '4QAM' if amc_bit == 1 else 'BPSK'
                scsi_state = (reliable_bits[1] << 1) | reliable_bits[2]
                
                current_scsi_state = scsi_state 
                new_state_bits = f"{current_scsi_state:02b}"
                print(f"    <- [上行波形反馈] MRC软解码成功:{reliable_bits} | 下帧命令 -> 调制:{target_mod}, 状态:{current_scsi_state} ({new_state_bits})")

            send_msg(client_socket, {'type': 'control_frame', 'payload': PATTERNS['EOT']})
            time.sleep(0.5)

    except KeyboardInterrupt: print("\n[系统] 用户中断传输。")
    finally: client_socket.close()

if __name__ == "__main__":
    main()