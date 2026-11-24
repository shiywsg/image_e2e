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

# 【新增引入】：decode_feedback_from_waveform 波形软判决解码函数
from comm_utils import send_msg, recv_msg, decode_feedback_from_waveform
from image_e2e_model import ImageSemanticEncoder

PATTERNS = {
    'EOT':     np.tile(np.array([3.0, -3.0, 3.0, -3.0], dtype=np.float32), 128),
    '4QAM':    np.tile(np.array([3.0, 3.0, -3.0, -3.0], dtype=np.float32), 128),
    'BPSK':    np.tile(np.array([3.0, -3.0, -3.0, 3.0], dtype=np.float32), 128)
}

# (移除了冗余的 SNR_HISTORY 和 get_smoothed_snr 函数，因为发送端不再需要自己算信噪比)
# (移除了 determine_modulation 函数，因为调制直接由接收端通过 1-bit 反馈下达指令)

def str_to_physical_symbols(text):
    b_data = text.encode('utf-8')
    checksum = sum(b_data) % 256
    all_bytes = list(b_data) + [checksum]
    bits = np.unpackbits(np.array(all_bytes, dtype=np.uint8))
    return np.where(bits == 0, -1.0, 1.0).astype(np.float32)

def main():
    with open('config.json', 'r', encoding='utf-8') as f:
        CONFIG = json.load(f)

    parser = argparse.ArgumentParser(description="端到端语义发送端 (智能执行节点)")
    parser.add_argument("--dir", type=str, default=None, help="发送的源图片文件夹")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="接收端 IP")
    args = parser.parse_args()
    
    # 动态确定源目录
    tx_dir = args.dir if args.dir else CONFIG.get("tx_source_dir", "./dataset/test_images")
    
    tx_model = ImageSemanticEncoder().eval()
    try:
        tx_model.load_state_dict(torch.load(CONFIG["paths"]["tx_weights"], map_location='cpu'))
        print("\033[92m[系统] 成功加载发送端语义编码器权重！\033[0m")
    except Exception as e:
        print("\033[93m[警告] 未找到预训练权重，使用随机初始化。\033[0m")
        
    # 智能扫描所有常见图片格式
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

    # 初始化状态
    current_mod = '4QAM' 
    target_mod = '4QAM'   # 由接收端 1-bit 反馈决定
    MOD_LEVELS = ['4QAM', 'BPSK']

    # 指数退火的非线性掩码温度分配 (由接收端 2-bit 状态反馈决定)
    TEMP_MAPPING = {3: 1.0, 2: 0.8, 1: 0.3, 0: 0.1}
    current_scsi_state = 3 

    try:
        for file_path in image_files:
            rel_path = os.path.relpath(file_path, tx_dir) if not os.path.isabs(tx_dir) else os.path.join(os.path.basename(os.path.normpath(tx_dir)), os.path.relpath(file_path, tx_dir))
            rel_path_unix = rel_path.replace('\\', '/') 
            print(f"\n>>> 开始任务: {rel_path_unix}")

            # 1. 执行接收端要求的物理层调制切换
            if target_mod != current_mod:
                send_msg(client_socket, {'type': 'control_frame', 'payload': PATTERNS[target_mod]})
                current_mod = target_mod
                time.sleep(0.2)

            # 2. 强鲁棒的文件元数据 (Metadata) 握手发送
            metadata_success = False
            start_mod_idx = MOD_LEVELS.index(current_mod)
            for attempt_mod in MOD_LEVELS[start_mod_idx:]:
                if attempt_mod != current_mod:
                    send_msg(client_socket, {'type': 'control_frame', 'payload': PATTERNS[attempt_mod]})
                    current_mod = attempt_mod
                    time.sleep(0.2)
                for try_idx in range(3):
                    meta_syms = str_to_physical_symbols(rel_path_unix)
                    send_msg(client_socket, {'type': 'metadata', 'payload': meta_syms})
                    msg = recv_msg(client_socket)
                    if msg and msg.get('type') == 'ack' and msg.get('valid'):
                        metadata_success = True
                        break
                if metadata_success: break

            # 3. 提取接收端要求的掩码温度并执行图像语义编码
            current_temp = TEMP_MAPPING.get(current_scsi_state, 1.0)

            img = Image.open(file_path).convert('RGB')
            w, h = img.size
            new_w, new_h = (w // 4) * 4, (h // 4) * 4
            transform = transforms.Compose([transforms.Resize((new_h, new_w)), transforms.ToTensor()])
            img_tensor = transform(img).unsqueeze(0)

            with torch.no_grad():
                tx_symbols = tx_model(img_tensor, mod_type=current_mod, temperature=current_temp)
            
            payload = tx_symbols.cpu().numpy().flatten()
            state_bits = f"{current_scsi_state:02b}"
            print(f"[下行数据发送] 调制:{current_mod} | S-CSI状态:{state_bits} -> 执行掩码温度(Temp):{current_temp:.2f}")
            send_msg(client_socket, {'type': 'data', 'shape': list(tx_symbols.shape), 'payload': payload})
            
            # ==============================================================
            # 4. 【核心重构】：处理来自接收端的上行控制信令 (纯物理波形软解码)
            # ==============================================================
            msg = recv_msg(client_socket)
            if msg and msg.get('type') == 'feedback_waveform':
                # 收到的不再是字典明文，而是掺杂了物理干扰的 BPSK 浮点数波形序列
                noisy_fb_waveform = msg.get('payload')
                
                # 执行物理层软判决合并 (MRC)，恢复出金刚不坏的 3 bit 信息
                reliable_bits = decode_feedback_from_waveform(noisy_fb_waveform, num_bits=3, repeat=31)
                
                # 第 0 位：解析 1-bit AMC 调制指示 (1 为 4QAM，0 为 BPSK)
                amc_bit = reliable_bits[0]
                target_mod = '4QAM' if amc_bit == 1 else 'BPSK'
                
                # 第 1、2 位：解析 2-bit S-CSI 语义状态指示 (拼接为 0~3 的整数)
                scsi_state = (reliable_bits[1] << 1) | reliable_bits[2]
                
                # 更新本端状态，供下一张图片使用
                current_scsi_state = scsi_state 
                new_state_bits = f"{current_scsi_state:02b}"
                print(f"    <- [上行波形反馈] MRC软解码成功:{reliable_bits} | 下帧命令 -> 调制:{target_mod}, 状态:{current_scsi_state} ({new_state_bits})")

            # 5. 发送图像结束符
            send_msg(client_socket, {'type': 'control_frame', 'payload': PATTERNS['EOT']})
            time.sleep(0.5)

    except KeyboardInterrupt: print("\n[系统] 用户中断传输。")
    finally: client_socket.close()

if __name__ == "__main__":
    main()