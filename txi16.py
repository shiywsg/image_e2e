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
import struct
import sys

from comm_utils import decode_feedback_from_waveform
from image_e2e_model import ImageSemanticEncoder

DEBUG_MODE = True
def debug_print(msg):
    if DEBUG_MODE: print(f"\033[90m[TX-DEBUG] {msg}\033[0m")

# ================= 核心：交织器初始化 =================
TARGET_SIZE = 131072
rnd_state = np.random.RandomState(2026)
INTERLEAVE_IDX = rnd_state.permutation(TARGET_SIZE)
# ======================================================

class UplinkChannel:
    def __init__(self, rx_ip, port=9999, timeout=5.0, use_real_channel=0):
        self.use_real_channel = use_real_channel
        if self.use_real_channel == 0:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(timeout)
            print(f"\033[96m[Uplink] (仿真) 正在连接 {rx_ip}:{port}...\033[0m")
            try:
                self.sock.connect((rx_ip, port))
                print("\033[92m[Uplink] 连接成功！\033[0m")
            except Exception as e:
                print(f"\033[91m[错误] 无法连接反馈通道: {e}\033[0m")
                sys.exit(1)

    def recv_feedback(self, expected_bytes=4096):
        if self.use_real_channel == 0:
            try:
                data = self.sock.recv(expected_bytes)
                if not data: return None
                return np.frombuffer(data, dtype=np.float32)
            except: return None
        return None

    def close(self):
        if self.use_real_channel == 0 and hasattr(self, 'sock'):
            self.sock.close()

class SDRDownlinkTX:
    def __init__(self, proxy_ip='127.0.0.1', proxy_port=8001):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024 * 8)
        self.proxy_addr = (proxy_ip, proxy_port)
        
        tone_cplx = np.array([[0.707, 0.707], [0.707, -0.707], [-0.707, -0.707], [-0.707, 0.707]], dtype=np.float32)
        self.agc_pad = np.tile(tone_cplx, (40, 1))
        
        barker_13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=np.float32)
        self.preamble = np.column_stack((barker_13, barker_13)) / np.sqrt(2)
        self.preamble = np.tile(self.preamble, (4, 1))
        
        sync_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1], dtype=np.uint8)
        sync_syms = np.where(sync_bits == 1, 0.707, -0.707).astype(np.float32)
        self.sync_word = np.column_stack((sync_syms, sync_syms))

    def _pack_mac_header(self, msg_type, chunk_idx, total_chunks, length):
        header_bytes = struct.pack('!HHHHI', 43690, msg_type, chunk_idx, total_chunks, length) 
        bits = np.unpackbits(np.frombuffer(header_bytes, dtype=np.uint8))
        repeated_bits = np.repeat(bits, 8)
        syms = np.where(repeated_bits == 0, -1.0, 1.0).astype(np.float32) * 0.707
        return np.column_stack((syms, syms))

    def send_frame(self, msg_type, chunk_idx, total_chunks, payload_1d_floats):
        if len(payload_1d_floats) % 2 != 0:
            payload_1d_floats = np.append(payload_1d_floats, 0.0)
            
        payload_iq = payload_1d_floats.reshape(-1, 2)
        payload_len = len(payload_iq)
        
        mac_header = self._pack_mac_header(msg_type, chunk_idx, total_chunks, payload_len)
        frame = np.vstack((self.agc_pad, self.preamble, self.sync_word, mac_header, payload_iq)).astype(np.float32)
        
        CHUNK_SIZE = 4000
        for i in range(0, len(frame), CHUNK_SIZE):
            self.sock.sendto(frame[i:i+CHUNK_SIZE].tobytes(), self.proxy_addr)
            time.sleep(0.005)
            
        if msg_type == 2:
            print(f"[MAC层] 发送包 [{chunk_idx+1}/{total_chunks}], 载荷 {payload_len} IQ。")
        else:
            print(f"[MAC层] 发送控制帧 (Type={msg_type})")
        time.sleep(0.04) 

def main():
    with open('config.json', 'r', encoding='utf-8') as f: CONFIG = json.load(f)
    parser = argparse.ArgumentParser(description="语义图片发送端 (带特征交织)")
    parser.add_argument("--dir", type=str, default=None, help="发送的源图片文件夹")
    args = parser.parse_args()
    
    tx_dir = args.dir if args.dir else CONFIG.get("tx_source_dir", "./dataset/test_images")
    tx_model = ImageSemanticEncoder().eval()
    try: tx_model.load_state_dict(torch.load(CONFIG["paths"]["tx_weights"], map_location='cpu'))
    except Exception: print("警告：未找到权重。")
        
    image_files = [f for f in glob.glob(os.path.join(tx_dir, "**/*.*"), recursive=True) if f.lower().endswith(('.png', '.jpg'))]
    if not image_files: return

    sdr_cfg = CONFIG.get("sdr_platform", {})
    use_real_fb = sdr_cfg.get("feedback_use_real_channel", 0)
    fb_cfg = sdr_cfg.get("feedback_sim_channel", {})
    rx_host_ip = fb_cfg.get("rx_host_ip", "127.0.0.1")
    rx_host_port = fb_cfg.get("rx_host_port", 9999)
    timeout_sec = fb_cfg.get("timeout_sec", 5.0)
    tx_proxy_ip = sdr_cfg.get("downlink_tx_proxy", {}).get("ip", "127.0.0.1")
    tx_proxy_port = sdr_cfg.get("downlink_tx_proxy", {}).get("port", 8001)

    uplink = UplinkChannel(rx_ip=rx_host_ip, port=rx_host_port, timeout=timeout_sec, use_real_channel=use_real_fb)
    downlink = SDRDownlinkTX(proxy_ip=tx_proxy_ip, proxy_port=tx_proxy_port)

    current_mod = '4QAM'
    current_scsi_state = 3 
    TEMP_MAPPING = {3: 1.0, 2: 0.8, 1: 0.3, 0: 0.1}
    TOTAL_CHUNKS = 16

    try:
        print("\n[系统] 正在唤醒 SDR...")
        dummy_data = np.zeros(8000, dtype=np.float32).tobytes()
        for _ in range(10):
            downlink.sock.sendto(dummy_data, downlink.proxy_addr)
            time.sleep(0.05)
        print("[系统] 预热完毕！")

        for file_path in image_files:
            print(f"\n>>> 发送图像: {os.path.basename(file_path)}")
            current_temp = TEMP_MAPPING.get(current_scsi_state, 1.0)
            img = Image.open(file_path).convert('RGB')
            w, h = img.size
            new_w, new_h = (w // 4) * 4, (h // 4) * 4
            transform = transforms.Compose([transforms.Resize((new_h, new_w)), transforms.ToTensor()])
            img_tensor = transform(img).unsqueeze(0)

            with torch.no_grad():
                tx_symbols = tx_model(img_tensor, mod_type=current_mod, temperature=current_temp)

            payload_floats = tx_symbols.cpu().numpy().flatten()
            
            # 严格对齐大小到 TARGET_SIZE 131072，防止尺寸异常
            if len(payload_floats) < TARGET_SIZE:
                payload_floats = np.pad(payload_floats, (0, TARGET_SIZE - len(payload_floats)))
            else:
                payload_floats = payload_floats[:TARGET_SIZE]

            # ================= 核心：应用特征交织 =================
            payload_floats = payload_floats[INTERLEAVE_IDX]
            print(f"[交织器] 131072 个特征点已全局打乱 (Seed: 2026)")
            # ======================================================
            
            chunk_len = len(payload_floats) // TOTAL_CHUNKS
            for c_idx in range(TOTAL_CHUNKS):
                start_idx = c_idx * chunk_len
                end_idx = start_idx + chunk_len
                chunk_payload = payload_floats[start_idx:end_idx]
                downlink.send_frame(msg_type=2, chunk_idx=c_idx, total_chunks=TOTAL_CHUNKS, payload_1d_floats=chunk_payload)

            time.sleep(0.2)
            eot_floats = np.tile(np.array([3.0, -3.0], dtype=np.float32), 64)
            for _ in range(2):
                downlink.send_frame(msg_type=3, chunk_idx=0, total_chunks=1, payload_1d_floats=eot_floats)
                time.sleep(0.1)
            
            noisy_fb_waveform = uplink.recv_feedback()
            if noisy_fb_waveform is not None and len(noisy_fb_waveform) > 10:
                try:
                    reliable_bits = decode_feedback_from_waveform(noisy_fb_waveform, num_bits=3, repeat=31)
                    current_scsi_state = (reliable_bits[1] << 1) | reliable_bits[2]
                    print(f"    <- [上行反馈] 下张图策略 -> 状态:{current_scsi_state}")
                except Exception: pass
            time.sleep(1.0) 

    except KeyboardInterrupt: print("\n[系统] 用户中断。")
    finally: uplink.close()

if __name__ == "__main__": main()