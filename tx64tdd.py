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

class SDRDownlinkTX:
    def __init__(self, proxy_ip='127.0.0.1', proxy_port=8001):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024 * 8)
        self.proxy_addr = (proxy_ip, proxy_port)
        
        tone_cplx = np.array([[0.707, 0.707], [0.707, -0.707], [-0.707, -0.707], [-0.707, 0.707]], dtype=np.float32)
        self.agc_pad = np.tile(tone_cplx, (30, 1)) 
        
        barker_13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=np.float32)
        self.preamble = np.tile(np.column_stack((barker_13, barker_13)) / np.sqrt(2), (4, 1))
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
            print(f"[MAC层] 发送图像分片 [{chunk_idx+1}/{total_chunks}], 载荷 {payload_len} IQ。")
        elif msg_type == 3:
            print(f"[MAC层] 发送控制帧 EOT (Type={msg_type})")
            
        time.sleep(0.011)  

class SDRUplinkRX:
    def __init__(self, listen_ip='127.0.0.1', listen_port=8004):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024 * 8)
        self.sock.bind((listen_ip, listen_port))
        
        barker_13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=np.float32)
        self.template = np.conj(np.tile((barker_13 + 1j * barker_13) / np.sqrt(2), 4)[::-1])
        sync_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1], dtype=np.uint8)
        self.sync_template_cplx = np.conj(np.where(sync_bits == 1, 0.707+0.707j, -0.707-0.707j).astype(np.complex64)[::-1])
        self.stream_buffer = np.zeros((0, 2), dtype=np.float32)
        self.state = 0 

    def flush_buffer(self):
        self.sock.setblocking(False)
        while True:
            try: self.sock.recv(65536)
            except: break
        self.sock.setblocking(True)
        self.stream_buffer = np.zeros((0, 2), dtype=np.float32)
        self.state = 0

    def recv_feedback_frame(self, timeout_sec=5.0):
        self.sock.settimeout(0.5)
        start_t = time.time()
        
        while time.time() - start_t < timeout_sec:
            try:
                data, _ = self.sock.recvfrom(65536)
                self.stream_buffer = np.vstack((self.stream_buffer, np.frombuffer(data, dtype=np.float32).reshape(-1, 2)))
            except socket.timeout:
                pass
            
            while True:
                if self.state == 0 and len(self.stream_buffer) >= 200:
                    search_len = min(len(self.stream_buffer), 1200)
                    cplx_buf = self.stream_buffer[:search_len, 0] + 1j * self.stream_buffer[:search_len, 1]
                    corr = np.abs(np.convolve(cplx_buf, self.template, mode='valid'))
                    if len(corr) > 0:
                        max_c = np.max(corr)
                        mean_c = np.mean(corr)
                        if max_c > 2.0 * mean_c and max_c > 15.0:
                            start_idx = np.argmax(corr) + 52 
                            if start_idx < len(self.stream_buffer):
                                self.stream_buffer = self.stream_buffer[start_idx:]
                                self.state = 1
                                continue
                            else:
                                self.stream_buffer = np.zeros((0,2), dtype=np.float32)
                        else:
                            self.stream_buffer = self.stream_buffer[len(corr):]
                            continue 
                elif self.state == 1 and len(self.stream_buffer) >= 1000:
                    search_cplx = self.stream_buffer[:30, 0] + 1j * self.stream_buffer[:30, 1]
                    best_idx, best_rot, max_corr = -1, 1.0+0j, 0.0
                    for rot in [1+0j, 0+1j, -1+0j, 0-1j]:
                        corr = np.real(np.convolve(search_cplx * rot, self.sync_template_cplx, mode='valid'))
                        if np.max(corr) > max_corr: max_corr, best_idx, best_rot = np.max(corr), np.argmax(corr), rot
                    
                    if max_corr > 4.0 and best_idx < 12:
                        self.current_rotation = best_rot
                        header_iq_raw = self.stream_buffer[best_idx + 16 : best_idx + 16 + 768]
                        header_cplx = (header_iq_raw[:, 0] + 1j * header_iq_raw[:, 1]) * self.current_rotation
                        self.stream_buffer = self.stream_buffer[best_idx + 16 + 768:]
                        
                        header_soft = np.real(header_cplx) + np.imag(header_cplx)
                        votes = np.sum(header_soft.reshape(96, 8), axis=1)
                        header_bytes = np.packbits(np.where(votes > 0, 1, 0).astype(np.uint8)).tobytes()
                        
                        try:
                            magic, msg_type, chunk_idx, total_chunks, payload_len = struct.unpack('!HHHHI', header_bytes)
                            if magic == 43690 and msg_type == 4: # MsgType 4 代表反馈帧
                                self.current_msg_type = msg_type
                                self.current_payload_len = payload_len
                                self.state = 2
                                continue 
                            else: self.state = 0
                        except Exception: self.state = 0
                    else:
                        self.state = 0
                        self.stream_buffer = self.stream_buffer[50:]
                        continue
                elif self.state == 2 and len(self.stream_buffer) >= self.current_payload_len:
                    payload_iq = self.stream_buffer[:self.current_payload_len]
                    self.stream_buffer = self.stream_buffer[self.current_payload_len:]
                    self.state = 0
                    payload_cplx = (payload_iq[:, 0] + 1j * payload_iq[:, 1]) * self.current_rotation
                    return np.column_stack((np.real(payload_cplx), np.imag(payload_cplx))).flatten()
                else:
                    break # 缓存不足，继续接收 UDP
        return None

def main():
    with open('config.json', 'r', encoding='utf-8') as f: CONFIG = json.load(f)
    parser = argparse.ArgumentParser(description="语义图片发送端 (TDD 半双工物理反馈版)")
    parser.add_argument("--dir", type=str, default=None, help="发送的源图片文件夹")
    args = parser.parse_args()
    
    tx_dir = args.dir if args.dir else CONFIG.get("tx_source_dir", "./dataset/test_images")
    tx_model = ImageSemanticEncoder().eval()
    try: tx_model.load_state_dict(torch.load(CONFIG["paths"]["tx_weights"], map_location='cpu'))
    except Exception: print("警告：未找到权重。")
        
    image_files = [f for f in glob.glob(os.path.join(tx_dir, "**/*.*"), recursive=True) if f.lower().endswith(('.png', '.jpg'))]
    if not image_files: return

    sdr_cfg = CONFIG.get("sdr_platform", {})
    fb_cfg = sdr_cfg.get("feedback_sim_channel", {})
    timeout_sec = fb_cfg.get("timeout_sec", 5.0)
    
    downlink = SDRDownlinkTX(proxy_ip=sdr_cfg.get("downlink_tx_proxy", {}).get("ip", "127.0.0.1"), 
                             proxy_port=sdr_cfg.get("downlink_tx_proxy", {}).get("port", 8001))
    uplink = SDRUplinkRX(listen_ip=sdr_cfg.get("uplink_rx_proxy", {}).get("ip", "127.0.0.1"),
                         listen_port=sdr_cfg.get("uplink_rx_proxy", {}).get("port", 8004))

    current_mod = '4QAM'
    current_scsi_state = 3 
    TEMP_MAPPING = {3: 1.0, 2: 0.8, 1: 0.3, 0: 0.1}
    TOTAL_CHUNKS = 64

    stats_feedbacks = {0: 0, 1: 0, 2: 0, 3: 0}
    total_images_sent = 0
    missed_feedbacks = 0

    def print_stats_report():
        print("\n\033[92m" + "="*45)
        print(" 📊 本次实验发送端统计报告 (TX Statistics)")
        print("="*45)
        print(f" 🖼️ 总计尝试发送图片数量    : {total_images_sent} 张")
        print(f" 📥 成功接收物理反馈总数 : {sum(stats_feedbacks.values())} 次")
        print(f" ❌ 丢失/超时未收到反馈次数 : {missed_feedbacks} 次")
        print("-" * 45)
        print(f" [分类] S-CSI = 0 (极差) : {stats_feedbacks[0]} 次")
        print(f" [分类] S-CSI = 1 (较差) : {stats_feedbacks[1]} 次")
        print(f" [分类] S-CSI = 2 (良好) : {stats_feedbacks[2]} 次")
        print(f" [分类] S-CSI = 3 (极佳) : {stats_feedbacks[3]} 次")
        print("="*45 + "\033[0m\n")

    try:
        print("\n[系统] 正在唤醒并预热 USRP 射频前端...")
        dummy_data = np.zeros(8000, dtype=np.float32).tobytes()
        for _ in range(10):
            downlink.sock.sendto(dummy_data, downlink.proxy_addr)
            time.sleep(0.15)
        print("[系统] 预热完毕，开始传输真实图片！")

        for idx, file_path in enumerate(image_files):
            total_images_sent += 1
            is_last_file = (idx == len(image_files) - 1)
            
            rel_path_unix = os.path.relpath(file_path, tx_dir).replace('\\', '/')
            print(f"\n>>> 抽取并发送图像 ({idx+1}/{len(image_files)}): {rel_path_unix}")

            current_temp = TEMP_MAPPING.get(current_scsi_state, 1.0)
            img = Image.open(file_path).convert('RGB')
            w, h = img.size
            new_w, new_h = (w // 4) * 4, (h // 4) * 4
            transform = transforms.Compose([transforms.Resize((new_h, new_w)), transforms.ToTensor()])
            img_tensor = transform(img).unsqueeze(0)

            with torch.no_grad():
                tx_symbols = tx_model(img_tensor, mod_type=current_mod, temperature=current_temp)

            payload_floats = tx_symbols.cpu().numpy().flatten()
            print(f"[下行射频] S-CSI:{current_scsi_state:02b} | 正在流水线推送 ({TOTAL_CHUNKS} 包)...")
            
            chunk_len = len(payload_floats) // TOTAL_CHUNKS
            for c_idx in range(TOTAL_CHUNKS):
                start_idx = c_idx * chunk_len
                end_idx = start_idx + chunk_len if c_idx < TOTAL_CHUNKS - 1 else len(payload_floats)
                downlink.send_frame(msg_type=2, chunk_idx=c_idx, total_chunks=TOTAL_CHUNKS, payload_1d_floats=payload_floats[start_idx:end_idx])

            time.sleep(0.2)
            eot_floats = np.tile(np.array([3.0, -3.0], dtype=np.float32), 64)
            print("[下行射频] 图像发送完毕，发送 EOT 并进入 TDD 避让期...")
            
            for _ in range(2): 
                downlink.send_frame(msg_type=3, chunk_idx=0, total_chunks=1, payload_1d_floats=eot_floats)
                time.sleep(0.1)
            
            # 【TDD 核心时序】：清理残留底噪并给硬件预留切换时间
            time.sleep(0.5) 
            uplink.flush_buffer()
            
            wait_time = 10.0 if is_last_file else timeout_sec
            debug_print(f"进入空中监听模式，等待接收反馈帧 (超时 {wait_time}s)...")
                
            fb_waveform = uplink.recv_feedback_frame(timeout_sec=wait_time)
            
            if fb_waveform is not None and len(fb_waveform) > 10:
                try:
                    reliable_bits = decode_feedback_from_waveform(fb_waveform, num_bits=3, repeat=31)
                    scsi_state = (reliable_bits[1] << 1) | reliable_bits[2]
                    
                    if scsi_state in stats_feedbacks:
                        stats_feedbacks[scsi_state] += 1
                        current_scsi_state = scsi_state 
                        print(f"    <- [上行反馈] ✅ 成功捕获真实物理反馈！状态:{current_scsi_state}")
                    else:
                        raise ValueError("未知反馈状态")
                except Exception:
                    missed_feedbacks += 1
                    print("    <- [上行反馈] ⚠️ 信号严重畸变解码失败，保持策略不变。")
            else:
                missed_feedbacks += 1
                print("    <- [上行反馈] ❌ 未收到有效反馈 (超时)，保持策略不变。")

            time.sleep(1.0) 
            
        print("\n[系统] 所有图片切片已推入发送队列。")
        print_stats_report()

    except KeyboardInterrupt: 
        print("\n\033[91m========================================\033[0m")
        print("\033[91m[系统] 检测到 Ctrl+C 手动中断！正在执行收尾...\033[0m")
        print_stats_report()
        
if __name__ == "__main__":
    main()