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

    def set_timeout(self, timeout):
        """动态修改超时时间，用于最后一张图的 10 秒等待"""
        if self.use_real_channel == 0 and hasattr(self, 'sock'):
            self.sock.settimeout(timeout)

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
        
        tone_cplx = np.array([
            [0.707, 0.707],
            [0.707, -0.707],
            [-0.707, -0.707],
            [-0.707, 0.707]
        ], dtype=np.float32)
        
        # 🔧【微调点 1】：AGC 热身垫片长度。
        # 建议范围: 20 ~ 50。越长 AGC 越平稳，但可能导致 RX 相位环对单音过拟合。
        self.agc_pad = np.tile(tone_cplx, (30, 1)) 
        
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
        
        # 🔧【微调点 2】：UDP 分片大小。建议范围: 1000 ~ 8000。
        # 影响向 TX Proxy 灌数据的吞吐稳定性，4000 是通常较优的值。
        CHUNK_SIZE = 4000
        for i in range(0, len(frame), CHUNK_SIZE):
            self.sock.sendto(frame[i:i+CHUNK_SIZE].tobytes(), self.proxy_addr)
            # 🔧【微调点 3】：UDP 分片间隙。建议范围: 0.001 ~ 0.01 秒。
            time.sleep(0.005) 
            
        if msg_type == 2:
            print(f"[MAC层] 发送包 [{chunk_idx+1}/{total_chunks}], 载荷 {payload_len} IQ。")
        else:
            print(f"[MAC层] 发送控制帧 (Type={msg_type})")
            
        # 🔧【微调点 4】：整包发送间隙 (跨包退避)。建议范围: 0.02 ~ 0.06 秒。
        # 太长(0.05)会导致 RX 锁相环失锁；太短会导致 TX Buffer 溢出尾部被截断。
        time.sleep(0.011)  

def main():
    with open('config.json', 'r', encoding='utf-8') as f: CONFIG = json.load(f)
    parser = argparse.ArgumentParser(description="语义图片发送端 (64块对照实验 - 统计增强版)")
    parser.add_argument("--dir", type=str, default=None, help="发送的源图片文件夹")
    args = parser.parse_args()
    
    tx_dir = args.dir if args.dir else CONFIG.get("tx_source_dir", "./dataset/test_images")
    tx_model = ImageSemanticEncoder().eval()
    try: 
        tx_model.load_state_dict(torch.load(CONFIG["paths"]["tx_weights"], map_location='cpu'))
    except Exception: 
        print("警告：未找到权重。")
        
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
    
    TOTAL_CHUNKS = 64

    # =============== 新增统计变量 ===============
    stats_feedbacks = {0: 0, 1: 0, 2: 0, 3: 0}
    total_images_sent = 0
    missed_feedbacks = 0

    def print_stats_report():
        """打印反馈统计信息报告"""
        print("\n\033[92m" + "="*45)
        print(" 📊 本次实验发送端统计报告 (TX Statistics)")
        print("="*45)
        print(f" 🖼️ 总计尝试发送图片数量    : {total_images_sent} 张")
        print(f" 📥 成功接收 S-CSI 反馈总数 : {sum(stats_feedbacks.values())} 次")
        print(f" ❌ 丢失/超时未收到反馈次数 : {missed_feedbacks} 次")
        print("-" * 45)
        print(f" [分类统计] S-CSI = 0 (极差) : {stats_feedbacks[0]} 次")
        print(f" [分类统计] S-CSI = 1 (较差) : {stats_feedbacks[1]} 次")
        print(f" [分类统计] S-CSI = 2 (良好) : {stats_feedbacks[2]} 次")
        print(f" [分类统计] S-CSI = 3 (极佳) : {stats_feedbacks[3]} 次")
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
            print(f"\n>>> 抽取并发送图像数据 ({idx+1}/{len(image_files)}): {rel_path_unix}")

            current_temp = TEMP_MAPPING.get(current_scsi_state, 1.0)
            img = Image.open(file_path).convert('RGB')
            w, h = img.size
            new_w, new_h = (w // 4) * 4, (h // 4) * 4
            transform = transforms.Compose([transforms.Resize((new_h, new_w)), transforms.ToTensor()])
            img_tensor = transform(img).unsqueeze(0)

            with torch.no_grad():
                tx_symbols = tx_model(img_tensor, mod_type=current_mod, temperature=current_temp)

            payload_floats = tx_symbols.cpu().numpy().flatten()
            state_bits = f"{current_scsi_state:02b}"
            print(f"[下行射频] 调制:4QAM | S-CSI:{state_bits} | 正在流水线推送 ({TOTAL_CHUNKS} 包)...")
            
            chunk_len = len(payload_floats) // TOTAL_CHUNKS
            for c_idx in range(TOTAL_CHUNKS):
                start_idx = c_idx * chunk_len
                end_idx = start_idx + chunk_len if c_idx < TOTAL_CHUNKS - 1 else len(payload_floats)
                chunk_payload = payload_floats[start_idx:end_idx]
                downlink.send_frame(msg_type=2, chunk_idx=c_idx, total_chunks=TOTAL_CHUNKS, payload_1d_floats=chunk_payload)

            time.sleep(0.2)
            eot_floats = np.tile(np.array([3.0, -3.0], dtype=np.float32), 64)
            print("[下行射频] 图像发送完毕，发送 EOT 并等待反馈...")
            
            # 🔧【微调点 5】：EOT (End of Transmission) 帧发送次数。建议范围: 1 ~ 4。
            # 防止单次 EOT 遭遇突发衰落导致接收端一直阻塞等待。
            for _ in range(2): 
                downlink.send_frame(msg_type=3, chunk_idx=0, total_chunks=1, payload_1d_floats=eot_floats)
                time.sleep(0.1)
            
            # --- 动态超时逻辑处理 ---
            if is_last_file:
                uplink.set_timeout(10.0)
                debug_print("进入 Socket 阻塞等待上行反馈阶段 (最后一张图，超时 10s)...")
            else:
                uplink.set_timeout(timeout_sec)
                debug_print(f"进入 Socket 阻塞等待上行反馈阶段 (超时 {timeout_sec}s)...")
                
            noisy_fb_waveform = uplink.recv_feedback()
            
            if noisy_fb_waveform is not None and len(noisy_fb_waveform) > 10:
                try:
                    reliable_bits = decode_feedback_from_waveform(noisy_fb_waveform, num_bits=3, repeat=31)
                    scsi_state = (reliable_bits[1] << 1) | reliable_bits[2]
                    
                    # 保护逻辑：只接收 0,1,2,3 状态
                    if scsi_state in stats_feedbacks:
                        stats_feedbacks[scsi_state] += 1
                        current_scsi_state = scsi_state 
                        new_state_bits = f"{current_scsi_state:02b}"
                        print(f"    <- [上行反馈] 解码成功 | 收到状态:{current_scsi_state} ({new_state_bits})")
                    else:
                        raise ValueError("未知反馈状态")
                except Exception:
                    missed_feedbacks += 1
                    print("    <- [上行反馈] 信号严重畸变，视为未收到，保持策略不变。")
            else:
                missed_feedbacks += 1
                print("    <- [上行反馈] 未收到有效反馈 (超时)，视为未收到，保持策略不变。")

            time.sleep(1.0) 
            
        print("\n[系统] 所有图片切片已推入发送队列。")
        print("[系统] 等待底层 SDR 硬件缓冲区排空 (Flush)...")
        time.sleep(1.5)
        
        # 正常传输结束，打印报告
        print_stats_report()

    except KeyboardInterrupt: 
        print("\n\033[91m========================================\033[0m")
        print("\033[91m[系统] 检测到 Ctrl+C 手动中断！正在执行收尾...\033[0m")
        print_stats_report()
        
    finally: 
        uplink.close()

if __name__ == "__main__":
    main()