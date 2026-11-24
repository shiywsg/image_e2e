import socket
import numpy as np
import torch
import torchvision.utils as vutils
import os
import argparse
import json
import joblib
import xgboost as xgb
import warnings
import struct
import time
import threading

from image_e2e_model import ImageSemanticDecoder
from comm_utils import encode_feedback_to_waveform

warnings.filterwarnings("ignore")

DEBUG_MODE = True
def debug_print(msg, color="93"):
    if DEBUG_MODE: print(f"\033[{color}m[RX-DEBUG] {msg}\033[0m")

class UplinkChannel:
    def __init__(self, port=9999, use_real_channel=0):
        self.use_real_channel = use_real_channel
        self.conn = None
        if self.use_real_channel == 0:
            self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_sock.bind(('0.0.0.0', port))
            self.server_sock.listen(1)
            print(f"\033[96m[Uplink] 反馈通道已启动(端口{port})，等待发送端...\033[0m")
            self.conn, self.addr = self.server_sock.accept()

    def send_feedback(self, fb_waveform_floats):
        if self.use_real_channel == 0 and self.conn:
            try: self.conn.sendall(fb_waveform_floats.tobytes())
            except Exception: pass

    def close(self):
        if self.use_real_channel == 0:
            if self.conn: self.conn.close()
            if hasattr(self, 'server_sock'): self.server_sock.close()

class SDRDownlinkRX:
    def __init__(self, listen_ip='127.0.0.1', listen_port=8002):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024 * 8)
        self.sock.settimeout(1.0) 
        self.sock.bind((listen_ip, listen_port))
        
        self.stream_buffer = np.zeros((0, 2), dtype=np.float32)
        barker_13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=np.float32)
        barker_complex = (barker_13 + 1j * barker_13) / np.sqrt(2)
        self.template = np.conj(np.tile(barker_complex, 4)[::-1])
        
        sync_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1], dtype=np.uint8)
        self.sync_cplx_ideal = np.where(sync_bits == 1, 0.707+0.707j, -0.707-0.707j).astype(np.complex64)
        self.sync_template_cplx = np.conj(self.sync_cplx_ideal[::-1])
        
        self.state = 0 

    def _calculate_evm_snr(self, header_iq):
        I, Q = header_iq[:, 0], header_iq[:, 1]
        amplitude_ref = np.mean(np.abs(I) + np.abs(Q)) / 1.414 
        ideal_I = np.sign(I) * amplitude_ref
        ideal_Q = np.sign(Q) * amplitude_ref
        evm_sq = (I - ideal_I)**2 + (Q - ideal_Q)**2
        valid_evm = evm_sq[evm_sq <= np.percentile(evm_sq, 95)]
        
        signal_pwr = 2 * (amplitude_ref ** 2)
        noise_pwr = np.mean(valid_evm)
        if noise_pwr <= 0: return 40.0
        return float(10 * np.log10(signal_pwr / noise_pwr))

    def recv_frame(self):
        while True:
            try:
                data, _ = self.sock.recvfrom(65536)
            except socket.timeout:
                return -99, 0, 1, np.array([]), 0.0

            new_pairs = np.frombuffer(data, dtype=np.float32).reshape(-1, 2)
            self.stream_buffer = np.vstack((self.stream_buffer, new_pairs))
            
            if len(self.stream_buffer) > 300000:
                self.stream_buffer = self.stream_buffer[-100000:]
            
            if self.state == 0 and len(self.stream_buffer) >= 200:
                cplx_buf = self.stream_buffer[:, 0] + 1j * self.stream_buffer[:, 1]
                corr = np.abs(np.convolve(cplx_buf, self.template, mode='valid'))
                if len(corr) > 0:
                    max_c = np.max(corr)
                    mean_c = np.mean(corr)
                    
                    if DEBUG_MODE and max_c > 2.5 * mean_c and max_c > 12.0:
                        debug_print(f"[ST0-Barker] 探测峰值: max={max_c:.2f}, mean={mean_c:.2f} (目标门限需>{3.0 * mean_c:.2f} 且 >15.0)", "90")

                    # 修复1：门限倍数下调至3.0，抵抗深衰落！
                    if max_c > 3.0 * mean_c and max_c > 15.0:
                        debug_print(f"[ST0-Barker] 🚀 成功锁定 Barker! max_corr={max_c:.2f}", "92")
                        start_idx = np.argmax(corr) + 52 
                        self.stream_buffer = self.stream_buffer[start_idx:]
                        self.state = 1
                    else:
                        self.stream_buffer = self.stream_buffer[-100:]
                    
            elif self.state == 1 and len(self.stream_buffer) >= 1000: 
                search_cplx = self.stream_buffer[:30, 0] + 1j * self.stream_buffer[:30, 1]
                best_idx, best_rot, max_corr = -1, 1.0+0j, 0.0
                
                for rot in [1+0j, 0+1j, -1+0j, 0-1j]:
                    test_cplx = search_cplx * rot
                    corr = np.real(np.convolve(test_cplx, self.sync_template_cplx, mode='valid'))
                    if np.max(corr) > max_corr:
                        max_corr = np.max(corr)
                        best_idx = np.argmax(corr)
                        best_rot = rot
                
                debug_print(f"[ST1-Sync] 检查 0xAAAA: 最大相关值={max_corr:.2f}, 偏移={best_idx} (门限>4.0)", "90")
                
                # 修复2：Sync同步门限放宽至 4.0，偏移界限放宽至 12，包容衰落变形
                if max_corr > 4.0 and best_idx < 12:
                    debug_print(f"[ST1-Sync] 🟢 Sync对齐成功！提取 Header 进行解码... 偏移:{best_idx}", "92")
                    self.current_rotation = best_rot
                    header_iq_raw = self.stream_buffer[best_idx + 16 : best_idx + 16 + 768]
                    header_cplx = (header_iq_raw[:, 0] + 1j * header_iq_raw[:, 1]) * self.current_rotation
                    self.stream_buffer = self.stream_buffer[best_idx + 16 + 768:]
                    
                    header_iq = np.column_stack((np.real(header_cplx), np.imag(header_cplx)))
                    self.current_snr_db = self._calculate_evm_snr(header_iq)
                    
                    header_soft = np.real(header_cplx) + np.imag(header_cplx)
                    votes = np.sum(header_soft.reshape(96, 8), axis=1)
                    final_bits = np.where(votes > 0, 1, 0).astype(np.uint8)
                    header_bytes = np.packbits(final_bits).tobytes()
                    
                    try:
                        _, msg_type, chunk_idx, total_chunks, payload_len = struct.unpack('!HHHHI', header_bytes)
                        
                        # 修复3：硬件级位宽纠错！利用 payload 长度倒推并强行纠正被翻转的 msg_type
                        if payload_len == 64: msg_type = 3
                        elif 260 < payload_len < 30000: msg_type = 2 #32块对应250，8块对应1000
                        
                        debug_print(f"[MAC解析] 🟢 成功! Type={msg_type}, 包={chunk_idx+1}/{total_chunks}, 长度={payload_len}", "92")
                        
                        # 增加严密的越界检查
                        if payload_len <= 0 or payload_len > 1_000_000 or msg_type not in [2, 3]:
                            debug_print("[MAC解析] ❌ 字段越界异常，丢弃。", "91")
                            self.state = 0
                        elif msg_type == 2 and (chunk_idx >= total_chunks or total_chunks <= 0):
                            debug_print("[MAC解析] ❌ Chunk 序号超界异常，丢弃。", "91")
                            self.state = 0
                        else:
                            self.current_msg_type = msg_type
                            self.current_chunk_idx = chunk_idx
                            self.current_total_chunks = total_chunks
                            self.current_payload_len = payload_len
                            self.state = 2
                    except Exception as e:
                        debug_print(f"[MAC解析] ❌ struct 解包失败 (杂乱比特): {e}", "91")
                        self.state = 0
                else:
                    debug_print("[ST1-Sync] ❌ Sync峰值太低或偏移过大，假唤醒，丢弃！", "91")
                    self.state = 0
                    self.stream_buffer = self.stream_buffer[50:]

            elif self.state == 2:
                if len(self.stream_buffer) >= self.current_payload_len:
                    payload_iq = self.stream_buffer[:self.current_payload_len]
                    self.stream_buffer = self.stream_buffer[self.current_payload_len:]
                    self.state = 0
                    
                    payload_cplx = (payload_iq[:, 0] + 1j * payload_iq[:, 1]) * self.current_rotation
                    payload_1d = np.column_stack((np.real(payload_cplx), np.imag(payload_cplx))).flatten()
                    
                    return self.current_msg_type, self.current_chunk_idx, self.current_total_chunks, payload_1d, self.current_snr_db

def calculate_soft_evm(noisy_payload):
    if len(noisy_payload) == 0: return 1.0
    ideal = np.sign(noisy_payload)
    ideal[ideal == 0] = 1.0 
    return np.mean(np.abs(noisy_payload - ideal))

def main():
    with open('config.json', 'r', encoding='utf-8') as f: CONFIG = json.load(f)
    parser = argparse.ArgumentParser(description="语义图片接收端")
    parser.add_argument("--outdir", type=str, default=None, help="图片输出目录") 
    args = parser.parse_args()

    rx_outdir = args.outdir if args.outdir else CONFIG.get("rx_target_dir", "./output_images")
    rx_model = ImageSemanticDecoder().eval()
    try: 
        rx_model.load_state_dict(torch.load(CONFIG["paths"]["rx_weights"], map_location='cpu'))
    except Exception: 
        print("权重加载失败。")

    try:
        scsi_scaler = joblib.load('scsi_scaler.pkl')
        scsi_xgb = xgb.XGBClassifier()
        scsi_xgb.load_model('scsi_xgboost.json')
    except Exception:
        print("ML 模型加载失败。")
        return

    sdr_cfg = CONFIG.get("sdr_platform", {})
    use_real_fb = sdr_cfg.get("feedback_use_real_channel", 0)
    rx_host_port = sdr_cfg.get("feedback_sim_channel", {}).get("rx_host_port", 9999)
    rx_proxy_ip = sdr_cfg.get("downlink_rx_proxy", {}).get("ip", "127.0.0.1")
    rx_proxy_port = sdr_cfg.get("downlink_rx_proxy", {}).get("port", 8002)

    uplink = UplinkChannel(port=rx_host_port, use_real_channel=use_real_fb)
    downlink = SDRDownlinkRX(listen_ip=rx_proxy_ip, listen_port=rx_proxy_port)

    TARGET_SIZE = 131072
    FALLBACK_SHAPE = (1, 32, 64, 64)
    image_buffer = np.zeros(TARGET_SIZE, dtype=np.float32)
    chunk_received = set()
    snr_accumulator = []
    evm_accumulator = []
    
    image_counter = 0
    last_chunk_idx = -1  
    last_frame_time = time.time()
    timeout_triggered = False

    def async_inference_task(buf_copy, chunks_count, snr_copy, evm_copy, current_counter):
        rx_tensor = torch.tensor(buf_copy, dtype=torch.float32).view(*FALLBACK_SHAPE)
        with torch.no_grad():
            reconstructed_img, _ = rx_model(rx_tensor, return_sparsity=True)

        avg_snr = np.mean(snr_copy) if snr_copy else 0.0
        avg_evm = np.mean(evm_copy) if evm_copy else 1.0
        global_erasure_rate = np.mean(buf_copy == 0.0) * 100.0
        
        features = np.array([[avg_evm, avg_snr, global_erasure_rate]])
        scsi_state = int(scsi_xgb.predict(scsi_scaler.transform(features))[0])
        
        print(f"[AI 异步推理完毕] 拼凑 {chunks_count} 个切片 | 擦除:{global_erasure_rate:.1f}% -> 推理反馈: {scsi_state:02b}")
        
        feedback_bits = [1, (scsi_state >> 1) & 1, scsi_state & 1]
        fb_waveform = encode_feedback_to_waveform(feedback_bits, repeat=31)
        
        if use_real_fb == 0:
            uplink_noise_pwr = 1.0 / (10 ** (avg_snr / 10.0)) if avg_snr > 0 else 1.0
            noisy_fb = fb_waveform + np.sqrt(uplink_noise_pwr) * np.random.randn(*fb_waveform.shape).astype(np.float32)
            uplink.send_feedback(noisy_fb)

        final_name = f"pic{current_counter:03d}.png"
        save_path = os.path.join(os.getcwd(), rx_outdir, final_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"\033[92m[存盘] 落盘成功: {save_path}\033[0m\n")
        vutils.save_image(reconstructed_img, save_path)

    def trigger_flush():
        nonlocal image_buffer, chunk_received, snr_accumulator, evm_accumulator, image_counter, last_chunk_idx
        if len(chunk_received) == 0: return

        buf_copy = image_buffer.copy()
        chunks_count = len(chunk_received)
        snr_copy = list(snr_accumulator)
        evm_copy = list(evm_accumulator)
        
        image_counter += 1
        current_counter = image_counter
        
        threading.Thread(target=async_inference_task, args=(buf_copy, chunks_count, snr_copy, evm_copy, current_counter), daemon=True).start()
        
        image_buffer = np.zeros(TARGET_SIZE, dtype=np.float32)
        chunk_received.clear()
        snr_accumulator.clear()
        evm_accumulator.clear()
        last_chunk_idx = -1

    print("\033[96m[系统] 接收端初始化完毕，进入极简盲收模式...\033[0m")
    try:
        while True:
            msg_type, chunk_idx, total_chunks, payload, physical_snr = downlink.recv_frame()
            
            current_time = time.time()
            gap_time = current_time - last_frame_time
            last_frame_time = current_time
            
            if msg_type == -99:
                if len(chunk_received) > 0 and not timeout_triggered:
                    if gap_time > 10.0:
                        print("\n\033[91m[超时保护] 超过 10 秒未收到空口数据，触发强制脱壳清空缓冲！\033[0m")
                        trigger_flush()
                        timeout_triggered = True
                continue
                
            timeout_triggered = False

            if msg_type == 2: 
                # 修复4：帧间隙探测。如果在正常数据包前发现了 > 0.5 秒的停顿，无条件说明这是另一张新图片！强制断代
                if len(chunk_received) > 0 and gap_time > 0.5:
                    print(f"\n\033[93m[MAC 容错] 探测到 {gap_time:.2f}s 的停顿(>0.5s)，判定为新图起始，强制闭环上一张！\033[0m")
                    trigger_flush()
                # 保留原有序列号回绕兜底
                elif chunk_idx <= last_chunk_idx and len(chunk_received) > 0:
                    print(f"\n\033[93m[MAC 容错] 检测到包序号回绕 ({chunk_idx} <= {last_chunk_idx})，强制脱壳存盘！\033[0m")
                    trigger_flush()
                
                print(f"  -> 收到包 [{chunk_idx+1}/{total_chunks}] 长度: {len(payload)}")
                
                fixed_tau = CONFIG.get("tau_config", {}).get('4QAM', 0.35)
                snr_linear = 10 ** (physical_snr / 10.0)
                decode_tau = min(( (1.0/snr_linear) * np.log(3)) / (2.0 * 0.7071), 0.7)
                
                ternary_payload = np.where(np.abs(payload) < decode_tau, 0.0, np.sign(payload))
                
                expected_chunk_len = TARGET_SIZE // total_chunks
                start_idx = chunk_idx * expected_chunk_len
                end_idx = min(start_idx + len(ternary_payload), TARGET_SIZE)
                
                if start_idx < TARGET_SIZE:
                    valid_len = end_idx - start_idx
                    image_buffer[start_idx:end_idx] = ternary_payload[:valid_len]
                    chunk_received.add(chunk_idx)
                    snr_accumulator.append(physical_snr)
                    evm_accumulator.append(calculate_soft_evm(payload))
                
                last_chunk_idx = chunk_idx

                if chunk_idx == total_chunks - 1:
                    print(f"\033[93m[MAC 容错] 收到尾包 ({chunk_idx+1}/{total_chunks})，快速触发存盘！\033[0m")
                    trigger_flush()

            elif msg_type == 3: 
                if len(chunk_received) > 0:
                    print("\033[96m[下行射频] 收到常规 EOT 标识，完美闭环存盘！\033[0m")
                    trigger_flush()

    except KeyboardInterrupt:
        print("\n[系统] 停止接收。")
    finally:
        uplink.close()

if __name__ == "__main__":
    main()