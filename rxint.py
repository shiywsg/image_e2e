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

# ================= 核心：解交织器初始化 =================
TARGET_SIZE = 131072
rnd_state = np.random.RandomState(2026)
INTERLEAVE_IDX = rnd_state.permutation(TARGET_SIZE)
DEINTERLEAVE_IDX = np.argsort(INTERLEAVE_IDX)
# ========================================================

class UplinkChannel:
    def __init__(self, port=9999, use_real_channel=0):
        self.use_real_channel = use_real_channel
        self.conn = None
        if self.use_real_channel == 0:
            self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_sock.bind(('0.0.0.0', port))
            self.server_sock.listen(1)
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
        noise_pwr = np.mean(valid_evm)
        return float(10 * np.log10(2 * (amplitude_ref ** 2) / max(noise_pwr, 1e-6)))

    def recv_frame(self):
        while True:
            try: data, _ = self.sock.recvfrom(65536)
            except socket.timeout: return -99, 0, 1, np.array([]), 0.0

            new_pairs = np.frombuffer(data, dtype=np.float32).reshape(-1, 2)
            self.stream_buffer = np.vstack((self.stream_buffer, new_pairs))
            if len(self.stream_buffer) > 300000: self.stream_buffer = self.stream_buffer[-100000:]
            
            if self.state == 0 and len(self.stream_buffer) >= 200:
                cplx_buf = self.stream_buffer[:, 0] + 1j * self.stream_buffer[:, 1]
                corr = np.abs(np.convolve(cplx_buf, self.template, mode='valid'))
                if len(corr) > 0:
                    max_c = np.max(corr)
                    mean_c = np.mean(corr)
                    if max_c > 3.0 * mean_c and max_c > 15.0:
                        start_idx = np.argmax(corr) + 52 
                        self.stream_buffer = self.stream_buffer[start_idx:]
                        self.state = 1
                    else: self.stream_buffer = self.stream_buffer[-100:]
                    
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
                    self.current_snr_db = self._calculate_evm_snr(np.column_stack((np.real(header_cplx), np.imag(header_cplx))))
                    
                    header_soft = np.real(header_cplx) + np.imag(header_cplx)
                    votes = np.sum(header_soft.reshape(96, 8), axis=1)
                    header_bytes = np.packbits(np.where(votes > 0, 1, 0).astype(np.uint8)).tobytes()
                    
                    try:
                        _, msg_type, chunk_idx, total_chunks, payload_len = struct.unpack('!HHHHI', header_bytes)
                        if payload_len == 64: msg_type = 3
                        elif 1000 < payload_len < 30000: msg_type = 2
                        
                        if msg_type not in [2, 3] or payload_len > 1_000_000: self.state = 0
                        else:
                            self.current_msg_type, self.current_chunk_idx, self.current_total_chunks, self.current_payload_len = msg_type, chunk_idx, total_chunks, payload_len
                            self.state = 2
                    except: self.state = 0
                else:
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
    parser = argparse.ArgumentParser(description="语义图片接收端 (带特征解交织)")
    parser.add_argument("--outdir", type=str, default=None, help="图片输出目录") 
    args = parser.parse_args()

    rx_outdir = args.outdir if args.outdir else CONFIG.get("rx_target_dir", "./output_images")
    rx_model = ImageSemanticDecoder().eval()
    try: rx_model.load_state_dict(torch.load(CONFIG["paths"]["rx_weights"], map_location='cpu'))
    except Exception: pass

    try:
        scsi_scaler = joblib.load('scsi_scaler.pkl')
        scsi_xgb = xgb.XGBClassifier()
        scsi_xgb.load_model('scsi_xgboost.json')
    except Exception: return

    sdr_cfg = CONFIG.get("sdr_platform", {})
    uplink = UplinkChannel(port=sdr_cfg.get("feedback_sim_channel", {}).get("rx_host_port", 9999), use_real_channel=sdr_cfg.get("feedback_use_real_channel", 0))
    downlink = SDRDownlinkRX(listen_ip=sdr_cfg.get("downlink_rx_proxy", {}).get("ip", "127.0.0.1"), listen_port=sdr_cfg.get("downlink_rx_proxy", {}).get("port", 8002))

    FALLBACK_SHAPE = (1, 32, 64, 64)
    image_buffer = np.zeros(TARGET_SIZE, dtype=np.float32)
    chunk_received = set()
    snr_accumulator, evm_accumulator = [], []
    image_counter, last_chunk_idx, last_frame_time, timeout_triggered = 0, -1, time.time(), False

    def async_inference_task(buf_copy, chunks_count, snr_copy, evm_copy, current_counter):
        # ================= 核心：应用特征解交织 =================
        buf_restored = buf_copy[DEINTERLEAVE_IDX]
        rx_tensor = torch.tensor(buf_restored, dtype=torch.float32).view(*FALLBACK_SHAPE)
        # ========================================================
        
        with torch.no_grad():
            reconstructed_img, _ = rx_model(rx_tensor, return_sparsity=True)

        avg_snr = np.mean(snr_copy) if snr_copy else 0.0
        avg_evm = np.mean(evm_copy) if evm_copy else 1.0
        global_erasure_rate = np.mean(buf_restored == 0.0) * 100.0
        
        features = np.array([[avg_evm, avg_snr, global_erasure_rate]])
        scsi_state = int(scsi_xgb.predict(scsi_scaler.transform(features))[0])
        print(f"[AI 异步推理完毕] 拼凑 {chunks_count} 个切片 | 擦除:{global_erasure_rate:.1f}% -> 推理反馈: {scsi_state:02b}")
        
        fb_waveform = encode_feedback_to_waveform([1, (scsi_state >> 1) & 1, scsi_state & 1], repeat=31)
        uplink.send_feedback(fb_waveform)

        save_path = os.path.join(os.getcwd(), rx_outdir, f"pic{current_counter:03d}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"\033[92m[存盘] 落盘成功: {save_path}\033[0m\n")
        vutils.save_image(reconstructed_img, save_path)

    def trigger_flush():
        nonlocal image_buffer, chunk_received, snr_accumulator, evm_accumulator, image_counter, last_chunk_idx
        if len(chunk_received) == 0: return
        image_counter += 1
        threading.Thread(target=async_inference_task, args=(image_buffer.copy(), len(chunk_received), list(snr_accumulator), list(evm_accumulator), image_counter), daemon=True).start()
        image_buffer, chunk_received, snr_accumulator, evm_accumulator, last_chunk_idx = np.zeros(TARGET_SIZE, dtype=np.float32), set(), [], [], -1

    print("\033[96m[系统] 接收端初始化完毕，进入交织容错盲收模式...\033[0m")
    try:
        while True:
            msg_type, chunk_idx, total_chunks, payload, physical_snr = downlink.recv_frame()
            current_time = time.time()
            gap_time, last_frame_time = current_time - last_frame_time, current_time
            
            if msg_type == -99:
                if len(chunk_received) > 0 and not timeout_triggered and gap_time > 10.0:
                    trigger_flush()
                    timeout_triggered = True
                continue
            timeout_triggered = False

            if msg_type == 2: 
                if len(chunk_received) > 0 and gap_time > 0.5: trigger_flush()
                elif chunk_idx <= last_chunk_idx and len(chunk_received) > 0: trigger_flush()
                
                snr_linear = 10 ** (physical_snr / 10.0)
                decode_tau = min(( (1.0/snr_linear) * np.log(3)) / (2.0 * 0.7071), 0.7)
                ternary_payload = np.where(np.abs(payload) < decode_tau, 0.0, np.sign(payload))
                
                expected_chunk_len = TARGET_SIZE // total_chunks
                start_idx, end_idx = chunk_idx * expected_chunk_len, min((chunk_idx + 1) * expected_chunk_len, TARGET_SIZE)
                
                if start_idx < TARGET_SIZE:
                    image_buffer[start_idx:end_idx] = ternary_payload[:end_idx - start_idx]
                    chunk_received.add(chunk_idx)
                    snr_accumulator.append(physical_snr)
                    evm_accumulator.append(calculate_soft_evm(payload))
                
                last_chunk_idx = chunk_idx
                if chunk_idx == total_chunks - 1: trigger_flush()
            elif msg_type == 3 and len(chunk_received) > 0: trigger_flush()

    except KeyboardInterrupt: pass
    finally: uplink.close()

if __name__ == "__main__": main()