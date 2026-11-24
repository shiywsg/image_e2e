import socket
import numpy as np
import time
import argparse
import struct

class TDDSlowNode:
    def __init__(self, mode):
        self.mode = mode
        if mode == 'master':
            self.tx_port, self.rx_port = 8001, 8004
            self.my_msg_type, self.expect_msg_type = 2, 3
            print("[Master Node A] 启动 (慢动作诊断版)")
        else:
            self.tx_port, self.rx_port = 8003, 8002
            self.my_msg_type, self.expect_msg_type = 3, 2
            print("[Slave Node B] 启动 (慢动作诊断版)")
            
        self.sock_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_rx.bind(("127.0.0.1", self.rx_port))
        
        # 【极度宽裕的唤醒序列】
        # 1. 200 个单音符号让 AGC 彻底锁死幅度
        self.agc_pad = np.tile(np.array([[0.707, 0.707], [0.707, -0.707], [-0.707, -0.707], [-0.707, 0.707]], dtype=np.float32), (50, 1))
        # 2. 100 个交替符号让 Costas 环彻底锁死相位
        wu_1d = np.array([1.0, -1.0] * 50, dtype=np.float32)
        self.warm_up = np.column_stack((wu_1d, wu_1d)) * 0.707
        
        b13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=np.float32)
        self.preamble = np.tile(np.column_stack((b13, b13)) / np.sqrt(2), (4, 1))
        s_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1], dtype=np.uint8)
        self.sync_word = np.column_stack((np.where(s_bits==1, 0.707, -0.707), np.where(s_bits==1, 0.707, -0.707))).astype(np.float32)
        
        self.rx_template = np.conj(np.tile((b13 + 1j * b13) / np.sqrt(2), 4)[::-1])
        self.rx_sync_ideal = np.conj(np.where(s_bits==1, 0.707+0.707j, -0.707-0.707j).astype(np.complex64)[::-1])

    def generate_packet(self, payload_len):
        header = struct.pack('!HHHHI', 43690, self.my_msg_type, 0, 1, payload_len)
        bits = np.repeat(np.unpackbits(np.frombuffer(header, dtype=np.uint8)), 8)
        mac = np.column_stack((np.where(bits==0, -0.707, 0.707), np.where(bits==0, -0.707, 0.707))).astype(np.float32)
        payload = np.ones((payload_len, 2), dtype=np.float32) * 0.707 
        return np.vstack((self.agc_pad, self.warm_up, self.preamble, self.sync_word, mac, payload))

    def send_data(self, frame, chunks=1):
        for c in range(chunks):
            for i in range(0, len(frame), 4000):
                self.sock_tx.sendto(frame[i:i+4000].tobytes(), ("127.0.0.1", self.tx_port))
                time.sleep(0.005)
            time.sleep(0.1)

    def receive_data(self, timeout=10.0):
        # 【核心：延后洗胃】确保硬件毛刺已经全部掉进 UDP 缓冲后，再一次性抽干！
        self.sock_rx.setblocking(False)
        flush_count = 0
        while True:
            try: 
                self.sock_rx.recv(65536)
                flush_count += 1
            except: break
        self.sock_rx.setblocking(True)
        self.sock_rx.settimeout(1.0) 
        if flush_count > 0:
            print(f"  [系统] 成功清洗了 {flush_count} 块硬件切换残留垃圾。")
        
        buf = np.zeros((0, 2), dtype=np.float32)
        state = 0
        start_t = time.time()
        last_print_t = time.time()
        
        while time.time() - start_t < timeout:
            try:
                data, _ = self.sock_rx.recvfrom(65536)
                buf = np.vstack((buf, np.frombuffer(data, dtype=np.float32).reshape(-1, 2)))
            except socket.timeout: 
                continue
            
            while True:
                if state == 0 and len(buf) >= 400:
                    search_len = min(len(buf), 1500)
                    cplx = buf[:search_len, 0] + 1j * buf[:search_len, 1]
                    corr = np.abs(np.convolve(cplx, self.rx_template, mode='valid'))
                    if len(corr) > 0:
                        max_c = np.max(corr)
                        mean_c = np.mean(corr)
                        
                        # 【全天候雷达】：每隔 0.5 秒强行汇报一次当前看到的最高相关峰
                        if time.time() - last_print_t > 0.5:
                            print(f"  [扫描中...] 当前视野最高峰值: {max_c:.1f} (门限>6.0)")
                            last_print_t = time.time()

                        if max_c > 2.0 * mean_c and max_c > 6.0:
                            print(f"\n  [雷达] 🎯 锁定真实数据包! (峰值: {max_c:.1f}, 均值: {mean_c:.1f})")
                            idx = np.argmax(corr) + 52
                            if idx < len(buf):
                                buf = buf[idx:]
                                state = 1
                            else: buf = np.zeros((0,2), dtype=np.float32)
                        else:
                            buf = buf[len(corr):]
                    else: break
                
                elif state == 1 and len(buf) >= 800:
                    search_cplx = buf[:30, 0] + 1j * buf[:30, 1]
                    best_idx, max_corr, best_rot = -1, 0.0, 1.0+0j
                    for rot in [1+0j, 0+1j, -1+0j, 0-1j]:
                        corr = np.real(np.convolve(search_cplx * rot, self.rx_sync_ideal, mode='valid'))
                        if np.max(corr) > max_corr: max_corr, best_idx, best_rot = np.max(corr), np.argmax(corr), rot
                    
                    if max_corr > 4.0 and best_idx < 12:
                        if best_idx + 16 + 768 > len(buf): break
                            
                        hdr_iq = buf[best_idx+16 : best_idx+16+768]
                        hdr_cplx = (hdr_iq[:, 0] + 1j * hdr_iq[:, 1]) * best_rot
                        buf = buf[best_idx+16+768:]
                        
                        I, Q = np.real(hdr_cplx), np.imag(hdr_cplx)
                        amp = np.mean(np.abs(I) + np.abs(Q)) / 1.414 
                        votes = np.sum((I + Q).reshape(96, 8), axis=1)
                        hdr_bytes = np.packbits(np.where(votes > 0, 1, 0).astype(np.uint8)).tobytes()
                        
                        try:
                            magic, msg_type, _, _, _ = struct.unpack('!HHHHI', hdr_bytes)
                            if magic == 43690 and msg_type == self.expect_msg_type:
                                snr = 10 * np.log10( (2*amp**2) / max(np.mean((I - np.sign(I)*amp)**2 + (Q - np.sign(Q)*amp)**2), 1e-6) )
                                return True, snr
                            else:
                                print(f"  [雷达] ⚠️ MAC 校验失败。")
                        except: pass
                        state = 0
                    else:
                        print(f"  [雷达] ❌ 帧头匹配失败。")
                        state = 0
                        buf = buf[50:]
                else:
                    break 
                    
        return False, 0.0

def run_master():
    node = TDDSlowNode('master')
    for i in range(10):
        print(f"\n================ [Round {i+1}] ================")
        print("[Master] 🚀 发射图片载荷...")
        node.send_data(node.generate_packet(2048), chunks=10)
            
        print("[Master] ⏳ 强制物理避让 2 秒 (等待电涌消散)...")
        time.sleep(2.0)
        
        print("[Master] 🎧 开始监听 ACK...")
        success, snr = node.receive_data(timeout=10.0)
        if success: print(f"\033[92m[Master] ✅ ACK 成功接收！上行 SNR = {snr:.2f} dB\033[0m")
        else: print("\033[91m[Master] ❌ ACK 接收失败！\033[0m")
        time.sleep(1.0)

def run_slave():
    node = TDDSlowNode('slave')
    print("Slave 监听中...")
    while True:
        success, snr = node.receive_data(timeout=100.0)
        if success:
            print(f"\033[92m[Slave] 📥 载荷成功接收！下行 SNR = {snr:.2f} dB\033[0m")
            print("[Slave] ⏳ 强制物理避让 2 秒 (等待 Master 切回 RX)...")
            time.sleep(2.0) 
            print("[Slave] 📤 发射短 ACK...")
            node.send_data(node.generate_packet(64), chunks=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=['master', 'slave'])
    args = parser.parse_args()
    if args.mode == 'master': run_master()
    else: run_slave()