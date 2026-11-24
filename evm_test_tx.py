import socket
import numpy as np
import json
import time

def main():
    with open('config.json', 'r', encoding='utf-8') as f: CONFIG = json.load(f)
    sdr_cfg = CONFIG.get("sdr_platform", {})
    rx_host_ip = sdr_cfg.get("feedback_sim_channel", {}).get("rx_host_ip", "127.0.0.1")
    rx_host_port = sdr_cfg.get("feedback_sim_channel", {}).get("rx_host_port", 9999)
    tx_proxy_ip = sdr_cfg.get("downlink_tx_proxy", {}).get("ip", "127.0.0.1")
    tx_proxy_port = sdr_cfg.get("downlink_tx_proxy", {}).get("port", 8001)

    NUM_SYMBOLS = 40000
    bits = np.random.randint(0, 2, NUM_SYMBOLS * 2)
    syms = np.where(bits == 0, -1.0, 1.0).astype(np.float32) * 0.7071
    truth_iq = syms.reshape(-1, 2)
    
    tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_sock.connect((rx_host_ip, rx_host_port))
    tcp_sock.sendall(truth_iq.tobytes())
    tcp_sock.close()
    time.sleep(1.0)

    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # 【修复 4】：也扩大发送端的 UDP 缓冲
    udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024 * 8)
    
    barker = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=np.float32)
    preamble = np.column_stack((barker, barker)) / np.sqrt(2)
    preamble = np.tile(preamble, (4, 1))
    
    sync_bits = np.array([1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0], dtype=np.uint8)
    sync_syms = np.where(sync_bits == 1, 0.707, -0.707).astype(np.float32)
    sync_word = np.column_stack((sync_syms, sync_syms))
    
    frame = np.vstack((preamble, sync_word, truth_iq)).astype(np.float32)
    
    print("\033[96m[EVM 测试] 开始通过 SDR 代理循环发送物理波形...\033[0m")
    try:
        while True:
            # 【修复 5】：平滑发送，减小分块体积并增加微弱延时，避免瞬间打爆系统缓冲
            CHUNK_SIZE = 4000 
            for i in range(0, len(frame), CHUNK_SIZE):
                udp_sock.sendto(frame[i:i+CHUNK_SIZE].tobytes(), (tx_proxy_ip, tx_proxy_port))
                time.sleep(0.005) # 极短睡眠，保障流水线稳定
            print(">>> 发送一轮测试脉冲")
            time.sleep(2.0)
    except KeyboardInterrupt: print("\n测试停止。")

if __name__ == "__main__":
    main()