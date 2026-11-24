import socket
import numpy as np
import json
import time

def main():
    # 读取配置
    with open('config.json', 'r', encoding='utf-8') as f: CONFIG = json.load(f)
    sdr_cfg = CONFIG.get("sdr_platform", {})
    tx_proxy_ip = sdr_cfg.get("downlink_tx_proxy", {}).get("ip", "127.0.0.1")
    tx_proxy_port = sdr_cfg.get("downlink_tx_proxy", {}).get("port", 8001)

    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024 * 8)
    
    # 1. 构造全新的 AGC 保护垫 (1/4 采样率正弦波)
    tone_cplx = np.array([[0.707, 0.707], [0.707, -0.707], [-0.707, -0.707], [-0.707, 0.707]], dtype=np.float32)
    agc_pad = np.tile(tone_cplx, (40, 1)) # 160 个符号
    
    # 2. 构造 Barker 码
    barker_13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=np.float32)
    preamble = np.column_stack((barker_13, barker_13)) / np.sqrt(2)
    preamble = np.tile(preamble, (4, 1))
    
    # 3. 构造非周期的黄金同步字 (Sync Word)
    sync_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1], dtype=np.uint8)
    sync_syms = np.where(sync_bits == 1, 0.707, -0.707).astype(np.float32)
    sync_word = np.column_stack((sync_syms, sync_syms))
    
    # 4. 构造随机 4QAM 载荷 (4000 个符号，足够画星座图了)
    NUM_SYMBOLS = 4000
    bits = np.random.randint(0, 2, NUM_SYMBOLS * 2)
    syms = np.where(bits == 0, -1.0, 1.0).astype(np.float32) * 0.7071
    payload_iq = syms.reshape(-1, 2)
    
    # 拼装测试帧
    frame = np.vstack((agc_pad, preamble, sync_word, payload_iq)).astype(np.float32)
    
    print(f"\033[96m[PHY 诊断 TX] 正在向 USRP ({tx_proxy_ip}:{tx_proxy_port}) 持续发射测试物理帧...\033[0m")
    try:
        frame_counter = 0
        while True:
            CHUNK_SIZE = 4000 
            for i in range(0, len(frame), CHUNK_SIZE):
                udp_sock.sendto(frame[i:i+CHUNK_SIZE].tobytes(), (tx_proxy_ip, tx_proxy_port))
                time.sleep(0.002) 
                
            frame_counter += 1
            print(f">>> 发送探测帧 #{frame_counter}")
            time.sleep(0.5) # 控制发送频率为 2fps，让接收端画图跟得上
            
    except KeyboardInterrupt: 
        print("\n测试停止。")
    finally:
        udp_sock.close()

if __name__ == "__main__":
    main()