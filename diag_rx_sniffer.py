import socket
import numpy as np
import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True, help="监听的 UDP 端口 (Node A=8004, Node B=8002)")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", args.port))
    sock.settimeout(0.5)

    print(f"\n[RX Sniffer] 🎧 正在监听端口 {args.port}...")
    print("[RX Sniffer] 只要收到哪怕一丝能量波动，就会打印出来！\n")

    idle_count = 0
    while True:
        try:
            data, _ = sock.recvfrom(65536)
            arr = np.frombuffer(data, dtype=np.float32)
            if len(arr) == 0: continue
            
            # 计算这一块数据的平均能量和峰值
            mean_amp = np.mean(np.abs(arr))
            max_amp = np.max(np.abs(arr))
            
            # 只有当能量超过静默底噪时才打印，过滤掉全 0 的无用输出
            if mean_amp > 0.05:
                print(f"[{time.strftime('%H:%M:%S')}] ⚡ 发现信号波峰！| 长度: {len(arr):<5} | 平均能量: {mean_amp:.3f} | 峰值能量: {max_amp:.3f}")
                idle_count = 0
                
        except socket.timeout:
            idle_count += 1
            if idle_count % 4 == 0:  # 每 2 秒打印一次存活心跳
                print(f"[{time.strftime('%H:%M:%S')}] 💤 监听中，当前信道一片死寂...")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()