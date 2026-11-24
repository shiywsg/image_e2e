import socket
import numpy as np
import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True, help="发送的 UDP 端口 (Node A=8001, Node B=8003)")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    print(f"\n[TX Pulser] 🚀 准备向端口 {args.port} 发射射频脉冲...")

    # 生成极其粗暴的满功率随机 BPSK 信号
    np.random.seed(42)
    raw_payload = np.random.choice([-0.707, 0.707], size=(10000, 2)).astype(np.float32)

    try:
        count = 1
        while True:
            print(f"[{time.strftime('%H:%M:%S')}] 💥 正在发射第 {count} 发测试脉冲...")
            # 切碎成 UDP 块发送，模拟真实发送流程
            for i in range(0, len(raw_payload), 4000):
                sock.sendto(raw_payload[i:i+4000].tobytes(), ("127.0.0.1", args.port))
                time.sleep(0.002)
                
            count += 1
            time.sleep(2.0)  # 每 2 秒发射一次
            
    except KeyboardInterrupt:
        print("\n[TX Pulser] 停止发射。")

if __name__ == "__main__":
    main()