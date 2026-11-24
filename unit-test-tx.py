import socket
import time
import numpy as np

TX_PROXY_PORT = 8001
TX_PROXY_IP = "127.0.0.1"

def run_logic_sender():
    sock_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print("🚀 [AI_TX] 逻辑发送端已启动。")
    print(f"目标代理: {TX_PROXY_IP}:{TX_PROXY_PORT}")
    
    # 4QAM 理想星座点库
    qam_points = np.array([
        [ 1.0,  1.0],  # 第一象限 (I>0, Q>0)
        [-1.0,  1.0],  # 第二象限 (I<0, Q>0)
        [-1.0, -1.0],  # 第三象限 (I<0, Q<0)
        [ 1.0, -1.0]   # 第四象限 (I>0, Q<0)
    ], dtype=np.float32)
    
    burst_count = 1
    try:
        while True:
            # 随机生成 4000 个 4QAM 符号
            random_indices = np.random.randint(0, 4, size=4000)
            payload = qam_points[random_indices]
            
            # 统计本次发送四个象限的数量
            counts = [np.sum(random_indices == i) for i in range(4)]
            
            print(f"\n>>> [{time.strftime('%H:%M:%S')}] 正在发送突发数据包 #{burst_count}")
            print(f"    内容: 共 4000 符号。随机分布: Q1:{counts[0]} Q2:{counts[1]} Q3:{counts[2]} Q4:{counts[3]}")
            
            sock_tx.sendto(payload.tobytes(), (TX_PROXY_IP, TX_PROXY_PORT))
            burst_count += 1
            
            # 发完休息 3 秒
            time.sleep(3)
    except KeyboardInterrupt:
        print("\n[AI_TX] 测试结束。")
    finally:
        sock_tx.close()

if __name__ == '__main__':
    run_logic_sender()