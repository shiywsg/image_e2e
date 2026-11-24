import socket
import numpy as np
import sys

RX_LISTEN_PORT = 8002
RX_LISTEN_IP = "127.0.0.1"

EXPECTED_FRAME_LEN = 4000 

# 构建用于滑动互相关的本地 Barker 模板
barker_13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=np.float32)
barker_complex = (barker_13 + 1j * barker_13) / np.sqrt(2)
sync_template = np.tile(barker_complex, 4) # 52 符号
template_conj_rev = np.conj(sync_template[::-1])

def run_logic_receiver():
    sock_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_rx.bind((RX_LISTEN_IP, RX_LISTEN_PORT))
    
    print("📡 [AI_RX] 逻辑接收端已启动，应用层帧同步(Barker)与重组机制启用...")
    
    stream_buffer = np.zeros(0, dtype=np.complex64)
    state = 0  # 状态机：0 = 寻找包头，1 = 提取定长载荷
    
    try:
        while True:
            data, _ = sock_rx.recvfrom(65536)
            received_pairs = np.frombuffer(data, dtype=np.float32).reshape(-1, 2)
            # 在 AI 端重新合成为复数，便于做互相关和能量计算
            received_cplx = received_pairs[:, 0] + 1j * received_pairs[:, 1]
            
            stream_buffer = np.concatenate((stream_buffer, received_cplx))
            
            if state == 0:
                # 【状态 0】：滑动寻找 Barker 码
                # 缓冲区累积到一定长度再找，避免频繁计算
                if len(stream_buffer) >= 200:
                    corr = np.abs(np.convolve(stream_buffer, template_conj_rev, mode='valid'))
                    max_corr = np.max(corr)
                    
                    # 理论峰值是 52，这里设置 25 作为检测门限
                    if max_corr > 25.0:
                        max_idx = np.argmax(corr)
                        # 包头找到了！有效载荷从包头后面开始
                        payload_start = max_idx + 52
                        
                        print(f"\n[同步] 🎯 捕获帧头！峰值: {max_corr:.1f}，进入重组模式...")
                        # 丢弃包头之前的数据
                        stream_buffer = stream_buffer[payload_start:]
                        state = 1  # 切换状态
                    else:
                        # 没找到，说明当前全是底噪或失锁残骸。
                        # 保留最后 100 个点（防止下一个包头刚好在这个边界上被切断），其余丢掉
                        stream_buffer = stream_buffer[-100:]
                        sys.stdout.write(".")
                        sys.stdout.flush()
                        
            elif state == 1:
                # 【状态 1】：提取 4000 个符号的载荷
                if len(stream_buffer) >= EXPECTED_FRAME_LEN:
                    frame_data = stream_buffer[:EXPECTED_FRAME_LEN]
                    # 从缓冲区移除已提取的数据
                    stream_buffer = stream_buffer[EXPECTED_FRAME_LEN:]
                    
                    I = np.real(frame_data)
                    Q = np.imag(frame_data)
                    power = np.mean(I**2 + Q**2)
                    
                    bar_len = int(power * 20)
                    bar = "█" * min(bar_len, 40)
                    
                    print(f"✨ [激活] 完美重组定长包！ (P={power:.2f}) {bar}")
                    
                    q1 = np.sum((I > 0) & (Q > 0))
                    q2 = np.sum((I < 0) & (Q > 0))
                    q3 = np.sum((I < 0) & (Q < 0))
                    q4 = np.sum((I > 0) & (Q < 0))
                    
                    print(f"    解析本包 {len(frame_data)} 个符号的象限分布:")
                    print(f"    Q1:{q1}  Q2:{q2}  Q3:{q3}  Q4:{q4}")
                    print("-" * 50)
                    
                    # 提取完一包，立刻回到寻找下一包头状态
                    state = 0
                    
    except KeyboardInterrupt:
        print("\n[AI_RX] 接收测试结束。")
    finally:
        sock_rx.close()

if __name__ == '__main__':
    run_logic_receiver()