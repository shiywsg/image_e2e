import socket
import numpy as np
import json
import time
import matplotlib.pyplot as plt

def main():
    # 读取配置
    with open('config.json', 'r', encoding='utf-8') as f: CONFIG = json.load(f)
    sdr_cfg = CONFIG.get("sdr_platform", {})
    rx_proxy_ip = sdr_cfg.get("downlink_rx_proxy", {}).get("ip", "127.0.0.1")
    rx_proxy_port = sdr_cfg.get("downlink_rx_proxy", {}).get("port", 8002)

    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024 * 8)
    udp_sock.bind((rx_proxy_ip, rx_proxy_port))
    udp_sock.settimeout(0.5)
    
    # 构建理想模板
    barker_13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=np.float32)
    barker_cplx = (barker_13 + 1j * barker_13) / np.sqrt(2)
    template = np.conj(np.tile(barker_cplx, 4)[::-1])
    
    sync_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1], dtype=np.uint8)
    sync_cplx_ideal = np.where(sync_bits == 1, 0.707+0.707j, -0.707-0.707j).astype(np.complex64)
    sync_template_cplx = np.conj(sync_cplx_ideal[::-1])

    # === Matplotlib 实时画图初始化 ===
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.canvas.manager.set_window_title('物理信道实时 X 光机')
    
    sc1 = ax1.scatter([], [], s=5, alpha=0.5, color='blue')
    ax1.set_xlim(-1.8, 1.8); ax1.set_ylim(-1.8, 1.8)
    ax1.set_title("1. 原始接收 (未解频偏)")
    ax1.grid(True); ax1.axhline(0, color='black', lw=0.5); ax1.axvline(0, color='black', lw=0.5)
    
    sc2 = ax2.scatter([], [], s=5, alpha=0.5, color='green')
    ax2.set_xlim(-1.8, 1.8); ax2.set_ylim(-1.8, 1.8)
    ax2.set_title("2. 补偿后 (评估信噪比)")
    ax2.grid(True); ax2.axhline(0, color='black', lw=0.5); ax2.axvline(0, color='black', lw=0.5)
    
    text_info = fig.text(0.5, 0.05, '等待捕获信号...', ha='center', fontsize=12)
    plt.show(block=False)

    stream_buffer = np.zeros((0, 2), dtype=np.float32)
    state = 0
    payload_len = 4000
    
    print("\033[96m[PHY 诊断 RX] 正在监听 SDR 信道，星座图视窗已开启...\033[0m")
    
    try:
        while True:
            try:
                data, _ = udp_sock.recvfrom(65536)
                new_pairs = np.frombuffer(data, dtype=np.float32).reshape(-1, 2)
                stream_buffer = np.vstack((stream_buffer, new_pairs))
            except socket.timeout:
                fig.canvas.flush_events()
                continue
            
            if len(stream_buffer) > 200000:
                stream_buffer = stream_buffer[-50000:]
            
            if state == 0 and len(stream_buffer) >= 200:
                cplx_buf = stream_buffer[:, 0] + 1j * stream_buffer[:, 1]
                corr = np.abs(np.convolve(cplx_buf, template, mode='valid'))
                if len(corr) > 0:
                    max_c = np.max(corr)
                    mean_c = np.mean(corr)
                    
                    # 采用降低后的门限
                    if max_c > 3.2 * mean_c and max_c > 15.0:
                        start_idx = np.argmax(corr) + 52
                        stream_buffer = stream_buffer[start_idx:]
                        state = 1
                    else:
                        stream_buffer = stream_buffer[-100:]
                    
            elif state == 1 and len(stream_buffer) >= 100:
                search_cplx = stream_buffer[:30, 0] + 1j * stream_buffer[:30, 1]
                best_idx, max_corr, best_rot = -1, 0.0, 1.0+0j
                for rot in [1+0j, 0+1j, -1+0j, 0-1j]:
                    corr = np.real(np.convolve(search_cplx * rot, sync_template_cplx, mode='valid'))
                    if np.max(corr) > max_corr:
                        max_corr = np.max(corr)
                        best_idx = np.argmax(corr)
                        best_rot = rot
                
                if max_corr > 5.0 and best_idx < 10:
                    stream_buffer = stream_buffer[best_idx + 16:]
                    state = 2
                else:
                    state = 0
                    stream_buffer = stream_buffer[50:]
                    
            elif state == 2 and len(stream_buffer) >= payload_len:
                payload_iq = stream_buffer[:payload_len]
                stream_buffer = stream_buffer[payload_len:]
                state = 0
                
                # 原始接收 IQ (仅同步相位)
                rx_cplx_raw = (payload_iq[:, 0] + 1j * payload_iq[:, 1]) * best_rot
                
                # CFO (载波频偏) 补偿运算
                rx_4th = rx_cplx_raw ** 4
                fft_val = np.fft.fft(rx_4th)
                peak_idx = np.argmax(np.abs(fft_val))
                N = len(rx_cplx_raw)
                if peak_idx > N / 2: peak_idx -= N
                cfo_normalized = peak_idx / (4.0 * N)
                
                t = np.arange(N)
                rx_cplx_comp = rx_cplx_raw * np.exp(-1j * 2 * np.pi * cfo_normalized * t)
                
                # AGC 幅度归一化 (让画面好看)
                amp = np.mean(np.abs(rx_cplx_comp)) / 0.7071
                if amp > 0.01:
                    rx_cplx_raw = rx_cplx_raw / amp
                    rx_cplx_comp = rx_cplx_comp / amp

                # --- 更新界面 ---
                sc1.set_offsets(np.column_stack((np.real(rx_cplx_raw), np.imag(rx_cplx_raw))))
                sc2.set_offsets(np.column_stack((np.real(rx_cplx_comp), np.imag(rx_cplx_comp))))
                
                info_str = f"捕获成功! Barker峰值正常 | CFO估计: {cfo_normalized*1e6:.1f} PPM"
                text_info.set_text(info_str)
                print(f"[UI] 更新帧数据 - {info_str}")
                
                fig.canvas.draw()
                fig.canvas.flush_events()

    except KeyboardInterrupt:
        print("\n\033[93m[测试中止]\033[0m")
    finally:
        udp_sock.close()
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()