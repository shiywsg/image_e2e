import socket
import numpy as np
import json
import sys

def calculate_metrics(rx_cplx, tx_cplx):
    error_vector = rx_cplx - tx_cplx
    evm_rms = np.sqrt(np.mean(np.abs(error_vector)**2) / np.mean(np.abs(tx_cplx)**2)) * 100
    
    rx_hard_i = np.where(np.real(rx_cplx) > 0, 1.0, -1.0) * 0.7071
    rx_hard_q = np.where(np.imag(rx_cplx) > 0, 1.0, -1.0) * 0.7071
    rx_hard = rx_hard_i + 1j * rx_hard_q
    
    tx_hard_i = np.where(np.real(tx_cplx) > 0, 1.0, -1.0) * 0.7071
    tx_hard_q = np.where(np.imag(tx_cplx) > 0, 1.0, -1.0) * 0.7071
    tx_hard = tx_hard_i + 1j * tx_hard_q
    
    errors = np.sum(rx_hard != tx_hard)
    ser = (errors / len(tx_cplx)) * 100
    return evm_rms, ser

def print_statistics(metrics):
    print("\n" + "="*55)
    print("\033[93m📊 [EVM 单元测试 - 最终统计报告 (抗假唤醒优化版)]\033[0m")
    
    total_rounds = len(metrics['evm'])
    if total_rounds == 0:
        print("没有收集到足够的数据。")
        print("="*55 + "\n")
        return
        
    evm_arr = np.array(metrics['evm'])
    ser_arr = np.array(metrics['ser'])
    cfo_arr = np.array(metrics['cfo'])
    
    outlier_threshold_ser = 10.0 
    outliers = np.sum(ser_arr > outlier_threshold_ser)
    outlier_rate = (outliers / total_rounds) * 100

    print(f"总计评估有效帧数: {total_rounds}")
    print(f"突发严重失误次数: {outliers} (占比: {outlier_rate:.2f}%)  <-- 评估假唤醒控制效果")
    print("-" * 55)
    
    print("\033[96m[EVM (RMS) 统计]\033[0m")
    print(f"  平均值 (Mean)  : {np.mean(evm_arr):6.2f} %")
    print(f"  中位数 (Median): {np.median(evm_arr):6.2f} %")
    print(f"  标准差 (StdDev): {np.std(evm_arr):6.2f} %")
    print(f"  最小值 (Min)   : {np.min(evm_arr):6.2f} %")
    print(f"  最大值 (Max)   : {np.max(evm_arr):6.2f} %")
    print("-" * 55)
    
    print("\033[96m[SER (符号误码率) 统计]\033[0m")
    print(f"  平均值 (Mean)  : {np.mean(ser_arr):6.4f} %")
    print(f"  中位数 (Median): {np.median(ser_arr):6.4f} %")
    print("-" * 55)
    
    print("\033[96m[CFO (载波频偏) 统计]\033[0m")
    print(f"  平均绝对频偏   : {np.mean(np.abs(cfo_arr)):6.2f} PPM")
    print("="*55 + "\n")

def main():
    with open('config.json', 'r', encoding='utf-8') as f: CONFIG = json.load(f)
    sdr_cfg = CONFIG.get("sdr_platform", {})
    rx_host_port = sdr_cfg.get("feedback_sim_channel", {}).get("rx_host_port", 9999)
    rx_proxy_ip = sdr_cfg.get("downlink_rx_proxy", {}).get("ip", "127.0.0.1")
    rx_proxy_port = sdr_cfg.get("downlink_rx_proxy", {}).get("port", 8004)

    tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    tcp_sock.bind(('0.0.0.0', rx_host_port))
    tcp_sock.listen(1)
    print(f"\033[96m[EVM 测试] 等待发送端通过 TCP 传输基准序列...\033[0m")
    conn, addr = tcp_sock.accept()
    
    raw_data = b""
    while True:
        chunk = conn.recv(65536)
        if not chunk: break
        raw_data += chunk
    truth_iq = np.frombuffer(raw_data, dtype=np.float32).reshape(-1, 2)
    truth_cplx = truth_iq[:, 0] + 1j * truth_iq[:, 1]
    expected_len = len(truth_cplx)
    print(f"\033[92m[EVM 测试] 收到基准序列，长度: {expected_len} 符号\033[0m")
    conn.close(); tcp_sock.close()

    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024 * 8)
    udp_sock.bind((rx_proxy_ip, rx_proxy_port))
    
    barker = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=np.float32)
    barker_cplx = (barker + 1j * barker) / np.sqrt(2)
    template = np.conj(np.tile(barker_cplx, 4)[::-1])
    
    sync_bits = np.array([1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0], dtype=np.uint8)
    sync_cplx_ideal = np.where(sync_bits == 1, 0.707+0.707j, -0.707-0.707j).astype(np.complex64)
    sync_template_cplx = np.conj(sync_cplx_ideal[::-1])

    stream_buffer = np.zeros((0, 2), dtype=np.float32)
    state = 0
    metrics_history = {'evm': [], 'ser': [], 'cfo': []}
    
    print("\033[96m[EVM 测试] 正在监听 SDR 物理信道，按 [Ctrl+C] 结束测试并打印统计报告...\033[0m")
    
    try:
        while True:
            data, _ = udp_sock.recvfrom(65536)
            new_pairs = np.frombuffer(data, dtype=np.float32).reshape(-1, 2)
            stream_buffer = np.vstack((stream_buffer, new_pairs))
            
            # 缓冲区防爆保护：如果积累的垃圾数据过多（超过 20 万个点还没同步上），主动砍掉前一半防内存溢出
            if len(stream_buffer) > 300000:
                stream_buffer = stream_buffer[-100000:]
            
            if state == 0 and len(stream_buffer) >= 200:
                cplx_buf = stream_buffer[:, 0] + 1j * stream_buffer[:, 1]
                corr = np.abs(np.convolve(cplx_buf, template, mode='valid'))
                
                # 【硬化 1】：将 Barker 码触发门限从 5.0 提高到 7.0，严格抑制噪声假唤醒
                if len(corr) > 0 and np.max(corr) > 7.0 * np.mean(corr) and np.max(corr) > 3.0:
                    start_idx = np.argmax(corr) + 52
                    stream_buffer = stream_buffer[start_idx:]
                    state = 1
                else:
                    stream_buffer = stream_buffer[-100:]
                    
            elif state == 1 and len(stream_buffer) >= 100:
                search_cplx = stream_buffer[:100, 0] + 1j * stream_buffer[:100, 1]
                best_idx, max_corr, best_rot = -1, 0.0, 1.0+0j
                for rot in [1+0j, 0+1j, -1+0j, 0-1j]:
                    corr = np.real(np.convolve(search_cplx * rot, sync_template_cplx, mode='valid'))
                    if np.max(corr) > max_corr:
                        max_corr = np.max(corr)
                        best_idx = np.argmax(corr)
                        best_rot = rot
                
                # 【硬化 2】：提高 0xAAAA 同步字的确认门限从 4.5 到 5.5
                # 如果相关峰小于 5.5，说明刚才的 Barker 码是假唤醒，立刻打回原形重新找！
                if max_corr > 5.5:
                    stream_buffer = stream_buffer[best_idx + 16:]
                    state = 2
                else:
                    state = 0
                    stream_buffer = stream_buffer[50:] # 丢弃一小段噪声，继续往下找
                    
            elif state == 2 and len(stream_buffer) >= expected_len:
                payload_iq = stream_buffer[:expected_len]
                rx_cplx = (payload_iq[:, 0] + 1j * payload_iq[:, 1]) * best_rot
                
                valid_len = min(2000, expected_len)
                amp_gain = np.mean(np.abs(truth_cplx[:valid_len])) / (np.mean(np.abs(rx_cplx[:valid_len])) + 1e-6)
                rx_cplx = rx_cplx * amp_gain
                
                rx_4th = rx_cplx ** 4
                fft_val = np.fft.fft(rx_4th)
                peak_idx = np.argmax(np.abs(fft_val))
                N = len(rx_cplx)
                if peak_idx > N / 2: peak_idx -= N
                cfo_normalized = peak_idx / (4.0 * N)

                # 【修改处】：增加 CFO 硬限幅过滤
                # 如果算出离谱值(大于 50 PPM)，说明是噪声导致或相位卷绕，强制归零
                if abs(cfo_normalized * 1e6) > 50.0:
                    cfo_normalized = 0.0
                
                t = np.arange(N)
                rx_cplx = rx_cplx * np.exp(-1j * 2 * np.pi * cfo_normalized * t)
                
                evm_rms, ser = calculate_metrics(rx_cplx, truth_cplx)
                
                metrics_history['evm'].append(evm_rms)
                metrics_history['ser'].append(ser)
                metrics_history['cfo'].append(cfo_normalized * 1e6)
                
                round_num = len(metrics_history['evm'])
                print(f"[Round {round_num:03d}] EVM: {evm_rms:6.2f}% | SER: {ser:6.4f}%")
                
                stream_buffer = stream_buffer[expected_len:]
                state = 0

    except KeyboardInterrupt:
        print("\n\033[93m[测试中止] 正在生成统计报告...\033[0m")
    finally:
        udp_sock.close()
        print_statistics(metrics_history)

if __name__ == "__main__":
    main()