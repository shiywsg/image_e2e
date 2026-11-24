import socket
import numpy as np
import uhd
import numba as nb
import threading
import queue

AI_LOGIC_IP = "127.0.0.1"
AI_LOGIC_PORT = 8002
USRP_IP = "192.168.10.2"
RF_PORT = "TX/RX"
FREQ = 2.4e9
RATE = 500e3  
SPS = 4

@nb.njit
def run_costas_loop(iq_cfo, init_phase, init_freq, alpha, beta):
    N = len(iq_cfo)
    out_iq = np.zeros(N, dtype=np.complex64)
    costas_phase = init_phase
    costas_freq = init_freq
    for i in range(N):
        c_p, s_p = np.cos(-costas_phase), np.sin(-costas_phase)
        sample_real = iq_cfo[i].real * c_p - iq_cfo[i].imag * s_p
        sample_imag = iq_cfo[i].real * s_p + iq_cfo[i].imag * c_p
        sample = sample_real + 1j * sample_imag
        out_iq[i] = sample
        I, Q = sample_real, sample_imag
        
        # 仅在幅度足够大时更新频率估计，防止在纯噪声中迷失
        if abs(I) + abs(Q) > 0.1:
            error = np.sign(I)*Q - np.sign(Q)*I
            costas_freq += beta * error
            
            # 限制物理频偏的容忍度 (归一化界限)
            if costas_freq > 0.05: costas_freq = 0.05
            elif costas_freq < -0.05: costas_freq = -0.05
            
            costas_phase += costas_freq + alpha * error
            
            # 相位卷绕保护
            if costas_phase > np.pi: costas_phase -= 2*np.pi
            elif costas_phase < -np.pi: costas_phase += 2*np.pi
            
    return out_iq, costas_phase, costas_freq

print(f"[RX Proxy] 正在初始化 USRP...")
usrp = uhd.usrp.MultiUSRP(f"addr={USRP_IP},recv_frame_size=1472")
usrp.set_rx_rate(RATE); usrp.set_rx_freq(uhd.types.TuneRequest(FREQ))
usrp.set_rx_antenna(RF_PORT, 0); usrp.set_rx_gain(20)
st_args = uhd.usrp.StreamArgs("fc32", "sc16")
st_args.channels = [0]
rx_streamer = usrp.get_rx_stream(st_args)
metadata = uhd.types.RXMetadata()

# 预热 JIT
_, _, _ = run_costas_loop(np.array([1+1j, 1-1j], dtype=np.complex64), 0.0, 0.0, 0.1, 0.005)
iq_queue = queue.Queue(maxsize=200)

def usrp_receiver():
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    stream_cmd.stream_now = True
    rx_streamer.issue_stream_cmd(stream_cmd)
    recv_buffer = np.zeros((1, 10000 * SPS), dtype=np.complex64)
    while True:
        num_samps = rx_streamer.recv(recv_buffer, metadata)
        if num_samps > 0 and not iq_queue.full():
            iq_queue.put(recv_buffer[0][:num_samps].copy())

def dsp_processor():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024 * 8)
    costas_phase, costas_freq = 0.0, 0.0
    global_amp = 1.0 
    
    while True:
        iq_raw = iq_queue.get()
        iq_len = (len(iq_raw) // SPS) * SPS
        if iq_len == 0: continue
        
        # 简单固定降采样 (后续若要解决相关峰递减问题，需升级此处为Gardner环)
        iq_symbols = np.mean(iq_raw[:iq_len].reshape(-1, SPS)[:, 1:3], axis=1)
        iq_symbols = iq_symbols - np.mean(iq_symbols)
        chunk_amp = np.mean(np.abs(iq_symbols))
        
        if chunk_amp < 0.01:
            # 【核心修改】：静默期可以重置相位，但【绝对不能重置频率(costas_freq)】！
            # 必须让它记住 USRP 的物理频偏，下一个包一来立刻无缝对接
            costas_phase = 0.0
            out_pairs = np.zeros((len(iq_symbols), 2), dtype=np.float32)
        else:
            global_amp = 0.8 * global_amp + 0.2 * chunk_amp
            iq_symbols = iq_symbols / (global_amp + 1e-6)
            
            # alpha 减半至 0.015，beta 减小至 0.0001，进一步过滤相位噪声
            out_iq, costas_phase, costas_freq = run_costas_loop(iq_symbols, costas_phase, costas_freq, 0.015, 0.0001)
            
            out_pairs = np.column_stack((np.real(out_iq), np.imag(out_iq))).astype(np.float32)
        
        for i in range(0, len(out_pairs), 4000):
            sock.sendto(out_pairs[i : i + 4000].tobytes(), (AI_LOGIC_IP, AI_LOGIC_PORT))

t_rx = threading.Thread(target=usrp_receiver, daemon=True)
t_dsp = threading.Thread(target=dsp_processor, daemon=True)
t_rx.start(); t_dsp.start()

try: t_rx.join()
except KeyboardInterrupt: rx_streamer.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))