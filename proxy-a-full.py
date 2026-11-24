import socket
import numpy as np
import uhd
import numba as nb
import threading
import queue
import time

# --- 网络与硬件配置 ---
# TX 通道 (图像下行)
UDP_IMG_TX_PORT = 8001
FREQ_IMG_TX = 2.4e9
CH_TX = 0  # RF0

# RX 通道 (反馈上行)
UDP_FB_RX_PORT = 8004
FREQ_FB_RX = 4.32e8
CH_RX = 1  # RF1

USRP_IP = "192.168.10.2"
RATE = 500e3  
SPS = 4

@nb.njit
def run_costas_loop(iq_cfo, init_phase, init_freq, alpha, beta):
    N = len(iq_cfo)
    out_iq = np.zeros(N, dtype=np.complex64)
    costas_phase, costas_freq = init_phase, init_freq
    for i in range(N):
        c_p, s_p = np.cos(-costas_phase), np.sin(-costas_phase)
        sample = (iq_cfo[i].real * c_p - iq_cfo[i].imag * s_p) + 1j * (iq_cfo[i].real * s_p + iq_cfo[i].imag * c_p)
        out_iq[i] = sample
        I, Q = sample.real, sample.imag
        if abs(I) + abs(Q) > 0.1:
            error = np.sign(I)*Q - np.sign(Q)*I
            costas_freq += beta * error
            if costas_freq > 0.05: costas_freq = 0.05
            elif costas_freq < -0.05: costas_freq = -0.05
            costas_phase += costas_freq + alpha * error
            if costas_phase > np.pi: costas_phase -= 2*np.pi
            elif costas_phase < -np.pi: costas_phase += 2*np.pi
    return out_iq, costas_phase, costas_freq

print(f"[Node A Proxy] 初始化 USRP ({USRP_IP}) FDD 模式...")
usrp = uhd.usrp.MultiUSRP(f"addr={USRP_IP},send_frame_size=1472,recv_frame_size=1472")

# 设置 TX (RF0, 2.4GHz)
usrp.set_tx_rate(RATE, CH_TX)
usrp.set_tx_freq(uhd.types.TuneRequest(FREQ_IMG_TX), CH_TX)
usrp.set_tx_antenna("TX/RX", CH_TX)
usrp.set_tx_gain(20, CH_TX)

# 设置 RX (RF1, 432MHz)
usrp.set_rx_rate(RATE, CH_RX)
usrp.set_rx_freq(uhd.types.TuneRequest(FREQ_FB_RX), CH_RX)
usrp.set_rx_antenna("TX/RX", CH_RX)  # 注意：如果RF1没有RX2丝印，请改为 TX/RX
usrp.set_rx_gain(20, CH_RX)

# 开启双流
st_args_tx = uhd.usrp.StreamArgs("fc32", "sc16")
st_args_tx.channels = [CH_TX]
tx_streamer = usrp.get_tx_stream(st_args_tx)

st_args_rx = uhd.usrp.StreamArgs("fc32", "sc16")
st_args_rx.channels = [CH_RX]
rx_streamer = usrp.get_rx_stream(st_args_rx)

tx_queue = queue.Queue(maxsize=100)
iq_queue = queue.Queue(maxsize=200)
zeros_chunk = np.zeros(4000 * SPS, dtype=np.complex64)

# ================= TX 线程组 (处理图像下行) =================
def udp_listener_tx():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024 * 8)
    sock.bind(("127.0.0.1", UDP_IMG_TX_PORT))
    while True:
        try:
            data, _ = sock.recvfrom(65536)
            arr = np.frombuffer(data, dtype=np.float32).reshape(-1, 2)
            payload_signal = np.repeat((arr[:, 0] + 1j * arr[:, 1]) / np.sqrt(2), SPS).astype(np.complex64)
            if not tx_queue.full(): tx_queue.put(payload_signal)
        except Exception: pass

def usrp_sender():
    meta = uhd.types.TXMetadata()
    tx_streamer.send(zeros_chunk, meta)
    while True:
        try: sig = tx_queue.get(timeout=0.05)
        except queue.Empty: sig = zeros_chunk
        tx_streamer.send(sig, meta)

# ================= RX 线程组 (处理反馈上行) =================
def usrp_receiver():
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    stream_cmd.stream_now = True
    rx_streamer.issue_stream_cmd(stream_cmd)
    meta = uhd.types.RXMetadata()
    recv_buffer = np.zeros((1, 10000 * SPS), dtype=np.complex64)
    while True:
        n = rx_streamer.recv(recv_buffer, meta)
        if n > 0 and not iq_queue.full(): iq_queue.put(recv_buffer[0][:n].copy())

def dsp_processor_rx():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024 * 8)
    costas_phase, costas_freq, global_amp = 0.0, 0.0, 1.0 
    
    # 预热 JIT
    run_costas_loop(np.array([1+1j, 1-1j], dtype=np.complex64), 0.0, 0.0, 0.1, 0.005)
    
    while True:
        iq_raw = iq_queue.get()
        iq_len = (len(iq_raw) // SPS) * SPS
        if iq_len == 0: continue
        
        iq_matrix = iq_raw[:iq_len].reshape(-1, SPS)
        best_phase = np.argmax(np.sum(np.abs(iq_matrix)**2, axis=0))
        iq_symbols = iq_matrix[:, best_phase]
        iq_symbols -= np.mean(iq_symbols)
        chunk_amp = np.mean(np.abs(iq_symbols))
        
        if chunk_amp < 0.02:
            costas_phase, costas_freq = 0.0, costas_freq * 0.99
            out_pairs = np.zeros((len(iq_symbols), 2), dtype=np.float32)
        else:
            global_amp = 0.2 * global_amp + 0.8 * chunk_amp 
            iq_symbols = iq_symbols / (global_amp + 1e-6)
            out_iq, costas_phase, costas_freq = run_costas_loop(iq_symbols, costas_phase, costas_freq, 0.015, 0.0001)
            out_pairs = np.column_stack((np.real(out_iq), np.imag(out_iq))).astype(np.float32)
        
        for i in range(0, len(out_pairs), 4000):
            sock.sendto(out_pairs[i : i + 4000].tobytes(), ("127.0.0.1", UDP_FB_RX_PORT))

threads = [
    threading.Thread(target=udp_listener_tx, daemon=True),
    threading.Thread(target=usrp_sender, daemon=True),
    threading.Thread(target=usrp_receiver, daemon=True),
    threading.Thread(target=dsp_processor_rx, daemon=True)
]
for t in threads: t.start()

print("[Node A Proxy] 双向全双工引擎启动完毕。按 Ctrl+C 退出。")
try:
    while True: time.sleep(1)
except KeyboardInterrupt: 
    rx_streamer.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))
    print("\n[Node A Proxy] 安全关闭。")