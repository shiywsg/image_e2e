import socket
import numpy as np
import uhd
import numba as nb
import threading
import queue
import time

UDP_IMG_TX_PORT = 8001  
UDP_FB_RX_PORT = 8004   
USRP_IP = "192.168.10.2"
FREQ = 2.4e9
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

print(f"[Node A Proxy] 初始化 USRP ({USRP_IP}) 包级定时锁定版...")
usrp = uhd.usrp.MultiUSRP(f"addr={USRP_IP},send_frame_size=1472,recv_frame_size=1472")

usrp.set_tx_rate(RATE, 0); usrp.set_rx_rate(RATE, 0)
usrp.set_tx_freq(uhd.types.TuneRequest(FREQ), 0); usrp.set_rx_freq(uhd.types.TuneRequest(FREQ), 0)
usrp.set_tx_antenna("TX/RX", 0); usrp.set_rx_antenna("TX/RX", 0)
usrp.set_tx_gain(35, 0); usrp.set_rx_gain(35, 0)

st_args_tx = uhd.usrp.StreamArgs("fc32", "sc16"); st_args_tx.channels = [0]
tx_streamer = usrp.get_tx_stream(st_args_tx)
st_args_rx = uhd.usrp.StreamArgs("fc32", "sc16"); st_args_rx.channels = [0]
rx_streamer = usrp.get_rx_stream(st_args_rx)

tx_queue = queue.Queue(maxsize=100)
iq_queue = queue.Queue(maxsize=200)
is_transmitting = False

def udp_listener_tx():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024 * 8)
    sock.bind(("127.0.0.1", UDP_IMG_TX_PORT))
    while True:
        try:
            data, _ = sock.recvfrom(65536)
            arr = np.frombuffer(data, dtype=np.float32).reshape(-1, 2)
            payload = np.repeat((arr[:, 0] + 1j * arr[:, 1]) / np.sqrt(2), SPS).astype(np.complex64)
            if not tx_queue.full(): tx_queue.put(payload)
        except Exception: pass

def usrp_sender():
    global is_transmitting
    meta = uhd.types.TXMetadata()
    in_burst = False
    while True:
        try:
            sig = tx_queue.get(timeout=0.1)
            is_transmitting = True
            if not in_burst:
                meta.start_of_burst = True; in_burst = True
            else:
                meta.start_of_burst = False
            meta.end_of_burst = False
            tx_streamer.send(sig, meta)
        except queue.Empty:
            if in_burst:
                meta.start_of_burst = False; meta.end_of_burst = True
                tx_streamer.send(np.zeros(10, dtype=np.complex64), meta)
                in_burst = False; time.sleep(0.05); is_transmitting = False

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
    
    # DSP 状态机变量
    costas_phase, costas_freq = 0.0, 0.0
    global_amp = 0.0 
    was_transmitting, blind_samples = False, 0
    
    # 核心新增：包级别定时锁定变量
    packet_active = False
    locked_phase = 0
    
    while True:
        iq_raw = iq_queue.get()
        if is_transmitting: 
            was_transmitting = True
            continue
            
        if was_transmitting:
            while not iq_queue.empty():
                try: iq_queue.get_nowait()
                except queue.Empty: break
            was_transmitting = False
            costas_phase, global_amp = 0.0, 0.0
            blind_samples = 10000 
            packet_active = False # 切回接收时重置包状态
            continue
            
        if blind_samples > 0:
            blind_samples -= len(iq_raw)
            continue
        
        iq_len = (len(iq_raw) // SPS) * SPS
        if iq_len == 0: continue
        
        iq_matrix = iq_raw[:iq_len].reshape(-1, SPS)
        phase_energies = np.sum(np.abs(iq_matrix)**2, axis=0)
        temp_phase = np.argmax(phase_energies)
        temp_symbols = iq_matrix[:, temp_phase]
        chunk_amp = np.mean(np.abs(temp_symbols))
        
        # 门限设为 0.01，过滤底噪
        if chunk_amp < 0.01:
            packet_active = False # 空气中没信号，解锁定时状态
            costas_phase, costas_freq = 0.0, costas_freq * 0.99
            global_amp = 0.0 
            out_pairs = np.zeros((len(temp_symbols), 2), dtype=np.float32)
        else:
            if not packet_active:
                # 侦测到数据包降临，瞬间死锁最佳采样相位！
                locked_phase = temp_phase
                packet_active = True
                global_amp = chunk_amp
                
            global_amp = 0.5 * global_amp + 0.5 * chunk_amp 
            # 使用锁定的相位进行采样，绝不滑动边缘！
            iq_symbols = iq_matrix[:, locked_phase]
            iq_symbols = iq_symbols / (global_amp + 1e-6)
            out_iq, costas_phase, costas_freq = run_costas_loop(iq_symbols, costas_phase, costas_freq, 0.015, 0.0001)
            out_pairs = np.column_stack((np.real(out_iq), np.imag(out_iq))).astype(np.float32)
        
        for i in range(0, len(out_pairs), 4000):
            sock.sendto(out_pairs[i : i + 4000].tobytes(), ("127.0.0.1", UDP_FB_RX_PORT))

threads = [threading.Thread(target=f, daemon=True) for f in [udp_listener_tx, usrp_sender, usrp_receiver, dsp_processor_rx]]
for t in threads: t.start()
print("[Node A Proxy] 启动完毕。按 Ctrl+C 退出。")
while True: time.sleep(1)