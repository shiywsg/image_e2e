import socket
import numpy as np
import uhd
import threading
import queue
import time

UDP_IP = "127.0.0.1"
UDP_PORT = 8001
USRP_IP = "192.168.10.2"
RF_PORT = "TX/RX"
FREQ = 2.4e9
RATE = 500e3  
SPS = 4

print(f"[TX Proxy] 正在初始化 USRP ({USRP_IP})...")
usrp = uhd.usrp.MultiUSRP(f"addr={USRP_IP},send_frame_size=1472")
usrp.set_tx_rate(RATE)
usrp.set_tx_freq(uhd.types.TuneRequest(FREQ))
usrp.set_tx_antenna(RF_PORT, 0)
usrp.set_tx_gain(20)

st_args = uhd.usrp.StreamArgs("fc32", "sc16")
st_args.channels = [0]
tx_streamer = usrp.get_tx_stream(st_args)
metadata = uhd.types.TXMetadata()

tx_queue = queue.Queue(maxsize=100)
zeros_chunk = np.zeros(4000 * SPS, dtype=np.complex64)

def udp_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # 【修复 1】：强行扩大 UDP 缓冲区至 8MB，防止阵列错位
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024 * 8)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"[TX Proxy] UDP 透明管道已启动 (端口 {UDP_PORT})")
    
    while True:
        try:
            data, _ = sock.recvfrom(65536)
            arr = np.frombuffer(data, dtype=np.float32).reshape(-1, 2)
            complex_syms = (arr[:, 0] + 1j * arr[:, 1]) / np.sqrt(2)
            payload_signal = np.repeat(complex_syms, SPS).astype(np.complex64)
            if not tx_queue.full():
                tx_queue.put(payload_signal)
        except Exception: pass

def usrp_sender():
    print("[TX Proxy] 射频发射线程已启动。")
    tx_streamer.send(zeros_chunk, metadata)
    while True:
        try: signal_to_send = tx_queue.get(timeout=0.05)
        except queue.Empty: signal_to_send = zeros_chunk
        tx_streamer.send(signal_to_send, metadata)

t_udp = threading.Thread(target=udp_listener, daemon=True)
t_usrp = threading.Thread(target=usrp_sender, daemon=True)
t_udp.start(); t_usrp.start()

try: t_udp.join()
except KeyboardInterrupt: print("\n[TX Proxy] 安全关闭。")