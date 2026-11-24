import uhd
import numpy as np
import time

mapping = {'00': -1-1j, '01': -1+1j, '10': 1-1j, '11': 1+1j}
symbols = np.array([mapping[b] for b in ['00', '01', '10', '11']], dtype=np.complex64) / np.sqrt(2)

# 【核心修改】：过采样 (Sps = 4)
# 将 [A, B, C, D] 变成 [A, A, A, A, B, B, B, B, ...]
sps = 4
symbols_upsampled = np.repeat(symbols, sps)

# 生成足够长的数据块，保证 USRP 不会断流
tx_data = np.tile(symbols_upsampled, 25000) # 总长 40万个采样点

print("正在初始化 USRP 发送端...")
usrp = uhd.usrp.MultiUSRP("addr=192.168.10.2,send_frame_size=1472,recv_frame_size=1472")

usrp.set_tx_rate(1e6)
usrp.set_tx_freq(uhd.types.TuneRequest(2.4e9))
usrp.set_tx_antenna("TX/RX", 0) 
usrp.set_tx_gain(20)  # 保持 20

st_args = uhd.usrp.StreamArgs("fc32", "sc16")
st_args.channels = [0]
tx_streamer = usrp.get_tx_stream(st_args)
metadata = uhd.types.TXMetadata()

print("开始发送 4QAM 信号 (按 Ctrl+C 停止)...")
try:
    while True:
        tx_streamer.send(tx_data, metadata)
except KeyboardInterrupt:
    print("发送已停止。")