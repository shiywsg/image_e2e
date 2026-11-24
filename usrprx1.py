import uhd
import numpy as np
import matplotlib.pyplot as plt

print("正在初始化 USRP 接收端...")
usrp = uhd.usrp.MultiUSRP("addr=192.168.10.2,recv_frame_size=1472,send_frame_size=1472")

usrp.set_rx_rate(1e6)
usrp.set_rx_freq(uhd.types.TuneRequest(2.4e9))
usrp.set_rx_antenna("RX2", 0)
usrp.set_rx_gain(20)

st_args = uhd.usrp.StreamArgs("fc32", "sc16")
st_args.channels = [0]
rx_streamer = usrp.get_rx_stream(st_args)

recv_buffer = np.zeros((1, 40000), dtype=np.complex64)
metadata = uhd.types.RXMetadata()

stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
stream_cmd.stream_now = True
rx_streamer.issue_stream_cmd(stream_cmd)

print("正在捕获信号...")
for i in range(10):
    num_samps = rx_streamer.recv(recv_buffer, metadata)

rx_streamer.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))
iq_raw = recv_buffer[0][:num_samps]

# 1. 符号定时提取 (保持之前的滑动平均)
sps = 4
iq_len = (len(iq_raw) // sps) * sps
iq_trimmed = iq_raw[:iq_len]
iq_symbols = np.mean(iq_trimmed.reshape(-1, sps)[:, 1:3], axis=1)

# 2. 粗略 CFO 估计与补偿 (四次方 FFT)
fs_sym = 1e6 / sps
N = len(iq_symbols)

iq_4 = iq_symbols ** 4
fft_iq_4 = np.fft.fft(iq_4)
fft_iq_4[0:10] = 0
fft_iq_4[-10:] = 0

freqs = np.fft.fftfreq(N, 1/fs_sym)
peak_idx = np.argmax(np.abs(fft_iq_4))
f_offset_est = freqs[peak_idx] / 4.0

t = np.arange(N) / fs_sym
cfo_phasor = np.exp(-1j * 2 * np.pi * f_offset_est * t)
iq_cfo_corrected = iq_symbols * cfo_phasor

print(f"👉 粗略频偏补偿: {f_offset_est:.2f} Hz")

# ================= 核心升级：4QAM Costas 环 =================
print("🔄 正在运行 Costas 环进行残余相位跟踪...")

# Costas 环参数
alpha = 0.05    # 比例增益 (控制响应速度)
beta = 0.001    # 积分增益 (控制稳态误差)

out_iq = np.zeros(N, dtype=np.complex64)
phase = 0.0
freq = 0.0

for i in range(N):
    # 接收当前样本
    in_sample = iq_cfo_corrected[i]
    
    # 乘以内部振荡器产生的补偿相位
    out_sample = in_sample * np.exp(-1j * phase)
    out_iq[i] = out_sample
    
    # 4QAM 误差检测 (Hard Decision Directed)
    # 判断当前样本离哪个理想星座点最近，然后计算角度差
    # 为了简化，我们使用一种对幅值不敏感的误差计算公式：e = sign(I)*Q - sign(Q)*I
    # 注意：这里的误差与相位差成正比，用于驱动环路滤波器
    I = np.real(out_sample)
    Q = np.imag(out_sample)
    
    # 这里加入一个保护，如果幅度太小（过渡带的拖尾），我们不更新环路，防止干扰锁定
    if abs(I) + abs(Q) < 0.1:
        error = 0.0
    else:
        error = np.sign(I) * Q - np.sign(Q) * I
    
    # 环路滤波器 (PI 控制器)
    freq += beta * error
    phase += freq + alpha * error
    
    # 保持相位在 -pi 到 pi 之间
    phase = (phase + np.pi) % (2 * np.pi) - np.pi


# ================= 绘制对比图 =================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# 我们截取后半段数据绘图，因为 Costas 环在最开始需要一点时间来“锁定” (Lock Time)
start_idx = N // 4

ax1.scatter(np.real(iq_cfo_corrected[start_idx:]), np.imag(iq_cfo_corrected[start_idx:]), alpha=0.3, color='orange', s=5)
ax1.set_title("Only Block CFO Compensation (Residual Spreading)")
ax1.grid(True)
ax1.axhline(0, color='black', linewidth=0.5); ax1.axvline(0, color='black', linewidth=0.5)
ax1.set_xlim([-1.0, 1.0]); ax1.set_ylim([-1.0, 1.0])

ax2.scatter(np.real(out_iq[start_idx:]), np.imag(out_iq[start_idx:]), alpha=0.3, color='blue', s=5)
ax2.set_title("After Costas Loop Phase Tracking (Locked)")
ax2.grid(True)
ax2.axhline(0, color='black', linewidth=0.5); ax2.axvline(0, color='black', linewidth=0.5)
ax2.set_xlim([-1.0, 1.0]); ax2.set_ylim([-1.0, 1.0])

plt.show()