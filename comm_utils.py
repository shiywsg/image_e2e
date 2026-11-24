# comm_utils.py (支撑底层网络传输与纯物理波形级反馈)
import struct
import pickle
import numpy as np

# ==========================================
# 1. 基础 Socket 传输模块 (解决 TCP 粘包)
# ==========================================
def send_msg(sock, msg):
    """序列化并发送消息 (支持 Numpy 数组波形传输)"""
    msg_data = pickle.dumps(msg)
    sock.sendall(struct.pack('>I', len(msg_data)) + msg_data)

def recv_msg(sock):
    """接收并反序列化消息"""
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    return pickle.loads(recvall(sock, msglen))

def recvall(sock, n):
    """辅助函数：循环接收直到达到指定字节数"""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

# ==========================================
# 2. 上行控制信令物理层前端 (BPSK 调制与扩频)
# ==========================================
def encode_feedback_to_waveform(bits, repeat=31):
    """
    发射端 (接收机反馈侧) 处理：
    1. 极强冗余编码 (Repetition Code)
    2. BPSK 物理调制 (1 -> +1.0, 0 -> -1.0)
    
    bits: list of ints, e.g., [1, 0, 1, 1]
    返回: np.array (float32 波形序列)
    """
    waveform = []
    for b in bits:
        # BPSK 映射
        sym = 1.0 if b == 1 else -1.0
        # 扩频/重复
        waveform.extend([sym] * repeat)
    
    return np.array(waveform, dtype=np.float32)

def decode_feedback_from_waveform(noisy_waveform, num_bits=4, repeat=31):
    """
    接收端 (发送机解析侧) 处理：
    软判决合并解码 (Soft-decision Maximum Ratio Combining)
    
    不使用硬切少数服从多数，而是直接在物理层累加能量，这在对抗 AWGN 
    和脉冲干扰时，比简单的逻辑表决能获得额外约 3dB 的编码增益！
    """
    decoded_bits = []
    
    # 确保传入的是 numpy 数组
    noisy_waveform = np.array(noisy_waveform)
    
    for i in range(num_bits):
        # 截取属于当前 bit 的波形片段
        chunk = noisy_waveform[i * repeat : (i + 1) * repeat]
        
        # 软合并：直接对带有正负能量的浮点数求和
        energy_sum = np.sum(chunk)
        
        # 判决极性
        if energy_sum > 0:
            decoded_bits.append(1)
        else:
            decoded_bits.append(0)
            
    return decoded_bits