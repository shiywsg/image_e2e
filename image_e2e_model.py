import torch
import torch.nn as nn

class DynamicQuantize_STE(nn.Module):
    """发送端的严格二值化直通估计器"""
    def __init__(self):
        super().__init__()

    def forward(self, x, mod_type='BPSK'):
        x_quant = torch.sign(x)
        x_quant[x_quant == 0] = 1.0 # 消除绝对的 0
        if self.training:
            return (x_quant - x).detach() + x
        return x_quant

class SpatialAttention(nn.Module):
    """
    终极版空间注意力模块 (堵死所有作弊漏洞)
    """
    def __init__(self, in_channels):
        super().__init__()
        # 1. 移除 bias，防止网络利用常数偏移作弊
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=False)
        # 2. 强制 affine=False，绝不给网络学习缩放和平移的机会，严格均值 0
        self.in_norm = nn.InstanceNorm2d(1, affine=False)

    def forward(self, x, temperature=1.0):
        logits = self.conv(x)
        norm_logits = self.in_norm(logits)
        temp = max(float(temperature), 0.05)
        
        # 因为 norm_logits 严格均值为 0：
        # 当 T=0.1 时，一半的特征会逼近绝对的 1.0，另一半会被无情地打压成绝对的 0.0
        mask = torch.sigmoid(norm_logits / temp)
        return mask

class ImageSemanticEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1), nn.Tanh())
        
        # 【架构革命】：注意力掩码作用于最后一层，通道数匹配 conv3 的 32
        self.attention = SpatialAttention(in_channels=32)
        self.quantizer = DynamicQuantize_STE()

    def forward(self, x, mod_type='BPSK', temperature=1.0):
        feat1 = self.conv1(x)
        feat2 = self.conv2(feat1)
        feat3 = self.conv3(feat2)
        
        # 1. 首先生成满载的 BPSK/4QAM 物理符号 (+1 / -1)
        raw_symbols = self.quantizer(feat3, mod_type)
        
        # 2. 根据最终特征计算物理截断掩码
        mask = self.attention(feat3, temperature)
        
        # 3. 【真正的物理静噪】：直接用掩码对符号进行功率截断！
        # 如果 mask 趋近于 0，发送的物理能量就是 0。
        # 此时接收端只会收到纯底噪，完美触发接收端的“三值擦除 (Tau)”机制！
        symbols = raw_symbols * mask
        return symbols

class ImageSemanticDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() 
        )

    def forward(self, received_symbols, return_sparsity=False):
        x1 = self.layer1(received_symbols)
        x2 = self.layer2(x1) 
        
        out_img = self.layer3(x2)
        
        if return_sparsity:
            l1_norm = torch.norm(x2, p=1)
            l2_norm = torch.norm(x2, p=2)
            sparsity_val = (l1_norm / (l2_norm + 1e-9)).item()
            return out_img, sparsity_val
            
        return out_img