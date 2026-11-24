import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import json
import random
# 引入自动混合精度 (AMP) 用于节省显存和加速
from torch.amp import GradScaler, autocast 

from image_e2e_model import ImageSemanticEncoder, ImageSemanticDecoder

def get_best_device():
    """智能硬件探测器：适配 4090, T400, 以及 AMD CPU"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[硬件探测] 发现 NVIDIA GPU: {gpu_name}")
        return torch.device("cuda")
    
    # AMD 机器 torch-directml 兼容
    try:
        import torch_directml
        if torch_directml.is_available():
            print("[硬件探测] 发现 DirectML 设备 (AMD GPU兼容模式)")
            return torch_directml.device()
    except ImportError:
        pass
        
    print("[硬件探测] 未检测到可用 GPU 训练环境，安全回退到 CPU 模式")
    return torch.device("cpu")

class AnalogEndToEndSystem(nn.Module):
    """虚拟端到端纯模拟物理链路 (Analog Deep JSCC 2019 开山之作基线)"""
    def __init__(self, tx_model, rx_model):
        super().__init__()
        self.encoder = tx_model
        self.decoder = rx_model

    def forward(self, img, snr_db):
        # 1. 发送端：提取深层语义特征 (固定温度 T=1.0，关闭所有掩码和动态机制)
        # 注意：这里输出的是未经离散化的浮点特征
        z = self.encoder(img, mod_type='BPSK', temperature=1.0)
        
        # 2. 物理层可微功率归一化 (Power Normalization)
        # 保证每个 Batch 的特征矩阵平均发送功率严格为 1
        B = z.size(0)
        z_flat = z.view(B, -1)
        k = z_flat.size(1) # 对应 131072 个物理符号
        
        norm_val = torch.norm(z_flat, p=2, dim=1, keepdim=True)
        x_analog = np.sqrt(k) * (z_flat / (norm_val + 1e-9))
        x_analog = x_analog.view_as(z)
        
        # 3. 物理信道：加性高斯白噪声 (AWGN)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = 1.0 / snr_linear
        noise = torch.sqrt(torch.tensor(noise_power, device=z.device)) * torch.randn_like(x_analog)
        y_analog = x_analog + noise
        
        # 4. 接收端解码器：直接摄入带有连续模拟噪声的张量，不设任何阈值/不执行硬判决
        reconstructed_img = self.decoder(y_analog)
        return reconstructed_img

def main():
    print("\033[96m启动 Analog Deep JSCC (2019 基线) 端到端联合训练 (硬件自适应 | AMP加速)\033[0m")
    
    try:
        with open('config.json', 'r', encoding='utf-8') as f: CONFIG = json.load(f)
        print("[系统] 成功加载 config.json")
    except FileNotFoundError:
        print("\033[91m[错误] 找不到 config.json\033[0m")
        return

    train_params = CONFIG["train_params"]
    device = get_best_device()

    tx = ImageSemanticEncoder().to(device)
    rx = ImageSemanticDecoder().to(device)
    # 实例化专属的纯模拟传输系统
    e2e_model = AnalogEndToEndSystem(tx, rx)

    criterion = nn.MSELoss()
    
    # 彻底移除 L2 正则化 (weight_decay 全部置 0)，保持与你之前实验的一致性
    actual_model = e2e_model.module if isinstance(e2e_model, nn.DataParallel) else e2e_model
    optimizer = optim.Adam([
        {'params': actual_model.encoder.parameters(), 'weight_decay': 0.0},
        {'params': actual_model.decoder.parameters(), 'weight_decay': 0.0}
    ], lr=train_params["learning_rate"])

    # 引入梯度缩放器用于 AMP (混合精度训练)
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))

    transform = transforms.Compose([
        transforms.Resize((train_params["img_size"], train_params["img_size"])),
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.ToTensor()
    ])

    data_dir = CONFIG["train_data_dir"]
    try:
        full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        print(f"[数据] 成功加载数据集，共找到 {len(full_dataset)} 张图片。")
    except FileNotFoundError:
        print(f"\033[91m[错误] 找不到数据集目录 {data_dir}/\033[0m")
        return

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 加载数据集
    train_loader = DataLoader(train_dataset, batch_size=train_params["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=train_params["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    best_val_loss = float('inf')
    
    # 提前创建权重保存目录
    analog_tx_path = CONFIG["paths"]["analog_tx_weights"]
    analog_rx_path = CONFIG["paths"]["analog_rx_weights"]
    os.makedirs(os.path.dirname(analog_tx_path), exist_ok=True)

    print("\n\033[93m>>> 开始 Analog Deep JSCC 基线训练...\033[0m")
    for epoch in range(train_params["epochs"]):
        e2e_model.train()
        running_loss = 0.0
        
        for i, (inputs, _) in enumerate(train_loader):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            
            # 随机在设定区间内波动 SNR，提升模型对未知噪声强度的泛化性
            dynamic_snr = np.random.uniform(train_params["sim_snr_min"], train_params["sim_snr_max"])

            # 开启 AMP 自动混合精度
            with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                outputs = e2e_model(inputs, snr_db=dynamic_snr)
                loss = criterion(outputs, inputs)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            
        train_loss = running_loss / len(train_loader)

        # ==================================
        # 验证阶段 (Validation)
        # ==================================
        e2e_model.eval()
        val_loss_high = 0.0
        val_loss_low = 0.0
        
        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(device)
                
                with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                    # 测试优质信道表现 (SNR=15dB)
                    outputs_high = e2e_model(inputs, snr_db=15.0)
                    val_loss_high += criterion(outputs_high, inputs).item()
                    
                    # 测试恶劣信道表现 (SNR=0dB)
                    outputs_low = e2e_model(inputs, snr_db=0.0)
                    val_loss_low += criterion(outputs_low, inputs).item()
                
        val_loss_high /= len(val_loader)
        val_loss_low /= len(val_loader)
        val_loss_avg = (val_loss_high + val_loss_low) / 2.0
        
        print(f"[Epoch {epoch + 1:03d}/{train_params['epochs']}] Train MSE: {train_loss:.4f} | Val MSE(15dB): {val_loss_high:.4f} | Val MSE(0dB): {val_loss_low:.4f}", end="")

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(tx.state_dict(), analog_tx_path)
            torch.save(rx.state_dict(), analog_rx_path)
            print("  \033[92m-> 已保存 Analog 最佳权重！\033[0m")
        else:
            print("")

    print(f"\n\033[92mAnalog JSCC 训练彻底结束！历史最佳 Val Avg MSE: {best_val_loss:.5f}\033[0m")
    print(f"📁 编码器权重: {analog_tx_path}")
    print(f"📁 解码器权重: {analog_rx_path}")

if __name__ == "__main__":
    main()