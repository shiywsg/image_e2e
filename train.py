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

MOD_POWER = {'4QAM': 1.0, 'BPSK': 1.0}

def get_best_device():
    """智能硬件探测器：适配 4090, T400, 以及 AMD CPU"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[硬件探测] 发现 NVIDIA GPU: {gpu_name}")
        return torch.device("cuda")
    
    # 如果未来你在 AMD 机器上安装了 torch-directml，这里也能自动兼容
    try:
        import torch_directml
        if torch_directml.is_available():
            print("[硬件探测] 发现 DirectML 设备 (AMD GPU兼容模式)")
            return torch_directml.device()
    except ImportError:
        pass
        
    print("[硬件探测] 未检测到可用 GPU 训练环境，安全回退到 CPU 模式")
    return torch.device("cpu")

class EndToEndSystem(nn.Module):
    """虚拟端到端物理链路 (包含物理层硬发送与接收端三值擦除)"""
    def __init__(self, tx_model, rx_model, tau_config):
        super().__init__()
        self.encoder = tx_model
        self.decoder = rx_model
        self.tau_config = tau_config

    def forward(self, img, snr_db, mod_type='BPSK', temp=1.0):
        # 1. 发送端：输出严格的硬判决二值信号 (-1, 1)，带有动态温度
        symbols = self.encoder(img, mod_type=mod_type, temperature=temp)
        
        # 2. 物理信道：加性高斯白噪声
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = MOD_POWER.get(mod_type, 1.0) / snr_linear
        noise = torch.sqrt(torch.tensor(noise_power, device=symbols.device)) * torch.randn_like(symbols)
        noisy_symbols = symbols + noise
        
        # 3. 接收端前端：三值解调判决
        tau = self.tau_config.get(mod_type, 0.4)
        is_ambiguous = torch.abs(noisy_symbols) < tau
        
        hard_symbols = torch.sign(noisy_symbols)
        ternary_symbols = torch.where(is_ambiguous, torch.zeros_like(noisy_symbols), hard_symbols)
        
        # 4. 关键点：信道层的 Straight-Through Estimator (STE)
        rx_input = (ternary_symbols - noisy_symbols).detach() + noisy_symbols
        
        # 5. 接收端解码器
        reconstructed_img = self.decoder(rx_input)
        return reconstructed_img

def main():
    print("\033[96m启动端到端联合训练 (实例归一化 IN 版 | 硬件自适应 | AMP加速)\033[0m")
    
    try:
        with open('config.json', 'r') as f: CONFIG = json.load(f)
        print("[系统] 成功加载 config.json")
    except FileNotFoundError:
        print("\033[91m[错误] 找不到 config.json\033[0m")
        return

    train_params = CONFIG["train_params"]
    device = get_best_device()

    tx = ImageSemanticEncoder().to(device)
    rx = ImageSemanticDecoder().to(device)
    e2e_model = EndToEndSystem(tx, rx, CONFIG["tau_config"])

    criterion = nn.MSELoss()
    
    # 【核心修复 1】：彻底移除无效的 L2 正则化 (weight_decay 全部置 0)
    actual_model = e2e_model.module if isinstance(e2e_model, nn.DataParallel) else e2e_model
    optimizer = optim.Adam([
        {'params': actual_model.encoder.parameters(), 'weight_decay': 0.0},
        {'params': actual_model.decoder.parameters(), 'weight_decay': 0.0}
    ], lr=train_params["learning_rate"])

    # 【工程优化】：引入梯度缩放器用于 AMP (混合精度训练)
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

    # Windows 下如果出现 BrokenPipeError，可以将 num_workers 改为 0
    train_loader = DataLoader(train_dataset, batch_size=train_params["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=train_params["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    best_val_loss = float('inf')
    os.makedirs(os.path.dirname(CONFIG["paths"]["tx_weights"]), exist_ok=True)

    TEMPERATURE_STATES = [1.0, 0.8, 0.3, 0.1]

    print("\n\033[93m>>> 开始端到端训练...\033[0m")
    for epoch in range(train_params["epochs"]):
        e2e_model.train()
        running_loss = 0.0
        
        for i, (inputs, _) in enumerate(train_loader):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            
            dynamic_snr = np.random.uniform(train_params["sim_snr_min"], train_params["sim_snr_max"])
            target_temp = random.choice(TEMPERATURE_STATES)

            # 【工程优化】：开启 AMP 自动混合精度，极大降低 T400 显存压力，提升 4090 速度
            with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                outputs = e2e_model(inputs, snr_db=dynamic_snr, mod_type=train_params["sim_mod"], temp=target_temp)
                loss = criterion(outputs, inputs)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            
        train_loss = running_loss / len(train_loader)

        e2e_model.eval()
        val_loss_high = 0.0
        val_loss_low = 0.0
        
        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(device)
                
                with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                    outputs_high = e2e_model(inputs, snr_db=15.0, mod_type=train_params["sim_mod"], temp=1.0)
                    val_loss_high += criterion(outputs_high, inputs).item()
                    
                    outputs_low = e2e_model(inputs, snr_db=0.0, mod_type=train_params["sim_mod"], temp=0.1)
                    val_loss_low += criterion(outputs_low, inputs).item()
                
        val_loss_high /= len(val_loader)
        val_loss_low /= len(val_loader)
        val_loss_avg = (val_loss_high + val_loss_low) / 2.0
        
        print(f"[Epoch {epoch + 1:03d}/{train_params['epochs']}] Train MSE: {train_loss:.4f} | Val MSE(T=1.0): {val_loss_high:.4f} | Val MSE(T=0.1): {val_loss_low:.4f}", end="")

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(tx.state_dict(), CONFIG["paths"]["tx_weights"])
            torch.save(rx.state_dict(), CONFIG["paths"]["rx_weights"])
            print("  \033[92m-> 已保存最佳权重！\033[0m")
        else:
            print("")

    print(f"\n\033[92m训练彻底结束！历史最佳 Val Avg MSE: {best_val_loss:.5f}\033[0m")

if __name__ == "__main__":
    main()