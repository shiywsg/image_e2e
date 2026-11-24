import os
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import torchvision.transforms as transforms
import torchvision.models as models
import json
import pandas as pd
from datetime import datetime

# --- 新增的依赖 ---
import lpips
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance

VALID_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')

def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2).item()
    if mse == 0: return float('inf')
    return 10 * np.log10(1.0 / mse)

def get_semantic_extractor(device):
    print("\033[90m[系统] 正在加载 ResNet50 语义特征提取器...\033[0m")
    weights = models.ResNet50_Weights.DEFAULT
    resnet = models.resnet50(weights=weights).to(device).eval()
    semantic_model = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
    return semantic_model

def calculate_ssnr_db(tensor_src, tensor_dst, semantic_model, device):
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    src_input = preprocess(tensor_src).unsqueeze(0).to(device)
    dst_input = preprocess(tensor_dst).unsqueeze(0).to(device)

    with torch.no_grad():
        V_orig = semantic_model(src_input).squeeze()
        V_dec = semantic_model(dst_input).squeeze()

    power_s = torch.sum(V_orig ** 2).item()
    noise_vec = V_dec - V_orig
    power_n = max(torch.sum(noise_vec ** 2).item(), 1e-9)
    return 10 * np.log10(power_s / power_n)

def get_all_images(directory, sort_by='name'):
    files_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(VALID_EXTS):
                files_list.append(os.path.join(root, file))
    if sort_by == 'name': files_list.sort()
    elif sort_by == 'time': files_list.sort(key=lambda x: os.path.getmtime(x))
    return files_list

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = {}
    try:
        with open('config.json', 'r', encoding='utf-8') as f: config = json.load(f)
    except Exception: pass

    parser = argparse.ArgumentParser(description="学术级语义图片传输质量评估")
    parser.add_argument("--src", type=str, default=None, help="完美参考源图片文件夹 (计算指标的基准)")
    parser.add_argument("--dst", type=str, default=None, help="接收端重建图片文件夹")
    parser.add_argument("--out", type=str, default="eval_results", help="输出数据表格的前缀名")
    args = parser.parse_args()

    src_dir = args.src if args.src else config.get("tx_source_dir")
    dst_dir = args.dst if args.dst else config.get("rx_target_dir")

    if not src_dir or not dst_dir:
        print("\033[91m[错误] 必须提供源目录与接收目录！\033[0m")
        return

    print(f"\033[96m开始深度语义质量评估\n[参考原图]: {src_dir}\n[重建评估图]: {dst_dir}\033[0m")

    src_files = get_all_images(src_dir, sort_by='name')
    dst_files = get_all_images(dst_dir, sort_by='name')

    if not src_files or not dst_files:
        print("\033[91m[错误] 未找到图片。\033[0m")
        return

    semantic_model = get_semantic_extractor(device)
    
    # --- 初始化新增的评估指标模型 ---
    print("\033[90m[系统] 正在加载 LPIPS (AlexNet), MS-SSIM 与 FID 评估模型...\033[0m")
    loss_fn_lpips = lpips.LPIPS(net='alex', verbose=False).to(device).eval()
    ms_ssim_metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)

    results_data = []
    src_dict = {os.path.basename(p): p for p in src_files}
    
    for dst_path in dst_files:
        filename = os.path.basename(dst_path)
        if filename not in src_dict: continue
        
        src_path = src_dict[filename]
        try:
            img_dst = Image.open(dst_path).convert('RGB')
            img_src = Image.open(src_path).convert('RGB')
            
            tensor_dst = transforms.ToTensor()(img_dst)
            tensor_src = transforms.ToTensor()(img_src)
            
            if tensor_src.shape != tensor_dst.shape:
                 tensor_src = transforms.Resize(tensor_dst.shape[1:])(tensor_src)

            # 扩展维度为 Batch Size = 1 以适配 PyTorch 指标计算
            src_batch = tensor_src.unsqueeze(0).to(device)
            dst_batch = tensor_dst.unsqueeze(0).to(device)

            # --- 计算指标 ---
            # 1. PSNR & SSIM
            psnr_val = calculate_psnr(tensor_src, tensor_dst)
            np_src, np_dst = tensor_src.permute(1, 2, 0).numpy(), tensor_dst.permute(1, 2, 0).numpy()
            ssim_val = ssim(np_src, np_dst, data_range=1.0, channel_axis=-1)
            
            with torch.no_grad():
                # 2. MS-SSIM
                try:
                    ms_ssim_val = ms_ssim_metric(dst_batch, src_batch).item()
                except Exception:
                    ms_ssim_val = float('nan') # 图像小于 160x160 时可能会抛出异常

                # 3. LPIPS (需将像素范围从 [0, 1] 映射到 [-1, 1])
                src_lpips = src_batch * 2.0 - 1.0
                dst_lpips = dst_batch * 2.0 - 1.0
                lpips_val = loss_fn_lpips(src_lpips, dst_lpips).item()

                # 4. 累计 FID 特征 (要求输入为 uint8 的 [0, 255])
                src_fid = (src_batch * 255).to(torch.uint8)
                dst_fid = (dst_batch * 255).to(torch.uint8)
                fid_metric.update(src_fid, real=True)
                fid_metric.update(dst_fid, real=False)

            # 5. SSNR
            ssnr_val = calculate_ssnr_db(tensor_src, tensor_dst, semantic_model, device)

            # 暂不填入 FID，因为需等遍历完毕后统一计算
            results_data.append({
                "image_name": filename,
                "psnr_db": round(psnr_val, 4),
                "ssim": round(ssim_val, 4),
                "ms_ssim": round(ms_ssim_val, 4),
                "lpips": round(lpips_val, 4),
                "fid": None, 
                "ssnr_db": round(ssnr_val, 4)
            })

            print(f"[评估 {filename}] PSNR={psnr_val:>5.2f}dB | SSIM={ssim_val:.4f} | MS-SSIM={ms_ssim_val:.4f} | LPIPS={lpips_val:.4f} | S-SNR={ssnr_val:>5.2f}dB")
            
        except Exception as e:
            print(f"\033[91m[错误] 处理 {filename} 时异常: {e}\033[0m")

    if results_data:
        print("\033[90m[系统] 正在计算数据集全局 FID 分数...\033[0m")
        try:
            global_fid = fid_metric.compute().item()
        except Exception as e:
            global_fid = float('nan')
            print(f"\033[91m[错误] FID 计算失败 (可能由于样本量过少): {e}\033[0m")

        # 将全局 FID 分数补全到每一行的结果中
        for row in results_data:
            row["fid"] = round(global_fid, 4) if not np.isnan(global_fid) else global_fid

        df = pd.DataFrame(results_data)
        
        # 严格按照要求重新排序列
        columns_order = ["image_name", "psnr_db", "ssim", "ms_ssim", "lpips", "fid", "ssnr_db"]
        df = df[columns_order]

        csv_filename = f"{args.out}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_filename, index=False, encoding='utf-8')

        print("\n\033[92m" + "="*50)
        print("最终全局评估结果:")
        print(f"平均 PSNR:    {df['psnr_db'].mean():.2f} dB")
        print(f"平均 SSIM:    {df['ssim'].mean():.4f}")
        print(f"平均 MS-SSIM: {df['ms_ssim'].mean():.4f}")
        print(f"平均 LPIPS:   {df['lpips'].mean():.4f}  (越低越好)")
        print(f"全局 FID:     {global_fid:.2f}  (越低越好)")
        print(f"平均 S-SNR:   {df['ssnr_db'].mean():.2f} dB")
        print(f"[成功] 数据已落盘至: {csv_filename}\033[0m")

if __name__ == "__main__":
    main()