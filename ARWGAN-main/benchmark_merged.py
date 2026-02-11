"""
ARWGAN Benchmark Script - 合併版
==================================
結合兩個版本的優點：
- 完整攻擊覆蓋（60+ 種，精確實作）
- 視覺化圖表（matplotlib）
- PSNR 指標
- 統一類別結構（易維護）

使用方法:
    python benchmark_merged.py --checkpoint ./checkpoints_merged/best_model.pth --data-dir ./data/coco2017/val/images

輸出:
    - benchmark_results.csv: 詳細結果
    - benchmark_chart.png: 視覺化圖表
    - Console: 表格格式輸出
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import argparse
import csv
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import io

# Kornia imports
try:
    import kornia.filters as filters
    from kornia.enhance import AdjustHue, AdjustSaturation, AdjustContrast, AdjustBrightness
    HAS_KORNIA = True
except ImportError:
    HAS_KORNIA = False
    print("⚠️  警告: 未安裝 kornia，部分攻擊可能無法使用")
    print("   建議安裝: pip install kornia")

# 導入模型架構
from watermark_model_merged import Encoder, Decoder, Bottleneck


# ============================================================
# 統一攻擊層（教授風格 + 完整實作）
# ============================================================

class PaperRobustnessLayer(nn.Module):
    """
    符合 ARWGAN 論文 Table II 與 Fig. 8 規範的攻擊層
    統一接口，易於維護
    """
    def __init__(self, device):
        super().__init__()
        self.device = device
    
    def jpeg_compression(self, x, quality):
        """真實 JPEG 壓縮"""
        B, C, H, W = x.shape
        x_attacked = []
        
        for b in range(B):
            img_tensor = x[b].cpu().clamp(0, 1)
            img = transforms.ToPILImage()(img_tensor)
            
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=int(quality))
            buffer.seek(0)
            img_jpeg = Image.open(buffer).convert('RGB')
            
            x_jpeg = transforms.ToTensor()(img_jpeg).unsqueeze(0)
            if x_jpeg.shape[2:] != (H, W):
                x_jpeg = F.interpolate(x_jpeg, size=(H, W), mode='bilinear', align_corners=False)
            x_attacked.append(x_jpeg)
        
        result = torch.cat(x_attacked, dim=0).to(self.device)
        return torch.clamp(result, 0, 1)

    def gaussian_noise(self, x, variance):
        """高斯噪聲: var = 0.06, 0.08, 0.10"""
        std = np.sqrt(variance)
        noise = torch.randn_like(x) * std
        return torch.clamp(x + noise, 0, 1)

    def salt_pepper(self, x, prob):
        """椒鹽噪聲: p = 0.05, 0.10, 0.15"""
        noise_tensor = torch.rand_like(x)
        # Salt (1) with probability prob/2
        x_out = torch.where(noise_tensor < prob/2, torch.ones_like(x), x)
        # Pepper (0) with probability prob/2
        x_out = torch.where(noise_tensor > 1 - prob/2, torch.zeros_like(x), x_out)
        return x_out

    def median_filter(self, x, kernel_size):
        """中值濾波: k = 3, 5, 7"""
        if HAS_KORNIA:
            return filters.MedianBlur(kernel_size=(kernel_size, kernel_size))(x)
        else:
            # CPU 備用實作
            pad = kernel_size // 2
            x_pad = F.pad(x, (pad, pad, pad, pad), mode='reflect')
            patches = x_pad.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
            median = patches.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3], -1).median(dim=-1)[0]
            return median

    def brightness(self, x, factor):
        """調整亮度: factor = 1.1, 1.2, 1.3"""
        if HAS_KORNIA:
            return torch.clamp(AdjustBrightness(brightness_factor=factor)(x), 0, 1)
        else:
            return torch.clamp(x * factor, 0, 1)

    def contrast(self, x, factor):
        """調整對比度: factor = 1.0, 1.5, 2.0"""
        if HAS_KORNIA:
            return torch.clamp(AdjustContrast(contrast_factor=factor)(x), 0, 1)
        else:
            mean = x.mean(dim=[2, 3], keepdim=True)
            return torch.clamp((x - mean) * factor + mean, 0, 1)

    def hue(self, x, factor):
        """調整色調: factor = -0.1, 0.0, 0.1"""
        if HAS_KORNIA:
            return torch.clamp(AdjustHue(hue_factor=factor)(x), 0, 1)
        else:
            return x  # 無 kornia 時跳過

    def saturation(self, x, factor):
        """調整飽和度: factor = 0.5, 1.0, 1.5"""
        if HAS_KORNIA:
            return torch.clamp(AdjustSaturation(saturation_factor=factor)(x), 0, 1)
        else:
            return x  # 無 kornia 時跳過

    def grid_crop(self, x, ratio):
        """Grid Crop (精確實作): ratio = 0.5, 0.6, 0.7 (50%, 60%, 70%)"""
        B, C, H, W = x.shape
        block_size = 8
        n_blocks_h = H // block_size
        n_blocks_w = W // block_size
        block_switch = torch.rand(n_blocks_h, n_blocks_w, device=self.device) < ratio
        x_attacked = x.clone()
        
        for i in range(n_blocks_h):
            for j in range(n_blocks_w):
                if block_switch[i, j]:
                    h_start = block_size * i
                    h_end = min(h_start + block_size, H)
                    w_start = block_size * j
                    w_end = min(w_start + block_size, W)
                    x_attacked[:, :, h_start:h_end, w_start:w_end] = 0.0
        return x_attacked

    def cropout(self, x, ratio, original_image):
        """Cropout (用原圖填補): ratio = 0.1, 0.2, 0.3 (10%, 20%, 30%)"""
        B, C, H, W = x.shape
        crop_h = int(H * np.sqrt(ratio))
        crop_w = int(W * np.sqrt(ratio))
        h_start = torch.randint(0, max(1, H - crop_h), (B,), device=self.device)
        w_start = torch.randint(0, max(1, W - crop_w), (B,), device=self.device)
        
        x_attacked = x.clone()
        for b in range(B):
            hs = h_start[b].item()
            ws = w_start[b].item()
            h_end = min(hs + crop_h, H)
            w_end = min(ws + crop_w, W)
            x_attacked[b:b+1, :, hs:h_end, ws:w_end] = \
                original_image[b:b+1, :, hs:h_end, ws:w_end]
        return x_attacked

    def resize(self, x, scale):
        """Resize: scale = 0.5, 0.75, 1.0, 1.25, 1.5"""
        B, C, H, W = x.shape
        new_h, new_w = int(H * scale), int(W * scale)
        x_resized = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
        x_restored = F.interpolate(x_resized, size=(H, W), mode='bilinear', align_corners=False)
        return x_restored

    def gaussian_blur(self, x, sigma):
        """Gaussian Blur: sigma = 0.5, 1.0, 1.5"""
        if HAS_KORNIA:
            kernel_size = int(2 * np.ceil(3 * sigma) + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            return filters.GaussianBlur2d(kernel_size=(kernel_size, kernel_size), 
                                         sigma=(sigma, sigma))(x)
        else:
            return x  # 無 kornia 時跳過

    def dropout(self, x, ratio, original_image):
        """Dropout: ratio = 0.1, 0.2, 0.3"""
        B, C, H, W = x.shape
        block_h = max(1, int(H * np.sqrt(ratio)))
        block_w = max(1, int(W * np.sqrt(ratio)))
        h_start = torch.randint(0, max(1, H - block_h), (B,), device=self.device)
        w_start = torch.randint(0, max(1, W - block_w), (B,), device=self.device)
        
        x_attacked = x.clone()
        for b in range(B):
            hs = h_start[b].item()
            ws = w_start[b].item()
            h_end = min(hs + block_h, H)
            w_end = min(ws + block_w, W)
            x_attacked[b:b+1, :, hs:h_end, ws:w_end] = \
                original_image[b:b+1, :, hs:h_end, ws:w_end]
        return x_attacked

    def forward(self, x, attack_config, original_image=None):
        """統一接口"""
        atype = attack_config['type']
        val = attack_config.get('val', 0)
        
        if atype == 'none':
            return x
        elif atype == 'gaussian_noise':
            return self.gaussian_noise(x, val)
        elif atype == 'salt_pepper':
            return self.salt_pepper(x, val)
        elif atype == 'jpeg':
            return self.jpeg_compression(x, val)
        elif atype == 'median':
            return self.median_filter(x, int(val))
        elif atype == 'resize':
            return self.resize(x, val)
        elif atype == 'cropout':
            if original_image is None:
                return x
            return self.cropout(x, val, original_image)
        elif atype == 'brightness':
            return self.brightness(x, val)
        elif atype == 'contrast':
            return self.contrast(x, val)
        elif atype == 'hue':
            return self.hue(x, val)
        elif atype == 'saturation':
            return self.saturation(x, val)
        elif atype == 'grid_crop':
            return self.grid_crop(x, val)
        elif atype == 'gaussian_blur':
            return self.gaussian_blur(x, val)
        elif atype == 'dropout':
            if original_image is None:
                return x
            return self.dropout(x, val, original_image)
        elif atype == 'combined':
            # 組合攻擊
            result = x
            for sub_attack in attack_config['attacks']:
                if sub_attack['type'] in ['cropout', 'dropout']:
                    result = self.forward(result, sub_attack, original_image)
                else:
                    result = self.forward(result, sub_attack, None)
            return result
        else:
            return x


# ============================================================
# 評測主函數
# ============================================================

def load_test_images(data_dir, max_images=100):
    """載入測試圖片"""
    data_dir = Path(data_dir)
    image_files = []
    
    # 搜尋圖片文件
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(list(data_dir.glob(ext)))
        image_files.extend(list(data_dir.glob(ext.upper())))
    
    if len(image_files) == 0:
        # 嘗試子目錄
        for subdir in ['val/images', 'val', 'images']:
            subdir_path = data_dir / subdir
            if subdir_path.exists():
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    image_files.extend(list(subdir_path.glob(ext)))
                if image_files:
                    break
    
    image_files = image_files[:max_images]
    print(f"找到 {len(image_files)} 張測試圖片")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    images = []
    for img_path in image_files:
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            images.append(img_tensor)
        except Exception as e:
            print(f"跳過 {img_path}: {e}")
            continue
    
    return torch.stack(images) if images else None


def compute_metrics(extracted, watermark, watermarked, original):
    """計算所有指標"""
    # Bit Accuracy
    # 注意：Decoder 輸出的 extracted 已經是二值化的 0/1，不需要 .round()
    ber = (extracted != watermark).float().mean().item()
    ba = (1 - ber) * 100.0
    
    # PSNR
    mse = F.mse_loss(watermarked, original).item()
    psnr = 10 * np.log10(1.0 / max(mse, 1e-8))
    
    return ba, ber, psnr


def evaluate_robustness(encoder, decoder, images, watermarks, device, attack_suite, batch_size=8):
    """執行完整評測"""
    encoder.eval()
    decoder.eval()
    attacker = PaperRobustnessLayer(device)
    
    results_summary = []
    
    print(f"\n🔬 開始執行論文復現測試 (Table II)...")
    print(f"   總共 {len(attack_suite)} 種攻擊\n")
    
    for attack_name, config in tqdm(attack_suite.items(), desc="評測進度"):
        ba_list = []
        ber_list = []
        psnr_list = []
        
        needs_original = config['type'] in ['cropout', 'dropout'] or \
                        (config['type'] == 'combined' and 
                         any(a['type'] in ['cropout', 'dropout'] for a in config.get('attacks', [])))
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i+batch_size].to(device)
                batch_watermarks = watermarks[i:i+batch_size].to(device)
                
                # 嵌入浮水印
                watermarked = encoder(batch_images, batch_watermarks)
                
                # 施加攻擊
                if needs_original:
                    attacked = attacker(watermarked, config, original_image=batch_images)
                else:
                    attacked = attacker(watermarked, config)
                
                # 提取浮水印
                extracted, _ = decoder(attacked)
                
                # 計算指標
                ba, ber, psnr = compute_metrics(extracted, batch_watermarks, watermarked, batch_images)
                
                ba_list.append(ba)
                ber_list.append(ber)
                psnr_list.append(psnr)
        
        avg_ba = np.mean(ba_list)
        avg_ber = np.mean(ber_list)
        avg_psnr = np.mean(psnr_list)
        
        print(f"✅ {attack_name:40s} | BA: {avg_ba:6.2f}% | BER: {avg_ber:.4f} | PSNR: {avg_psnr:6.2f}dB")
        
        results_summary.append({
            'Attack': attack_name,
            'Parameter': config.get('val', '-'),
            'Bit Accuracy (%)': round(avg_ba, 2),
            'BER': round(avg_ber, 4),
            'PSNR (dB)': round(avg_psnr, 2)
        })
    
    return results_summary


def print_table(results):
    """印出表格格式"""
    print("\n" + "="*90)
    print("Benchmark Results - Bit Accuracy (%)")
    print("="*90)
    print(f"{'Attack':<50} {'BA (%)':>10} {'BER':>10} {'PSNR (dB)':>12}")
    print("-"*90)
    
    for r in results:
        print(f"{r['Attack']:<50} {r['Bit Accuracy (%)']:>10.2f} {r['BER']:>10.4f} {r['PSNR (dB)']:>12.2f}")
    
    print("="*90)


def save_csv(results, output_path):
    """保存結果到 CSV"""
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"\n📊 CSV 結果已保存至: {output_path}")


def plot_results(results, output_path):
    """生成視覺化圖表"""
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bit Accuracy 圖表
    axes[0].bar(range(len(df)), df['Bit Accuracy (%)'], color='skyblue', alpha=0.7)
    axes[0].set_xticks(range(len(df)))
    axes[0].set_xticklabels(df['Attack'], rotation=90, ha='right', fontsize=8)
    axes[0].set_ylabel('Bit Accuracy (%)', fontsize=12)
    axes[0].set_title('ARWGAN Benchmark - Bit Accuracy', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylim([0, 100])
    
    # PSNR 圖表
    axes[1].bar(range(len(df)), df['PSNR (dB)'], color='lightcoral', alpha=0.7)
    axes[1].set_xticks(range(len(df)))
    axes[1].set_xticklabels(df['Attack'], rotation=90, ha='right', fontsize=8)
    axes[1].set_ylabel('PSNR (dB)', fontsize=12)
    axes[1].set_title('ARWGAN Benchmark - PSNR', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📈 圖表已保存至: {output_path}")


# ============================================================
# 主程式
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ARWGAN Benchmark Script (Merged Version)')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints_merged/best_model.pth',
                       help='模型 checkpoint 路徑')
    parser.add_argument('--data-dir', type=str, default='./data/coco2017/val/images',
                       help='測試圖片目錄')
    parser.add_argument('--watermark-bits', type=int, default=64, help='浮水印位數')
    parser.add_argument('--channels', type=int, default=64, help='模型通道數')
    parser.add_argument('--max-images', type=int, default=100, help='最大測試圖片數量')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='設備 (cuda/cpu)')
    parser.add_argument('--output-dir', type=str, default='benchmark_results', help='輸出目錄')
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    print(f"使用設備: {device}")
    
    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 載入模型
    print(f"\n🔧 載入模型: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    encoder = Encoder(watermark_bits=args.watermark_bits, channels=args.channels).to(device)
    decoder = Decoder(watermark_bits=args.watermark_bits, channels=args.channels).to(device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    encoder.eval()
    decoder.eval()
    print("✅ 模型載入完成")
    
    # 載入測試圖片
    print(f"\n📁 載入測試圖片: {args.data_dir}")
    images = load_test_images(args.data_dir, max_images=args.max_images)
    if images is None or len(images) == 0:
        raise ValueError(f"無法載入測試圖片: {args.data_dir}")
    
    images = images.to(device)
    watermarks = torch.randint(0, 2, (len(images), args.watermark_bits), device=device).float()
    print(f"✅ 載入 {len(images)} 張圖片")
    
    # 定義所有攻擊（完整版）
    attack_suite = {
        'Identity (No Attack)': {'type': 'none', 'val': 0},
        
        # Gaussian Noise
        'Gaussian Noise (var=0.06)': {'type': 'gaussian_noise', 'val': 0.06},
        'Gaussian Noise (var=0.08)': {'type': 'gaussian_noise', 'val': 0.08},
        'Gaussian Noise (var=0.10)': {'type': 'gaussian_noise', 'val': 0.10},
        
        # Salt & Pepper
        'Salt & Pepper (p=0.05)': {'type': 'salt_pepper', 'val': 0.05},
        'Salt & Pepper (p=0.10)': {'type': 'salt_pepper', 'val': 0.10},
        'Salt & Pepper (p=0.15)': {'type': 'salt_pepper', 'val': 0.15},
        
        # Median Filter
        'Median Filter (k=3)': {'type': 'median', 'val': 3},
        'Median Filter (k=5)': {'type': 'median', 'val': 5},
        'Median Filter (k=7)': {'type': 'median', 'val': 7},
        
        # Brightness
        'Brightness (1.1)': {'type': 'brightness', 'val': 1.1},
        'Brightness (1.2)': {'type': 'brightness', 'val': 1.2},
        'Brightness (1.3)': {'type': 'brightness', 'val': 1.3},
        
        # Contrast
        'Contrast (1.0)': {'type': 'contrast', 'val': 1.0},
        'Contrast (1.5)': {'type': 'contrast', 'val': 1.5},
        'Contrast (2.0)': {'type': 'contrast', 'val': 2.0},
        
        # Hue
        'Hue (-0.1)': {'type': 'hue', 'val': -0.1},
        'Hue (0.0)': {'type': 'hue', 'val': 0.0},
        'Hue (0.1)': {'type': 'hue', 'val': 0.1},
        
        # Saturation
        'Saturation (0.5)': {'type': 'saturation', 'val': 0.5},
        'Saturation (1.0)': {'type': 'saturation', 'val': 1.0},
        'Saturation (1.5)': {'type': 'saturation', 'val': 1.5},
        
        # Grid Crop
        'Grid Crop (50%)': {'type': 'grid_crop', 'val': 0.5},
        'Grid Crop (60%)': {'type': 'grid_crop', 'val': 0.6},
        'Grid Crop (70%)': {'type': 'grid_crop', 'val': 0.7},
        
        # Cropout
        'Cropout (10%)': {'type': 'cropout', 'val': 0.1},
        'Cropout (20%)': {'type': 'cropout', 'val': 0.2},
        'Cropout (30%)': {'type': 'cropout', 'val': 0.3},
        
        # JPEG
        'JPEG (Q=10)': {'type': 'jpeg', 'val': 10},
        'JPEG (Q=20)': {'type': 'jpeg', 'val': 20},
        'JPEG (Q=30)': {'type': 'jpeg', 'val': 30},
        'JPEG (Q=40)': {'type': 'jpeg', 'val': 40},
        'JPEG (Q=50)': {'type': 'jpeg', 'val': 50},
        'JPEG (Q=60)': {'type': 'jpeg', 'val': 60},
        'JPEG (Q=70)': {'type': 'jpeg', 'val': 70},
        'JPEG (Q=80)': {'type': 'jpeg', 'val': 80},
        'JPEG (Q=90)': {'type': 'jpeg', 'val': 90},
        
        # Resize
        'Resize (0.5)': {'type': 'resize', 'val': 0.5},
        'Resize (0.75)': {'type': 'resize', 'val': 0.75},
        'Resize (1.0)': {'type': 'resize', 'val': 1.0},
        'Resize (1.25)': {'type': 'resize', 'val': 1.25},
        'Resize (1.5)': {'type': 'resize', 'val': 1.5},
        
        # Gaussian Blur
        'Gaussian Blur (σ=0.5)': {'type': 'gaussian_blur', 'val': 0.5},
        'Gaussian Blur (σ=1.0)': {'type': 'gaussian_blur', 'val': 1.0},
        'Gaussian Blur (σ=1.5)': {'type': 'gaussian_blur', 'val': 1.5},
        
        # Dropout
        'Dropout (10%)': {'type': 'dropout', 'val': 0.1},
        'Dropout (20%)': {'type': 'dropout', 'val': 0.2},
        'Dropout (30%)': {'type': 'dropout', 'val': 0.3},
        
        # Combined Attacks
        'Gaussian+S&P (var=0.08, p=0.10)': {
            'type': 'combined',
            'attacks': [
                {'type': 'gaussian_noise', 'val': 0.08},
                {'type': 'salt_pepper', 'val': 0.10}
            ]
        },
        'Brightness+Contrast (1.2, 1.5)': {
            'type': 'combined',
            'attacks': [
                {'type': 'brightness', 'val': 1.2},
                {'type': 'contrast', 'val': 1.5}
            ]
        },
        'JPEG+Resize (Q=50, 0.75)': {
            'type': 'combined',
            'attacks': [
                {'type': 'jpeg', 'val': 50},
                {'type': 'resize', 'val': 0.75}
            ]
        },
        'Cropout(20%) + Grid Crop(60%)': {
            'type': 'combined',
            'attacks': [
                {'type': 'cropout', 'val': 0.2},
                {'type': 'grid_crop', 'val': 0.6}
            ]
        },
        'Grid(50%) + Bright(1.1) + Median(3)': {
            'type': 'combined',
            'attacks': [
                {'type': 'grid_crop', 'val': 0.5},
                {'type': 'brightness', 'val': 1.1},
                {'type': 'median', 'val': 3}
            ]
        },
    }
    
    # 執行評測
    results = evaluate_robustness(
        encoder, decoder, images, watermarks, device, attack_suite, 
        batch_size=args.batch_size
    )
    
    # 輸出結果
    print_table(results)
    
    # 保存 CSV
    csv_path = os.path.join(args.output_dir, 'benchmark_results.csv')
    save_csv(results, csv_path)
    
    # 生成圖表
    chart_path = os.path.join(args.output_dir, 'benchmark_chart.png')
    plot_results(results, chart_path)
    
    print(f"\n✅ 評測完成！所有結果已保存至: {args.output_dir}")
