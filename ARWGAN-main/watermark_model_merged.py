"""
ARWGAN Merged Model - 合併版
============================
此檔案將已驗證可收斂的 ARWGAN 原始模型架構，
移植至改進的訓練框架（漸進式 Noise Layer、Warm-up、Checkpointing）。

【架構來源】
- Encoder/Decoder/Discriminator: model/encoder.py, model/decoder.py, model/discriminator.py
- Dense Block (Bottleneck): model/Dense_block.py

【訓練框架來源】
- NoiseLayer (漸進攻擊): watermark_model_better.py
- train_model (Warm-up, Checkpoint, CSV Logging): watermark_model_better.py

【重要適配】
- 原始 ARWGAN 使用圖像範圍 [-1, 1]（PSNR 計算 MAX^2=4）
- 本檔案適配為 [0, 1] 範圍以相容現有 DataLoader 和 Loss
- Encoder 輸出加入 clamp(0, 1)

運行: python watermark_model_merged.py --train --epochs 100 --batch 16 --use_vgg --data-dir data/coco2017
（已啟用混合精度訓練 AMP，batch 16 應可在 24GB GPU 運行；若仍 OOM，請改用 --batch 8）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
import torchvision
import numpy as np
import random
import os
import csv
import time
import multiprocessing
from pathlib import Path
from PIL import Image


# ============================================================
# Dense Block (Bottleneck) - 來自 model/Dense_block.py
# ============================================================
class Bottleneck(nn.Module):
    """
    DenseNet 風格的 Bottleneck 模組
    - 1x1 Conv 降維 → 3x3 Conv 特徵提取
    - 支援 Dense Connection (last=False) 或單獨輸出 (last=True)
    """
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3, padding=1, bias=False)

    def forward(self, x, last=False):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        if last:
            return out
        else:
            return torch.cat((x, out), 1)


# ============================================================
# Encoder - 來自 model/encoder.py (已適配)
# ============================================================
class Encoder(nn.Module):
    """
    ARWGAN 原始 Encoder 架構
    
    特點：
    - Dense Connection 保留多層特徵
    - Attention Mask (Softmax) 引導嵌入位置
    - 殘差連接 (im_w + image)
    
    適配：
    - 移除 HiDDenConfiguration 依賴
    - 輸出 clamp 到 [0, 1] 以相容現有訓練流程
      （原始 ARWGAN 使用 [-1, 1] 範圍）
    """
    
    def conv2(self, in_channel, out_channel):
        return nn.Conv2d(in_channels=in_channel,
                         out_channels=out_channel,
                         stride=1,
                         kernel_size=3,
                         padding=1)

    def __init__(self, watermark_bits=64, channels=64):
        super(Encoder, self).__init__()
        self.watermark_bits = watermark_bits
        self.conv_channels = channels
        
        # 第一層：提取初始特徵
        self.first_layer = nn.Sequential(
            self.conv2(3, self.conv_channels)
        )

        self.second_layer = nn.Sequential(
            self.conv2(self.conv_channels, self.conv_channels),
            nn.BatchNorm2d(self.conv_channels),
            nn.LeakyReLU(inplace=True),
        )

        self.third_layer = nn.Sequential(
            self.conv2(self.conv_channels * 2, self.conv_channels),
            nn.BatchNorm2d(self.conv_channels),
            nn.LeakyReLU(inplace=True),
            self.conv2(self.conv_channels, self.conv_channels),
            nn.BatchNorm2d(self.conv_channels),
            nn.LeakyReLU(inplace=True),
        )

        self.fourth_layer = nn.Sequential(
            self.conv2(self.conv_channels * 3 + watermark_bits, self.conv_channels),
            nn.BatchNorm2d(self.conv_channels),
            nn.LeakyReLU(inplace=True)
        )

        # Dense Blocks（融合 watermark）
        self.Dense_block1 = Bottleneck(self.conv_channels + watermark_bits, self.conv_channels)
        self.Dense_block2 = Bottleneck(self.conv_channels * 2 + watermark_bits, self.conv_channels)
        self.Dense_block3 = Bottleneck(self.conv_channels * 3 + watermark_bits, self.conv_channels)
        
        # Dense Blocks（Attention 分支）
        self.Dense_block_a1 = Bottleneck(self.conv_channels, self.conv_channels)
        self.Dense_block_a2 = Bottleneck(self.conv_channels * 2, self.conv_channels)
        self.Dense_block_a3 = Bottleneck(self.conv_channels * 3, self.conv_channels)

        # 第五層：生成 watermark 特徵
        self.fifth_layer = nn.Sequential(
            nn.BatchNorm2d(self.conv_channels + watermark_bits),
            nn.ReLU(inplace=True),
            self.conv2(self.conv_channels + watermark_bits, self.conv_channels),
            nn.BatchNorm2d(self.conv_channels),
            nn.ReLU(inplace=True),
            self.conv2(self.conv_channels, watermark_bits),
        )
        
        # 第六層：Attention Mask 生成
        self.sixth_layer = nn.Sequential(
            nn.BatchNorm2d(self.conv_channels),
            nn.ReLU(inplace=True),
            self.conv2(self.conv_channels, self.conv_channels),
            nn.BatchNorm2d(self.conv_channels),
            nn.ReLU(inplace=True),
            self.conv2(self.conv_channels, watermark_bits),
            nn.Softmax(dim=1)  # Attention mask: 學習嵌入位置
        )

        # 最終層：從 watermark 特徵生成 RGB 殘差
        self.final_layer = nn.Sequential(
            nn.Conv2d(watermark_bits, 3, kernel_size=3, padding=1),
        )

    def forward(self, image, message):
        """
        Args:
            image: (B, 3, H, W) 範圍 [0, 1]
            message: (B, watermark_bits) 二進制浮水印
        Returns:
            watermarked_image: (B, 3, H, W) 範圍 [0, 1]
        """
        H, W = image.size()[2], image.size()[3]

        # 擴展 message 到空間維度
        expanded_message = message.unsqueeze(-1).unsqueeze(-1)
        expanded_message = expanded_message.expand(-1, -1, H, W)

        # 主幹：Dense Connection + Watermark 融合
        feature0 = self.first_layer(image)
        feature1 = self.Dense_block1(torch.cat((feature0, expanded_message), 1), last=True)
        feature2 = self.Dense_block2(torch.cat((feature0, expanded_message, feature1), 1), last=True)
        feature3 = self.Dense_block3(torch.cat((feature0, expanded_message, feature1, feature2), 1), last=True)
        feature3 = self.fifth_layer(torch.cat((feature3, expanded_message), 1))
        
        # Attention 分支：學習嵌入位置
        feature_attention = self.Dense_block_a3(
            self.Dense_block_a2(
                self.Dense_block_a1(feature0)
            ), 
            last=True
        )
        # Attention mask * 30 放大（原始 ARWGAN 設計）
        feature_mask = self.sixth_layer(feature_attention) * 30
        
        # 特徵 × Attention Mask
        feature = feature3 * feature_mask
        
        # 生成 RGB 殘差並加到原圖
        im_w = self.final_layer(feature)
        im_w = im_w + image
        
        # ============================================================
        # 【適配】Clamp 到 [0, 1]
        # 原始 ARWGAN 使用 [-1, 1] 範圍（無 clamp）
        # 為了相容現有訓練框架，在此加入 clamp
        # ============================================================
        clamped = torch.clamp(im_w, 0, 1)
        return clamped


# ============================================================
# Decoder - 來自 model/decoder.py (已適配)
# ============================================================
class Decoder(nn.Module):
    """
    ARWGAN 原始 Decoder 架構
    
    特點：
    - Dense Connection 保留多層特徵
    - AdaptiveAvgPool + Linear 輸出 logits
    
    輸出：
    - extracted: 二值化後的 watermark (B, watermark_bits)
    - logits: 原始 logits，用於 BCE Loss (B, watermark_bits)
    """
    
    def conv2(self, in_channel, out_channel):
        return nn.Conv2d(in_channels=in_channel,
                         out_channels=out_channel,
                         stride=1,
                         kernel_size=3,
                         padding=1)

    def __init__(self, watermark_bits=64, channels=64):
        super(Decoder, self).__init__()
        self.watermark_bits = watermark_bits
        self.channels = channels

        self.first_layer = nn.Sequential(
            self.conv2(3, self.channels),
            nn.BatchNorm2d(self.channels),
            nn.LeakyReLU(inplace=True)
        )

        self.second_layer = nn.Sequential(
            self.conv2(self.channels, self.channels),
            nn.BatchNorm2d(self.channels),
            nn.LeakyReLU(inplace=True)
        )

        self.third_layer = nn.Sequential(
            self.conv2(self.channels * 2, self.channels),
            nn.BatchNorm2d(self.channels),
            nn.LeakyReLU(inplace=True)
        )

        self.fourth_layer = nn.Sequential(
            self.conv2(self.channels * 3, self.channels),
            nn.BatchNorm2d(self.channels),
            nn.LeakyReLU(inplace=True)
        )

        # Dense Blocks
        self.Dense_block1 = Bottleneck(self.channels, self.channels)
        self.Dense_block2 = Bottleneck(self.channels * 2, self.channels)
        self.Dense_block3 = Bottleneck(self.channels * 3, self.channels)

        self.fifth_layer = nn.Sequential(
            self.conv2(self.channels, watermark_bits),
            nn.BatchNorm2d(watermark_bits),
            nn.ReLU(inplace=True)
        )

        self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear(watermark_bits, watermark_bits)

    def forward(self, image_with_wm):
        """
        Args:
            image_with_wm: (B, 3, H, W) 含浮水印的圖像
        Returns:
            extracted: (B, watermark_bits) 二值化後的 watermark
            logits: (B, watermark_bits) 原始 logits（用於 BCE Loss）
        """
        feature0 = self.first_layer(image_with_wm)
        feature1 = self.second_layer(feature0)
        feature2 = self.third_layer(torch.cat([feature0, feature1], dim=1))
        feature3 = self.fourth_layer(torch.cat([feature0, feature1, feature2], dim=1))
        x = self.fifth_layer(feature3)
        x = self.pooling(x)
        logits = self.linear(x.squeeze(3).squeeze(2))
        
        # 二值化輸出（用於計算 BER）
        extracted = (torch.sigmoid(logits) > 0.5).float()
        
        return extracted, logits


# ============================================================
# Discriminator - 來自 model/discriminator.py (已適配)
# ============================================================
class Discriminator(nn.Module):
    """
    ARWGAN 原始 Discriminator 架構
    
    特點：
    - Dense Connection
    - AdaptiveAvgPool + Linear 輸出單一 scalar
    - 適用於 WGAN-GP 或標準 GAN
    """
    
    def conv2(self, in_channel, out_channel):
        return nn.Conv2d(in_channels=in_channel,
                         out_channels=out_channel,
                         stride=1,
                         kernel_size=3,
                         padding=1)

    def __init__(self, watermark_bits=64, channels=64):
        super(Discriminator, self).__init__()
        self.channels = channels

        self.first_layer = nn.Sequential(
            self.conv2(3, self.channels),
            nn.BatchNorm2d(self.channels),
            nn.LeakyReLU(inplace=True)
        )

        self.second_layer = nn.Sequential(
            self.conv2(self.channels, self.channels),
            nn.BatchNorm2d(self.channels),
            nn.LeakyReLU(inplace=True)
        )

        self.third_layer = nn.Sequential(
            self.conv2(self.channels * 2, self.channels),
            nn.BatchNorm2d(self.channels),
            nn.LeakyReLU(inplace=True)
        )

        self.fourth_layer = nn.Sequential(
            self.conv2(self.channels * 3, self.channels),
            nn.BatchNorm2d(self.channels),
            nn.LeakyReLU(inplace=True)
        )

        # Dense Blocks
        self.Dense_block1 = Bottleneck(self.channels, self.channels)
        self.Dense_block2 = Bottleneck(self.channels * 2, self.channels)
        self.Dense_block3 = Bottleneck(self.channels * 3, self.channels)
        
        self.fifth_layer = nn.Sequential(
            self.conv2(self.channels, watermark_bits),
            nn.BatchNorm2d(watermark_bits),
            nn.LeakyReLU(inplace=True)
        )

        self.average = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear(watermark_bits, 1)

    def forward(self, image):
        """
        Args:
            image: (B, 3, H, W)
        Returns:
            scalar: 判別分數（用於 GAN Loss）
        """
        feature0 = self.first_layer(image)
        feature1 = self.second_layer(feature0)
        feature2 = self.third_layer(torch.cat([feature0, feature1], dim=1))
        feature3 = self.fourth_layer(torch.cat([feature0, feature1, feature2], dim=1))
        x = self.fifth_layer(feature3)
        x = self.average(x)
        x = x.squeeze(3).squeeze(2)
        x = self.linear(x)
        result = x.mean()  # 返回 batch 平均（WGAN 風格）
        return result


# ============================================================
# 以下為 watermark_model_better.py 的訓練框架（保留）
# ============================================================

# JPEG 噪聲模擬器
class JPEGNoiseSimulator(nn.Module):
    """
    簡化的 JPEG 壓縮模擬器（高斯噪聲模擬）
    """
    def __init__(self, device):
        super(JPEGNoiseSimulator, self).__init__()
        self.device = device
    
    def forward(self, x, quality_factor=50):
        quality_scale = (100 - quality_factor) / 100.0
        noise_std = 0.02 + quality_scale * 0.08
        noised = x + torch.randn_like(x) * noise_std
        return torch.clamp(noised, 0, 1)


# Noise Layer（漸進式攻擊）
class NoiseLayer(nn.Module):
    """
    漸進式 Noise Layer
    - warmup_epochs: 前 N 個 epoch 關閉攻擊
    - noise_ramp_epochs: 攻擊強度線性增長
    """
    def __init__(self, device, attacks=['gaussian', 'jpeg', 'crop', 'dropout', 'resize'], warmup_epochs=5):
        super(NoiseLayer, self).__init__()
        self.attacks = attacks
        self.device = device
        self.jpeg_simulator = JPEGNoiseSimulator(device)
        self.warmup_epochs = warmup_epochs
        self.noise_ramp_epochs = 10
        self.current_epoch = 0
        self.enable_attacks = False
        self.attack_prob = 0.0

    def gaussian_noise(self, x, std=0.05):
        noise = torch.randn_like(x) * std
        return torch.clamp(x + noise, 0, 1)

    def jpeg_compression(self, x, quality=50):
        return self.jpeg_simulator(x, quality_factor=quality)

    def crop(self, x, ratio=0.1):
        B, C, H, W = x.shape
        crop_h = max(1, int(H * ratio))
        crop_w = max(1, int(W * ratio))
        start_h = random.randint(0, max(0, H - crop_h))
        start_w = random.randint(0, max(0, W - crop_w))
        cropped = x[:, :, start_h:start_h+crop_h, start_w:start_w+crop_w]
        pad_left, pad_right = start_w, W - start_w - crop_w
        pad_top, pad_bottom = start_h, H - start_h - crop_h
        padded = F.pad(cropped, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        assert padded.shape == (B, C, H, W), f"crop pad shape mismatch: got {padded.shape}, expected ({B}, {C}, {H}, {W})"
        return padded

    def dropout(self, x, original_image, ratio=0.1):
        B, C, H, W = x.shape
        block_h = max(1, int(H * ratio))
        block_w = max(1, int(W * ratio))
        start_h = random.randint(0, max(1, H - block_h))
        start_w = random.randint(0, max(1, W - block_w))
        x_clone = x.clone()
        x_clone[:, :, start_h:start_h+block_h, start_w:start_w+block_w] = original_image[:, :, start_h:start_h+block_h, start_w:start_w+block_w]
        return x_clone

    def resize(self, x, scale=0.5):
        return F.interpolate(F.interpolate(x, scale_factor=scale, mode='bicubic', align_corners=False), 
                           size=x.shape[2:], mode='bicubic', align_corners=False)

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        self.enable_attacks = (epoch >= self.warmup_epochs)
        if epoch < self.warmup_epochs:
            self.attack_prob = 0.0
        else:
            ramp = (epoch - self.warmup_epochs) / max(1, self.noise_ramp_epochs)
            self.attack_prob = min(1.0, ramp)
    
    def forward(self, x, original_image=None):
        if not self.enable_attacks:
            return x
        if random.random() >= self.attack_prob:
            return x
        attack = random.choice(self.attacks)
        if attack == 'gaussian':
            return self.gaussian_noise(x)
        elif attack == 'jpeg':
            return self.jpeg_compression(x)
        elif attack == 'crop':
            return self.crop(x)
        elif attack == 'dropout' and original_image is not None:
            return self.dropout(x, original_image)
        elif attack == 'resize':
            return self.resize(x)
        return x


# VGG 感知損失
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        # 直接使用 features 的前 16 個層（避免 children() 可能的問題）
        # VGG16 features 結構：Conv2d -> ReLU -> Conv2d -> ReLU -> MaxPool2d -> ... (共 30 層)
        # 前 16 層對應到第 3 個 block 的 ReLU（feature map size: 64x64）
        self.vgg_layers = nn.Sequential(*list(vgg16.features)[:16])
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

    def forward(self, x):
        x_norm = (x - self.mean) / self.std
        return self.vgg_layers(x_norm)


# SSIM Loss
def ssim_loss(img1, img2):
    mu1 = F.avg_pool2d(img1, 11, stride=1, padding=5)
    mu2 = F.avg_pool2d(img2, 11, stride=1, padding=5)
    sigma1_sq = F.avg_pool2d(img1**2, 11, stride=1, padding=5) - mu1**2
    sigma2_sq = F.avg_pool2d(img2**2, 11, stride=1, padding=5) - mu2**2
    sigma12 = F.avg_pool2d(img1*img2, 11, stride=1, padding=5) - mu1*mu2
    C1, C2 = 0.01**2, 0.03**2
    sigma1_sq = sigma1_sq.clamp(min=0)
    sigma2_sq = sigma2_sq.clamp(min=0)
    ssim = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8)
    return 1 - ssim.mean()


# WGAN-GP Loss
def wgan_gp_loss(discriminator, real_imgs, fake_imgs, lambda_gp=10):
    """
    計算 WGAN-GP 梯度懲罰
    注意：確保所有 tensor 在同一設備上，避免多進程下的記憶體問題
    """
    batch_size = real_imgs.size(0)
    device = real_imgs.device
    
    # 確保所有 tensor 在同一設備上
    alpha = torch.rand(batch_size, 1, 1, 1, device=device, requires_grad=False)
    alpha = alpha.expand_as(real_imgs)
    
    # 創建插值樣本
    interpolates = alpha * real_imgs + (1 - alpha) * fake_imgs
    interpolates = interpolates.requires_grad_(True)
    
    # 計算判別器輸出
    disc_interpolates = discriminator(interpolates)
    
    # Discriminator 返回 scalar (0-d tensor)，grad_outputs 也必須是 scalar
    # 使用 torch.tensor(1.0) 而不是 torch.ones(1) 來創建 scalar
    if disc_interpolates.dim() == 0:
        # scalar tensor，grad_outputs 也必須是 scalar
        grad_outputs = torch.tensor(1.0, device=device, requires_grad=False)
    else:
        grad_outputs = torch.ones_like(disc_interpolates)
    
    # 計算梯度
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # 計算梯度懲罰
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    
    return gradient_penalty


# CSV 記錄
def write_losses_to_csv(file_name, losses_dict, epoch, duration):
    file_exists = os.path.exists(file_name) and os.path.getsize(file_name) > 0
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            row_to_write = ['epoch'] + list(losses_dict.keys()) + ['duration']
            writer.writerow(row_to_write)
        row_to_write = [epoch] + ['{:.4f}'.format(v) for v in losses_dict.values()] + ['{:.0f}'.format(duration)]
        writer.writerow(row_to_write)


# Dataset
TARGET_IMAGE_SIZE = (256, 256)

class WatermarkDataset(Dataset):
    def __init__(self, root_dir='./data/coco/images/train2017', transform=None, watermark_bits=64):
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize(TARGET_IMAGE_SIZE),
            transforms.ToTensor(),
        ])
        
        if not os.path.exists(root_dir):
            raise ValueError(f"數據集目錄不存在: {root_dir}")
        
        if not os.path.isdir(root_dir):
            raise ValueError(f"指定的路徑不是目錄: {root_dir}")
        
        self.image_list = []
        
        try:
            all_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for f in all_files:
                img_path = os.path.join(root_dir, f)
                if os.path.exists(img_path):
                    self.image_list.append(f)
        except (OSError, PermissionError):
            pass
        
        if len(self.image_list) == 0:
            common_subdirs = ['train/images', 'train', 'images', 'train2017', 'val/images', 'val']
            for subdir in common_subdirs:
                search_path = os.path.join(root_dir, subdir)
                if os.path.exists(search_path) and os.path.isdir(search_path):
                    try:
                        files = [f for f in os.listdir(search_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        for f in files:
                            img_path = os.path.join(search_path, f)
                            if os.path.exists(img_path):
                                self.image_list.append(f)
                        if len(self.image_list) > 0:
                            self.root_dir = search_path
                            print(f"在子目錄 {subdir} 中找到圖片文件")
                            break
                    except (OSError, PermissionError):
                        continue
        
        if len(self.image_list) == 0:
            print(f"在 {root_dir} 中未找到圖片，開始遞迴搜索...")
            for root, dirs, files in os.walk(root_dir):
                for f in files:
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(root, f)
                        if os.path.exists(img_path):
                            rel_path = os.path.relpath(img_path, root_dir)
                            self.image_list.append(rel_path)
                if len(self.image_list) > 100:
                    break
        
        if len(self.image_list) == 0:
            raise ValueError(f"在 {root_dir} 中找不到有效的圖片文件！")
        
        self.image_list = [f for f in self.image_list if f and isinstance(f, str) and len(f) > 0]
        
        if len(self.image_list) == 0:
            raise ValueError(f"過濾後沒有有效的圖片文件！")
        
        print(f"找到 {len(self.image_list)} 個有效的圖片文件（在 {self.root_dir}）")
        self.watermark_bits = watermark_bits

    def __len__(self):
        return len(self.image_list)

    def _ensure_size(self, tensor):
        if tensor.shape[-2:] != TARGET_IMAGE_SIZE:
            tensor = F.interpolate(
                tensor.unsqueeze(0), size=TARGET_IMAGE_SIZE, mode='bilinear', align_corners=False
            ).squeeze(0)
        return tensor

    def __getitem__(self, idx):
        max_retries = 10
        
        for attempt in range(max_retries):
            try:
                current_idx = idx if attempt == 0 else random.randint(0, len(self.image_list) - 1)
                img_file = self.image_list[current_idx]
                
                if not isinstance(img_file, str):
                    img_file = str(img_file) if img_file is not None else ""
                
                if not img_file:
                    continue
                
                if os.path.isabs(img_file):
                    img_path = img_file
                else:
                    img_path = os.path.join(self.root_dir, img_file)
                
                if not isinstance(img_path, str):
                    img_path = str(img_path)
                
                image = Image.open(img_path)
                image = image.convert('RGB')
                image.load()
                
                if image.size is None or image.size[0] <= 0 or image.size[1] <= 0:
                    raise ValueError(f"無效的圖片尺寸")
                
                image_tensor = self.transform(image)
                image.close()
                del image
                image_tensor = self._ensure_size(image_tensor)
                image_tensor = image_tensor.clone()
                
                watermark = torch.randint(0, 2, (self.watermark_bits,)).float()
                return image_tensor, watermark
                
            except Exception:
                continue
        
        fallback_image = Image.new('RGB', TARGET_IMAGE_SIZE, color=(0, 0, 0))
        image_tensor = self.transform(fallback_image)
        fallback_image = None
        image_tensor = self._ensure_size(image_tensor)
        watermark = torch.randint(0, 2, (self.watermark_bits,)).float()
        return image_tensor.clone(), watermark


# ============================================================
# Training Function（改進版）
# ============================================================
def train_model(epochs=100, batch_size=16, lr=None, device='cuda', 
                save_dir='./checkpoints_merged', use_vgg=True, resume_from_checkpoint=None,
                data_dir=None, watermark_bits=64, channels=64):
    """
    訓練 ARWGAN 合併模型
    
    Args:
        epochs: 訓練 epochs
        batch_size: Batch size
        device: 'cuda' or 'cpu'
        save_dir: Checkpoint 保存目錄
        use_vgg: 是否使用 VGG 感知損失
        resume_from_checkpoint: 從檢查點恢復訓練
        data_dir: 數據集目錄
        watermark_bits: 浮水印位數（默認 64）
        channels: 模型通道數（默認 64）
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    train_csv_path = save_dir / 'train.csv'
    validation_csv_path = save_dir / 'validation.csv'
    
    # 自動檢測數據集路徑
    if data_dir is None:
        possible_paths = [
            './data/coco2017/train/images',
            './data/coco/images/train2017',
            './data/train',
            './data/coco/train',
        ]
        data_dir = None
        for path in possible_paths:
            if os.path.exists(path) and os.path.isdir(path):
                try:
                    files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if len(files) > 0:
                        data_dir = path
                        print(f"自動檢測到數據集路徑: {data_dir} (找到 {len(files)} 個圖片文件)")
                        break
                except (OSError, PermissionError):
                    continue
        
        if data_dir is None:
            raise ValueError(f"無法找到有效的數據集！請使用 --data-dir 參數指定數據集路徑。")
    else:
        if not os.path.exists(data_dir):
            raise ValueError(f"指定的數據集路徑不存在: {data_dir}")
        print(f"使用指定的數據集路徑: {data_dir}")
    
    # 資料集
    dataset = WatermarkDataset(root_dir=data_dir, watermark_bits=watermark_bits)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    num_workers = 4
    pin_memory = torch.cuda.is_available()
    use_persistent_workers = num_workers > 0
    mp_context = multiprocessing.get_context('fork') if num_workers > 0 else None
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=use_persistent_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        multiprocessing_context=mp_context,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=use_persistent_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        multiprocessing_context=mp_context,
    )
    print(f"DataLoader 配置: num_workers={num_workers}, pin_memory={pin_memory}, multiprocessing_context=fork")

    # 階段常數（與 NoiseLayer.warmup_epochs 一致，避免重複定義）
    NOISE_WARMUP_EPOCHS = 5
    GAN_WARMUP_EPOCHS = 15

    # ============================================================
    # 模型初始化（使用原始 ARWGAN 架構）
    # ============================================================
    print(f"\n初始化 ARWGAN 原始架構模型...")
    print(f"  - watermark_bits: {watermark_bits}")
    print(f"  - channels: {channels}")
    
    encoder = Encoder(watermark_bits=watermark_bits, channels=channels).to(device)
    noise_layer = NoiseLayer(device, warmup_epochs=NOISE_WARMUP_EPOCHS).to(device)
    decoder = Decoder(watermark_bits=watermark_bits, channels=channels).to(device)
    discriminator = Discriminator(watermark_bits=watermark_bits, channels=channels).to(device)
    
    # 計算模型參數量
    enc_params = sum(p.numel() for p in encoder.parameters())
    dec_params = sum(p.numel() for p in decoder.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())
    print(f"  - Encoder 參數量: {enc_params:,}")
    print(f"  - Decoder 參數量: {dec_params:,}")
    print(f"  - Discriminator 參數量: {disc_params:,}")
    print(f"  - 總參數量: {enc_params + dec_params + disc_params:,}\n")
    
    # VGG Loss
    vgg_loss_fn = VGGLoss().to(device) if use_vgg else None

    # 優化器（差分學習率）
    opt_gen = optim.Adam([
        {'params': encoder.parameters(), 'lr': 1e-4},
        {'params': decoder.parameters(), 'lr': 1e-3}
    ], betas=(0.5, 0.999))
    opt_disc = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    
    # 學習率調度器
    scheduler_gen = optim.lr_scheduler.StepLR(opt_gen, step_size=30, gamma=0.5)
    scheduler_disc = optim.lr_scheduler.StepLR(opt_disc, step_size=30, gamma=0.5)
    
    # 混合精度訓練 (AMP) - 降低顯存使用，允許使用更大的 batch size
    scaler_gen = GradScaler()
    scaler_disc = GradScaler()
    
    # 損失函數
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    
    # 從檢查點恢復
    start_epoch = 0
    best_val_ber = float('inf')
    patience_counter = 0

    if resume_from_checkpoint is not None and Path(resume_from_checkpoint).exists():
        try:
            print(f"\n從檢查點恢復訓練: {resume_from_checkpoint}")
            checkpoint = torch.load(resume_from_checkpoint, map_location=device, weights_only=False)
            
            required_keys = ['encoder_state_dict', 'decoder_state_dict', 'discriminator_state_dict', 'epoch']
            missing_keys = [k for k in required_keys if k not in checkpoint]
            if missing_keys:
                raise KeyError(f"Checkpoint 缺少必要的鍵: {missing_keys}")
            
            encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
            decoder.load_state_dict(checkpoint['decoder_state_dict'], strict=False)
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'], strict=False)
            
            if 'opt_gen_state_dict' in checkpoint:
                try:
                    opt_gen.load_state_dict(checkpoint['opt_gen_state_dict'])
                except Exception as e:
                    print(f"⚠️  優化器狀態載入失敗，將重新初始化: {e}")
            if 'opt_disc_state_dict' in checkpoint:
                try:
                    opt_disc.load_state_dict(checkpoint['opt_disc_state_dict'])
                except Exception as e:
                    print(f"⚠️  Discriminator 優化器狀態載入失敗: {e}")
            
            if 'scheduler_gen_state_dict' in checkpoint:
                try:
                    scheduler_gen.load_state_dict(checkpoint['scheduler_gen_state_dict'])
                except Exception:
                    pass
            if 'scheduler_disc_state_dict' in checkpoint:
                try:
                    scheduler_disc.load_state_dict(checkpoint['scheduler_disc_state_dict'])
                except Exception:
                    pass
            
            # 恢復混合精度 scaler 狀態（如果存在）
            # 注意：舊的檢查點可能沒有 scaler 狀態（AMP 是後來加入的），這是正常的
            scaler_loaded = False
            if 'scaler_gen_state_dict' in checkpoint:
                try:
                    scaler_gen.load_state_dict(checkpoint['scaler_gen_state_dict'])
                    scaler_loaded = True
                except Exception:
                    print(f"⚠️  Generator scaler 狀態載入失敗，將重新初始化")
            if 'scaler_disc_state_dict' in checkpoint:
                try:
                    scaler_disc.load_state_dict(checkpoint['scaler_disc_state_dict'])
                    scaler_loaded = True
                except Exception:
                    print(f"⚠️  Discriminator scaler 狀態載入失敗，將重新初始化")
            
            if not scaler_loaded:
                print(f"ℹ️  檢查點中沒有 AMP scaler 狀態（可能是舊版本保存的）")
                print(f"   GradScaler 將從初始狀態開始，會自動適應訓練過程")
            
            start_epoch = checkpoint['epoch'] + 1
            patience_counter = checkpoint.get('patience_counter', 0)

            # Unleash Strategy: 如果從 Epoch >= 30 恢復，重置最佳紀錄
            # 因為釋放期的目標（高 PSNR）與前期不同，不應與 Phase 2 的最佳 BER 比較
            UNLEASH_EPOCH = 30
            if start_epoch >= UNLEASH_EPOCH:
                print(f"🚀 檢測到釋放期（Epoch >= {UNLEASH_EPOCH}），重置最佳紀錄以適應新的權重策略")
                best_val_ber = float('inf')
                patience_counter = 0
                print(f"   - best_val_ber 已重置為 inf")
                print(f"   - patience_counter 已重置為 0")
            elif 'best_val_ber' in checkpoint:
                best_val_ber = checkpoint['best_val_ber']
            # patience_counter 已於上方由 checkpoint.get('patience_counter', 0) 載入

            print(f"✓ 已恢復到 Epoch {start_epoch}")
            ber_str = f"{best_val_ber:.4f}" if best_val_ber != float('inf') else "inf (已重置)"
            print(f"✓ 最佳驗證 BER: {ber_str}\n")
            
        except Exception as e:
            print(f"⚠️  載入 checkpoint 失敗: {e}")
            print("   將從頭開始訓練...\n")
            start_epoch = 0
            best_val_ber = float('inf')
            patience_counter = 0
            
    elif resume_from_checkpoint is not None:
        print(f"⚠️  警告: 檢查點文件不存在: {resume_from_checkpoint}")
        print("   將從頭開始訓練...\n")
    
    print(f"開始訓練... 訓練集: {train_size}, 驗證集: {val_size}")
    if start_epoch > 0:
        print(f"從 Epoch {start_epoch} 繼續訓練，總共 {epochs} epochs\n")

    # ============================================================
    # 階段式 Warm-up 設定（NOISE_WARMUP_EPOCHS, GAN_WARMUP_EPOCHS 已於上方定義）
    # ============================================================
    early_stopping_patience = 15
    if start_epoch == 0:
        patience_counter = 0
    # 若從 checkpoint 恢復，patience_counter 已在 resume 區塊從 checkpoint 載入
    UNLEASH_EPOCH = 30  # 釋放期起始 epoch（resume 時若 start_epoch >= 30 已在上面重置 best_val_ber）

    for epoch in range(start_epoch, epochs):
        noise_layer.set_epoch(epoch)
        gan_enabled = (epoch >= GAN_WARMUP_EPOCHS)
        
        # 顯示當前訓練階段
        if epoch < NOISE_WARMUP_EPOCHS:
            phase_name = "Phase 1: 純通訊系統"
            print(f"🔥 {phase_name} (Epoch {epoch+1}/{NOISE_WARMUP_EPOCHS}): 無 Noise, 無 GAN")
        elif epoch < GAN_WARMUP_EPOCHS:
            phase_name = "Phase 2: 抗攻擊訓練"
            if epoch == NOISE_WARMUP_EPOCHS:
                print(f"\n{'='*60}")
                print(f"✅ Phase 1 完成！從 Epoch {epoch + 1} 開始啟用 Noise Layer 攻擊")
                print(f"{'='*60}\n")
            print(f"🔥 {phase_name} (Epoch {epoch+1}): 有 Noise, 無 GAN")
        else:
            phase_name = "Phase 3: 完整訓練"
            if epoch == GAN_WARMUP_EPOCHS:
                print(f"\n{'='*60}")
                print(f"✅ Phase 2 完成！從 Epoch {epoch + 1} 開始啟用 GAN")
                print(f"{'='*60}\n")
                # 清理 GPU 快取，為 GAN 訓練騰出空間
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    print("🧹 已清理 GPU 快取")
        
        # ============= 訓練階段 =============
        encoder.train()
        decoder.train()
        discriminator.train()
        
        epoch_start_time = time.time()
        train_losses = {'g_loss': 0, 'd_loss': 0, 'ber': 0, 'psnr': 0}
        num_batches = 0
        
        for batch_idx, (images, watermarks) in enumerate(train_loader):
            images, watermarks = images.to(device), watermarks.to(device)
            
            # GAN 訓練時定期清理快取（每 50 batch）
            if gan_enabled and batch_idx % 50 == 0:
                torch.cuda.empty_cache()
            
            # Train Discriminator (WGAN-GP)
            # 注意：梯度懲罰不使用 autocast，避免混合精度造成數值問題
            if gan_enabled:
                for _ in range(1):
                    opt_disc.zero_grad()
                    with autocast():
                        watermarked = encoder(images, watermarks)
                        d_real = discriminator(images)
                        d_fake = discriminator(watermarked.detach())
                    
                    # 梯度懲罰在 autocast 外計算（減少記憶體使用 + 數值穩定）
                    # 確保使用 .float() 避免混合精度問題，並確保在同一設備上
                    gp = wgan_gp_loss(
                        discriminator, 
                        images.float().detach(), 
                        watermarked.detach().float()
                    )
                    d_loss = -d_real + d_fake + gp
                    
                    scaler_disc.scale(d_loss).backward()
                    scaler_disc.step(opt_disc)
                    scaler_disc.update()
                    
                    # 釋放 Discriminator 訓練的中間變數，避免記憶體累積
                    del d_real, d_fake, gp, watermarked
            else:
                d_loss = torch.tensor(0.0, device=device)
            
            # Train Generator - 使用混合精度
            opt_gen.zero_grad()
            with autocast():
                watermarked = encoder(images, watermarks)
                noised = noise_layer(watermarked, original_image=images)
                extracted, logits = decoder(noised)
                
                # Losses
                mse_img_loss = mse_loss(watermarked, images)
                ssim_img_loss = ssim_loss(watermarked, images)
                wm_loss = bce_loss(logits, watermarks)
                
                if gan_enabled:
                    g_gan_loss = -discriminator(watermarked)
                else:
                    g_gan_loss = torch.tensor(0.0, device=device)
                
                if vgg_loss_fn is not None:
                    vgg_real = vgg_loss_fn(images)
                    vgg_fake = vgg_loss_fn(watermarked)
                    vgg_perceptual_loss = mse_loss(vgg_fake, vgg_real)
                    img_loss = 0.5 * mse_img_loss + 0.3 * ssim_img_loss + 0.2 * vgg_perceptual_loss
                else:
                    img_loss = mse_img_loss + ssim_img_loss
                
                # 損失權重排程 (Unleash Strategy + Plan B + Golden Balance)
                if gan_enabled:
                    if epoch < 30:
                        # 高壓期：偏向浮水印
                        current_wm_weight = 8.0
                        current_img_weight = 1.0
                    elif epoch < 50:
                        # 釋放期 I：wm 降權，專注畫質
                        current_wm_weight = 2.0
                        current_img_weight = 1.0
                    elif epoch < 52:
                        # Epoch 50–52: Plan B 強制美顏模式（BER 犧牲換 PSNR）
                        current_wm_weight = 1.0
                        current_img_weight = 5.0
                    else:
                        # Epoch 53+: 黃金平衡 (Golden Balance)
                        # 稍微拉回 BER：提高 wm 權重、降低畫質懲罰
                        current_wm_weight = 4.0
                        current_img_weight = 2.0
                    
                    current_gan_weight = 0.001
                else:
                    # Phase 1-2: 降低 img 權重、拉高 wm，讓 Encoder 敢嵌入、Decoder 能學到，目標 ber_clean < 0.1
                    current_img_weight = 0.3   # 允許改圖，否則 Encoder 被懲罰不敢嵌入，BER 卡在 0.5
                    current_wm_weight = 25.0    # 強迫優先優化 BER，無攻擊時應可降到 ~0.05
                    current_gan_weight = 0.0
                
                g_loss = current_img_weight * img_loss + current_wm_weight * wm_loss + current_gan_weight * g_gan_loss
            
            scaler_gen.scale(g_loss).backward()
            # 梯度裁剪需要在 scaler 的 unscale 之後，但 scaler.scale().backward() 後可以直接 clip
            scaler_gen.unscale_(opt_gen)  # 先 unscale 才能 clip
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            scaler_gen.step(opt_gen)
            scaler_gen.update()
            
            with torch.no_grad():
                # extracted 已經是 0/1 二值化結果，不需要 .round()
                ber = (extracted != watermarks).float().mean().item()
                psnr = 10 * torch.log10(1.0 / mse_img_loss.clamp(min=1e-8)).item()
                
            train_losses['g_loss'] += g_loss.item()
            train_losses['d_loss'] += d_loss.item() if isinstance(d_loss, torch.Tensor) else d_loss
            train_losses['ber'] += ber
            train_losses['psnr'] += psnr
            num_batches += 1
            
            if batch_idx % 50 == 0:
                if gan_enabled:
                    phase_str = "Phase3"
                elif noise_layer.enable_attacks:
                    phase_str = "Phase2"
                else:
                    phase_str = "Phase1"
                print(f"[{phase_str}] Epoch [{epoch}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] "
                      f"G_loss: {g_loss.item():.4f}, D_loss: {d_loss.item() if isinstance(d_loss, torch.Tensor) else d_loss:.4f}, "
                      f"BER: {ber:.4f}, PSNR: {psnr:.2f}dB")
        
        if num_batches > 0:
            for key in train_losses:
                train_losses[key] /= num_batches
        
        train_duration = time.time() - epoch_start_time
        write_losses_to_csv(train_csv_path, train_losses, epoch + 1, train_duration)
        
        # ============= 驗證階段 =============
        encoder.eval()
        decoder.eval()
        discriminator.eval()
        
        val_losses = {'ber': 0, 'ber_clean': 0, 'psnr': 0, 'ssim': 0}
        num_val_batches = 0
        
        with torch.no_grad():
            for images, watermarks in val_loader:
                images, watermarks = images.to(device), watermarks.to(device)
                
                watermarked = encoder(images, watermarks)
                noised = noise_layer(watermarked, original_image=images)
                extracted, _ = decoder(noised)
                extracted_clean, _ = decoder(watermarked)
                
                # extracted 和 extracted_clean 已經是 0/1 二值化結果，不需要 .round()
                ber = (extracted != watermarks).float().mean().item()
                ber_clean = (extracted_clean != watermarks).float().mean().item()
                mse = mse_loss(watermarked, images).item()
                psnr = 10 * np.log10(1.0 / max(mse, 1e-8))
                ssim_val = 1 - ssim_loss(watermarked, images).item()
                
                val_losses['ber'] += ber
                val_losses['ber_clean'] += ber_clean
                val_losses['psnr'] += psnr
                val_losses['ssim'] += ssim_val
                num_val_batches += 1
        
        if num_val_batches > 0:
            for key in val_losses:
                val_losses[key] /= num_val_batches
        
        total_duration = time.time() - epoch_start_time
        write_losses_to_csv(validation_csv_path, val_losses, epoch + 1, total_duration)
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{epochs} 完成")
        print(f"訓練 - G_loss: {train_losses['g_loss']:.4f}, BER: {train_losses['ber']:.4f}, PSNR: {train_losses['psnr']:.2f}dB")
        print(f"驗證 - BER(含攻擊): {val_losses['ber']:.4f}, BER(無攻擊): {val_losses['ber_clean']:.4f}, PSNR: {val_losses['psnr']:.2f}dB, SSIM: {val_losses['ssim']:.4f}")
        print(f"{'='*80}\n")
        
        scheduler_gen.step()
        scheduler_disc.step()
        
        # Unleash Strategy: 在進入釋放期時重置最佳紀錄
        if epoch == UNLEASH_EPOCH:
            print(f"\n{'='*60}")
            print(f"🚀 進入釋放期 (Epoch {epoch + 1})，重置最佳紀錄")
            old_ber_str = f"{best_val_ber:.4f}" if best_val_ber != float('inf') else "inf"
            print(f"   舊最佳 BER: {old_ber_str}")
            best_val_ber = float('inf')
            patience_counter = 0
            print(f"   新最佳 BER: inf (已重置)")
            print(f"   patience_counter: 0 (已重置)")
            print(f"{'='*60}\n")
        
        # 保存最佳模型
        if val_losses['ber'] < best_val_ber:
            best_val_ber = val_losses['ber']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'opt_gen_state_dict': opt_gen.state_dict(),
                'opt_disc_state_dict': opt_disc.state_dict(),
                'best_val_ber': best_val_ber,
                'val_psnr': val_losses['psnr'],
                'val_ssim': val_losses['ssim'],
                'patience_counter': patience_counter,
            }, save_dir / 'best_model.pth')
            ber_str = f"{best_val_ber:.4f}" if best_val_ber != float('inf') else "inf"
            print(f"✓ 保存最佳模型 (BER: {ber_str})")
        else:
            patience_counter += 1
            print(f"⏳ 驗證 BER 未改善 ({patience_counter}/{early_stopping_patience})")
        
        # 每個 epoch 保存 checkpoint（包含 scaler 狀態）
        torch.save({
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'opt_gen_state_dict': opt_gen.state_dict(),
            'opt_disc_state_dict': opt_disc.state_dict(),
            'scheduler_gen_state_dict': scheduler_gen.state_dict(),
            'scheduler_disc_state_dict': scheduler_disc.state_dict(),
            'scaler_gen_state_dict': scaler_gen.state_dict(),
            'scaler_disc_state_dict': scaler_disc.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_ber': best_val_ber,
            'patience_counter': patience_counter,
        }, save_dir / f'checkpoint_epoch_{epoch}.pth')
        print(f"✓ 保存檢查點: checkpoint_epoch_{epoch}.pth")
        
        # 早停
        if epoch >= GAN_WARMUP_EPOCHS and patience_counter >= early_stopping_patience:
            print(f"\n{'='*60}")
            print(f"🛑 早停觸發：驗證 BER 在 {early_stopping_patience} epochs 內未改善")
            ber_str = f"{best_val_ber:.4f}" if best_val_ber != float('inf') else "inf"
            print(f"   最佳 BER: {ber_str}")
            print(f"{'='*60}\n")
            break
    
    print("\n訓練完成！")
    return encoder, decoder, discriminator


# ============================================================
# Test Function
# ============================================================
def test_model(checkpoint_path, image_path, watermark_bits=64, channels=64, device='cuda', save_dir='./test_results'):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder = Encoder(watermark_bits, channels).to(device)
    decoder = Decoder(watermark_bits, channels).to(device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    encoder.eval()
    decoder.eval()
    
    noise_layer = NoiseLayer(device).to(device)
    noise_layer.set_epoch(10)
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    image = transform(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)
    watermark = torch.randint(0, 2, (1, watermark_bits)).float().to(device)
    
    print(f"\n{'='*80}")
    print(f"測試圖像: {image_path}")
    print(f"水印位數: {watermark_bits}")
    print(f"{'='*80}\n")
    
    with torch.no_grad():
        watermarked = encoder(image, watermark)
        
        mse = F.mse_loss(watermarked, image).item()
        psnr = 10 * np.log10(1.0 / max(mse, 1e-8))
        ssim_val = 1 - ssim_loss(watermarked, image).item()
        
        print(f"原始嵌入品質:")
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  SSIM: {ssim_val:.4f}")
        print(f"  MSE:  {mse:.6f}\n")
        
        attacks = ['gaussian', 'jpeg', 'crop', 'dropout', 'resize']
        print(f"攻擊魯棒性測試:")
        print(f"{'-'*80}")
        
        for attack in attacks:
            noise_layer.attacks = [attack]
            noised = noise_layer(watermarked, original_image=image)
            extracted, _ = decoder(noised)
            # extracted 已經是 0/1 二值化結果，不需要 .round()
            ber = (extracted != watermark).float().mean().item()
            print(f"  {attack:15s}: BER = {ber:.4f} ({int(ber * watermark_bits)}/{watermark_bits} bits)")
        
        extracted_clean, _ = decoder(watermarked)
        # extracted_clean 已經是 0/1 二值化結果，不需要 .round()
        ber_clean = (extracted_clean != watermark).float().mean().item()
        print(f"  {'no_attack':15s}: BER = {ber_clean:.4f} ({int(ber_clean * watermark_bits)}/{watermark_bits} bits)")
        print(f"{'-'*80}\n")
        
        transforms.ToPILImage()(watermarked[0].cpu()).save(save_dir / 'watermarked.png')
        transforms.ToPILImage()(image[0].cpu()).save(save_dir / 'original.png')
        print(f"✓ 結果已保存至 {save_dir}")
        
        diff = torch.abs(watermarked - image) * 10
        transforms.ToPILImage()(diff[0].cpu()).save(save_dir / 'difference_x10.png')
        
    return {'psnr': psnr, 'ssim': ssim_val, 'ber_clean': ber_clean}


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ARWGAN 合併版水印模型')
    parser.add_argument('--train', action='store_true', help='訓練模式')
    parser.add_argument('--test', action='store_true', help='測試模式')
    parser.add_argument('--image', type=str, default='test.jpg', help='測試圖像路徑')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints_merged/best_model.pth', help='checkpoint 路徑（測試用）')
    parser.add_argument('--resume', type=str, default=None, help='從檢查點恢復訓練（訓練用）')
    parser.add_argument('--epochs', type=int, default=100, help='訓練 epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size（已啟用混合精度 AMP，24GB GPU 建議 16；若仍 OOM 請改用 8）')
    parser.add_argument('--use_vgg', action='store_true', help='使用 VGG 感知損失')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_merged', help='模型保存目錄')
    parser.add_argument('--data-dir', type=str, default=None, help='數據集目錄路徑')
    parser.add_argument('--watermark-bits', type=int, default=64, help='浮水印位數')
    parser.add_argument('--channels', type=int, default=64, help='模型通道數')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用設備: {device}")

    if args.train:
        print("\n" + "="*60)
        print("ARWGAN 合併版 - 原始架構 + 改進訓練框架")
        print("="*60 + "\n")
        train_model(
            epochs=args.epochs, 
            batch_size=args.batch, 
            device=device,
            save_dir=args.save_dir,
            use_vgg=args.use_vgg,
            resume_from_checkpoint=args.resume,
            data_dir=args.data_dir,
            watermark_bits=args.watermark_bits,
            channels=args.channels
        )
    
    if args.test:
        print("\n開始測試模型...")
        if not Path(args.checkpoint).exists():
            print(f"錯誤: checkpoint 不存在: {args.checkpoint}")
        else:
            test_model(
                checkpoint_path=args.checkpoint,
                image_path=args.image,
                watermark_bits=args.watermark_bits,
                channels=args.channels,
                device=device
            )
