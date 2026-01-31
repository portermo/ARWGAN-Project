#!/usr/bin/env python3
"""
浮水印模型管線診斷腳本 (debug_pipeline.py)
==============================================
執行一次 Forward-Backward Pass，印出關鍵節點的「生命體徵」。

用途：診斷為何 Encoder 有嵌入訊號但 Decoder 無法解碼 (BER 卡在 0.4+)

執行方式：
    python debug_pipeline.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ============================================================
# 複製模型架構（與 watermark_model_better.py 相同）
# ============================================================

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(in_planes, in_planes // ratio, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_planes // ratio, in_planes, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x).view(x.size(0), -1))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x).view(x.size(0), -1))))
        out = avg_out + max_out
        return self.sigmoid(out).view(x.size(0), x.size(1), 1, 1) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_input = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        return attention * x_input

class CBAM(nn.Module):
    def __init__(self, channels, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channels, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

# 修改版 Encoder：加入中間輸出以供診斷（含特徵正規化修復）
class EncoderDebug(nn.Module):
    def __init__(self, watermark_bits=64):
        super(EncoderDebug, self).__init__()
        self.watermark_bits = watermark_bits
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.dense1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.dense2 = nn.Conv2d(96, 32, kernel_size=3, padding=1)
        self.dense3 = nn.Conv2d(128, 32, kernel_size=3, padding=1)
        self.dense4 = nn.Conv2d(160, 64, kernel_size=3, padding=1)
        
        self.cbam = CBAM(64)
        self.wm_embed = nn.Conv2d(watermark_bits, 64, kernel_size=1)
        
        # 特徵尺度正規化（修復 attended/wm_embedded 尺度不匹配）
        self.bn_attended = nn.BatchNorm2d(64)
        self.bn_wm = nn.BatchNorm2d(64)
        
        self.to_rgb = nn.Conv2d(128, 3, kernel_size=1)
        # 小隨機初始化（不能用零初始化，否則梯度斷裂）
        nn.init.normal_(self.to_rgb.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.to_rgb.bias)
        self.residual_scale = 0.1
        
    def forward(self, image, watermark):
        x = self.relu(self.bn1(self.conv1(image)))
        
        d1 = self.relu(self.dense1(x))
        x = torch.cat([x, d1], dim=1)
        d2 = self.relu(self.dense2(x))
        x = torch.cat([x, d2], dim=1)
        d3 = self.relu(self.dense3(x))
        x = torch.cat([x, d3], dim=1)
        d4 = self.relu(self.dense4(x))
        
        attended = self.cbam(d4)
        
        B, _, H, W = image.shape
        wm_repeated = watermark.unsqueeze(2).unsqueeze(3).repeat(1, 1, H, W)
        wm_embedded = self.wm_embed(wm_repeated.float())
        
        # 特徵正規化
        attended_norm = self.bn_attended(attended)
        wm_embedded_norm = self.bn_wm(wm_embedded)
        
        fused = torch.cat([attended_norm, wm_embedded_norm], dim=1)
        residual = self.to_rgb(fused) * self.residual_scale
        watermarked = image + residual
        
        # 返回中間結果供診斷
        return torch.clamp(watermarked, 0, 1), {
            'attended': attended,
            'wm_embedded': wm_embedded,
            'attended_norm': attended_norm,
            'wm_embedded_norm': wm_embedded_norm,
            'fused': fused,
            'residual': residual
        }

# CNN 分類器架構 Decoder（適合 Image → Bits 任務）
class Decoder(nn.Module):
    def __init__(self, watermark_bits=64):
        super(Decoder, self).__init__()
        self.watermark_bits = watermark_bits
        
        # 連續下採樣 CNN
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, watermark_bits),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = self.features(x)
        pooled = self.global_pool(features)
        pooled = pooled.view(pooled.size(0), -1)
        logits = self.classifier(pooled)
        extracted = (self.sigmoid(logits) > 0.5).float()
        return extracted, logits

# ============================================================
# 診斷函數
# ============================================================

def print_tensor_stats(name, tensor):
    """印出 tensor 的統計數據"""
    if tensor is None:
        print(f"  {name}: None")
        return
    if tensor.grad is not None:
        grad = tensor.grad
        print(f"  {name}:")
        print(f"    值: mean={tensor.mean().item():.6f}, std={tensor.std().item():.6f}, "
              f"min={tensor.min().item():.6f}, max={tensor.max().item():.6f}")
        print(f"    梯度: mean={grad.mean().item():.6f}, std={grad.std().item():.6f}, "
              f"min={grad.min().item():.6f}, max={grad.max().item():.6f}")
    else:
        print(f"  {name}: mean={tensor.mean().item():.6f}, std={tensor.std().item():.6f}, "
              f"min={tensor.min().item():.6f}, max={tensor.max().item():.6f}")

def print_grad_stats(name, param):
    """印出參數的梯度統計"""
    if param.grad is None:
        print(f"  {name}.grad: ❌ None (梯度斷裂!)")
    else:
        grad = param.grad
        is_zero = grad.abs().max().item() < 1e-10
        status = "⚠️ 全為零!" if is_zero else "✅"
        print(f"  {name}.grad: mean={grad.mean().item():.8f}, "
              f"std={grad.std().item():.8f}, max={grad.abs().max().item():.8f} {status}")

def run_diagnostic():
    print("=" * 70)
    print("浮水印模型管線診斷")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用裝置: {device}")
    
    # 初始化模型
    watermark_bits = 64
    encoder = EncoderDebug(watermark_bits).to(device)
    decoder = Decoder(watermark_bits).to(device)
    
    encoder.train()
    decoder.train()
    
    # 隨機生成測試數據
    batch_size = 2
    images = torch.rand(batch_size, 3, 256, 256, device=device)
    watermarks = torch.randint(0, 2, (batch_size, watermark_bits), device=device).float()
    
    print(f"\n輸入數據:")
    print(f"  images: shape={images.shape}, range=[{images.min():.3f}, {images.max():.3f}]")
    print(f"  watermarks: shape={watermarks.shape}, 0的比例={1 - watermarks.mean().item():.2f}, 1的比例={watermarks.mean().item():.2f}")
    
    # ============================================================
    # 1. Forward Pass
    # ============================================================
    print("\n" + "=" * 70)
    print("1. ENCODER 中間輸出檢查")
    print("=" * 70)
    
    watermarked, intermediates = encoder(images, watermarks)
    
    attended = intermediates['attended']
    wm_embedded = intermediates['wm_embedded']
    fused = intermediates['fused']
    residual = intermediates['residual']
    
    print("\n[1.1] Watermark Embedding 檢查 (正規化前):")
    print_tensor_stats("attended (圖像特徵)", attended)
    print_tensor_stats("wm_embedded (浮水印特徵)", wm_embedded)
    
    # 計算比例
    attended_scale = attended.abs().mean().item()
    wm_scale = wm_embedded.abs().mean().item()
    ratio = attended_scale / (wm_scale + 1e-10)
    
    print(f"\n  ⚡ 特徵尺度比 (正規化前): attended/wm_embedded = {ratio:.2f}")
    if ratio > 10:
        print(f"     ⚠️ 警告: 圖像特徵遠大於浮水印特徵，浮水印可能被淹沒!")
    elif ratio < 0.1:
        print(f"     ⚠️ 警告: 浮水印特徵遠大於圖像特徵，可能導致圖像失真過大!")
    else:
        print(f"     ✅ 兩者在同一數量級，融合應該有效")
    
    # 檢查正規化後的特徵
    if 'attended_norm' in intermediates:
        attended_norm = intermediates['attended_norm']
        wm_embedded_norm = intermediates['wm_embedded_norm']
        
        print("\n[1.1b] Watermark Embedding 檢查 (正規化後):")
        print_tensor_stats("attended_norm", attended_norm)
        print_tensor_stats("wm_embedded_norm", wm_embedded_norm)
        
        attended_norm_scale = attended_norm.abs().mean().item()
        wm_norm_scale = wm_embedded_norm.abs().mean().item()
        ratio_norm = attended_norm_scale / (wm_norm_scale + 1e-10)
        
        print(f"\n  ⚡ 特徵尺度比 (正規化後): attended/wm_embedded = {ratio_norm:.2f}")
        if 0.5 < ratio_norm < 2.0:
            print(f"     ✅ 正規化成功！兩者現在在同一尺度")
        else:
            print(f"     ⚠️ 正規化後仍有差異")
    
    print("\n[1.2] Residual 輸出檢查:")
    print_tensor_stats("residual", residual)
    
    residual_energy = residual.abs().mean().item()
    if residual_energy < 0.001:
        print(f"  ⚠️ 警告: Residual 幾乎為零 (mean={residual_energy:.6f})，Encoder 可能沒在修改圖像!")
    else:
        print(f"  ✅ Residual 有非零變化 (mean={residual_energy:.6f})")
    
    print("\n[1.3] Watermarked 輸出:")
    print_tensor_stats("watermarked", watermarked)
    
    # 計算 PSNR
    mse = F.mse_loss(watermarked, images)
    psnr = 10 * torch.log10(1.0 / mse.clamp(min=1e-10))
    print(f"  PSNR: {psnr.item():.2f} dB")
    
    # ============================================================
    # 2. Decoder 檢查
    # ============================================================
    print("\n" + "=" * 70)
    print("2. DECODER 輸出檢查")
    print("=" * 70)
    
    extracted, logits = decoder(watermarked)
    
    print("\n[2.1] Decoder Logits (進入 Sigmoid 前):")
    print_tensor_stats("logits", logits)
    
    logits_range = logits.max().item() - logits.min().item()
    if logits_range < 0.1:
        print(f"  ⚠️ 警告: Logits 範圍太小 ({logits_range:.4f})，Decoder 可能沒學到東西!")
    else:
        print(f"  ✅ Logits 有變化範圍: {logits_range:.4f}")
    
    print("\n[2.2] BER 計算:")
    ber = (extracted != watermarks).float().mean().item()
    print(f"  BER: {ber:.4f}")
    if ber > 0.45:
        print(f"  ⚠️ BER 接近 0.5，幾乎等於隨機猜測!")
    elif ber > 0.3:
        print(f"  ⚠️ BER 偏高，Decoder 學習不佳")
    else:
        print(f"  ✅ BER 在可接受範圍")
    
    # ============================================================
    # 3. Loss 計算
    # ============================================================
    print("\n" + "=" * 70)
    print("3. LOSS 計算")
    print("=" * 70)
    
    mse_loss_fn = nn.MSELoss()
    bce_loss_fn = nn.BCEWithLogitsLoss()
    
    img_loss = mse_loss_fn(watermarked, images)
    wm_loss = bce_loss_fn(logits, watermarks)
    
    # 使用目前的權重設定
    img_weight = 0.01
    wm_weight = 10.0
    
    total_loss = img_weight * img_loss + wm_weight * wm_loss
    
    print(f"\n  img_loss (MSE): {img_loss.item():.6f}")
    print(f"  wm_loss (BCE):  {wm_loss.item():.6f}")
    print(f"  ")
    print(f"  權重設定: img_weight={img_weight}, wm_weight={wm_weight}")
    print(f"  加權後: img_contrib={img_weight * img_loss.item():.6f}, wm_contrib={wm_weight * wm_loss.item():.6f}")
    print(f"  total_loss: {total_loss.item():.6f}")
    
    # ============================================================
    # 4. Backward Pass - 梯度檢查
    # ============================================================
    print("\n" + "=" * 70)
    print("4. 梯度流動檢查 (Gradient Flow)")
    print("=" * 70)
    
    # 清除之前的梯度
    encoder.zero_grad()
    decoder.zero_grad()
    
    # 重新計算 loss 並反向傳播
    watermarked2, _ = encoder(images, watermarks)
    extracted2, logits2 = decoder(watermarked2)
    
    img_loss2 = mse_loss_fn(watermarked2, images)
    wm_loss2 = bce_loss_fn(logits2, watermarks)
    total_loss2 = img_weight * img_loss2 + wm_weight * wm_loss2
    
    total_loss2.backward()
    
    print("\n[4.1] Encoder 關鍵層梯度:")
    print_grad_stats("encoder.to_rgb.weight", encoder.to_rgb.weight)
    print_grad_stats("encoder.wm_embed.weight", encoder.wm_embed.weight)
    print_grad_stats("encoder.conv1.weight", encoder.conv1.weight)
    print_grad_stats("encoder.dense4.weight", encoder.dense4.weight)
    
    print("\n[4.2] Decoder 關鍵層梯度:")
    # 新架構使用 Sequential，取第一個 Conv2d 層
    print_grad_stats("decoder.features[0].weight (第一層Conv)", decoder.features[0].weight)
    print_grad_stats("decoder.classifier[3].weight (輸出層)", decoder.classifier[3].weight)
    
    # ============================================================
    # 5. 額外診斷：檢查浮水印是否真的被嵌入
    # ============================================================
    print("\n" + "=" * 70)
    print("5. 額外診斷：浮水印嵌入驗證")
    print("=" * 70)
    
    # 用不同的浮水印嵌入，看輸出是否不同
    watermarks_alt = 1 - watermarks  # 翻轉浮水印
    watermarked_alt, _ = encoder(images, watermarks_alt)
    
    diff = (watermarked - watermarked_alt).abs()
    print(f"\n  使用不同浮水印時的輸出差異:")
    print(f"    mean diff: {diff.mean().item():.6f}")
    print(f"    max diff:  {diff.max().item():.6f}")
    
    if diff.max().item() < 0.01:
        print(f"  ❌ 嚴重問題: 不同浮水印產生幾乎相同的輸出!")
        print(f"     這代表 Encoder 忽略了浮水印輸入!")
    else:
        print(f"  ✅ 不同浮水印產生不同輸出，Encoder 有在處理浮水印")
    
    # ============================================================
    # 6. 總結
    # ============================================================
    print("\n" + "=" * 70)
    print("診斷總結")
    print("=" * 70)
    
    issues = []
    
    if ratio > 10 or ratio < 0.1:
        issues.append("特徵尺度不匹配")
    if residual_energy < 0.001:
        issues.append("Residual 幾乎為零")
    if logits_range < 0.1:
        issues.append("Decoder logits 無變化")
    if ber > 0.45:
        issues.append("BER 接近隨機")
    if encoder.to_rgb.weight.grad is None or encoder.to_rgb.weight.grad.abs().max().item() < 1e-10:
        issues.append("梯度斷裂或為零")
    if diff.max().item() < 0.01:
        issues.append("Encoder 忽略浮水印")
    
    if issues:
        print(f"\n⚠️ 發現 {len(issues)} 個問題:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("\n✅ 未發現明顯的架構問題")
        print("   如果訓練仍不收斂，問題可能在於:")
        print("   - 學習率設定")
        print("   - Loss 權重平衡")
        print("   - 訓練策略 (如 GAN 干擾)")
    
    print("\n" + "=" * 70)
    print("診斷完成")
    print("=" * 70)

if __name__ == "__main__":
    run_diagnostic()
