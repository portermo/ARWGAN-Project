#!/usr/bin/env python3
"""
Sanity Check: 過擬合測試
目的：驗證 Encoder-Decoder 架構是否能正確傳遞浮水印資訊。
方法：只用 1 個 Batch (16 張圖)，重複訓練 1000 次。
預期：
  - 如果 BER 能降到 0 → 架構正確
  - 如果 BER 卡住 → 程式碼邏輯有 Bug
"""
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from watermark_model_better import Encoder, Decoder

def sanity_check_overfit(iterations=1000, batch_size=16, device='cuda'):
    print("=" * 60)
    print("Sanity Check: 單 Batch 過擬合測試")
    print("=" * 60)
    
    # 初始化模型
    encoder = Encoder(watermark_bits=64).to(device)
    decoder = Decoder(watermark_bits=64).to(device)
    
    # 優化器
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=1e-3  # 較高學習率，加速過擬合
    )
    
    # Loss
    bce_loss = nn.BCEWithLogitsLoss()
    
    # 固定的測試數據（模擬 1 個 Batch）
    torch.manual_seed(42)
    fixed_images = torch.rand(batch_size, 3, 256, 256).to(device)
    fixed_watermarks = torch.randint(0, 2, (batch_size, 64)).float().to(device)
    
    print(f"Batch Size: {batch_size}")
    print(f"Fixed Images shape: {fixed_images.shape}")
    print(f"Fixed Watermarks shape: {fixed_watermarks.shape}")
    print(f"Watermark 範例 (第 0 張): {fixed_watermarks[0][:10].tolist()}...")
    print()
    
    # Sanity Check: Encoder residual_scale
    print(f"[Check] Encoder residual_scale: {encoder.residual_scale}")
    print(f"[Check] Encoder to_rgb.weight mean: {encoder.to_rgb.weight.mean().item():.6f}")
    print()
    
    encoder.train()
    decoder.train()
    
    print("開始過擬合訓練...")
    print("-" * 60)
    
    for i in range(iterations):
        optimizer.zero_grad()
        
        # Forward
        watermarked = encoder(fixed_images, fixed_watermarks)
        extracted, logits = decoder(watermarked)
        
        # Loss
        loss = bce_loss(logits, fixed_watermarks)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # 計算 BER
        with torch.no_grad():
            ber = (extracted != fixed_watermarks).float().mean().item()
            psnr = 10 * torch.log10(1.0 / ((watermarked - fixed_images) ** 2).mean().clamp(min=1e-8)).item()
        
        # 每 100 次輸出一次
        if i % 100 == 0 or i == iterations - 1:
            print(f"Iter {i:4d}/{iterations} | Loss: {loss.item():.4f} | BER: {ber:.4f} | PSNR: {psnr:.2f}dB")
        
        # 提前終止：BER = 0
        if ber == 0.0:
            print()
            print("=" * 60)
            print(f"SUCCESS! BER = 0 at iteration {i}")
            print("架構驗證通過：Encoder-Decoder 能正確傳遞浮水印資訊")
            print("=" * 60)
            return True
    
    # 訓練結束後的最終檢查
    print()
    print("=" * 60)
    if ber < 0.1:
        print(f"PARTIAL SUCCESS: BER = {ber:.4f} (< 0.1)")
        print("架構基本可用，但可能需要更多迭代或調整")
    else:
        print(f"FAILURE: BER = {ber:.4f} (仍然很高)")
        print("架構存在問題！可能的原因：")
        print("  1. Encoder 輸出的浮水印訊號太弱 (residual_scale 太小)")
        print("  2. Decoder 無法提取浮水印特徵")
        print("  3. 資料流有 Bug (watermark 和 image 對不上)")
    print("=" * 60)
    
    return ber < 0.1


def detailed_debug(device='cuda'):
    """更詳細的除錯：逐層檢查數據流"""
    print("\n" + "=" * 60)
    print("詳細除錯：檢查數據流")
    print("=" * 60)
    
    encoder = Encoder(watermark_bits=64).to(device)
    decoder = Decoder(watermark_bits=64).to(device)
    
    # 測試數據
    image = torch.rand(1, 3, 256, 256).to(device)
    watermark = torch.randint(0, 2, (1, 64)).float().to(device)
    
    print(f"\n輸入:")
    print(f"  Image: {image.shape}, range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"  Watermark: {watermark.shape}, unique: {watermark.unique().tolist()}")
    
    # Encoder 輸出
    with torch.no_grad():
        watermarked = encoder(image, watermark)
        residual = watermarked - image
    
    print(f"\nEncoder 輸出:")
    print(f"  Watermarked: {watermarked.shape}, range: [{watermarked.min():.3f}, {watermarked.max():.3f}]")
    print(f"  Residual: mean={residual.mean():.6f}, std={residual.std():.6f}, max_abs={residual.abs().max():.6f}")
    
    # Decoder 輸出
    with torch.no_grad():
        extracted, logits = decoder(watermarked)
    
    print(f"\nDecoder 輸出:")
    print(f"  Logits: {logits.shape}, range: [{logits.min():.3f}, {logits.max():.3f}]")
    print(f"  Extracted: {extracted.shape}")
    
    # BER
    ber = (extracted != watermark).float().mean().item()
    print(f"\n初始 BER (未訓練): {ber:.4f} (預期 ~0.5)")
    
    if abs(ber - 0.5) > 0.2:
        print("  ⚠️ 警告：初始 BER 偏離 0.5 太多，可能有問題")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用設備: {device}\n")
    
    # 先做詳細除錯
    detailed_debug(device)
    
    print("\n")
    
    # 過擬合測試
    success = sanity_check_overfit(iterations=1000, batch_size=16, device=device)
    
    if success:
        print("\n建議：可以開始正式訓練了！")
    else:
        print("\n建議：請先修復架構問題再進行正式訓練")
