#!/usr/bin/env python3
"""
驗證修復後代碼的正確性
"""
import sys
import torch
import torch.nn as nn

print("\n" + "="*70)
print("驗證修復後的代碼...")
print("="*70 + "\n")

try:
    # 測試導入
    print("✓ 步驟 1: 導入模組")
    sys.path.insert(0, '/mnt/nvme/p3/Project/arwgan/ARWGAN-Project/ARWGAN-main')
    from watermark_model_better import (
        ChannelAttention, SpatialAttention, CBAM,
        Encoder, Decoder, Discriminator,
        DiffJPEG, NoiseLayer,
        VGGLoss, ssim_loss, wgan_gp_loss
    )
    print("  所有模組成功導入\n")
    
    # 測試 CBAM
    print("✓ 步驟 2: 測試 CBAM Attention")
    cbam = CBAM(channels=64)
    x = torch.randn(2, 64, 32, 32)
    out = cbam(x)
    assert out.shape == x.shape, "CBAM 輸出形狀錯誤"
    print(f"  輸入: {x.shape}, 輸出: {out.shape} ✓\n")
    
    # 測試 Encoder
    print("✓ 步驟 3: 測試 Encoder（含修復的輸出層）")
    encoder = Encoder(watermark_bits=64)
    image = torch.randn(2, 3, 256, 256)
    watermark = torch.randint(0, 2, (2, 64)).float()
    watermarked = encoder(image, watermark)
    assert watermarked.shape == image.shape, "Encoder 輸出形狀錯誤"
    assert torch.all((watermarked >= 0) & (watermarked <= 1)), "Encoder 輸出範圍錯誤"
    print(f"  輸入: {image.shape}, 水印: {watermark.shape}, 輸出: {watermarked.shape} ✓")
    print(f"  輸出範圍: [{watermarked.min():.3f}, {watermarked.max():.3f}] ✓\n")
    
    # 測試 DiffJPEG
    print("✓ 步驟 4: 測試可微分 JPEG")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    diff_jpeg = DiffJPEG(device).to(device)
    x = torch.randn(2, 3, 256, 256).to(device)
    x.requires_grad = True
    out = diff_jpeg(x, quality_factor=50)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "DiffJPEG 不可微分！"
    print(f"  梯度存在: {x.grad is not None} ✓")
    print(f"  梯度範數: {x.grad.norm().item():.6f} ✓\n")
    
    # 測試 NoiseLayer
    print("✓ 步驟 5: 測試 NoiseLayer（含邊界檢查）")
    noise_layer = NoiseLayer(device).to(device)
    x = torch.randn(2, 3, 128, 128).to(device)  # 小尺寸測試
    original = x.clone()
    
    # 測試各種攻擊
    for attack in ['gaussian', 'jpeg', 'crop', 'dropout', 'resize']:
        noise_layer.attacks = [attack]
        try:
            noised = noise_layer(x, original_image=original)
            assert noised.shape == x.shape, f"{attack} 攻擊輸出形狀錯誤"
            print(f"  {attack:10s}: 形狀 {noised.shape} ✓")
        except Exception as e:
            print(f"  {attack:10s}: ❌ {str(e)}")
            raise
    print()
    
    # 測試 Decoder
    print("✓ 步驟 6: 測試 Decoder")
    decoder = Decoder(watermark_bits=64)
    watermarked_img = torch.randn(2, 3, 256, 256)
    extracted, logits = decoder(watermarked_img)
    assert extracted.shape == (2, 64), "Decoder 輸出形狀錯誤"
    assert torch.all((extracted == 0) | (extracted == 1)), "Decoder 輸出不是二進位"
    print(f"  輸入: {watermarked_img.shape}")
    print(f"  提取水印: {extracted.shape}, 二進位: {torch.all((extracted == 0) | (extracted == 1))} ✓\n")
    
    # 測試 VGG Loss
    print("✓ 步驟 7: 測試 VGG 感知損失")
    vgg_loss = VGGLoss()
    x1 = torch.randn(2, 3, 256, 256)
    x2 = torch.randn(2, 3, 256, 256)
    feat1 = vgg_loss(x1)
    feat2 = vgg_loss(x2)
    assert feat1.shape == feat2.shape, "VGG 特徵形狀不一致"
    print(f"  輸入: {x1.shape}, VGG 特徵: {feat1.shape} ✓\n")
    
    # 測試 SSIM Loss
    print("✓ 步驟 8: 測試 SSIM Loss")
    img1 = torch.rand(2, 3, 256, 256)
    img2 = img1 + torch.randn_like(img1) * 0.1
    ssim_val = ssim_loss(img1, img2)
    assert 0 <= ssim_val <= 1, "SSIM 值超出範圍"
    print(f"  SSIM Loss: {ssim_val.item():.4f} (範圍正確) ✓\n")
    
    # 測試 WGAN-GP Loss
    print("✓ 步驟 9: 測試 WGAN-GP Loss")
    discriminator = Discriminator()
    real = torch.randn(2, 3, 256, 256)
    fake = torch.randn(2, 3, 256, 256)
    gp = wgan_gp_loss(discriminator, real, fake)
    assert gp.item() >= 0, "Gradient Penalty 為負"
    print(f"  Gradient Penalty: {gp.item():.4f} ✓\n")
    
    # 端到端測試（重點測試模型參數梯度）
    print("✓ 步驟 10: 端到端測試（梯度流通到模型參數）")
    encoder = Encoder(64).to(device)
    decoder = Decoder(64).to(device)
    noise_layer = NoiseLayer(device).to(device)
    
    image = torch.randn(2, 3, 256, 256).to(device)  # 數據不需要 requires_grad
    watermark = torch.randint(0, 2, (2, 64)).float().to(device)
    
    # Forward
    watermarked = encoder(image, watermark)
    noised = noise_layer(watermarked, original_image=image)
    extracted, logits = decoder(noised)
    
    # Loss
    loss = nn.MSELoss()(watermarked, image) + nn.BCEWithLogitsLoss()(logits, watermark)
    
    # Backward
    loss.backward()
    
    # 檢查模型參數是否有梯度
    encoder_has_grad = any(p.grad is not None for p in encoder.parameters() if p.requires_grad)
    decoder_has_grad = any(p.grad is not None for p in decoder.parameters() if p.requires_grad)
    
    assert encoder_has_grad, "Encoder 參數沒有梯度"
    assert decoder_has_grad, "Decoder 參數沒有梯度"
    
    print(f"  前向傳播: ✓")
    print(f"  Encoder 梯度: ✓")
    print(f"  Decoder 梯度: ✓")
    print(f"  損失值: {loss.item():.4f} ✓\n")
    
    print("="*70)
    print("✅ 所有測試通過！修復後的代碼可以正常工作。")
    print("="*70)
    print("\n建議接下來的步驟:")
    print("  1. 準備數據集（COCO 或其他）")
    print("  2. 開始小規模訓練測試（--epochs 5 --batch 4）")
    print("  3. 確認無誤後進行完整訓練（--epochs 100 --batch 16 --use_vgg）")
    print()
    
except Exception as e:
    print(f"\n❌ 測試失敗: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
