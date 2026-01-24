#!/usr/bin/env python3
"""
測試新的 DiffJPEG 實現
"""

import torch
import torch.nn as nn
from noise_layers.jpeg import DiffJPEG, Jpeg, quality_to_factor

def test_diff_jpeg():
    """測試 DiffJPEG 基本功能"""
    print("測試 DiffJPEG...")
    
    # 創建測試圖片 (batch=2, channels=3, height=128, width=128)
    # 輸入範圍: [0, 1]
    test_image = torch.rand(2, 3, 128, 128)
    test_image = torch.clamp(test_image, 0.0, 1.0)
    
    # 創建 DiffJPEG 模組
    diff_jpeg = DiffJPEG(factor=1.0)
    
    # 測試 forward pass
    noise_and_cover = [test_image.clone()]
    output = diff_jpeg(noise_and_cover)
    
    # 檢查輸出
    assert output[0].shape == test_image.shape, f"形狀不匹配: {output[0].shape} vs {test_image.shape}"
    assert torch.all(output[0] >= 0.0) and torch.all(output[0] <= 1.0), "輸出超出 [0, 1] 範圍"
    
    print("✅ DiffJPEG 測試通過！")
    print(f"   輸入範圍: [{test_image.min():.4f}, {test_image.max():.4f}]")
    print(f"   輸出範圍: [{output[0].min():.4f}, {output[0].max():.4f}]")
    print(f"   輸出形狀: {output[0].shape}")


def test_jpeg_wrapper():
    """測試 Jpeg wrapper 類"""
    print("\n測試 Jpeg wrapper...")
    
    # 測試不同的 factor 類型
    test_cases = [
        (1.0, "float"),
        ("1.5", "string"),
        (2.0, "float"),
    ]
    
    test_image = torch.rand(1, 3, 64, 64)
    test_image = torch.clamp(test_image, 0.0, 1.0)
    
    for factor, factor_type in test_cases:
        jpeg = Jpeg(factor)
        noise_and_cover = [test_image.clone()]
        output = jpeg(noise_and_cover)
        
        assert output[0].shape == test_image.shape, f"形狀不匹配 (factor={factor}, type={factor_type})"
        assert torch.all(output[0] >= 0.0) and torch.all(output[0] <= 1.0), f"輸出超出範圍 (factor={factor})"
        
        print(f"✅ Factor {factor} ({factor_type}) 測試通過")


def test_quality_to_factor():
    """測試 quality_to_factor 函數"""
    print("\n測試 quality_to_factor...")
    
    test_cases = [
        (10, 5.0),   # quality < 50: 50 / quality
        (25, 2.0),   # quality < 50: 50 / quality
        (50, 1.0),   # quality >= 50: 2.0 - quality * 0.02
        (75, 0.5),   # quality >= 50: 2.0 - quality * 0.02
        (100, 0.0),  # quality >= 50: 2.0 - quality * 0.02
    ]
    
    for quality, expected_factor in test_cases:
        factor = quality_to_factor(quality)
        print(f"   Quality {quality} -> Factor {factor:.4f} (預期: {expected_factor:.4f})")
        assert abs(factor - expected_factor) < 1e-6, f"Quality {quality} 轉換錯誤"


def test_gpu_compatibility():
    """測試 GPU 兼容性"""
    print("\n測試 GPU 兼容性...")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"   使用 GPU: {torch.cuda.get_device_name(0)}")
        
        test_image = torch.rand(2, 3, 128, 128, device=device)
        test_image = torch.clamp(test_image, 0.0, 1.0)
        
        diff_jpeg = DiffJPEG(factor=1.0).to(device)
        noise_and_cover = [test_image.clone()]
        output = diff_jpeg(noise_and_cover)
        
        assert output[0].device == device, "輸出不在 GPU 上"
        assert output[0].shape == test_image.shape, "形狀不匹配"
        
        print("✅ GPU 測試通過！")
    else:
        print("   跳過 GPU 測試（無可用 GPU）")


def test_gradient_flow():
    """測試梯度流動"""
    print("\n測試梯度流動...")
    
    test_image = torch.rand(1, 3, 64, 64, requires_grad=True)
    test_image = torch.clamp(test_image, 0.0, 1.0)
    
    diff_jpeg = DiffJPEG(factor=1.0)
    noise_and_cover = [test_image]
    output = diff_jpeg(noise_and_cover)
    
    # 計算損失並反向傳播
    loss = output[0].mean()
    loss.backward()
    
    assert test_image.grad is not None, "梯度未流動"
    assert not torch.all(test_image.grad == 0), "梯度全為零"
    
    print("✅ 梯度流動測試通過！")
    print(f"   梯度範圍: [{test_image.grad.min():.6f}, {test_image.grad.max():.6f}]")


if __name__ == '__main__':
    print("=" * 60)
    print("DiffJPEG 測試套件")
    print("=" * 60)
    
    try:
        test_diff_jpeg()
        test_jpeg_wrapper()
        test_quality_to_factor()
        test_gpu_compatibility()
        test_gradient_flow()
        
        print("\n" + "=" * 60)
        print("✅ 所有測試通過！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()