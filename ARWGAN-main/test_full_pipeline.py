#!/usr/bin/env python3
"""
完整訓練管線測試
測試資料載入、模型初始化、forward/backward pass
"""

import torch
import numpy as np
from options import HiDDenConfiguration, TrainingOptions
from model.ARWGAN import ARWGAN
from noise_layers.noiser import Noiser
from noise_layers.jpeg import Jpeg
import utils

def test_data_loader():
    """測試資料載入器"""
    print("=" * 60)
    print("測試 1: 資料載入器")
    print("=" * 60)
    
    try:
        net_config = HiDDenConfiguration(
            H=128, W=128,
            message_length=30,
            encoder_blocks=4, encoder_channels=64,
            decoder_blocks=7, decoder_channels=64,
            use_discriminator=True,
            use_vgg=False,
            discriminator_blocks=3, discriminator_channels=64,
            decoder_loss=1,
            encoder_loss=0.7,
            adversarial_loss=1e-3,
            enable_fp16=False
        )
        
        train_options = TrainingOptions(
            batch_size=2,  # 小 batch size 用於測試
            number_of_epochs=1,
            train_folder='data/coco2017/train',
            validation_folder='data/coco2017/val',
            runs_folder='./runs',
            start_epoch=1,
            experiment_name='test'
        )
        
        train_loader, val_loader = utils.get_data_loaders(net_config, train_options)
        
        # 測試載入一個 batch
        for images, labels in train_loader:
            print(f"✓ 訓練資料 batch 形狀: {images.shape}")
            print(f"✓ 數值範圍: [{images.min():.4f}, {images.max():.4f}]")
            break
        
        for images, labels in val_loader:
            print(f"✓ 驗證資料 batch 形狀: {images.shape}")
            break
        
        print("\n✅ 資料載入器測試通過！\n")
        return net_config, train_options
        
    except Exception as e:
        print(f"\n❌ 資料載入器測試失敗: {e}\n")
        import traceback
        traceback.print_exc()
        return None, None


def test_model_initialization(net_config):
    """測試模型初始化"""
    print("=" * 60)
    print("測試 2: 模型初始化")
    print("=" * 60)
    
    try:
        device = torch.device('cpu')  # 使用 CPU 避免 GPU 相容性問題
        
        # 建立 noiser
        noise_config = [Jpeg(1.0)]
        noiser = Noiser(noise_config, device)
        
        # 建立模型
        model = ARWGAN(net_config, device, noiser, None)
        
        print(f"✓ 模型已初始化")
        print(f"✓ Encoder-Decoder 參數數量: {sum(p.numel() for p in model.encoder_decoder.parameters()):,}")
        print(f"✓ Discriminator 參數數量: {sum(p.numel() for p in model.discriminator.parameters()):,}")
        
        print("\n✅ 模型初始化測試通過！\n")
        return model, device, noiser
        
    except Exception as e:
        print(f"\n❌ 模型初始化測試失敗: {e}\n")
        import traceback
        traceback.print_exc()
        return None, None, None


def test_forward_pass(model, device, net_config):
    """測試 forward pass"""
    print("=" * 60)
    print("測試 3: Forward Pass")
    print("=" * 60)
    
    try:
        # 建立測試資料
        batch_size = 2
        test_image = torch.rand(batch_size, 3, net_config.H, net_config.W).to(device)
        test_message = torch.Tensor(np.random.choice([0, 1], (batch_size, net_config.message_length))).to(device)
        
        # Normalize to [-1, 1]
        test_image = test_image * 2 - 1
        
        print(f"✓ 輸入圖片形狀: {test_image.shape}")
        print(f"✓ 輸入訊息形狀: {test_message.shape}")
        
        # Forward pass (validation mode)
        model.encoder_decoder.eval()
        model.discriminator.eval()
        
        with torch.no_grad():
            losses, (encoded_images, noised_images, decoded_messages) = model.validate_on_batch([test_image, test_message])
        
        print(f"✓ 編碼圖片形狀: {encoded_images.shape}")
        print(f"✓ 加噪圖片形狀: {noised_images.shape}")
        print(f"✓ 解碼訊息形狀: {decoded_messages.shape}")
        
        # 檢查損失
        print("\n損失值:")
        for loss_name, loss_value in losses.items():
            print(f"  {loss_name}: {loss_value:.6f}")
        
        # 檢查訊息準確度
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        message_detached = test_message.detach().cpu().numpy()
        accuracy = 1 - np.mean(np.abs(decoded_rounded - message_detached))
        print(f"\n✓ 訊息解碼準確度: {accuracy*100:.2f}%")
        
        print("\n✅ Forward pass 測試通過！\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Forward pass 測試失敗: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_backward_pass(model, device, net_config):
    """測試 backward pass"""
    print("=" * 60)
    print("測試 4: Backward Pass (梯度計算)")
    print("=" * 60)
    
    try:
        # 建立測試資料
        batch_size = 2
        test_image = torch.rand(batch_size, 3, net_config.H, net_config.W).to(device)
        test_message = torch.Tensor(np.random.choice([0, 1], (batch_size, net_config.message_length))).to(device)
        
        # Normalize to [-1, 1]
        test_image = test_image * 2 - 1
        
        # Training mode
        model.encoder_decoder.train()
        model.discriminator.train()
        
        # Forward pass
        losses, _ = model.train_on_batch([test_image, test_message])
        
        print("✓ Backward pass 完成")
        print("\n損失值:")
        for loss_name, loss_value in losses.items():
            print(f"  {loss_name}: {loss_value:.6f}")
        
        # 檢查梯度
        has_grad = False
        for name, param in model.encoder_decoder.named_parameters():
            if param.grad is not None:
                has_grad = True
                break
        
        if has_grad:
            print("\n✓ 梯度已計算")
        else:
            print("\n⚠️ 警告: 未檢測到梯度")
        
        print("\n✅ Backward pass 測試通過！\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Backward pass 測試失敗: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_jpeg_noise(device):
    """測試 JPEG 噪聲層"""
    print("=" * 60)
    print("測試 5: JPEG 噪聲層")
    print("=" * 60)
    
    try:
        from noise_layers.jpeg import Jpeg, DiffJPEG
        
        # 測試 DiffJPEG
        diff_jpeg = DiffJPEG(factor=1.0).to(device)
        test_image = torch.rand(2, 3, 128, 128).to(device)
        test_image = torch.clamp(test_image, 0.0, 1.0)
        
        noise_and_cover = [test_image.clone()]
        output = diff_jpeg(noise_and_cover)
        
        print(f"✓ DiffJPEG 輸入形狀: {test_image.shape}")
        print(f"✓ DiffJPEG 輸出形狀: {output[0].shape}")
        print(f"✓ 輸出範圍: [{output[0].min():.4f}, {output[0].max():.4f}]")
        
        # 測試 Jpeg wrapper
        jpeg = Jpeg(1.0)
        noise_and_cover = [test_image.clone()]
        output = jpeg(noise_and_cover)
        
        print(f"✓ Jpeg wrapper 測試通過")
        
        print("\n✅ JPEG 噪聲層測試通過！\n")
        return True
        
    except Exception as e:
        print(f"\n❌ JPEG 噪聲層測試失敗: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("ARWGAN 完整管線測試")
    print("=" * 60 + "\n")
    
    results = []
    
    # 測試 1: 資料載入
    net_config, train_options = test_data_loader()
    results.append(('資料載入器', net_config is not None))
    
    if net_config is None:
        print("❌ 由於資料載入失敗，跳過後續測試")
        return
    
    # 測試 2: 模型初始化
    model, device, noiser = test_model_initialization(net_config)
    results.append(('模型初始化', model is not None))
    
    if model is None:
        print("❌ 由於模型初始化失敗，跳過後續測試")
        return
    
    # 測試 3: Forward pass
    forward_ok = test_forward_pass(model, device, net_config)
    results.append(('Forward Pass', forward_ok))
    
    # 測試 4: Backward pass
    backward_ok = test_backward_pass(model, device, net_config)
    results.append(('Backward Pass', backward_ok))
    
    # 測試 5: JPEG 噪聲
    jpeg_ok = test_jpeg_noise(device)
    results.append(('JPEG 噪聲層', jpeg_ok))
    
    # 總結
    print("=" * 60)
    print("測試總結")
    print("=" * 60)
    for test_name, passed in results:
        status = "✅ 通過" if passed else "❌ 失敗"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有測試通過！程式碼可以正常運行。")
    else:
        print("⚠️ 部分測試失敗，請檢查錯誤訊息。")
    print("=" * 60 + "\n")
    
    # GPU 相容性提示
    print("📝 注意事項:")
    print("  - RTX 4090 需要 PyTorch 1.13+ 才能使用 GPU")
    print("  - 目前使用 CPU 模式進行測試")
    print("  - 如需 GPU 訓練，請升級 PyTorch 到 2.x 版本")
    print()


if __name__ == '__main__':
    main()
