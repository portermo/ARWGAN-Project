#!/usr/bin/env python3
"""
GPU 訓練測試 - 執行一個完整的訓練迭代
"""

import torch
import numpy as np
from options import HiDDenConfiguration, TrainingOptions
from model.ARWGAN import ARWGAN
from noise_layers.noiser import Noiser
from noise_layers.jpeg import Jpeg
import utils
import time

def test_gpu_training():
    print("=" * 60)
    print("GPU 訓練測試")
    print("=" * 60)
    print()
    
    # 檢查 GPU
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用")
        return False
    
    device = torch.device('cuda')
    print(f"✓ 使用設備: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA 版本: {torch.version.cuda}")
    print()
    
    # 配置
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
        batch_size=8,  # 使用較大的 batch size 測試 GPU
        number_of_epochs=1,
        train_folder='data/coco2017/train',
        validation_folder='data/coco2017/val',
        runs_folder='./runs',
        start_epoch=1,
        experiment_name='gpu_test'
    )
    
    print("載入資料集...")
    train_loader, val_loader = utils.get_data_loaders(net_config, train_options)
    print(f"✓ 訓練資料: {len(train_loader)} batches")
    print()
    
    # 建立模型
    print("初始化模型...")
    noise_config = [Jpeg(1.0)]
    noiser = Noiser(noise_config, device)
    model = ARWGAN(net_config, device, noiser, None)
    print(f"✓ 模型已載入到 GPU")
    print()
    
    # 訓練 3 個 batch
    print("開始訓練測試 (3 個 batches)...")
    print("-" * 60)
    
    model.encoder_decoder.train()
    model.discriminator.train()
    
    batch_count = 0
    start_time = time.time()
    
    for images, _ in train_loader:
        if batch_count >= 3:
            break
        
        batch_count += 1
        batch_start = time.time()
        
        # 準備資料
        images = images.to(device)
        messages = torch.Tensor(np.random.choice([0, 1], 
                               (images.shape[0], net_config.message_length))).to(device)
        
        # 訓練
        losses, _ = model.train_on_batch([images, messages])
        
        batch_time = time.time() - batch_start
        
        print(f"Batch {batch_count}/3:")
        print(f"  時間: {batch_time:.2f}s")
        print(f"  Loss: {losses.get('loss', 0):.4f}")
        if 'PSNR' in losses:
            print(f"  PSNR: {losses['PSNR']:.2f} dB")
        
        # 清理 GPU 記憶體
        torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    avg_time = total_time / batch_count
    
    print("-" * 60)
    print(f"\n✓ GPU 訓練測試完成")
    print(f"  總時間: {total_time:.2f}s")
    print(f"  平均每 batch: {avg_time:.2f}s")
    print(f"  預估每 epoch (CPU): ~{len(train_loader) * avg_time / 60:.1f} 分鐘")
    
    # GPU 記憶體使用
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"\nGPU 記憶體使用:")
        print(f"  已分配: {mem_allocated:.2f} GB")
        print(f"  已保留: {mem_reserved:.2f} GB")
    
    print()
    print("=" * 60)
    print("✅ GPU 訓練完全正常！")
    print("=" * 60)
    print()
    print("可以開始完整訓練:")
    print("  python main.py new -n my_experiment -d data/coco2017 -b 32 -e 100")
    print()
    
    return True


if __name__ == '__main__':
    try:
        success = test_gpu_training()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
