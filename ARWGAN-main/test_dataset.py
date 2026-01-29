#!/usr/bin/env python3
"""
快速測試資料集是否能正常載入
"""
import sys
sys.path.insert(0, '/mnt/nvme/p3/Project/arwgan/ARWGAN-Project/ARWGAN-main')

from watermark_model_better import WatermarkDataset
from torch.utils.data import DataLoader

print("測試資料集載入...")
print("-" * 60)

# 創建資料集
dataset = WatermarkDataset()
print(f"✓ 資料集路徑: {dataset.root_dir}")
print(f"✓ 總圖像數量: {len(dataset)}")

# 測試載入第一個樣本
try:
    image, watermark = dataset[0]
    print(f"✓ 圖像形狀: {image.shape}")
    print(f"✓ 水印形狀: {watermark.shape}")
    print(f"✓ 圖像範圍: [{image.min():.3f}, {image.max():.3f}]")
    print(f"✓ 水印值: {watermark[:10].tolist()}")
except Exception as e:
    print(f"❌ 載入失敗: {e}")
    sys.exit(1)

# 測試 DataLoader
try:
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    batch = next(iter(loader))
    images, watermarks = batch
    print(f"✓ Batch 圖像: {images.shape}")
    print(f"✓ Batch 水印: {watermarks.shape}")
except Exception as e:
    print(f"❌ DataLoader 失敗: {e}")
    sys.exit(1)

print("-" * 60)
print("✅ 資料集測試成功！可以開始訓練。")
