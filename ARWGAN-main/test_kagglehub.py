#!/usr/bin/env python3
"""
測試 kagglehub 是否正常運作
"""

import kagglehub

print("正在測試 kagglehub...")

try:
    # 下載 COCO 2017 資料集
    print("開始下載 COCO 2017 資料集...")
    path = kagglehub.dataset_download("awsaf49/coco-2017-dataset")
    
    print(f"✅ 下載成功！")
    print(f"資料集路徑: {path}")
    
except Exception as e:
    print(f"❌ 下載失敗: {e}")
    print("\n可能的解決方案:")
    print("1. 檢查 Kaggle API 認證:")
    print("   - 確保 ~/.kaggle/kaggle.json 存在")
    print("   - 或設置環境變數 KAGGLE_USERNAME 和 KAGGLE_KEY")
    print("2. 檢查網路連接")
    print("3. 檢查是否有足夠的磁碟空間")