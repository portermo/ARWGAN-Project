#!/usr/bin/env python3
"""
下載並準備 COCO 2017 資料集供 ARWGAN 訓練使用

此腳本會：
1. 使用 kagglehub 下載 COCO 2017 資料集
2. 將資料集組織成 ImageFolder 格式（train/val 目錄結構）
3. 創建符號連結或複製圖片到適當的目錄結構
"""

import os
import shutil
import kagglehub
from pathlib import Path
import zipfile
from PIL import Image
import argparse
import time
import json
from urllib3.exceptions import ReadTimeoutError
from requests.exceptions import ConnectionError, Timeout, ReadTimeout

# #region agent log
LOG_PATH = "/mnt/nvme/p3/Project/arwgan/ARWGAN-Project/.cursor/debug.log"
# #endregion


def log_debug(session_id, run_id, hypothesis_id, location, message, data):
    """記錄調試日誌"""
    # #region agent log
    try:
        log_entry = {
            "sessionId": session_id,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000)
        }
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # #endregion


def download_coco_dataset(max_retries=5, retry_delay=30):
    """下載 COCO 2017 資料集（帶重試機制）"""
    # #region agent log
    log_debug("debug-session", "run1", "A", "download_coco_dataset:20", "函數開始", {"max_retries": max_retries, "retry_delay": retry_delay})
    # #endregion
    
    print("正在下載 COCO 2017 資料集...")
    print("這可能需要一些時間，請耐心等待...")
    
    dataset_name = "awsaf49/coco-2017-dataset"
    last_error = None
    
    for attempt in range(1, max_retries + 1):
        # #region agent log
        log_debug("debug-session", "run1", "A", f"download_coco_dataset:26", "重試嘗試", {"attempt": attempt, "max_retries": max_retries})
        # #endregion
        
        try:
            if attempt > 1:
                print(f"\n重試下載 (嘗試 {attempt}/{max_retries})...")
                print(f"等待 {retry_delay} 秒後重試...")
                time.sleep(retry_delay)
            
            # #region agent log
            log_debug("debug-session", "run1", "A", f"download_coco_dataset:33", "調用 kagglehub.dataset_download", {"dataset_name": dataset_name, "attempt": attempt})
            # #endregion
            
            # Download latest version
            path = kagglehub.dataset_download(dataset_name)
            
            # #region agent log
            log_debug("debug-session", "run1", "A", f"download_coco_dataset:38", "下載成功", {"path": str(path), "attempt": attempt})
            # #endregion
            
            print(f"資料集下載完成，路徑: {path}")
            return path
            
        except (ReadTimeoutError, ConnectionError, Timeout, ReadTimeout) as e:
            # #region agent log
            log_debug("debug-session", "run1", "B", f"download_coco_dataset:45", "捕獲超時/連接錯誤", {"error_type": type(e).__name__, "error_msg": str(e), "attempt": attempt})
            # #endregion
            
            last_error = e
            error_type = type(e).__name__
            print(f"\n⚠️  下載過程中發生 {error_type} 錯誤 (嘗試 {attempt}/{max_retries})")
            print(f"錯誤訊息: {str(e)}")
            
            if attempt < max_retries:
                print(f"將在 {retry_delay} 秒後自動重試...")
                # kagglehub 支援斷點續傳，會自動從上次中斷的地方繼續
            else:
                print(f"\n❌ 已達到最大重試次數 ({max_retries})，下載失敗")
                raise
        
        except Exception as e:
            # #region agent log
            log_debug("debug-session", "run1", "C", f"download_coco_dataset:58", "捕獲其他錯誤", {"error_type": type(e).__name__, "error_msg": str(e), "attempt": attempt})
            # #endregion
            
            last_error = e
            error_type = type(e).__name__
            print(f"\n⚠️  下載過程中發生未預期的錯誤: {error_type}")
            print(f"錯誤訊息: {str(e)}")
            
            # 對於非超時錯誤，也進行重試
            if attempt < max_retries:
                print(f"將在 {retry_delay} 秒後自動重試...")
            else:
                print(f"\n❌ 已達到最大重試次數 ({max_retries})，下載失敗")
                raise
    
    # 如果所有重試都失敗
    if last_error:
        raise last_error


def extract_if_needed(dataset_path, extract_to=None):
    """如果需要，解壓縮資料集"""
    if extract_to is None:
        extract_to = os.path.join(os.path.dirname(dataset_path), "coco_extracted")
    
    # 檢查是否已經解壓縮
    if os.path.exists(extract_to) and os.listdir(extract_to):
        print(f"資料集似乎已經解壓縮到: {extract_to}")
        return extract_to
    
    print(f"正在解壓縮資料集到: {extract_to}")
    os.makedirs(extract_to, exist_ok=True)
    
    # 查找 zip 檔案
    zip_files = list(Path(dataset_path).glob("*.zip"))
    
    for zip_file in zip_files:
        print(f"解壓縮: {zip_file.name}")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    
    return extract_to


def organize_coco_for_imagefolder(dataset_path, output_dir="data/coco2017", max_train=None, max_val=None):
    """
    將 COCO 2017 資料集組織成 ImageFolder 格式
    
    ImageFolder 需要以下結構:
    data/
      train/
        images/
          image1.jpg
          image2.jpg
      val/
        images/
          image1.jpg
          image2.jpg
    """
    print(f"正在組織資料集為 ImageFolder 格式...")
    
    # 創建輸出目錄
    train_dir = os.path.join(output_dir, "train", "images")
    val_dir = os.path.join(output_dir, "val", "images")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # 查找圖片目錄
    dataset_path = Path(dataset_path)
    
    # 常見的 COCO 資料集結構
    possible_train_paths = [
        dataset_path / "train2017",
        dataset_path / "train",
        dataset_path / "coco" / "train2017",
        dataset_path / "coco2017" / "train2017",
    ]
    
    possible_val_paths = [
        dataset_path / "val2017",
        dataset_path / "val",
        dataset_path / "coco" / "val2017",
        dataset_path / "coco2017" / "val2017",
    ]
    
    # 查找訓練圖片目錄
    train_source = None
    for path in possible_train_paths:
        if path.exists() and path.is_dir():
            train_source = path
            print(f"找到訓練圖片目錄: {train_source}")
            break
    
    # 查找驗證圖片目錄
    val_source = None
    for path in possible_val_paths:
        if path.exists() and path.is_dir():
            val_source = path
            print(f"找到驗證圖片目錄: {val_source}")
            break
    
    if train_source is None:
        # 嘗試在整個資料集目錄中查找
        print("未找到標準的 train2017 目錄，正在搜索...")
        for root, dirs, files in os.walk(dataset_path):
            if "train" in root.lower() and any(f.lower().endswith(('.jpg', '.png', '.jpeg')) for f in files):
                train_source = Path(root)
                print(f"找到訓練圖片目錄: {train_source}")
                break
    
    if val_source is None:
        print("未找到標準的 val2017 目錄，正在搜索...")
        for root, dirs, files in os.walk(dataset_path):
            if "val" in root.lower() and any(f.lower().endswith(('.jpg', '.png', '.jpeg')) for f in files):
                val_source = Path(root)
                print(f"找到驗證圖片目錄: {val_source}")
                break
    
    # 複製或創建符號連結
    def copy_images(source_dir, target_dir, max_images=None):
        """複製圖片到目標目錄"""
        if source_dir is None:
            print(f"警告: 未找到源目錄，跳過")
            return 0
        
        source_path = Path(source_dir)
        target_path = Path(target_dir)
        
        # 查找所有圖片檔案
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in source_path.iterdir() 
                      if f.suffix.lower() in image_extensions and f.is_file()]
        
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"正在複製 {len(image_files)} 張圖片從 {source_dir} 到 {target_dir}...")
        
        copied = 0
        for img_file in image_files:
            try:
                # 使用符號連結以節省空間（如果支援）
                target_file = target_path / img_file.name
                if not target_file.exists():
                    try:
                        os.symlink(img_file, target_file)
                    except OSError:
                        # 如果不支援符號連結，則複製檔案
                        shutil.copy2(img_file, target_file)
                    copied += 1
                    
                    if copied % 1000 == 0:
                        print(f"  已處理 {copied}/{len(image_files)} 張圖片...")
            except Exception as e:
                print(f"  警告: 無法複製 {img_file}: {e}")
        
        print(f"完成！共複製 {copied} 張圖片")
        return copied
    
    # 複製訓練和驗證圖片
    train_count = copy_images(train_source, train_dir, max_images=max_train)
    val_count = copy_images(val_source, val_dir, max_images=max_val)
    
    print(f"\n資料集準備完成！")
    print(f"訓練圖片: {train_count} 張")
    print(f"驗證圖片: {val_count} 張")
    print(f"資料集路徑: {os.path.abspath(output_dir)}")
    print(f"\n使用方式:")
    print(f"  python main.py new -n experiment_name -d {os.path.abspath(output_dir)} -b 32 -e 100")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='下載並準備 COCO 2017 資料集')
    parser.add_argument('--output-dir', '-o', default='data/coco2017',
                       help='輸出目錄（預設: data/coco2017）')
    parser.add_argument('--max-train', type=int, default=None,
                       help='最大訓練圖片數量（用於測試，預設: 全部）')
    parser.add_argument('--max-val', type=int, default=None,
                       help='最大驗證圖片數量（用於測試，預設: 全部）')
    parser.add_argument('--skip-download', action='store_true',
                       help='跳過下載步驟（使用已下載的資料集）')
    parser.add_argument('--dataset-path', type=str, default=None,
                       help='已下載資料集的路徑（如果跳過下載）')
    
    args = parser.parse_args()
    
    if args.skip_download and args.dataset_path:
        dataset_path = args.dataset_path
        print(f"使用已下載的資料集: {dataset_path}")
    else:
        # 下載資料集
        dataset_path = download_coco_dataset()
    
    # 組織資料集
    output_dir = organize_coco_for_imagefolder(dataset_path, args.output_dir, 
                                                max_train=args.max_train, 
                                                max_val=args.max_val)
    
    print(f"\n✅ 資料集準備完成！")
    print(f"資料集位置: {os.path.abspath(output_dir)}")


if __name__ == '__main__':
    main()