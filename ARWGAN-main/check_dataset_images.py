#!/usr/bin/env python3
"""
圖片資料集檢查工具
檢查資料集中是否有損壞、無效或不符合要求的圖片
"""

import os
import sys
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm
from collections import defaultdict
import torch
from torchvision import transforms
import multiprocessing as mp
from functools import partial


class ImageChecker:
    """圖片檢查器"""
    
    def __init__(self, min_size=32, max_size=10000, allowed_formats=None):
        """
        初始化檢查器
        
        Args:
            min_size: 最小尺寸（寬或高）
            max_size: 最大尺寸（寬或高）
            allowed_formats: 允許的圖片格式，None 表示允許所有格式
        """
        self.min_size = min_size
        self.max_size = max_size
        self.allowed_formats = allowed_formats or ['JPEG', 'PNG', 'BMP', 'TIFF', 'WEBP']
        
    def check_single_image(self, image_path):
        """
        檢查單張圖片
        
        Returns:
            dict: 檢查結果
        """
        result = {
            'path': str(image_path),
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        try:
            # 1. 檢查檔案是否存在
            if not os.path.exists(image_path):
                result['valid'] = False
                result['errors'].append('檔案不存在')
                return result
            
            # 2. 檢查檔案大小
            file_size = os.path.getsize(image_path)
            if file_size == 0:
                result['valid'] = False
                result['errors'].append('檔案大小為 0')
                return result
            
            result['info']['file_size'] = file_size
            
            # 3. 嘗試打開圖片
            try:
                img = Image.open(image_path)
                img.load()  # 強制載入圖片數據
            except Exception as e:
                result['valid'] = False
                result['errors'].append(f'無法打開圖片: {str(e)}')
                return result
            
            # 4. 檢查圖片格式
            result['info']['format'] = img.format
            if img.format not in self.allowed_formats:
                result['warnings'].append(f'圖片格式 {img.format} 不在允許列表中')
            
            # 5. 檢查圖片尺寸
            width, height = img.size
            result['info']['size'] = (width, height)
            result['info']['mode'] = img.mode
            
            if width < self.min_size or height < self.min_size:
                result['valid'] = False
                result['errors'].append(f'圖片尺寸過小: {width}x{height} (最小: {self.min_size})')
            
            if width > self.max_size or height > self.max_size:
                result['warnings'].append(f'圖片尺寸過大: {width}x{height}')
            
            # 6. 檢查色彩模式
            if img.mode not in ['RGB', 'L', 'RGBA']:
                result['warnings'].append(f'非標準色彩模式: {img.mode}')
            
            # 7. 嘗試轉換為 RGB（訓練時會用到）
            try:
                if img.mode != 'RGB':
                    img_rgb = img.convert('RGB')
            except Exception as e:
                result['valid'] = False
                result['errors'].append(f'無法轉換為 RGB: {str(e)}')
            
            # 8. 嘗試轉換為 tensor（模擬訓練時的操作）
            try:
                transform = transforms.ToTensor()
                tensor = transform(img.convert('RGB'))
                
                # 檢查是否有 NaN 或 Inf
                if torch.isnan(tensor).any():
                    result['valid'] = False
                    result['errors'].append('圖片包含 NaN 值')
                
                if torch.isinf(tensor).any():
                    result['valid'] = False
                    result['errors'].append('圖片包含 Inf 值')
                
            except Exception as e:
                result['valid'] = False
                result['errors'].append(f'無法轉換為 tensor: {str(e)}')
            
            # 9. 檢查圖片是否為全黑或全白
            try:
                extrema = img.convert('L').getextrema()
                if extrema[0] == extrema[1]:
                    result['warnings'].append(f'圖片為單一顏色: {extrema[0]}')
            except:
                pass
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'未預期的錯誤: {str(e)}')
        
        return result


def check_image_wrapper(args):
    """包裝函數，用於多進程處理"""
    image_path, checker = args
    return checker.check_single_image(image_path)


def find_all_images(directory, extensions=None):
    """
    遞迴尋找目錄中的所有圖片
    
    Args:
        directory: 目錄路徑
        extensions: 允許的副檔名
    
    Returns:
        list: 圖片路徑列表
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    image_paths = []
    directory = Path(directory)
    
    for ext in extensions:
        # 不區分大小寫
        image_paths.extend(directory.rglob(f'*{ext}'))
        image_paths.extend(directory.rglob(f'*{ext.upper()}'))
    
    return sorted(set(image_paths))


def main():
    parser = argparse.ArgumentParser(
        description='檢查圖片資料集中的損壞或無效圖片',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 檢查整個資料集
  python check_dataset_images.py data/coco2017
  
  # 檢查並移除損壞的圖片
  python check_dataset_images.py data/coco2017 --remove-invalid
  
  # 檢查並將損壞的圖片移到其他目錄
  python check_dataset_images.py data/coco2017 --move-invalid corrupted_images/
  
  # 使用多進程加速
  python check_dataset_images.py data/coco2017 --workers 8
        """
    )
    
    parser.add_argument('directory', type=str,
                        help='要檢查的資料集目錄')
    parser.add_argument('--min-size', type=int, default=32,
                        help='最小圖片尺寸（預設: 32）')
    parser.add_argument('--max-size', type=int, default=10000,
                        help='最大圖片尺寸（預設: 10000）')
    parser.add_argument('--extensions', type=str, nargs='+',
                        default=['.jpg', '.jpeg', '.png'],
                        help='要檢查的副檔名（預設: .jpg .jpeg .png）')
    parser.add_argument('--remove-invalid', action='store_true',
                        help='刪除無效的圖片')
    parser.add_argument('--move-invalid', type=str, metavar='DIR',
                        help='將無效的圖片移到指定目錄')
    parser.add_argument('--workers', type=int, default=4,
                        help='並行處理的進程數（預設: 4）')
    parser.add_argument('--save-report', type=str, metavar='FILE',
                        help='將檢查報告儲存到檔案')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='顯示詳細資訊')
    
    args = parser.parse_args()
    
    # 檢查目錄是否存在
    if not os.path.exists(args.directory):
        print(f"❌ 錯誤: 目錄不存在: {args.directory}")
        return 1
    
    print("=" * 70)
    print("圖片資料集檢查工具")
    print("=" * 70)
    print(f"\n📁 檢查目錄: {args.directory}")
    
    # 尋找所有圖片
    print(f"\n🔍 搜尋圖片檔案...")
    image_paths = find_all_images(args.directory, args.extensions)
    
    if not image_paths:
        print(f"⚠️  未找到任何圖片檔案")
        return 0
    
    print(f"✓ 找到 {len(image_paths)} 張圖片")
    
    # 建立檢查器
    checker = ImageChecker(
        min_size=args.min_size,
        max_size=args.max_size
    )
    
    # 檢查圖片
    print(f"\n🔬 檢查圖片 (使用 {args.workers} 個進程)...")
    
    results = []
    if args.workers > 1:
        # 多進程處理
        with mp.Pool(processes=args.workers) as pool:
            check_args = [(path, checker) for path in image_paths]
            results = list(tqdm(
                pool.imap(check_image_wrapper, check_args),
                total=len(image_paths),
                desc="檢查進度",
                unit="張"
            ))
    else:
        # 單進程處理
        for path in tqdm(image_paths, desc="檢查進度", unit="張"):
            results.append(checker.check_single_image(path))
    
    # 統計結果
    print("\n" + "=" * 70)
    print("檢查結果")
    print("=" * 70)
    
    valid_images = [r for r in results if r['valid']]
    invalid_images = [r for r in results if not r['valid']]
    warning_images = [r for r in results if r['warnings'] and r['valid']]
    
    print(f"\n✅ 有效圖片: {len(valid_images)} 張 ({len(valid_images)/len(results)*100:.1f}%)")
    print(f"❌ 無效圖片: {len(invalid_images)} 張 ({len(invalid_images)/len(results)*100:.1f}%)")
    print(f"⚠️  警告圖片: {len(warning_images)} 張 ({len(warning_images)/len(results)*100:.1f}%)")
    
    # 顯示無效圖片詳情
    if invalid_images:
        print(f"\n❌ 無效圖片詳細列表:")
        print("-" * 70)
        
        # 統計錯誤類型
        error_types = defaultdict(int)
        for img in invalid_images:
            for error in img['errors']:
                error_types[error] += 1
        
        print("\n錯誤類型統計:")
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  • {error_type}: {count} 張")
        
        if args.verbose:
            print("\n詳細清單:")
            for i, img in enumerate(invalid_images, 1):
                print(f"\n{i}. {img['path']}")
                for error in img['errors']:
                    print(f"   ✗ {error}")
    
    # 顯示警告圖片
    if warning_images and args.verbose:
        print(f"\n⚠️  警告圖片詳細列表:")
        print("-" * 70)
        for i, img in enumerate(warning_images[:10], 1):  # 只顯示前 10 個
            print(f"\n{i}. {img['path']}")
            for warning in img['warnings']:
                print(f"   ⚠ {warning}")
        
        if len(warning_images) > 10:
            print(f"\n... 還有 {len(warning_images) - 10} 個警告（使用 --verbose 查看全部）")
    
    # 處理無效圖片
    if invalid_images:
        if args.remove_invalid:
            print(f"\n🗑️  刪除無效圖片...")
            for img in invalid_images:
                try:
                    os.remove(img['path'])
                    print(f"  ✓ 已刪除: {img['path']}")
                except Exception as e:
                    print(f"  ✗ 刪除失敗: {img['path']} ({e})")
        
        elif args.move_invalid:
            print(f"\n📦 移動無效圖片到: {args.move_invalid}")
            os.makedirs(args.move_invalid, exist_ok=True)
            
            for img in invalid_images:
                try:
                    src = Path(img['path'])
                    dst = Path(args.move_invalid) / src.name
                    
                    # 如果目標已存在，加上數字後綴
                    counter = 1
                    while dst.exists():
                        dst = Path(args.move_invalid) / f"{src.stem}_{counter}{src.suffix}"
                        counter += 1
                    
                    src.rename(dst)
                    print(f"  ✓ 已移動: {src.name}")
                except Exception as e:
                    print(f"  ✗ 移動失敗: {img['path']} ({e})")
    
    # 儲存報告
    if args.save_report:
        print(f"\n💾 儲存檢查報告到: {args.save_report}")
        try:
            with open(args.save_report, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write("圖片資料集檢查報告\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"檢查目錄: {args.directory}\n")
                f.write(f"總圖片數: {len(results)}\n")
                f.write(f"有效圖片: {len(valid_images)}\n")
                f.write(f"無效圖片: {len(invalid_images)}\n")
                f.write(f"警告圖片: {len(warning_images)}\n\n")
                
                if invalid_images:
                    f.write("=" * 70 + "\n")
                    f.write("無效圖片清單\n")
                    f.write("=" * 70 + "\n\n")
                    for img in invalid_images:
                        f.write(f"路徑: {img['path']}\n")
                        f.write(f"錯誤:\n")
                        for error in img['errors']:
                            f.write(f"  - {error}\n")
                        f.write("\n")
                
                if warning_images:
                    f.write("=" * 70 + "\n")
                    f.write("警告圖片清單\n")
                    f.write("=" * 70 + "\n\n")
                    for img in warning_images:
                        f.write(f"路徑: {img['path']}\n")
                        f.write(f"警告:\n")
                        for warning in img['warnings']:
                            f.write(f"  - {warning}\n")
                        f.write(f"資訊: {img['info']}\n")
                        f.write("\n")
            
            print(f"✓ 報告已儲存")
        except Exception as e:
            print(f"✗ 儲存報告失敗: {e}")
    
    # 最終總結
    print("\n" + "=" * 70)
    if invalid_images:
        print("⚠️  發現損壞的圖片，建議處理後再進行訓練")
        if not args.remove_invalid and not args.move_invalid:
            print("    使用 --remove-invalid 刪除或 --move-invalid DIR 移動")
    else:
        print("✅ 所有圖片都有效！資料集可以用於訓練。")
    print("=" * 70)
    
    return 1 if invalid_images else 0


if __name__ == '__main__':
    sys.exit(main())
