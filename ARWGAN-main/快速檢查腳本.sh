#!/bin/bash
# 快速檢查資料集的腳本

cd /mnt/nvme/Project/arwgan/ARWGAN-Project/ARWGAN-main
source venv/bin/activate

echo "開始檢查資料集..."
echo ""

# 檢查並移動損壞的圖片到備份目錄
python check_dataset_images.py data/coco2017 \
    --workers 8 \
    --move-invalid corrupted_images/ \
    --save-report dataset_check_$(date +%Y%m%d_%H%M%S).txt \
    --verbose

echo ""
echo "檢查完成！"
