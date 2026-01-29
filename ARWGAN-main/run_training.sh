#!/bin/bash
# 訓練模型的輔助腳本

cd "$(dirname "$0")"

# 檢查並激活虛擬環境
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✓ 虛擬環境已激活"
else
    echo "警告: 找不到 venv/bin/activate，將使用系統 Python"
fi

# 使用虛擬環境中的 Python（優先使用完整路徑）
if [ -f "venv/bin/python" ]; then
    PYTHON_CMD="venv/bin/python"
elif [ -f "venv/bin/python3" ]; then
    PYTHON_CMD="venv/bin/python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "錯誤: 找不到 Python 解釋器"
    exit 1
fi

echo "使用 Python: $PYTHON_CMD"
echo "Python 版本: $($PYTHON_CMD --version)"
echo ""

# 執行訓練腳本
# 注意: 要啟用 VGG 感知損失，請添加 --use_vgg 參數
# 例如: ./run_training.sh --train --epochs 10 --batch 16 --use_vgg --data-dir data/coco2017
$PYTHON_CMD watermark_model_better.py "$@"
