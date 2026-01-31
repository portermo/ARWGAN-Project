#!/bin/bash
# 延續訓練腳本 - 自動從最新檢查點繼續訓練

cd "$(dirname "$0")"

# 自動檢測最新的檢查點
LATEST_CHECKPOINT=$(ls -t checkpoints_improved/checkpoint_epoch_*.pth 2>/dev/null | head -1)

# 如果沒有找到檢查點，嘗試使用 best_model.pth
if [ -z "$LATEST_CHECKPOINT" ] || [ ! -f "$LATEST_CHECKPOINT" ]; then
    if [ -f "checkpoints_improved/best_model.pth" ]; then
        LATEST_CHECKPOINT="checkpoints_improved/best_model.pth"
        echo "⚠️  未找到 checkpoint_epoch_*.pth，使用 best_model.pth"
    else
        LATEST_CHECKPOINT=""
        echo "⚠️  未找到任何檢查點，將從頭開始訓練..."
    fi
fi

# 訓練參數
EPOCHS=100         # 總 epochs（會從檢查點的 epoch + 1 開始）
BATCH_SIZE=24      # Batch size（根據 GPU 記憶體調整）
USE_VGG=true       # 使用 VGG 感知損失
DATA_DIR="data/coco2017"
LEARNING_RATE=1e-4 # 學習率

echo "=========================================="
echo "延續訓練配置"
echo "=========================================="
if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "檢查點: $LATEST_CHECKPOINT"
    # 顯示檢查點信息
    if [ -f "$LATEST_CHECKPOINT" ]; then
        CHECKPOINT_SIZE=$(ls -lh "$LATEST_CHECKPOINT" | awk '{print $5}')
        CHECKPOINT_TIME=$(ls -lh "$LATEST_CHECKPOINT" | awk '{print $6, $7, $8}')
        echo "檢查點大小: $CHECKPOINT_SIZE"
        echo "檢查點時間: $CHECKPOINT_TIME"
    fi
else
    echo "檢查點: 無（從頭開始）"
fi
echo "總 Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "使用 VGG: $USE_VGG"
echo "學習率: $LEARNING_RATE"
echo "數據集: $DATA_DIR"
echo "=========================================="
echo ""

# 構建訓練命令
CMD="./run_training.sh --train --epochs $EPOCHS --batch $BATCH_SIZE --data-dir $DATA_DIR --lr $LEARNING_RATE"

if [ "$USE_VGG" = true ]; then
    CMD="$CMD --use_vgg"
fi

if [ -n "$LATEST_CHECKPOINT" ]; then
    CMD="$CMD --resume $LATEST_CHECKPOINT"
fi

# 執行訓練
echo "開始訓練..."
echo "命令: $CMD"
echo ""
eval $CMD
