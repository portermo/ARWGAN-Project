#!/bin/bash
# ARWGAN 訓練 tmux 會話啟動腳本

cd /mnt/nvme/p3/Project/arwgan/ARWGAN-Project/ARWGAN-main

# 檢查是否已經存在 train 會話
if tmux has-session -t train 2>/dev/null; then
    echo "會話 'train' 已存在，正在連接..."
    tmux attach -t train
else
    echo "創建新的 tmux 會話 'train'..."
    # 創建新會話並設置工作目錄
    tmux new-session -d -s train -c "$(pwd)"
    
    # 激活虛擬環境
    tmux send-keys -t train "source venv/bin/activate" C-m
    
    # 顯示提示信息
    tmux send-keys -t train "echo '✓ 虛擬環境已激活'" C-m
    tmux send-keys -t train "echo '✓ 當前目錄: $(pwd)'" C-m
    tmux send-keys -t train "echo '✓ Python: $(which python)'" C-m
    tmux send-keys -t train "echo ''" C-m
    tmux send-keys -t train "echo '可以開始訓練了！例如：'" C-m
    tmux send-keys -t train "echo '  python train.py -n my_experiment -d /path/to/data -b 32 -e 100'" C-m
    tmux send-keys -t train "echo ''" C-m
    
    # 連接到會話
    tmux attach -t train
fi
