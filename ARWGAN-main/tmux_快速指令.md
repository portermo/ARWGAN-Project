# Tmux 快速指令參考

## 當前會話狀態
您目前有一個編號為 0 的 tmux 會話。

## 連接會話

### 連接到現有會話（編號 0）
```bash
tmux attach -t 0
# 或簡寫
tmux a -t 0
```

### 創建新的 "train" 會話
```bash
cd /mnt/nvme/Project/arwgan/ARWGAN-Project/ARWGAN-main
tmux new -s train
# 然後在會話中：
source venv/bin/activate
```

### 或使用啟動腳本
```bash
./start_train_tmux.sh
```

## 常用操作

### 查看所有會話
```bash
tmux ls
```

### 在 tmux 會話內的操作（前綴鍵：Ctrl+b）
```bash
Ctrl+b d    # 分離會話（不關閉）
Ctrl+b c    # 創建新視窗
Ctrl+b n    # 下一個視窗
Ctrl+b p    # 上一個視窗
Ctrl+b "    # 水平分割面板
Ctrl+b %    # 垂直分割面板
Ctrl+b x    # 關閉當前面板
```

### 重新命名會話
```bash
# 將編號 0 的會話重新命名為 train
tmux rename-session -t 0 train

# 然後就可以用名稱連接
tmux attach -t train
```

## 實用範例

### 連接到現有會話並激活環境
```bash
tmux attach -t 0
source venv/bin/activate
```

### 創建訓練會話
```bash
tmux new -s train
source venv/bin/activate
python train.py [參數]
```

### 在背景運行訓練（分離會話）
```bash
# 在 tmux 中啟動訓練後
Ctrl+b d  # 分離會話，訓練繼續在背景運行

# 稍後重新連接查看進度
tmux attach -t train
```
