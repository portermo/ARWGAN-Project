# Tmux 指令參考

## 基本操作

### 啟動和退出
```bash
# 啟動新的 tmux 會話
tmux

# 啟動並命名會話
tmux new -s session_name

# 列出所有會話
tmux ls

# 附加到會話
tmux attach -t session_name
# 或簡寫
tmux a -t session_name

# 分離會話（在 tmux 內）
Ctrl+b d

# 殺死會話
tmux kill-session -t session_name
```

## 視窗（Window）操作

### 預設前綴鍵：`Ctrl+b`（以下簡稱 `prefix`）

```bash
# 創建新視窗
prefix c

# 切換到下一個視窗
prefix n

# 切換到上一個視窗
prefix p

# 列出所有視窗
prefix w

# 關閉當前視窗
prefix &

# 重新命名當前視窗
prefix ,
```

## 面板（Pane）操作

### 分割面板
```bash
# 水平分割（上下）
prefix "

# 垂直分割（左右）
prefix %

# 分割成當前面板的路徑
prefix :split-window -c "#{pane_current_path}"
```

### 切換面板
```bash
# 切換到下一個面板
prefix o

# 切換到上一個面板
prefix ;

# 使用方向鍵切換
prefix ↑  # 上
prefix ↓  # 下
prefix ←  # 左
prefix →  # 右
```

### 調整面板大小
```bash
# 開始調整大小模式
prefix Ctrl+方向鍵

# 或使用
prefix :resize-pane -L 10  # 向左調整 10 個字符
prefix :resize-pane -R 10  # 向右調整 10 個字符
prefix :resize-pane -U 10  # 向上調整 10 個字符
prefix :resize-pane -D 10  # 向下調整 10 個字符
```

### 面板操作
```bash
# 關閉當前面板
prefix x

# 最大化/還原面板
prefix z

# 交換面板位置
prefix {  # 與上一個交換
prefix }  # 與下一個交換
```

## 實用配置

### 在專案目錄啟動 tmux 並激活虛擬環境
```bash
# 創建一個啟動腳本
cd /mnt/nvme/Project/arwgan/ARWGAN-Project/ARWGAN-main
tmux new -s arwgan -c "$(pwd)" \; \
  send-keys "source venv/bin/activate" C-m \; \
  split-window -h -c "$(pwd)" \; \
  send-keys "source venv/bin/activate" C-m
```

### 快速啟動腳本（可選）
創建 `start_tmux.sh`：
```bash
#!/bin/bash
cd /mnt/nvme/Project/arwgan/ARWGAN-Project/ARWGAN-main
tmux new-session -d -s arwgan -c "$(pwd)"
tmux send-keys -t arwgan "source venv/bin/activate" C-m
tmux split-window -h -t arwgan -c "$(pwd)"
tmux send-keys -t arwgan "source venv/bin/activate" C-m
tmux attach -t arwgan
```

## 常用快捷鍵總結

| 功能 | 快捷鍵 |
|------|--------|
| 前綴鍵 | `Ctrl+b` |
| 分離會話 | `prefix d` |
| 新視窗 | `prefix c` |
| 切換視窗 | `prefix n` / `prefix p` |
| 水平分割 | `prefix "` |
| 垂直分割 | `prefix %` |
| 切換面板 | `prefix o` 或 `prefix 方向鍵` |
| 關閉面板 | `prefix x` |
| 最大化面板 | `prefix z` |
| 滾動 | `prefix [` 進入，`q` 退出 |
| 複製模式 | `prefix [` |

## 複製和粘貼

```bash
# 進入複製模式
prefix [

# 在複製模式中：
# - 使用方向鍵或 vim 鍵位移動
# - 空格鍵開始選擇
# - Enter 複製選中內容
# - q 退出複製模式

# 粘貼
prefix ]
```

## 自定義配置（~/.tmux.conf）

```bash
# 更改前綴鍵為 Ctrl+a（可選）
# unbind C-b
# set-option -g prefix C-a
# bind-key C-a send-prefix

# 啟用滑鼠支援
set -g mouse on

# 設置視窗和面板索引從 1 開始
set -g base-index 1
setw -g pane-base-index 1

# 重新加載配置
# prefix :source-file ~/.tmux.conf
```

## ARWGAN 專案專用指令

### 啟動訓練會話
```bash
cd /mnt/nvme/Project/arwgan/ARWGAN-Project/ARWGAN-main
tmux new -s train -c "$(pwd)"
# 在 tmux 中：
source venv/bin/activate
python train.py [參數]
```

### 監控訓練進度
```bash
# 在另一個面板中
watch -n 1 nvidia-smi  # 監控 GPU
# 或
tail -f runs/*/train.log  # 查看日誌
```

## 提示

1. **記住前綴鍵**：所有 tmux 指令都需要先按 `Ctrl+b`
2. **分離不關閉**：`prefix d` 只會分離會話，不會關閉，可以隨時重新連接
3. **會話持久化**：即使 SSH 斷開，tmux 會話也會繼續運行
4. **多個會話**：可以同時運行多個 tmux 會話，用不同名稱區分
