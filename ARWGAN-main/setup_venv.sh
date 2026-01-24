#!/bin/bash
# ARWGAN 虛擬環境安裝腳本

echo "正在建立 ARWGAN 虛擬環境..."

# 檢查 Python 3.10 是否可用（PyTorch 1.12.1 需要 Python 3.7-3.10）
PYTHON_CMD=""
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
    echo "使用 Python 3.10"
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
    if [[ $(echo "$PYTHON_VERSION >= 3.7" | bc -l) -eq 1 ]] && [[ $(echo "$PYTHON_VERSION <= 3.10" | bc -l) -eq 1 ]]; then
        PYTHON_CMD="python3"
        echo "使用 Python $PYTHON_VERSION"
    else
        echo "警告: 系統 Python 版本為 $PYTHON_VERSION，PyTorch 1.12.1 建議使用 Python 3.7-3.10"
        echo "將嘗試使用 Python 3.12，但可能需要安裝較新版本的 PyTorch"
        PYTHON_CMD="python3"
    fi
else
    echo "錯誤: 找不到 Python 3"
    exit 1
fi

# 檢查是否已安裝 python3-venv
if ! $PYTHON_CMD -m venv --help &> /dev/null; then
    echo "錯誤: 需要安裝 python3-venv 套件"
    echo "請執行以下命令安裝:"
    echo "  sudo apt install python3-venv"
    exit 1
fi

# 建立虛擬環境
echo "建立虛擬環境 'venv'..."
$PYTHON_CMD -m venv --without-pip venv

# 安裝 pip
echo "安裝 pip..."
$PYTHON_CMD -c "import urllib.request; urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', 'get-pip.py')"
venv/bin/python get-pip.py
rm get-pip.py

# 啟動虛擬環境並安裝套件
echo "啟動虛擬環境並安裝套件..."
source venv/bin/activate

# 升級 pip
pip install --upgrade pip

# 安裝 PyTorch (CUDA 10.2 版本)
echo "安裝 PyTorch 1.12.1 (CUDA 10.2)..."
pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 --extra-index-url https://download.pytorch.org/whl/cu102

# 安裝其他依賴套件
echo "安裝其他依賴套件..."
pip install tensorboard

# 安裝 kornia (使用與 PyTorch 1.12.1 相容的版本)
echo "安裝 kornia..."
pip install "kornia<0.7"

# 注意: Pillow 已由 torchvision 自動安裝，版本可能與 README 要求的不同
# 如果需要特定版本，可以執行: pip install Pillow==7.2.0

echo ""
echo "虛擬環境建立完成！"
echo ""
echo "要啟動虛擬環境，請執行:"
echo "  source venv/bin/activate"
echo ""
echo "要退出虛擬環境，請執行:"
echo "  deactivate"