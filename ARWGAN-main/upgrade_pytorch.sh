#!/bin/bash
# PyTorch 升級腳本 - 支援 RTX 4090

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║           PyTorch 升級工具 (for RTX 4090)                    ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# 檢查是否在正確的目錄
if [ ! -f "main.py" ]; then
    echo "❌ 錯誤: 請在 ARWGAN-main 目錄中執行此腳本"
    exit 1
fi

# 檢查虛擬環境
if [ ! -d "venv" ]; then
    echo "❌ 錯誤: 找不到虛擬環境 venv/"
    exit 1
fi

echo "📋 系統資訊檢查..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 檢查 GPU
GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "未檢測到 GPU")
echo "GPU: $GPU_INFO"

# 檢查 CUDA 版本
CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo "N/A")
echo "NVIDIA 驅動: $CUDA_VERSION"

echo ""
echo "⚠️  重要提示:"
echo "  - 此操作將升級 PyTorch 1.12.1 → 2.0.1"
echo "  - 將同時升級 torchvision 和 kornia"
echo "  - 舊版本的套件列表會被備份"
echo ""

read -p "確定要繼續嗎？(y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 已取消升級"
    exit 1
fi

echo ""
echo "🔄 開始升級流程..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 啟動虛擬環境
source venv/bin/activate

# 1. 備份
BACKUP_FILE="requirements_backup_$(date +%Y%m%d_%H%M%S).txt"
echo ""
echo "📦 步驟 1/5: 備份當前套件列表..."
pip freeze > "$BACKUP_FILE"
echo "   ✓ 備份已儲存: $BACKUP_FILE"

# 2. 卸載舊版本
echo ""
echo "🗑️  步驟 2/5: 卸載舊版 PyTorch..."
pip uninstall torch torchvision kornia -y > /dev/null 2>&1
echo "   ✓ 舊版本已卸載"

# 3. 安裝 PyTorch 2.0.1
echo ""
echo "⬇️  步驟 3/5: 安裝 PyTorch 2.0.1 + CUDA 11.8..."
echo "   (這可能需要幾分鐘，請稍候...)"
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118 > /dev/null 2>&1
echo "   ✓ PyTorch 2.0.1 安裝完成"

# 4. 安裝 kornia
echo ""
echo "⬇️  步驟 4/5: 安裝 kornia..."
pip install kornia > /dev/null 2>&1
echo "   ✓ kornia 安裝完成"

# 5. 驗證
echo ""
echo "✅ 步驟 5/5: 驗證安裝..."
python << 'PYTHON_EOF'
import torch
import torchvision
import kornia
import sys

print("")
print("   套件版本:")
print(f"     • PyTorch:     {torch.__version__}")
print(f"     • torchvision: {torchvision.__version__}")
print(f"     • kornia:      {kornia.__version__}")
print("")

cuda_available = torch.cuda.is_available()
print(f"   CUDA 可用: {'✓ 是' if cuda_available else '✗ 否'}")

if cuda_available:
    gpu_name = torch.cuda.get_device_name(0)
    cuda_version = torch.version.cuda
    print(f"   GPU 名稱: {gpu_name}")
    print(f"   CUDA 版本: {cuda_version}")
    print("")
    
    # 測試 GPU 運算
    try:
        x = torch.rand(1000, 1000).cuda()
        y = torch.matmul(x, x)
        print("   ✓ GPU 運算測試通過")
        del x, y
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"   ✗ GPU 運算測試失敗: {e}")
        sys.exit(1)
else:
    print("   ⚠️  警告: CUDA 不可用")
    sys.exit(1)
PYTHON_EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🎉 升級成功完成！"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "📋 下一步:"
    echo "  1. 執行測試以確保一切正常:"
    echo "     python test_full_pipeline.py"
    echo ""
    echo "  2. 開始 GPU 訓練:"
    echo "     python main.py new -n my_experiment -d data/coco2017 -b 32"
    echo ""
    echo "📝 回滾方法 (如果需要):"
    echo "  pip install -r $BACKUP_FILE --force-reinstall"
    echo ""
else
    echo ""
    echo "❌ 升級過程中發生錯誤"
    echo "   請檢查上述錯誤訊息"
    echo ""
    echo "   回滾到舊版本:"
    echo "   pip install -r $BACKUP_FILE --force-reinstall"
    echo ""
    exit 1
fi
