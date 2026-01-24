# GPU 訓練可用性說明

## ❌ 目前狀態：GPU 訓練**不可用**

---

## 問題分析

### 硬體資訊
```
GPU 型號: NVIDIA GeForce RTX 4090
顯存容量: 24 GB
驅動版本: 590.48.01
CUDA Compute Capability: sm_89 (8.9)
```

### 相容性問題

⚠️ **不相容原因:**

```
PyTorch 1.12.1+cu102 支援的 CUDA Compute Capabilities:
  - sm_37 (Kepler)
  - sm_50 (Maxwell)
  - sm_60 (Pascal)
  - sm_70 (Volta)

RTX 4090 需要:
  - sm_89 (Ada Lovelace)
```

**結論**: PyTorch 1.12.1 編譯時不包含 RTX 4090 所需的 CUDA 架構支援，因此無法在此 GPU 上執行運算。

---

## 解決方案

### 方案 1: 升級 PyTorch (強烈推薦) ⭐

升級到 PyTorch 2.x 以獲得 RTX 4090 完整支援。

#### 步驟

**1. 備份當前環境**
```bash
cd /mnt/nvme/Project/arwgan/ARWGAN-Project/ARWGAN-main
source venv/bin/activate
pip freeze > requirements_old.txt
```

**2. 升級 PyTorch**
```bash
# 卸載舊版本
pip uninstall torch torchvision -y

# 安裝 PyTorch 2.0+ (CUDA 11.8)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# 或安裝最新版 PyTorch 2.5+ (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**3. 驗證安裝**
```bash
python -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA 版本: {torch.version.cuda}')
    # 測試 GPU 運算
    x = torch.rand(1000, 1000).cuda()
    y = torch.matmul(x, x)
    print('✅ GPU 運算正常')
"
```

**4. 測試訓練管線**
```bash
python test_full_pipeline.py
```

#### 可能需要的程式碼調整

PyTorch 2.x 大部分 API 向後相容，但可能需要注意：

1. **AMP (Automatic Mixed Precision)**
   ```python
   # PyTorch 1.x
   from torch.cuda.amp import autocast, GradScaler
   
   # PyTorch 2.x (相同，但建議使用 torch.amp)
   from torch.amp import autocast, GradScaler
   ```

2. **某些已棄用的 API**
   - 大部分情況下不需要修改
   - 如果出現警告，按提示更新即可

---

### 方案 2: 使用 CPU 訓練 (臨時方案)

如果暫時不想升級 PyTorch，可以使用 CPU 模式。

#### 優點
- ✅ 無需修改環境
- ✅ 程式碼完全相容
- ✅ 可以進行小規模測試

#### 缺點
- ❌ 訓練速度**極慢** (約為 GPU 的 1/50 - 1/100)
- ❌ 不適合生產環境訓練
- ❌ 大 batch size 可能導致記憶體不足

#### 使用方式

程式會自動偵測並使用 CPU：

```bash
cd /mnt/nvme/Project/arwgan/ARWGAN-Project/ARWGAN-main
source venv/bin/activate

# 使用小 batch size 進行測試
python main.py new -n cpu_test -d data/coco2017 -b 4 -e 1
```

---

### 方案 3: 使用 Docker 容器 (進階)

使用預編譯好的 Docker 映像，包含最新 PyTorch 和 CUDA 支援。

```bash
# 拉取 PyTorch 官方映像
docker pull pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime

# 執行容器
docker run --gpus all -it \
  -v /mnt/nvme/Project/arwgan/ARWGAN-Project:/workspace \
  pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime \
  bash

# 在容器內
cd /workspace/ARWGAN-main
pip install -r requirements.txt
python test_full_pipeline.py
```

---

## 推薦方案比較

| 方案 | 難度 | 訓練速度 | 適用場景 |
|------|------|----------|----------|
| **升級 PyTorch** ⭐ | 🟡 中等 | 🟢 極快 | **生產環境、完整訓練** |
| CPU 訓練 | 🟢 簡單 | 🔴 極慢 | 小規模測試、除錯 |
| Docker 容器 | 🔴 困難 | 🟢 極快 | 隔離環境、多版本共存 |

---

## 升級 PyTorch 的詳細指南

### 檢查系統 CUDA 版本

```bash
nvidia-smi
```

查看 `CUDA Version` 欄位，選擇對應的 PyTorch 版本：

- **CUDA 11.8**: `pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118`
- **CUDA 12.1**: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
- **CUDA 12.4+**: `pip install torch torchvision` (預設)

### 完整升級腳本

建立 `upgrade_pytorch.sh`:

```bash
#!/bin/bash
# PyTorch 升級腳本

set -e

echo "🔄 開始升級 PyTorch..."

cd /mnt/nvme/Project/arwgan/ARWGAN-Project/ARWGAN-main
source venv/bin/activate

# 備份
echo "📦 備份當前套件列表..."
pip freeze > requirements_backup_$(date +%Y%m%d_%H%M%S).txt

# 卸載舊版本
echo "🗑️  卸載 PyTorch 1.12.1..."
pip uninstall torch torchvision kornia -y

# 安裝新版本
echo "⬇️  安裝 PyTorch 2.0.1..."
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# 安裝相容的 kornia
echo "⬇️  安裝 kornia..."
pip install kornia

# 驗證
echo "✅ 驗證安裝..."
python -c "
import torch
import torchvision
import kornia

print(f'PyTorch: {torch.__version__}')
print(f'torchvision: {torchvision.__version__}')
print(f'kornia: {kornia.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    x = torch.rand(100, 100).cuda()
    y = torch.matmul(x, x)
    print('✅ GPU 測試通過')
"

echo ""
echo "🎉 升級完成！"
echo ""
echo "下一步:"
echo "  1. 執行測試: python test_full_pipeline.py"
echo "  2. 開始訓練: python main.py new -n my_experiment -d data/coco2017"
```

### 執行升級

```bash
chmod +x upgrade_pytorch.sh
./upgrade_pytorch.sh
```

---

## 效能比較

### 預估訓練時間 (COCO 2017, 100 epochs)

| 硬體配置 | 每 epoch 時間 | 總時間 (100 epochs) |
|----------|--------------|---------------------|
| **RTX 4090 + PyTorch 2.x** | ~10 分鐘 | ~17 小時 |
| **CPU (AMD/Intel)** | ~8-15 小時 | ~33-62 天 |

**結論**: GPU 訓練快約 **120-240 倍**

---

## 快速決策指南

### ✅ 立即升級 PyTorch，如果：
- 你需要完整訓練模型
- 你想要合理的訓練時間
- 你的專案需要 GPU 加速

### ⏸️ 暫時使用 CPU，如果：
- 你只想測試程式碼邏輯
- 你在除錯特定功能
- 你的實驗只需要 1-2 個 epoch

---

## 常見問題

### Q: 升級 PyTorch 會破壞現有程式碼嗎？

**A**: 不太可能。PyTorch 2.x 大部分 API 向後相容 1.x。ARWGAN 的程式碼相對簡單，應該可以直接運行。如果有問題，通常只是警告，不影響功能。

### Q: 我可以同時保留兩個版本嗎？

**A**: 可以。建立另一個虛擬環境：

```bash
# 新環境 (PyTorch 2.x)
python3.10 -m venv venv_pytorch2
source venv_pytorch2/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 舊環境 (PyTorch 1.x)
source venv/bin/activate
```

### Q: 升級後還能降級回來嗎？

**A**: 可以。使用備份的 requirements:

```bash
pip install -r requirements_backup_YYYYMMDD_HHMMSS.txt --force-reinstall
```

---

## 總結與建議

### 🎯 強烈建議

**立即升級到 PyTorch 2.x**

理由：
1. ✅ RTX 4090 是高階 GPU，不用太浪費
2. ✅ 訓練速度快 100+ 倍
3. ✅ 升級風險低，相容性好
4. ✅ 未來專案都會用新版 PyTorch

### 📋 升級檢查清單

- [ ] 備份當前環境 (`pip freeze > backup.txt`)
- [ ] 檢查 CUDA 版本 (`nvidia-smi`)
- [ ] 升級 PyTorch 2.x
- [ ] 執行測試 (`python test_full_pipeline.py`)
- [ ] 開始 GPU 訓練

---

**下一步**: 如果決定升級，請告訴我，我會協助你完成整個過程！🚀
