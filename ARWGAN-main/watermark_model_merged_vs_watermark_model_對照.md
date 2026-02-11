# watermark_model_merged.py 與 watermark_model.py 對照說明

## 一、整體定位

| 項目 | watermark_model.py | watermark_model_merged.py |
|------|--------------------|---------------------------|
| **架構** | 自訂改進版（CBAM + 簡化 Dense） | **ARWGAN 論文原始架構**（model/encoder.py、decoder.py、discriminator.py、Dense_block.py） |
| **訓練流程** | 單一迴圈、無階段 | **改進訓練框架**（來自 watermark_model_better.py）：階段式 Warm-up、Checkpoint、驗證、CSV、早停 |

---

## 二、模型架構差異

### Encoder

| 項目 | watermark_model.py | watermark_model_merged.py |
|------|--------------------|---------------------------|
| **結構** | CBAM + 4 個 Dense-style Conv，單一分支 | **ARWGAN 原始**：Dense Block (Bottleneck) × 多層、**雙分支**（主幹 + Attention 分支） |
| **水印融合** | `attended * wm_embedded + attended`，再 `fused.mean(dim=1, keepdim=True)` 加回原圖 | 主幹特徵 × **Softmax Attention Mask × 30**，`final_layer` 出 RGB 殘差，`im_w = im_w + image` |
| **輸出** | `clamp(watermarked, 0, 1)` | `clamp(im_w, 0, 1)`（適配 [0,1]，原論文為 [-1,1]） |

### Decoder

| 項目 | watermark_model.py | watermark_model_merged.py |
|------|--------------------|---------------------------|
| **結構** | **U-Net like**：Conv 下採樣 + ConvTranspose 上採樣 + skip connections | **ARWGAN 原始**：Dense Block + 多層 Conv，**AdaptiveAvgPool2d(1,1) + Linear**，無 U-Net |
| **輸出** | `extracted, pooled`（pooled 當 logits） | `extracted, logits`（同上，用於 BCE Loss） |

### Discriminator

| 項目 | watermark_model.py | watermark_model_merged.py |
|------|--------------------|---------------------------|
| **結構** | **PatchGAN**：4 個 stride-2 Conv + 最後 1 個 Conv，輸出 `.mean()` | **ARWGAN 原始**：Dense Block + 多層 Conv，**AdaptiveAvgPool2d(1,1) + Linear(watermark_bits, 1)**，再 `.mean()` |
| **參數** | 無 `watermark_bits`/channels | `watermark_bits=64, channels=64` |

---

## 三、Noise Layer 差異

| 項目 | watermark_model.py | watermark_model_merged.py |
|------|--------------------|---------------------------|
| **攻擊類型** | gaussian, **jpeg(PIL)**、crop、dropout、resize、**adv(PGD)** | gaussian、**jpeg(高斯噪聲模擬)**、crop、dropout、resize（**無 adv**） |
| **時機** | **每次 forward 隨機選一種攻擊** | **漸進式**：Epoch 1–5 無攻擊；Epoch 5–15 攻擊機率線性 0→1；Epoch 15+ 100% 攻擊 |
| **JPEG** | 用 PIL 寫檔/讀檔模擬 | 可微分、用高斯噪聲模擬，無寫檔 |
| **介面** | `forward(x, original_image=None)`，無 `set_epoch` | `set_epoch(epoch)` + `forward(x, original_image=None)` |

---

## 四、訓練流程差異

| 項目 | watermark_model.py | watermark_model_merged.py |
|------|--------------------|---------------------------|
| **階段** | 無，從頭到尾同一套 loss | **Phase 1 (Epoch 0–4)**：無 Noise、無 GAN；**Phase 2 (5–14)**：有 Noise、無 GAN；**Phase 3 (15+)**：Noise + GAN |
| **Loss 權重** | 固定 `img + 10*wm + 0.1*g_gan` | **Phase 1**：img_weight=0.3, wm_weight=25, gan=0；**Phase 3**：img=1, wm=2, gan=0.001 |
| **D Loss 計算** | `d_loss = -d_real.mean() + d_fake.mean() + gp`（disc 已回傳 scalar，`.mean()` 多餘/易錯） | `d_loss = -d_real + d_fake + gp`（正確，因回傳已是 scalar） |
| **驗證集** | 無 | **90/10 切分**，每 epoch 算驗證 BER / PSNR / SSIM、**ber_clean**（無攻擊 BER） |
| **Checkpoint** | 每 10 epoch 存 encoder/decoder/discriminator 各一檔 | **best_model.pth**（最佳驗證 BER）+ **checkpoint_epoch_N.pth**（含 optimizer、scheduler、train/val losses） |
| **恢復訓練** | 無 | `--resume` 載入 checkpoint，從 `epoch+1` 繼續 |
| **早停** | 無 | 驗證 BER 連續 15 個 epoch 未改善則停止 |
| **CSV 記錄** | 無 | **train.csv**、**validation.csv**（含 epoch、loss、ber、psnr、ssim、duration 等） |
| **VGG 損失** | 無 | 可選 `--use_vgg`（VGG 感知損失） |
| **學習率** | 單一 lr=1e-4 | 差分學習率（Encoder 1e-4、Decoder 2e-3）+ StepLR（每 30 epoch ×0.5） |
| **預設 batch** | 16 | 8（合併版較吃顯存，24GB GPU 建議 8 或 4） |

---

## 五、Dataset 與參數

| 項目 | watermark_model.py | watermark_model_merged.py |
|------|--------------------|---------------------------|
| **Dataset** | 單一目錄 `listdir` 找 `.jpg`，無子目錄/遞迴 | 多路徑搜尋、子目錄、遞迴、`_ensure_size(256,256)`、載入失敗重試、fallback 黑圖 |
| **命令列** | `--train`, `--test`, `--image`, `--epochs`, `--batch` | 同上，另加 **`--data-dir`**、**`--use_vgg`**、**`--resume`**、**`--save_dir`**、**`--watermark-bits`**、**`--channels`**、**`--checkpoint`** |
| **測試** | 需分別提供 encoder / decoder 路徑 | 單一 **checkpoint**（如 best_model.pth），內含 encoder/decoder/discriminator |

---

## 六、小結

- **watermark_model.py**：較輕量的「改進版」架構（CBAM + U-Net Decoder + PatchGAN），訓練流程簡單，無階段、無驗證、無 checkpoint/CSV。
- **watermark_model_merged.py**：採用 **ARWGAN 論文原始 Encoder/Decoder/Discriminator**，搭配 **watermark_model_better.py 的訓練框架**（階段式 Noise、Warm-up、驗證、Checkpoint、CSV、早停、VGG、差分學習率），並適配圖像 [0,1]、預設較小 batch 以降低 OOM。

若要以「論文架構 + 完整訓練流程」做實驗，請用 **watermark_model_merged.py**；若只要快速試小模型與簡單訓練，可用 **watermark_model.py**。
