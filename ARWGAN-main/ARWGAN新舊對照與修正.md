# ARWGAN 新程式 vs 原版 對照與修正

## 新程式的優點 ✅

1. **mse_pixel_loss 用於 PSNR**  
   - 驗證時明確用 `mse_pixel_loss`（原圖 vs 編碼圖的 pixel MSE）算 PSNR，避免使用 VGG loss。  
   - 當 `use_vgg=True` 時，原版用 `g_loss_enc`（VGG MSE）算 PSNR，與「像素級畫質」不符；新版的作法正確。

2. **Phase 4 權重**  
   - `mse=2.0, ssim=0.5, decode=1.0` 延續你之前的調參結論，方向合理。

3. **程式結構**  
   - 註解、段落更清楚，可讀性較好。

---

## 必須修正的 Bug ❌

### 1. PSNR 公式錯誤（嚴重）

- **新程式**：`psnr = 10 * log10(1.0 / mse_pixel_loss)`
- **問題**：資料經過 `Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])`，數值在 **[-1, 1]**，動態範圍為 2，所以 `MAX^2 = 4`。
- **正確**：`psnr = 10 * log10(4.0 / mse_pixel_loss)`
- **影響**：用 1.0 會讓 PSNR 系統性低估約 **6 dB**（例如 28 dB 會被算成 22 dB）。

### 2. `to_string` 會導致 main 炸掉

- **新程式**：改為 `to_string`。
- **問題**：`main.py` 呼叫的是 `model.to_stirng()`（拼錯）。  
  若只改 ARWGAN 會 `AttributeError`。
- **作法**：維持 `to_stirng`，或同時改 `main.py` 的呼叫。建議先維持 `to_stirng`，確保可跑。

### 3. 驗證的 `ssim` 欄位語意改變

- **原版**：`'ssim': 1 - g_loss_enc_ssim` → 存的是 **SSIM loss**（1 - 相似度）。
- **新程式**：`'ssim': g_loss_enc_ssim` → 存的是 **SSIM 相似度**。  
  `SSIM` 回傳的是相似度，所以 `g_loss_enc_ssim` 本身就是 SSIM 值。
- **影響**：`validation.csv` 的 `ssim` 從「loss」變成「相似度」，數值會反向，舊的畫圖/分析腳本若不改會錯。
- **建議**：  
  - 若要與舊 CSVs、舊腳本相容：維持 `'ssim': 1 - g_loss_enc_ssim`。  
  - 若你確定要「直接顯示 SSIM 值」且會更新所有讀取程式：可改成 `g_loss_enc_ssim`。

---

## 可選 / 未完成部分

- **train 的 mse_pixel**：  
  新程式算了 `mse_pixel_loss` 想拿來「Log 觀察」，但 `losses` 裡仍只存 `g_loss_enc`。  
  若 `use_vgg=True`，可考慮改存 `mse_pixel_loss` 作為 `encoder_mse`，方便觀察純 pixel 誤差；  
  若保持與原版完全相同則不改也無妨。

---

## 結論

- 新程式的 **mse_pixel + PSNR 思路、權重、結構** 都比原版好，值得採用。  
- 但 **必須** 修正：  
  1. PSNR 用 `4.0 / mse_pixel_loss`；  
  2. 保留 `to_stirng`（或一併改 main）；  
  3. 依你是否要相容舊 CSV，決定 `ssim` 存 `1 - g_loss_enc_ssim` 還是 `g_loss_enc_ssim`。  

修正後，新程式會比原版更正確、也較符合你現在的訓練目標（Phase 4、PSNR 觀察）。

---

## 已套用的修正（model/ARWGAN.py）

- **PSNR**：改為 `10 * log10(4.0 / mse_pixel_loss)`，且驗證一律用 `mse_pixel_loss`。
- **`to_stirng`**：維持原拼法，與 `main.py` 相容。
- **`ssim`**：維持 `1 - g_loss_enc_ssim`，與既有 `validation.csv` 語意一致。
- **訓練**：加入 `mse_pixel_loss`，VGG 關閉時 `g_loss_enc = mse_pixel_loss`；VGG 開啟時仍用 VGG 做優化，但 PSNR 一律用 pixel MSE。
