#!/usr/bin/env python3
"""
debug_training_step.py — 診斷 "Model Collapse to Identity"

在訓練的一個 Step 中印出：
1. Encoder 輸出統計 (attended, wm_embedded, residual, watermarked vs images L1)
2. 梯度流檢查 (encoder.to_rgb.weight.grad, decoder.block1[0].weight.grad)
3. Loss 貢獻 (img_loss / wm_loss 原始數值)

用於判斷：權重沒更新 vs 數值太小（residual < 1e-5）。
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path

# 從 watermark_model_better 匯入模型與資料
from watermark_model_better import (
    Encoder,
    Decoder,
    NoiseLayer,
    Discriminator,
    VGGLoss,
    WatermarkDataset,
    ssim_loss,
    wgan_gp_loss,
)
from torch.utils.data import DataLoader


def _tensor_stats(t, name, grad=False):
    """印出 Tensor 的 Mean, Std, Min, Max；若 grad=True 則印 Grad Mean。"""
    if t is None:
        print(f"  {name}: None")
        return
    t = t.detach().float()
    mean, std = t.mean().item(), t.std().item()
    min_, max_ = t.min().item(), t.max().item()
    print(f"  {name}: mean={mean:.6e}, std={std:.6e}, min={min_:.6e}, max={max_:.6e}")
    if grad and t.requires_grad and t.grad is not None:
        g = t.grad
        print(f"    -> grad_mean={g.mean().item():.6e}, grad_std={g.std().item():.6e}")


def run_one_step_with_debug(
    encoder,
    decoder,
    noise_layer,
    discriminator,
    vgg_loss_fn,
    images,
    watermarks,
    device,
    gan_enabled=False,
):
    """
    執行一個訓練 Step（與 train_model 邏輯一致），並在過程中收集並印出診斷資訊。
    """
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    # ---------- 用 Hook 擷取 Encoder 中間輸出 ----------
    captured = {}

    def hook_cbam(module, inp, out):
        captured["attended"] = out.detach()

    def hook_wm_embed(module, inp, out):
        captured["wm_embedded"] = out.detach()

    def hook_to_rgb(module, inp, out):
        captured["fused"] = inp[0].detach()
        captured["to_rgb_out"] = out.detach()

    h_cbam = encoder.cbam.register_forward_hook(hook_cbam)
    h_wm = encoder.wm_embed.register_forward_hook(hook_wm_embed)
    h_rgb = encoder.to_rgb.register_forward_hook(hook_to_rgb)

    # ---------- Forward: Encoder ----------
    watermarked = encoder(images, watermarks)
    residual = captured["to_rgb_out"] * encoder.residual_scale
    l1_diff = (watermarked - images).abs().mean().item()

    h_cbam.remove()
    h_wm.remove()
    h_rgb.remove()

    # ---------- Forward: Noise + Decoder ----------
    noised = noise_layer(watermarked, original_image=images)
    extracted, logits = decoder(noised)

    # ---------- Losses（與 train_model 一致）----------
    mse_img_loss = mse_loss(watermarked, images)
    ssim_img_loss = ssim_loss(watermarked, images)
    wm_loss_raw = bce_loss(logits, watermarks)
    if vgg_loss_fn is not None:
        vgg_real = vgg_loss_fn(images)
        vgg_fake = vgg_loss_fn(watermarked)
        vgg_perceptual_loss = mse_loss(vgg_fake, vgg_real)
        img_loss = 0.5 * mse_img_loss + 0.3 * ssim_img_loss + 0.2 * vgg_perceptual_loss
    else:
        img_loss = mse_img_loss + ssim_img_loss
    if gan_enabled:
        g_gan_loss = -discriminator(watermarked).mean()
    else:
        g_gan_loss = torch.tensor(0.0, device=device)
    current_img_weight = 1.0
    current_wm_weight = 2.0
    current_gan_weight = 0.001 if gan_enabled else 0.0
    g_loss = current_img_weight * img_loss + current_wm_weight * wm_loss_raw + current_gan_weight * g_gan_loss

    # ---------- Backward ----------
    encoder.zero_grad()
    decoder.zero_grad()
    g_loss.backward()

    # ---------- 印出診斷報告 ----------
    print("\n" + "=" * 70)
    print("  [Debug] 單步訓練診斷 — Model Collapse to Identity")
    print("=" * 70)

    print("\n--- 1. Check Encoder Output ---")
    _tensor_stats(captured.get("attended"), "attended (CBAM output)")
    _tensor_stats(captured.get("wm_embedded"), "wm_embedded (Conv output)")
    _tensor_stats(residual, "residual (after scaling, 最重要)")
    print(f"  watermarked vs images 平均絕對差異 (L1 Diff): {l1_diff:.6e}")

    residual_abs_max = residual.abs().max().item()
    residual_mean_abs = residual.abs().mean().item()
    if residual_mean_abs < 1e-5:
        print(f"  >>> 警告: residual 平均絕對值 {residual_mean_abs:.2e} < 1e-5 → 幾乎未嵌入，需增大初始化或 residual_scale")
    if residual_abs_max < 1e-4:
        print(f"  >>> 警告: residual 最大絕對值 {residual_abs_max:.2e} < 1e-4 → 訊號過小")

    print("\n--- 2. Check Gradient Flow ---")
    if encoder.to_rgb.weight.grad is not None:
        g = encoder.to_rgb.weight.grad
        print(f"  encoder.to_rgb.weight.grad: mean={g.mean().item():.6e}, std={g.std().item():.6e}, norm={g.norm().item():.6e}")
        if g.abs().max().item() < 1e-8:
            print("  >>> 警告: to_rgb 梯度極小，可能未收到來自 Decoder 的浮水印梯度")
    else:
        print("  encoder.to_rgb.weight.grad: None (未更新)")
    if decoder.block1[0].weight.grad is not None:
        g = decoder.block1[0].weight.grad
        print(f"  decoder.block1[0].weight.grad: mean={g.mean().item():.6e}, std={g.std().item():.6e}, norm={g.norm().item():.6e}")
    else:
        print("  decoder.block1[0].weight.grad: None")

    print("\n--- 3. Check Loss Contribution (未加權前) ---")
    print(f"  img_loss (MSE+SSIM(+VGG)): {img_loss.item():.6e}")
    print(f"  wm_loss (BCE raw):         {wm_loss_raw.item():.6e}")
    print(f"  g_gan_loss:                {g_gan_loss.item():.6e}")
    print(f"  加權後 g_loss:             {g_loss.item():.6e} (img*{current_img_weight} + wm*{current_wm_weight} + gan*{current_gan_weight})")
    print("  若 img_loss 遠小於 wm_loss，且 residual 接近 0，代表 Encoder 在最小化 img_loss 時選擇不嵌入水印。")

    print("\n--- 4. Hypothesis 檢查 ---")
    to_rgb_w = encoder.to_rgb.weight.detach()
    print(f"  Encoder to_rgb 權重是否在更新: 梯度存在且 norm > 0 → {'是' if encoder.to_rgb.weight.grad is not None and encoder.to_rgb.weight.grad.norm().item() > 1e-10 else '否/極弱'}")
    print(f"  residual 是否 < 1e-5 (平均絕對): {residual_mean_abs:.2e} → {'是，需增大 scaling/初始化' if residual_mean_abs < 1e-5 else '否'}")
    print("=" * 70 + "\n")

    return {
        "residual_mean_abs": residual_mean_abs,
        "residual_abs_max": residual_abs_max,
        "l1_diff": l1_diff,
        "img_loss": img_loss.item(),
        "wm_loss_raw": wm_loss_raw.item(),
        "g_loss": g_loss.item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Debug training step for Model Collapse diagnosis")
    parser.add_argument("--data-dir", type=str, default=None, help="資料集路徑（與 train 相同）")
    parser.add_argument("--checkpoint", type=str, default=None, help="可選：載入檢查點以診斷已訓練模型")
    parser.add_argument("--batch", type=int, default=4, help="Batch size for debug")
    parser.add_argument("--no-vgg", action="store_true", help="關閉 VGG 損失（與部分 train 設定一致）")
    parser.add_argument("--gan", action="store_true", help="模擬 Phase3 啟用 GAN")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # 資料
    data_dir = args.data_dir
    if data_dir is None:
        for path in ["./data/coco2017/train/images", "./data/coco/images/train2017", "./data/train"]:
            if Path(path).exists():
                data_dir = path
                break
    if data_dir is None:
        raise FileNotFoundError("請指定 --data-dir 或將資料放在 ./data/coco2017/train/images 等路徑")
    dataset = WatermarkDataset(root_dir=data_dir)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=0)
    images, watermarks = next(iter(loader))
    images, watermarks = images.to(device), watermarks.to(device)

    # 模型
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    noise_layer = NoiseLayer(device).to(device)
    discriminator = Discriminator().to(device)
    vgg_loss_fn = None if args.no_vgg else VGGLoss().to(device)

    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        encoder.load_state_dict(ckpt["encoder_state_dict"], strict=False)
        decoder.load_state_dict(ckpt["decoder_state_dict"], strict=False)
        discriminator.load_state_dict(ckpt["discriminator_state_dict"], strict=False)
        print(f"已載入檢查點: {args.checkpoint}\n")

    encoder.train()
    decoder.train()
    noise_layer.set_epoch(10)

    run_one_step_with_debug(
        encoder,
        decoder,
        noise_layer,
        discriminator,
        vgg_loss_fn,
        images,
        watermarks,
        device,
        gan_enabled=args.gan,
    )


if __name__ == "__main__":
    main()
