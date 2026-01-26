"""
ARWGAN Attention Visualization Script
生成 3 欄式比較圖：原始圖像 | 注意力遮罩熱力圖 | 嵌入浮水印圖像
"""

import torch
import torch.nn as nn
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式後端，適合伺服器環境

import utils
from model.ARWGAN import ARWGAN
from noise_layers.noiser import Noiser
from PIL import Image
import torchvision.transforms.functional as TF


class AttentionHook:
    """用於提取注意力遮罩的 Hook 類別"""
    def __init__(self, hook_type='feature_attention'):
        """
        Args:
            hook_type: 'feature_attention' (在 Softmax 之前) 或 'sixth_layer' (在 Softmax 之後)
        """
        self.attention_mask = None
        self.hook_type = hook_type
    
    def __call__(self, module, input, output):
        """Hook 函數：提取空間注意力遮罩"""
        if self.hook_type == 'feature_attention':
            # feature_attention 的形狀是 (batch, channels, H, W)
            # 對 channels 維度取平均或最大值，得到 (batch, H, W) 的空間注意力圖
            self.attention_mask = output.mean(dim=1)  # 或使用 .max(dim=1)[0] 取最大值
        elif self.hook_type == 'sixth_layer_before_softmax':
            # sixth_layer 在 Softmax 之前的輸出形狀是 (batch, message_length, H, W)
            # 對 message_length 維度取最大值，得到每個空間位置的最大注意力值
            self.attention_mask = output.max(dim=1)[0]  # 取最大值而非平均
        else:  # 'sixth_layer_after_softmax'
            # sixth_layer 在 Softmax 之後的輸出形狀是 (batch, message_length, H, W)
            # 對 message_length 維度取最大值，得到每個空間位置的最大注意力值
            self.attention_mask = output.max(dim=1)[0]  # 取最大值而非平均
        return output


def extract_attention_mask(encoder, image, message, device):
    """
    提取注意力遮罩
    
    根據 encoder.py 的分析：
    - feature_attention: (batch, channels, H, W) - 真正的空間注意力特徵
    - sixth_layer 輸出: (batch, message_length, H, W) - 經過 Softmax(dim=1) 後，每個位置的值總和為 1
    
    我們優先使用 feature_attention，因為它代表真正的空間重要性。
    
    Args:
        encoder: Encoder 模型
        image: 輸入圖像 tensor (1, 3, H, W)
        message: 訊息 tensor (1, message_length)
        device: 設備
    
    Returns:
        attention_mask: 注意力遮罩 (H, W)
        encoded_image: 編碼後的圖像 (1, 3, H, W)
    """
    # 方法 1: Hook feature_attention（推薦）- 這是真正的空間注意力特徵
    hook_feature = AttentionHook(hook_type='feature_attention')
    
    # 方法 2: Hook sixth_layer 在 Softmax 之前（備選方案）
    # 我們需要 hook sixth_layer 內部的 conv2d 輸出（在 Softmax 之前）
    hook_sixth = AttentionHook(hook_type='sixth_layer_before_softmax')
    
    # Hook feature_attention（在 Dense_block_a3 之後）
    # 注意：我們需要 hook Dense_block_a3 的輸出
    handle_feature = None
    handle_sixth = None
    
    try:
        # Hook feature_attention: sixth_layer 的第一個模組 (BatchNorm2d) 的輸入就是 feature_attention
        # feature_attention 形狀: (batch, channels=64, H, W) - 這是真正的空間注意力特徵
        def hook_feature_input(module, input_tuple):
            # input_tuple[0] 是 feature_attention，形狀是 (batch, channels, H, W)
            feature_attn = input_tuple[0]
            # 對 channels 維度取最大值（而非平均），突出高注意力區域
            hook_feature.attention_mask = feature_attn.max(dim=1)[0]  # 取最大值以增強對比度
        
        # 使用 forward_pre_hook 來 hook 輸入
        handle_feature = encoder.sixth_layer[0].register_forward_pre_hook(hook_feature_input)
        
        # 同時 hook sixth_layer 的輸出（Softmax 之後）作為備選方案
        handle_sixth = encoder.sixth_layer.register_forward_hook(hook_sixth)
        
        # 執行前向傳播
        with torch.no_grad():
            encoded_image = encoder(image, message)
        
        # 優先使用 feature_attention（真正的空間注意力特徵）
        if hook_feature.attention_mask is not None:
            attention_mask = hook_feature.attention_mask.squeeze(0).cpu().numpy()  # (H, W)
            print("  ✓ Using feature_attention (spatial attention features from Dense_block_a3)")
        elif hook_sixth.attention_mask is not None:
            # 備選：使用 sixth_layer 輸出（取最大值而非平均，避免 Softmax 均勻分佈問題）
            attention_mask = hook_sixth.attention_mask.squeeze(0).cpu().numpy()  # (H, W)
            print("  ⚠ Using sixth_layer output (max over message_length) - may have Softmax uniformity issue")
        else:
            raise RuntimeError("Failed to extract attention mask from both hooks")
        
    finally:
        # 移除 hooks
        if handle_feature is not None:
            handle_feature.remove()
        if handle_sixth is not None:
            handle_sixth.remove()
    
    return attention_mask, encoded_image


def load_and_preprocess_image(image_path, target_size, device):
    """
    載入並預處理圖像
    
    Args:
        image_path: 圖像路徑
        target_size: 目標尺寸 (H, W)
        device: 設備
    
    Returns:
        image_tensor: 預處理後的圖像 tensor (1, 3, H, W)，範圍 [-1, 1]
        image_pil: PIL 圖像物件（用於顯示）
    """
    # 載入圖像
    image_pil = Image.open(image_path).convert('RGB')
    original_size = image_pil.size
    
    # Resize
    image_pil_resized = image_pil.resize(target_size, Image.LANCZOS)
    
    # 轉換為 Tensor [0, 1]
    image_tensor = TF.to_tensor(image_pil_resized).to(device)
    
    # 轉換到 [-1, 1] 範圍
    image_tensor = image_tensor * 2 - 1
    image_tensor = image_tensor.unsqueeze(0)  # (1, 3, H, W)
    
    return image_tensor, image_pil_resized


def tensor_to_image(tensor):
    """
    將 tensor 從 [-1, 1] 轉換為 [0, 1] 的 numpy 陣列，用於顯示
    
    Args:
        tensor: (1, 3, H, W) 或 (3, H, W)
    
    Returns:
        image: (H, W, 3) numpy 陣列，範圍 [0, 1]
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # 轉換到 [0, 1] 並轉置為 (H, W, 3)
    image = (tensor.cpu().numpy() + 1) / 2
    image = np.clip(image, 0, 1)
    image = np.transpose(image, (1, 2, 0))
    
    return image


def create_visualization(images_data, output_file, figsize=(12, 4)):
    """
    創建 3 欄式比較圖
    
    Args:
        images_data: 列表，每個元素為 (original, attention_mask, encoded, image_name)
        output_file: 輸出檔案路徑
        figsize: 圖像大小 (width, height)
    """
    num_images = len(images_data)
    
    # 創建子圖：num_images 行 x 3 列
    fig, axes = plt.subplots(num_images, 3, figsize=(figsize[0], figsize[1] * num_images))
    
    # 如果只有一張圖，確保 axes 是 2D
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    # 欄標題
    column_labels = ['(a) Original Image', '(b) Attention Mask', '(c) Encoded Image']
    
    for row_idx, (original, attention_mask, encoded, image_name) in enumerate(images_data):
        # Column (a): Original Image
        axes[row_idx, 0].imshow(original)
        axes[row_idx, 0].axis('off')
        if row_idx == 0:
            axes[row_idx, 0].set_title(column_labels[0], fontsize=12, pad=10)
        
        # Column (b): Attention Mask (熱力圖疊加在原始圖像上，符合論文 Fig. 7 樣式)
        # 將原始圖像轉為灰度作為背景
        original_gray = np.dot(original[...,:3], [0.2989, 0.5870, 0.1140])  # RGB to grayscale
        original_gray_3d = np.stack([original_gray, original_gray, original_gray], axis=-1)
        
        # 顯示灰度背景
        axes[row_idx, 1].imshow(original_gray_3d)
        
        # Debug: 印出數值範圍（在視覺化時再次確認）
        attention_min = attention_mask.min()
        attention_max = attention_mask.max()
        attention_range = attention_max - attention_min
        print(f"  [Visualization] Mask - Min: {attention_min:.6f}, Max: {attention_max:.6f}, Range: {attention_range:.6f}")
        
        # 強力歸一化 (Robust Percentile-based Normalization)
        # 使用 5th 和 95th 百分位數而非 min/max，去除極值影響並增強對比度
        if attention_range > 1e-8:
            p5 = np.percentile(attention_mask, 5)
            p95 = np.percentile(attention_mask, 95)
            percentile_range = p95 - p5
            
            if percentile_range > 1e-8:
                # 百分位數正規化，並 clip 到 [0, 1]
                attention_normalized = np.clip((attention_mask - p5) / percentile_range, 0, 1)
                print(f"  [Percentile Norm] P5: {p5:.6f}, P95: {p95:.6f}, Range: {percentile_range:.6f}")
                print(f"  [Normalized] Range: [{attention_normalized.min():.6f}, {attention_normalized.max():.6f}]")
            else:
                # 如果百分位數範圍也很小，使用標準 Min-Max
                attention_normalized = (attention_mask - attention_min) / attention_range
                print(f"  [Min-Max Norm] Using standard normalization")
        else:
            # 如果範圍太小（幾乎是常數），設為 0.5 並警告
            print(f"  [WARNING] Attention mask range too small ({attention_range:.2e}), using constant value")
            attention_normalized = np.ones_like(attention_mask) * 0.5
        
        # 疊加注意力熱力圖（使用 bicubic 插值讓熱力圖更平滑，消除馬賽克方塊）
        im = axes[row_idx, 1].imshow(attention_normalized, cmap='jet', alpha=0.6, interpolation='bicubic')
        axes[row_idx, 1].axis('off')
        if row_idx == 0:
            axes[row_idx, 1].set_title(column_labels[1], fontsize=12, pad=10)
            # 添加顏色條（只在第一行顯示）
            cbar = plt.colorbar(im, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)
            cbar.set_label('Attention Intensity', rotation=270, labelpad=15)
        
        # Column (c): Encoded Image
        axes[row_idx, 2].imshow(encoded)
        axes[row_idx, 2].axis('off')
        if row_idx == 0:
            axes[row_idx, 2].set_title(column_labels[2], fontsize=12, pad=10)
        
        # 在左側添加圖像名稱標籤
        if row_idx == 0:
            axes[row_idx, 0].text(-0.15, 0.5, image_name, 
                                  transform=axes[row_idx, 0].transAxes,
                                  rotation=90, va='center', ha='center',
                                  fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='ARWGAN Attention Visualization')
    parser.add_argument('--options-file', '-o', 
                       default='runs/arwgan_reproduce 2026.01.24--12-41-39/options-and-config.pickle',
                       type=str, help='Path to options-and-config.pickle')
    parser.add_argument('--checkpoint-file', '-c', required=True, type=str,
                       help='Path to model checkpoint file')
    parser.add_argument('--source-images', '-s', required=True, type=str,
                       help='Folder containing test images')
    parser.add_argument('--output-file', '-out', default='attention_visualization.png',
                       type=str, help='Output visualization file path')
    parser.add_argument('--image-names', '-i', nargs='+', default=None,
                       help='Specific image filenames to visualize (e.g., 4.2.03.tiff 4.2.05.tiff). If not specified, uses all images in folder.')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for message generation')
    
    args = parser.parse_args()
    
    # 設定隨機種子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 檢查設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. 載入設定和模型
    print("Loading configuration and model...")
    train_options, net_config, train_noise_config = utils.load_options(args.options_file)
    
    # 使用訓練時的 noiser 配置載入模型
    train_noiser = Noiser(train_noise_config, device).to(device)
    model = ARWGAN(net_config, device, train_noiser, None)
    
    # 載入權重
    checkpoint = torch.load(args.checkpoint_file, map_location=device)
    utils.model_from_checkpoint(model, checkpoint)
    
    model.encoder_decoder.eval()
    print(f"Model loaded from: {args.checkpoint_file}")
    print(f"Image size: {net_config.H}x{net_config.W}")
    print(f"Message length: {net_config.message_length}")
    
    # 2. 準備圖像列表
    if args.image_names:
        image_files = args.image_names
    else:
        # 獲取資料夾中的所有圖像
        all_files = os.listdir(args.source_images)
        image_files = [f for f in all_files 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))]
        image_files.sort()
    
    if len(image_files) == 0:
        print(f"Error: No images found in {args.source_images}")
        return
    
    print(f"\nProcessing {len(image_files)} images...")
    
    # 3. 處理每張圖像
    images_data = []
    
    with torch.no_grad():
        for image_file in image_files:
            image_path = os.path.join(args.source_images, image_file)
            
            if not os.path.exists(image_path):
                print(f"Warning: {image_path} not found, skipping...")
                continue
            
            print(f"Processing: {image_file}")
            
            # 載入並預處理圖像
            image_tensor, image_pil = load_and_preprocess_image(
                image_path, (net_config.H, net_config.W), device
            )
            
            # 生成隨機訊息
            message = torch.Tensor(
                np.random.choice([0, 1], (1, net_config.message_length))
            ).to(device)
            
            # 提取注意力遮罩和編碼圖像
            encoder = model.encoder_decoder.encoder
            attention_mask, encoded_image = extract_attention_mask(
                encoder, image_tensor, message, device
            )
            
            # Debug: 印出注意力遮罩的數值範圍
            mask_min = attention_mask.min()
            mask_max = attention_mask.max()
            mask_range = mask_max - mask_min
            print(f"  Attention Mask - Min: {mask_min:.6f}, Max: {mask_max:.6f}, Range: {mask_range:.6f}")
            
            # 轉換為顯示格式
            original_image = tensor_to_image(image_tensor)
            encoded_image_display = tensor_to_image(encoded_image)
            
            # 儲存數據
            image_name = os.path.splitext(image_file)[0]
            images_data.append((original_image, attention_mask, encoded_image_display, image_name))
    
    if len(images_data) == 0:
        print("Error: No images were successfully processed.")
        return
    
    # 4. 創建視覺化
    print(f"\nCreating visualization with {len(images_data)} images...")
    create_visualization(images_data, args.output_file)
    
    print("Visualization completed!")


if __name__ == '__main__':
    main()
