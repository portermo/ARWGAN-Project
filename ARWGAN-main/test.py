import torch
import torch.nn
import argparse
import os
import math
import numpy as np
import utils
from model.ARWGAN import ARWGAN
from noise_argparser import NoiseArgParser, parse_jpeg, parse_crop, parse_dropout, parse_resize, parse_gaussian
from noise_layers.noiser import Noiser
from PIL import Image
import torchvision
import torchvision.transforms.functional as TF


def calculate_accuracy(decoded, original):
    """計算位元準確率和錯誤率"""
    decoded_rounded = decoded.round().clip(0, 1)
    diff = np.abs(decoded_rounded - original)
    error_rate = np.mean(diff)
    accuracy = 1.0 - error_rate
    return accuracy, error_rate


def main():
    # 設定參數
    parser = argparse.ArgumentParser(description='ARWGAN Model Evaluation Script')
    parser.add_argument('--options-file', '-o', default='options-and-config.pickle', type=str,
                        help='The file where the simulation options are stored.')
    parser.add_argument('--checkpoint-file', '-c', required=True, type=str, 
                        help='Model checkpoint file')
    parser.add_argument('--source-images', '-s', required=True, type=str,
                        help='Folder containing test images')
    parser.add_argument('--output-folder', '-out', default='test_results', type=str,
                        help='Output folder for result images')
    parser.add_argument('--batch-size', '-b', default=1, type=int, 
                        help='Batch size (default: 1 for single image testing)')
    parser.add_argument('--mode', type=str, default='custom', 
                        choices=['paper', 'hell', 'clean', 'custom'],
                        help='Test mode: paper (Jpeg+Crop), hell (5 noises), clean (no noise), or custom (use --noise)')
    parser.add_argument("--noise", '-n', nargs="*", action=NoiseArgParser,
                        help='Custom noise configuration (only used in custom mode)')
    
    args = parser.parse_args()
    
    # 檢查 CUDA
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Testing on device: {device}")
    else:
        device = torch.device('cpu')
        print(f"Testing on device: {device} (CPU mode)")

    # 1. 載入設定（包含訓練時的 noise_config）
    train_options, net_config, train_noise_config = utils.load_options(args.options_file)
    
    # 2. 使用訓練時的 noise_config 建立模型並載入權重
    # （必須用相同配置才能正確載入 state_dict）
    train_noiser = Noiser(train_noise_config, device)
    checkpoint = torch.load(args.checkpoint_file, map_location=device)
    model = ARWGAN(net_config, device, train_noiser, None)
    utils.model_from_checkpoint(model, checkpoint)
    model.encoder_decoder.eval()
    model.discriminator.eval()
    
    # 3. 定義測試用的攻擊模式
    if args.mode == 'paper':
        print("=== Mode: Paper Reproduction (Jpeg(50) + Crop(3.5%)) ===")
        test_noise_config = [
            parse_jpeg('Jpeg(50)'),
            parse_crop('crop((0.965,0.965),(0.965,0.965))')
        ]
    elif args.mode == 'hell':
        print("=== Mode: Hell Mode (5 Noise Attacks) ===")
        test_noise_config = [
            parse_dropout('dropout(0.3,0.3)'),
            parse_crop('crop((0.965,0.965),(0.965,0.965))'),
            parse_resize('resize(0.8,0.8)'),
            parse_jpeg('Jpeg(50)'),
            parse_gaussian('gaussian(3,2.0)')
        ]
    elif args.mode == 'clean':
        print("=== Mode: Clean (No Noise) ===")
        test_noise_config = []
    else:  # custom mode
        if args.noise:
            test_noise_config = args.noise  # NoiseArgParser 已經解析成物件了
            print(f"=== Mode: Custom (User-defined noise) ===")
            print(f"Noise config: {[type(n).__name__ for n in test_noise_config]}")
        else:
            print("=== Mode: Custom (No noise specified, using clean mode) ===")
            test_noise_config = []

    # 4. 替換為測試用的 Noiser，並確保移動到正確的設備
    test_noiser = Noiser(test_noise_config, device).to(device)
    model.encoder_decoder.noiser = test_noiser
    
    print(f"Model loaded from: {args.checkpoint_file}")
    print(f"Image size: {net_config.H}x{net_config.W}")
    print(f"Message length: {net_config.message_length}")
    
    # 4. 準備數據
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        print(f"Created output folder: {args.output_folder}")
        
    # 獲取所有圖像檔案
    image_files = [f for f in os.listdir(args.source_images) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))]
    
    if len(image_files) == 0:
        print(f"Error: No images found in {args.source_images}")
        return
    
    print(f"\nFound {len(image_files)} images in {args.source_images}")
    
    # 統計變數
    total_psnr = 0
    total_ssim = 0
    total_acc = 0
    total_error = 0
    total_loss = 0
    
    # 固定隨機種子，方便重現結果
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("\n" + "="*80)
    print(f"{'Image':<30} | {'PSNR (dB)':<12} | {'SSIM':<8} | {'Error Rate':<12} | {'Accuracy':<10}")
    print("="*80)

    with torch.no_grad():
        for idx, filename in enumerate(image_files):
            # 讀取圖片
            img_path = os.path.join(args.source_images, filename)
            try:
                image_pil = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
                
            image_pil = image_pil.resize((net_config.H, net_config.W))
            
            # 轉 Tensor 並轉換到 [-1, 1] 範圍（與訓練一致）
            image_tensor = TF.to_tensor(image_pil).to(device)
            image_tensor = image_tensor * 2 - 1  # 轉換到 [-1, 1]
            image_tensor.unsqueeze_(0)
            
            # 產生隨機訊息
            message = torch.Tensor(np.random.choice([0, 1], (image_tensor.shape[0],
                                                             net_config.message_length))).to(device)

            # 模型推論（使用 validate_on_batch 獲取完整評估）
            losses, (encoded_images, noised_images, decoded_messages) = model.validate_on_batch(
                [image_tensor, message])

            # 從 losses 中提取指標（注意：key 帶有尾部空格用於格式化）
            psnr = losses.get('PSNR           ', 0)
            ssim = losses.get('encoded_ssim   ', 0)  # 這是真正的 SSIM 值
            bitwise_error = losses.get('bitwise-error  ', 0)
            accuracy = 1.0 - bitwise_error
            
            # 累加統計
            total_psnr += psnr
            total_ssim += ssim
            total_acc += accuracy
            total_error += bitwise_error
            total_loss += losses.get('loss           ', 0)
            
            # 顯示結果
            print(f"{filename[:28]:<30} | {psnr:>10.2f} | {ssim:>6.4f} | {bitwise_error:>10.4f} | {accuracy:>8.2%}")

            # 保存每張圖片的結果
            # 轉換回 [0, 1] 範圍以便保存
            image_tensor_01 = torch.clamp((image_tensor + 1) / 2, 0, 1)
            encoded_images_01 = torch.clamp((encoded_images + 1) / 2, 0, 1)
            
            # 保存原圖和編碼圖的對比
            base_name = os.path.splitext(filename)[0]
            stacked = torch.cat([image_tensor_01, encoded_images_01], dim=3)  # 水平拼接
            save_path = os.path.join(args.output_folder, f"{base_name}_compare.png")
            torchvision.utils.save_image(stacked, save_path)

    # 5. 最終報告
    num_images = len(image_files)
    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images
    avg_error = total_error / num_images
    avg_acc = total_acc / num_images
    avg_loss = total_loss / num_images

    print("\n" + "="*80)
    print(f"FINAL RESULTS ({args.mode.upper()} MODE)")
    print("="*80)
    print(f"Total images tested: {num_images}")
    print(f"Average Loss         : {avg_loss:.6f}")
    print(f"Average PSNR         : {avg_psnr:.4f} dB")
    print(f"Average SSIM          : {avg_ssim:.6f}")
    print(f"Average Error Rate    : {avg_error:.4%}")
    print(f"Average Accuracy      : {avg_acc:.4%}")
    print("="*80)
    print(f"Results saved to: {args.output_folder}")

if __name__ == '__main__':
    main()
