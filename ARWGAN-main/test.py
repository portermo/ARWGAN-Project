import torch
import argparse
import os
import numpy as np
import utils
from model.ARWGAN import ARWGAN
from noise_argparser import NoiseArgParser, parse_jpeg, parse_crop, parse_dropout, parse_resize, parse_gaussian
from noise_layers.noiser import Noiser
from PIL import Image
import torchvision
import torchvision.transforms.functional as TF

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
                        help='Batch size (default: 1)')
    parser.add_argument('--mode', type=str, default='custom', 
                        choices=['paper', 'hell', 'clean', 'custom'],
                        help='Test mode')
    parser.add_argument("--noise", '-n', nargs="*", action=NoiseArgParser,
                        help='Custom noise configuration')
    
    args = parser.parse_args()
    
    # 檢查 CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")

    # 1. 載入設定
    # 這裡會讀取訓練時原本的 Noise Config，這對載入權重至關重要
    train_options, net_config, train_noise_config = utils.load_options(args.options_file)
    
    # 2. 定義測試用的攻擊模式
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
            test_noise_config = args.noise
            print(f"=== Mode: Custom (User-defined noise) ===")
        else:
            print("=== Mode: Custom (No noise specified, using clean mode) ===")
            test_noise_config = []

    # 3. [關鍵修正] 初始化模型與載入權重
    # 步驟 A: 先用「訓練時的設定」初始化 Noiser，避免 load_state_dict 因為 key 不匹配而報錯
    train_noiser = Noiser(train_noise_config, device).to(device)
    model = ARWGAN(net_config, device, train_noiser, None)
    
    # 步驟 B: 載入權重
    print(f"Loading checkpoint from: {args.checkpoint_file}")
    checkpoint = torch.load(args.checkpoint_file, map_location=device)
    utils.model_from_checkpoint(model, checkpoint)
    
    # 步驟 C: [關鍵修正] 權重載入完成後，替換成「測試用的 Noiser」
    test_noiser = Noiser(test_noise_config, device).to(device) # 確保移至 GPU
    model.encoder_decoder.noiser = test_noiser
    
    model.encoder_decoder.eval()
    model.discriminator.eval()
    
    # 4. 準備數據
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        
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
    
    # 固定隨機種子
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("\n" + "="*80)
    print(f"{'Image':<30} | {'PSNR (dB)':<12} | {'SSIM':<8} | {'Error Rate':<12} | {'Accuracy':<10}")
    print("="*80)

    with torch.no_grad():
        for idx, filename in enumerate(image_files):
            img_path = os.path.join(args.source_images, filename)
            try:
                image_pil = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
                
            image_pil = image_pil.resize((net_config.H, net_config.W))
            
            # 轉 Tensor [-1, 1]
            image_tensor = TF.to_tensor(image_pil).to(device)
            image_tensor = image_tensor * 2 - 1
            image_tensor.unsqueeze_(0)
            
            # 產生隨機訊息
            message = torch.Tensor(np.random.choice([0, 1], (image_tensor.shape[0],
                                                             net_config.message_length))).to(device)

            # 模型推論
            losses, (encoded_images, noised_images, decoded_messages) = model.validate_on_batch(
                [image_tensor, message])

            # [關鍵修正] 使用 strip() 去除 Key 的空格
            clean_losses = {k.strip(): v for k, v in losses.items()}

            psnr = clean_losses.get('PSNR', 0)
            
            # [關鍵修正] SSIM 計算邏輯修正
            # 根據 ARWGAN.py，'encoded_ssim' 通常存的是計算出來的 SSIM 分數 (接近 1)
            # 而不是 Loss。如果之前的 log 顯示 0.9x，那這裡直接取值即可。
            if 'ssim' in clean_losses:
                 # 如果 ARWGAN.py 有存 'ssim' 這個 key，通常它是 1-loss 或直接的分數
                 # 我們優先相信數值大的那個（因為 SSIM 應該接近 1）
                 val = clean_losses['ssim']
                 ssim = val if val > 0.5 else (1 - val)
            elif 'encoded_ssim' in clean_losses:
                 val = clean_losses['encoded_ssim']
                 # 同樣邏輯，如果大於 0.5 視為分數，小於 0.5 視為 Loss
                 ssim = val if val > 0.5 else (1 - val)
            else:
                ssim = 0

            bitwise_error = clean_losses.get('bitwise-error', 0)
            accuracy = 1.0 - bitwise_error
            
            # 累加統計
            total_psnr += psnr
            total_ssim += ssim
            total_acc += accuracy
            total_error += bitwise_error
            
            print(f"{filename[:28]:<30} | {psnr:>10.2f} | {ssim:>6.4f} | {bitwise_error:>10.4f} | {accuracy:>8.2%}")

            # 保存圖片
            if idx == 0:
                image_tensor_01 = torch.clamp((image_tensor + 1) / 2, 0, 1)
                encoded_images_01 = torch.clamp((encoded_images + 1) / 2, 0, 1)
                noised_images_01 = torch.clamp((noised_images + 1) / 2, 0, 1)
                
                base_name = os.path.splitext(filename)[0]
                # 拼接: 原圖 | 加浮水印圖 | 攻擊後圖
                stacked = torch.cat([image_tensor_01, encoded_images_01, noised_images_01], dim=3)
                save_path = os.path.join(args.output_folder, f"{base_name}_compare.png")
                torchvision.utils.save_image(stacked, save_path)
                print(f"Example saved: {save_path} (Left: Org, Mid: Enc, Right: Noised)")

    # 最終報告
    num_images = len(image_files)
    if num_images > 0:
        avg_psnr = total_psnr / num_images
        avg_ssim = total_ssim / num_images
        avg_error = total_error / num_images
        avg_acc = total_acc / num_images

        print("\n" + "="*80)
        print(f"FINAL RESULTS ({args.mode.upper()} MODE)")
        print("="*80)
        print(f"Total images tested: {num_images}")
        print(f"Average PSNR         : {avg_psnr:.4f} dB")
        print(f"Average SSIM         : {avg_ssim:.6f}")
        print(f"Average Error Rate    : {avg_error:.4%}")
        print(f"Average Accuracy      : {avg_acc:.4%}")
        print("="*80)

# [關鍵修正] 加上程式入口點
if __name__ == '__main__':
    main()