"""
診斷 checkpoint 載入問題
執行: python check_checkpoint.py ./checkpoints_merged/checkpoint_epoch_29.pth
"""
import torch
import sys
from pathlib import Path

# 加入當前目錄以導入 watermark_model_merged
sys.path.insert(0, str(Path(__file__).parent))
from watermark_model_merged import Encoder, Decoder, Bottleneck

def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "./checkpoints_merged/checkpoint_epoch_29.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"檢查 checkpoint: {ckpt_path}")
    
    # 1. 載入 checkpoint
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        print("✓ Checkpoint 載入成功")
    except Exception as e:
        print(f"✗ Checkpoint 載入失敗: {e}")
        return
    
    # 2. 建立新模型（不載入權重）
    print("\n建立新模型...")
    encoder = Encoder(watermark_bits=64, channels=64).to(device)
    
    # 3. 測試 forward（無 checkpoint）
    print("測試 forward（未載入 checkpoint）...")
    try:
        x = torch.randn(2, 3, 256, 256, device=device)
        wm = torch.rand(2, 64, device=device)
        with torch.no_grad():
            out = encoder(x, wm)
        print(f"✓ Forward 成功，輸出 shape: {out.shape}")
    except Exception as e:
        print(f"✗ Forward 失敗: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. 載入 checkpoint 的 state_dict
    if 'encoder_state_dict' in ckpt:
        print("\n載入 encoder state_dict...")
        missing, unexpected = encoder.load_state_dict(ckpt['encoder_state_dict'], strict=False)
        print(f"  missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
        if missing:
            print(f"  missing: {missing[:5]}..." if len(missing) > 5 else f"  missing: {missing}")
    
    # 5. 再次測試 forward
    print("\n測試 forward（載入 checkpoint 後）...")
    try:
        with torch.no_grad():
            out = encoder(x, wm)
        print(f"✓ Forward 成功，輸出 shape: {out.shape}")
    except Exception as e:
        print(f"✗ Forward 失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
