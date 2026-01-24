import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiffJPEG(nn.Module):
    """
    Differentiable JPEG compression layer implemented in pure PyTorch.
    Fully vectorized, no loops (except initialization), GPU-optimized.
    """
    
    def __init__(self, factor=1.0):
        super(DiffJPEG, self).__init__()
        
        # Register quantization factor (can be learned or fixed)
        self.factor = factor
        
        # Pre-calculate DCT basis matrices (8x8)
        # These are registered as buffers so they move with the model
        self.register_buffer('dct_weights', self._build_dct_matrix())
        self.register_buffer('y_quant_table', self._build_y_quant_table())
        self.register_buffer('c_quant_table', self._build_c_quant_table())
        
    def _build_dct_matrix(self):
        """Build 8x8 DCT transformation matrix (64x64)."""
        # Create DCT basis: C(u,v) = alpha(u) * alpha(v) * cos((2x+1)u*pi/16) * cos((2y+1)v*pi/16)
        # where alpha(0) = 1/sqrt(2), alpha(k) = 1 for k > 0
        n = 8
        dct_matrix = torch.zeros(n * n, n * n, dtype=torch.float32)
        
        # Build DCT matrix: maps 8x8 pixel blocks to 8x8 DCT coefficients
        for u in range(n):
            for v in range(n):
                alpha_u = 1.0 / np.sqrt(2.0) if u == 0 else 1.0
                alpha_v = 1.0 / np.sqrt(2.0) if v == 0 else 1.0
                alpha = alpha_u * alpha_v * 0.25
                
                for x in range(n):
                    for y in range(n):
                        dct_val = alpha * np.cos((2 * x + 1) * u * np.pi / 16) * np.cos((2 * y + 1) * v * np.pi / 16)
                        # Map 2D position (x,y) to 1D index (row-major)
                        idx_in = x * n + y
                        # Map 2D frequency (u,v) to 1D index (row-major)
                        idx_out = u * n + v
                        dct_matrix[idx_out, idx_in] = dct_val
        
        # IDCT is the transpose of DCT (for orthonormal DCT)
        return dct_matrix
    
    def _build_y_quant_table(self):
        """Build luminance quantization table (standard JPEG)."""
        y_table = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=np.float32).T
        return torch.from_numpy(y_table)
    
    def _build_c_quant_table(self):
        """Build chrominance quantization table (standard JPEG)."""
        c_table = np.full((8, 8), 99, dtype=np.float32)
        c_table[:4, :4] = np.array([
            [17, 18, 24, 47],
            [18, 21, 26, 66],
            [24, 26, 56, 99],
            [47, 66, 99, 99]
        ], dtype=np.float32).T
        return torch.from_numpy(c_table)
    
    def _rgb_to_ycbcr(self, image):
        """Convert RGB to YCbCr color space (vectorized, matching utils.py)."""
        # Input: (B, 3, H, W) in range [0, 1]
        # Output: (B, 3, H, W) with Y in [0, 1], Cb/Cr in [0.5-0.5, 0.5+0.5]
        r, g, b = image[:, 0], image[:, 1], image[:, 2]
        
        # Standard ITU-R BT.601 coefficients (matching utils.py)
        delta = 0.5
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = (b - y) * 0.564 + delta
        cr = (r - y) * 0.713 + delta
        
        return torch.stack([y, cb, cr], dim=1)
    
    def _ycbcr_to_rgb(self, image):
        """Convert YCbCr to RGB color space (vectorized, matching utils.py)."""
        # Input: (B, 3, H, W)
        # Output: (B, 3, H, W) in range [0, 1]
        y, cb, cr = image[:, 0], image[:, 1], image[:, 2]
        
        # Standard ITU-R BT.601 coefficients (matching utils.py)
        delta = 0.5
        cb_shifted = cb - delta
        cr_shifted = cr - delta
        
        r = y + 1.403 * cr_shifted
        g = y - 0.714 * cr_shifted - 0.344 * cb_shifted
        b = y + 1.773 * cb_shifted
        
        return torch.clamp(torch.stack([r, g, b], dim=1), 0.0, 1.0)
    
    def _image_to_blocks(self, image):
        """Split image into 8x8 blocks (vectorized)."""
        # Input: (B, C, H, W)
        # Output: (B, C, H//8, W//8, 8, 8)
        B, C, H, W = image.shape
        
        # Pad to multiple of 8
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_h > 0 or pad_w > 0:
            image = F.pad(image, (0, pad_w, 0, pad_h), mode='reflect')
            H, W = H + pad_h, W + pad_w
        
        # Reshape to blocks: (B, C, H, W) -> (B, C, H//8, 8, W//8, 8) -> (B, C, H//8, W//8, 8, 8)
        image = image.view(B, C, H // 8, 8, W // 8, 8)
        image = image.permute(0, 1, 2, 4, 3, 5).contiguous()
        return image, H, W, pad_h, pad_w
    
    def _blocks_to_image(self, blocks, orig_h, orig_w, pad_h, pad_w):
        """Reconstruct image from 8x8 blocks (vectorized)."""
        # Input: (B, C, H//8, W//8, 8, 8)
        # Output: (B, C, orig_h, orig_w)
        B, C, n_h, n_w, _, _ = blocks.shape
        
        # Reshape back: (B, C, n_h, n_w, 8, 8) -> (B, C, n_h, 8, n_w, 8) -> (B, C, n_h*8, n_w*8)
        blocks = blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
        image = blocks.view(B, C, n_h * 8, n_w * 8)
        
        # Crop to original size
        if pad_h > 0 or pad_w > 0:
            image = image[:, :, :orig_h, :orig_w]
        
        return image
    
    def _dct_2d(self, blocks):
        """Apply 2D DCT to 8x8 blocks using matrix multiplication (fully vectorized)."""
        # Input: (B, C, n_h, n_w, 8, 8)
        # Output: (B, C, n_h, n_w, 8, 8)
        B, C, n_h, n_w, _, _ = blocks.shape
        
        # Flatten blocks to (B*C*n_h*n_w, 64)
        blocks_flat = blocks.view(-1, 64)
        
        # Center pixel values around 0 (JPEG standard: subtract 128 from [0,255] range)
        # Since our input is [0,1] range, we subtract 0.5 and scale to [-128, 127]
        blocks_flat = blocks_flat * 255.0 - 128.0
        
        # Apply DCT: output = DCT_matrix @ input (fully vectorized matrix multiplication)
        dct_flat = torch.matmul(self.dct_weights, blocks_flat.t()).t()
        
        # Reshape back
        dct_blocks = dct_flat.view(B, C, n_h, n_w, 8, 8)
        return dct_blocks
    
    def _idct_2d(self, dct_blocks):
        """Apply 2D IDCT to 8x8 blocks using matrix multiplication (fully vectorized)."""
        # Input: (B, C, n_h, n_w, 8, 8)
        # Output: (B, C, n_h, n_w, 8, 8)
        B, C, n_h, n_w, _, _ = dct_blocks.shape
        
        # Flatten to (B*C*n_h*n_w, 64)
        dct_flat = dct_blocks.view(-1, 64)
        
        # Apply IDCT: output = DCT_matrix^T @ input (fully vectorized)
        # Since DCT is orthonormal, IDCT = DCT^T
        idct_flat = torch.matmul(self.dct_weights.t(), dct_flat.t()).t()
        
        # Scale back: add 128 and normalize to [0, 1] range
        idct_flat = (idct_flat + 128.0) / 255.0
        
        # Reshape back
        idct_blocks = idct_flat.view(B, C, n_h, n_w, 8, 8)
        return idct_blocks
    
    def _quantize(self, dct_blocks, quant_table):
        """Differentiable quantization using Straight-Through Estimator (vectorized)."""
        # Input: (B, C, n_h, n_w, 8, 8)
        # Quant table: (8, 8)
        # Scale quantization table by factor
        quant_table_scaled = quant_table * self.factor
        
        # Divide by quantization table (broadcasted across all dimensions)
        quantized = dct_blocks / quant_table_scaled[None, None, None, None, :, :]
        
        # Differentiable rounding: use straight-through estimator
        # Forward: round, Backward: identity
        quantized_rounded = torch.round(quantized)
        quantized = quantized + (quantized_rounded - quantized).detach()
        
        return quantized
    
    def _dequantize(self, quantized_blocks, quant_table):
        """Dequantize DCT coefficients (vectorized)."""
        # Input: (B, C, n_h, n_w, 8, 8)
        # Quant table: (8, 8)
        quant_table_scaled = quant_table * self.factor
        
        # Multiply by quantization table (broadcasted)
        dequantized = quantized_blocks * quant_table_scaled[None, None, None, None, :, :]
        
        return dequantized
    
    def forward(self, noise_and_cover):
        """
        Forward pass of DiffJPEG compression/decompression.
        
        Args:
            noise_and_cover: Tuple/list where first element is image tensor (B, 3, H, W)
                           Expected input range: [0, 1] (Standard for Residual Learning)
        
        Returns:
            Same tuple/list with compressed/decompressed image in [0, 1] range
        """
        encoded_image = noise_and_cover[0]  # (B, 3, H, W)
        
        # 修正重點 1: 殘差連接可能會讓數值些微超出 [0, 1]，必須 Clamp (修剪)
        # 否則進入 Log 運算或 JPEG 轉換時會出錯
        image_01 = torch.clamp(encoded_image, 0.0, 1.0)
        
        # Store original shape
        B, C, orig_h, orig_w = image_01.shape
        
        # Step 1: RGB -> YCbCr
        ycbcr = self._rgb_to_ycbcr(image_01)
        
        # Step 2: Split into 8x8 blocks
        blocks, H, W, pad_h, pad_w = self._image_to_blocks(ycbcr)
        n_h, n_w = H // 8, W // 8
        
        # Step 3: DCT for all channels (fully vectorized)
        dct_blocks = self._dct_2d(blocks)
        
        # Step 4: Quantization (Y table for channel 0, C table for channels 1,2)
        quant_tables = torch.stack([
            self.y_quant_table,
            self.c_quant_table,
            self.c_quant_table
        ], dim=0)  # (3, 8, 8)
        
        # Quantize all channels at once
        quant_table_scaled = quant_tables * self.factor
        quant_table_expanded = quant_table_scaled[None, :, None, None, :, :]
        quantized = dct_blocks / quant_table_expanded
        
        # Differentiable rounding (STE)
        quantized_rounded = torch.round(quantized)
        quantized = quantized + (quantized_rounded - quantized).detach()
        
        # Step 5: Dequantization
        dequantized = quantized * quant_table_expanded
        
        # Step 6: IDCT
        compressed_blocks = self._idct_2d(dequantized)
        
        # Step 7: Reconstruct image
        compressed_ycbcr = self._blocks_to_image(compressed_blocks, orig_h, orig_w, pad_h, pad_w)
        
        # Step 8: YCbCr -> RGB
        compressed_rgb_01 = self._ycbcr_to_rgb(compressed_ycbcr)
        
        # Ensure output is in [0, 1] range
        compressed_rgb_01 = torch.clamp(compressed_rgb_01, 0.0, 1.0)
        
        # 修正重點 2: 不需要再轉回 [-1, 1]，直接輸出 [0, 1]
        noise_and_cover[0] = compressed_rgb_01
        
        return noise_and_cover


class Jpeg(nn.Module):
    """
    Wrapper class for DiffJPEG to match the original interface.
    Accepts factor parameter (quantization scale).
    """
    
    def __init__(self, factor):
        super(Jpeg, self).__init__()
        # Convert factor to float if it's a string
        if isinstance(factor, str):
            factor = float(factor)
        self.diff_jpeg = DiffJPEG(factor=factor)
        self.factor = factor
    
    def forward(self, noise_and_cover):
        return self.diff_jpeg(noise_and_cover)


def quality_to_factor(quality):
    """
    Convert JPEG quality (0-100) to quantization factor.
    This is used for backward compatibility.
    """
    if quality < 50:
        return 50.0 / quality
    else:
        return 2.0 - quality * 0.02
