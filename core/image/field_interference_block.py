import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ImageFieldInterference(nn.Module):
    """
    ImageFieldInterference applies low-rank, physics-inspired field mixing across heads using a learned coupler.
    Supports both token ([B, T, C]) and image ([B, C, H, W]) modes for 2D image processing.
    
    This is distinct from the previous TokenFieldInterference used for 1D time series.
    
    Args:
        num_heads: Number of field heads (channels per group)
    """
    def __init__(self, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.coupler = nn.Parameter(torch.randn(num_heads, num_heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C] (tokens) or [B, C, H, W] (images)
        Returns:
            Interference-mixed tensor of same shape as input
        """
        if x.dim() == 4:
            # Image mode: [B, C, H, W] -> [B, H*W, C]
            B, C, H, W = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
            out = self._interfere_tokens(x_flat)
            out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
            return out
        elif x.dim() == 3:
            # Token mode: [B, T, C]
            return self._interfere_tokens(x)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

    def _interfere_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C]
        Returns:
            [B, T, C] after interference
        """
        B, T, C = x.shape
        assert C % self.num_heads == 0, "C must be divisible by num_heads"
        head_dim = C // self.num_heads
        x_heads = x.view(B, T, self.num_heads, head_dim)  # [B, T, H, D]
        # Interference: pairwise mixing across heads
        # [B, T, H, D], [H, H] -> [B, T, H, D]
        mixed = torch.einsum('bthd,hk->btkd', x_heads, self.coupler)
        # Optionally, apply softmax normalization across heads
        mixed = F.softmax(mixed, dim=2)
        # Collapse head dimension
        out = mixed.reshape(B, T, C)
        return out 