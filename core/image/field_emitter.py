import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ImageFieldEmitter(nn.Module):
    """
    ImageFieldEmitter projects input image features into a spatial field using learnable amplitude, range, and dynamic centers.
    The field modulates the input before convolution, enabling input-adaptive receptive fields for 2D image processing.
    
    This is distinct from the previous token-based FieldEmitter used for 1D time series.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(1, in_channels, 1, 1))  # Amplitude for input channels
        self.sigma = nn.Parameter(torch.ones(1, in_channels, 1, 1))    # Range for input channels
        self.center = nn.Conv2d(in_channels, 2, kernel_size, padding=kernel_size//2)  # Dynamic centers (x, y)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, C, H, W]
        Returns:
            Output tensor of shape [B, out_channels, H, W]
        """
        B, C, H, W = x.shape
        # 1. Compute dynamic field centers
        centers = self.center(x)  # [B, 2, H, W]
        # 2. Generate coordinate grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=0).float()  # [2, H, W]
        # 3. Compute radial distances to dynamic centers
        # [B, 2, H, W] - [2, H, W] -> [B, 2, H, W]
        dist = torch.norm(grid[None, :, :, :] - centers, dim=1, keepdim=True)  # [B, 1, H, W]
        # 4. Emit field (Gaussian RBF) - generate for input channels
        # Broadcast alpha and sigma to match input channels
        alpha_broadcast = self.alpha.expand(B, C, H, W)  # [B, C, H, W]
        sigma_broadcast = self.sigma.expand(B, C, H, W)  # [B, C, H, W]
        field = alpha_broadcast * torch.exp(-dist**2 / (2 * sigma_broadcast**2 + 1e-6))  # [B, C, H, W]
        # 5. Field-modulated convolution
        return self.conv(x * field) 