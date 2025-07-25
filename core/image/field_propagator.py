import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ImageFieldPropagator(nn.Module):
    """
    ImageFieldPropagator evolves a 2D field over time using a stable, PDE-inspired update with a learnable diffusion matrix and clamped time step.
    Applies a finite-difference Laplacian for 2D image data.
    
    This is distinct from the previous FieldPropagator used for 1D time series.
    
    Args:
        steps: Number of propagation steps
    """
    def __init__(self, steps: int = 4):
        super().__init__()
        self.steps = steps
        self.diffusion = nn.Parameter(torch.eye(1))  # For multi-channel, can be nn.Parameter(torch.eye(C))
        self.dt = nn.Parameter(torch.tensor(0.05))   # Adaptive time step
        # Precompute Laplacian kernel for 2D convolution
        laplacian_kernel = torch.tensor([[[[0,1,0],[1,-4,1],[0,1,0]]]], dtype=torch.float32)  # [1,1,3,3]
        self.register_buffer('laplacian_kernel', laplacian_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, C, H, W]
        Returns:
            Evolved tensor of shape [B, C, H, W]
        """
        for _ in range(self.steps):
            # Laplacian (finite differences, per channel)
            laplacian = F.conv2d(x, self.laplacian_kernel.expand(x.shape[1], 1, 3, 3),
                                 padding=1, groups=x.shape[1])  # [B, C, H, W]
            # Field evolution: ∂F/∂t = ∇·(D∇F)
            # For now, D is scalar or diagonal (per channel)
            D = torch.clamp(self.diffusion, 0.01, 10.0)
            dt = torch.clamp(self.dt, 0.01, 0.1)
            x = x + dt * (laplacian * D)
        return x 