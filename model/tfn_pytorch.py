import torch
import torch.nn as nn
import torch.nn.functional as F
from core.image.field_emitter import ImageFieldEmitter
from core.image.field_interference_block import ImageFieldInterference
from core.image.field_propagator import ImageFieldPropagator

class ImageTFN(nn.Module):
    """
    Image Token Field Network (ImageTFN) for 2D images, integrating field emission, interference, and PDE-inspired propagation.
    Includes a classification head for image tasks.
    
    This is the new PyTorch implementation optimized for 2D image processing,
    distinct from the previous token-based TFN used for 1D time series.
    
    Args:
        in_ch: Number of input channels
        num_classes: Number of output classes
    """
    def __init__(self, in_ch: int = 3, num_classes: int = 10):
        super().__init__()
        # 1. Field Emission Stage
        self.emit1 = ImageFieldEmitter(in_ch, 64)
        self.emit2 = ImageFieldEmitter(64, 128)
        # 2. Interference Layers
        self.interfere1 = ImageFieldInterference(num_heads=8)
        self.interfere2 = ImageFieldInterference(num_heads=8)
        # 3. Physics-Aware Propagation
        self.propagate = ImageFieldPropagator(steps=4)
        # 4. Classification Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, in_ch, H, W]
        Returns:
            Logits tensor of shape [B, num_classes]
        """
        x = F.gelu(self.emit1(x))
        x = self.propagate(x)
        # Interference expects [B, C, H, W] -> [B, H*W, C] and back
        B, C, H, W = x.shape
        x = self.interfere1(x)
        x = F.gelu(self.emit2(x))
        x = self.interfere2(x)
        return self.head(x) 