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
        vocab_size: Vocabulary size (required by registry)
        embed_dim: Embedding dimension (required by registry)
        num_classes: Number of output classes
        num_layers: Number of TFN layers
        num_evolution_steps: Number of evolution steps
        field_dim: Dimension of field features
        grid_height: Height of the spatial grid
        grid_width: Width of the spatial grid
        use_dynamic_positions: Whether to use dynamic positional embeddings
        learnable_sigma: Whether sigma is learnable
        learnable_out_sigma: Whether output sigma is learnable
        out_sigma_scale: Scale factor for output sigma
        field_dropout: Dropout rate for field operations
        use_global_context: Whether to use global context
        dropout: General dropout rate
        multiscale: Whether to use multiscale processing
        kernel_mix: Whether to use kernel mixing
        kernel_mix_scale: Scale factor for kernel mixing
    """
    def __init__(self, 
                 vocab_size: int,  # Required by registry
                 embed_dim: int,   # Required by registry
                 num_classes: int = 10,
                 num_layers: int = 4,  # From registry defaults
                 num_evolution_steps: int = 5,  # From registry defaults
                 field_dim: int = 64,  # From registry defaults
                 grid_height: int = 32,  # From registry defaults
                 grid_width: int = 32,  # From registry defaults
                 use_dynamic_positions: bool = False,  # From registry defaults
                 learnable_sigma: bool = True,  # From registry defaults
                 learnable_out_sigma: bool = False,  # From registry defaults
                 out_sigma_scale: float = 2.0,  # From registry defaults
                 field_dropout: float = 0.0,  # From registry defaults
                 use_global_context: bool = False,  # From registry defaults
                 dropout: float = 0.1,  # From registry defaults
                 multiscale: bool = False,  # From registry defaults
                 kernel_mix: bool = False,  # From registry defaults
                 kernel_mix_scale: float = 2.0):  # From registry defaults
        super().__init__()
        
        # Store all parameters for potential use
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_evolution_steps = num_evolution_steps
        self.field_dim = field_dim
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.use_dynamic_positions = use_dynamic_positions
        self.learnable_sigma = learnable_sigma
        self.learnable_out_sigma = learnable_out_sigma
        self.out_sigma_scale = out_sigma_scale
        self.field_dropout = field_dropout
        self.use_global_context = use_global_context
        self.dropout = dropout
        self.multiscale = multiscale
        self.kernel_mix = kernel_mix
        self.kernel_mix_scale = kernel_mix_scale
        
        # Calculate input channels based on embed_dim (for backward compatibility)
        in_ch = max(3, embed_dim // 32)  # Ensure minimum 3 channels
        
        # 1. Field Emission Stage - now configurable
        self.emit1 = ImageFieldEmitter(in_ch, field_dim)
        self.emit2 = ImageFieldEmitter(field_dim, field_dim * 2)
        
        # 2. Interference Layers - configurable number of heads
        num_heads = max(1, embed_dim // 16)  # Adaptive number of heads
        self.interfere1 = ImageFieldInterference(num_heads=num_heads)
        self.interfere2 = ImageFieldInterference(num_heads=num_heads)
        
        # 3. Physics-Aware Propagation - configurable steps
        self.propagate = ImageFieldPropagator(steps=num_evolution_steps)
        
        # 4. Classification Head - configurable
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),  # Use configurable dropout
            nn.Linear(field_dim * 2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # 5. Additional configurable components
        if use_global_context:
            self.global_context = nn.AdaptiveAvgPool2d(1)
        
        if multiscale:
            self.multiscale_pool = nn.ModuleList([
                nn.AdaptiveAvgPool2d(size) for size in [1, 2, 4]
            ])
        
        if kernel_mix:
            self.kernel_mix_layer = nn.Linear(field_dim * 2, field_dim * 2)
        
        # Apply field dropout if specified
        if field_dropout > 0:
            self.field_dropout_layer = nn.Dropout2d(field_dropout)
        else:
            self.field_dropout_layer = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, in_ch, H, W]
        Returns:
            Logits tensor of shape [B, num_classes]
        """
        # Apply field dropout if enabled
        x = self.field_dropout_layer(x)
        
        # Field emission with configurable parameters
        x = F.gelu(self.emit1(x))
        x = self.propagate(x)
        
        # Interference expects [B, C, H, W] -> [B, H*W, C] and back
        B, C, H, W = x.shape
        x = self.interfere1(x)
        x = F.gelu(self.emit2(x))
        x = self.interfere2(x)
        
        # Apply multiscale processing if enabled
        if self.multiscale:
            multiscale_features = []
            for pool in self.multiscale_pool:
                pooled = pool(x)
                multiscale_features.append(pooled.flatten(1))
            if multiscale_features:
                x = torch.cat(multiscale_features, dim=1)
        
        # Apply kernel mixing if enabled
        if self.kernel_mix:
            B, C, H, W = x.shape
            x_flat = x.view(B, C, -1).transpose(1, 2)  # [B, H*W, C]
            x_mixed = self.kernel_mix_layer(x_flat)
            x = x_mixed.transpose(1, 2).view(B, C, H, W)
        
        # Apply global context if enabled
        if self.use_global_context:
            global_feat = self.global_context(x)
            x = x + global_feat.expand_as(x)
        
        return self.head(x) 