"""
Base TFN Layer Components

Clean, reusable model classes for Token Field Network layers.

DEPRECATED: This module is deprecated and will be removed in a future version.
Use the modern core components (UnifiedFieldDynamics, EnhancedTFNLayer) instead
for better performance and maintainability.
"""

import warnings
warnings.warn(
    "tfn_base.py is deprecated. Use UnifiedFieldDynamics and EnhancedTFNLayer "
    "from the core/ directory instead. This module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math
from typing import Optional, Tuple, Literal, Dict, Any, List

from core.field_projection import FieldProjector
from core.field_evolution import FieldEvolver
from core.field_sampling import FieldSampler
from .shared_layers import create_positional_embedding_strategy


class LearnableKernels(nn.Module):
    """Learnable kernel parameters for field projection."""
    
    def __init__(self, embed_dim: int, kernel_type: str = "rbf"):
        super().__init__()
        self.kernel_type = kernel_type
        self.embed_dim = embed_dim
        
        if kernel_type == "rbf":
            # Learnable sigma for RBF kernel
            self.sigma = nn.Parameter(torch.tensor(0.2))
        elif kernel_type == "compact":
            # Learnable radius for compact kernel
            self.radius = nn.Parameter(torch.tensor(0.3))
        elif kernel_type == "fourier":
            # Learnable frequency for Fourier kernel
            self.freq = nn.Parameter(torch.tensor(2.0))
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    def forward(self, grid_points: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Compute kernel values with learnable parameters."""
        B, M, P = grid_points.shape
        N = positions.shape[1]
        
        if self.kernel_type == "rbf":
            sigma = torch.clamp(self.sigma, min=0.01, max=2.0)  # Ensure positive
            kernel_params = sigma.expand(B, N, 1)
            return self._rbf_kernel(grid_points, positions, kernel_params)
        
        elif self.kernel_type == "compact":
            radius = torch.clamp(self.radius, min=0.01, max=1.0)  # Ensure positive
            kernel_params = radius.expand(B, N, 1)
            return self._compact_kernel(grid_points, positions, kernel_params)
        
        elif self.kernel_type == "fourier":
            freq = torch.clamp(self.freq, min=0.1, max=10.0)  # Ensure reasonable range
            kernel_params = freq.expand(B, N, 1)
            return self._fourier_kernel(grid_points, positions, kernel_params)
    
    def _rbf_kernel(self, grid_points: torch.Tensor, positions: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """RBF kernel with learnable sigma."""
        diff = grid_points.unsqueeze(1) - positions.unsqueeze(2)  # [B, N, M, P]
        dist_sq = torch.sum(diff ** 2, dim=-1)  # [B, N, M]
        return torch.exp(-dist_sq / (2 * sigma ** 2))
    
    def _compact_kernel(self, grid_points: torch.Tensor, positions: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        """Compact kernel with learnable radius."""
        diff = grid_points.unsqueeze(1) - positions.unsqueeze(2)  # [B, N, M, P]
        dist = torch.norm(diff, dim=-1)  # [B, N, M]
        return torch.where(dist <= radius, 1.0 - dist / radius, torch.zeros_like(dist))
    
    def _fourier_kernel(self, grid_points: torch.Tensor, positions: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        """Fourier kernel with learnable frequency."""
        diff = grid_points.unsqueeze(1) - positions.unsqueeze(2)  # [B, N, M, P]
        phase = torch.sum(diff * freq.unsqueeze(-1), dim=-1)  # [B, N, M]
        return torch.cos(phase)


class TrainableEvolution(nn.Module):
    """Learnable field evolution with different evolution types."""
    
    def __init__(self, embed_dim: int, evolution_type: str = "cnn", time_steps: int = 3, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.evolution_type = evolution_type
        self.time_steps = time_steps
        self.dropout = nn.Dropout(dropout)
        
        if evolution_type == "cnn":
            # CNN-based evolution
            self.conv1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm1d(embed_dim)
            self.bn2 = nn.BatchNorm1d(embed_dim)
            self.residual = nn.Linear(embed_dim, embed_dim)
            
        elif evolution_type == "pde":
            # PDE-based evolution parameters
            self.diffusion_coeff = nn.Parameter(torch.tensor(0.1))
            self.dt = nn.Parameter(torch.tensor(0.01))
            
        elif evolution_type == "wave":
            # Wave equation parameters
            self.wave_speed = nn.Parameter(torch.tensor(1.0))
            self.dt = nn.Parameter(torch.tensor(0.01))
            
        elif evolution_type == "schrodinger":
            # Schrödinger equation parameters
            self.hamiltonian = nn.Parameter(torch.randn(embed_dim, embed_dim))
            self.dt = nn.Parameter(torch.tensor(0.01))
            
        else:
            raise ValueError(f"Unknown evolution type: {evolution_type}")
    
    def forward(self, field: torch.Tensor, grid_points: torch.Tensor) -> torch.Tensor:
        """Evolve field using specified evolution type."""
        if self.evolution_type == "cnn":
            return self._cnn_evolution(field)
        elif self.evolution_type == "pde":
            return self._pde_evolution(field, grid_points)
        elif self.evolution_type == "wave":
            return self._wave_evolution(field, grid_points)
        elif self.evolution_type == "schrodinger":
            return self._schrodinger_evolution(field)
        else:
            raise ValueError(f"Unknown evolution type: {self.evolution_type}")
    
    def _cnn_evolution(self, field: torch.Tensor) -> torch.Tensor:
        """CNN-based field evolution."""
        batch_size, num_points, embed_dim = field.shape
        
        # Reshape for 1D convolution: [B, D, M]
        field_conv = field.transpose(1, 2)
        
        for _ in range(self.time_steps):
            # CNN evolution
            x = F.relu(self.bn1(self.conv1(field_conv)))
            x = F.relu(self.bn2(self.conv2(x)))
            
            # Residual connection
            field_conv = field_conv + self.residual(field_conv.transpose(1, 2)).transpose(1, 2)
            field_conv = field_conv + x
            
            # Apply dropout
            field_conv = self.dropout(field_conv)
        
        # Reshape back: [B, D, M] -> [B, M, D]
        return field_conv.transpose(1, 2)
    
    def _pde_evolution(self, field: torch.Tensor, grid_points: torch.Tensor) -> torch.Tensor:
        """PDE-based field evolution (diffusion equation)."""
        evolved = field.clone()
        dt = torch.clamp(self.dt, min=0.001, max=0.1)
        alpha = torch.clamp(self.diffusion_coeff, min=0.01, max=1.0)
        
        for _ in range(self.time_steps):
            # Compute Laplacian
            laplacian = torch.zeros_like(evolved)
            laplacian[:, 1:-1, :] = (evolved[:, 2:, :] - 2 * evolved[:, 1:-1, :] + evolved[:, :-2, :])
            
            # Update with learnable coefficients
            evolved = evolved + alpha * dt * laplacian
        
        return evolved
    
    def _wave_evolution(self, field: torch.Tensor, grid_points: torch.Tensor) -> torch.Tensor:
        """Wave equation evolution."""
        # Simplified wave equation implementation
        evolved = field.clone()
        dt = torch.clamp(self.dt, min=0.001, max=0.1)
        c = torch.clamp(self.wave_speed, min=0.1, max=10.0)
        
        for _ in range(self.time_steps):
            # Second derivative in space
            d2x = torch.zeros_like(evolved)
            d2x[:, 1:-1, :] = (evolved[:, 2:, :] - 2 * evolved[:, 1:-1, :] + evolved[:, :-2, :])
            
            # Wave equation: ∂²u/∂t² = c² ∂²u/∂x²
            evolved = evolved + c * c * dt * dt * d2x
        
        return evolved
    
    def _schrodinger_evolution(self, field: torch.Tensor) -> torch.Tensor:
        """Schrödinger equation evolution."""
        evolved = field.clone()
        dt = torch.clamp(self.dt, min=0.001, max=0.1)
        
        # Create Hermitian Hamiltonian
        H = self.hamiltonian
        H_real = (H + H.T) / 2  # Symmetric real part
        H_imag = (H - H.T) / 2  # Anti-Hermitian imaginary part
        
        for _ in range(self.time_steps):
            # Apply Hamiltonian: HF
            hamiltonian_evolution_real = torch.einsum('bnd,de->bne', evolved, H_real)
            hamiltonian_evolution_imag = torch.einsum('bnd,de->bne', evolved, H_imag)
            
            # For Schrödinger equation: i∂F/∂t = HF
            # ∂F/∂t = -i*HF = H_imag*F - i*H_real*F
            # Since we work with real fields, we take the real part
            evolution_real = hamiltonian_evolution_imag  # Real part of -i*HF
            
            evolved = evolved + dt * evolution_real
        
        return evolved


class TrainableTFNLayer(nn.Module):
    """Complete trainable TFN layer with all learnable components."""
    
    def __init__(self, embed_dim: int, kernel_type: str = "rbf", evolution_type: str = "cnn",
                 grid_size: int = 100, time_steps: int = 3, max_seq_len: int = 512,
                 dropout: float = 0.1, layer_norm_eps: float = 1e-5,
                 positional_embedding_strategy: str = "learned",
                 calendar_features: Optional[List[str]] = None,
                 feature_cardinalities: Optional[Dict[str, int]] = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.time_steps = time_steps
        
        # Learnable components
        self.kernels = LearnableKernels(embed_dim, kernel_type)
        # Pass the same dropout value to the evolution module so that users can
        # control *all* regularisation from a single CLI flag.
        self.evolution = TrainableEvolution(
            embed_dim,
            evolution_type,
            time_steps,
            dropout=dropout,
        )
        
        # Create positional embedding strategy
        self.pos_embeddings = create_positional_embedding_strategy(
            positional_embedding_strategy,
            max_seq_len,
            embed_dim,
            calendar_features=calendar_features,
            feature_cardinalities=feature_cardinalities
        )
        
        # Standard transformer components
        self.layer_norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        
        # Learnable grid (optional)
        self.learnable_grid = nn.Parameter(torch.linspace(0, 1, grid_size))

    def forward(
        self,
        embeddings: torch.Tensor,
        positions: torch.Tensor,
        use_learnable_grid: bool = False,
        *,
        add_pos_emb: bool = True,
        calendar_features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass of trainable TFN layer.
        
        Args:
            embeddings: [B, N, D] token embeddings
            positions: [B, N, P] token positions (P=1 for 1D)
            use_learnable_grid: Whether to use learnable grid points
            add_pos_emb: If *True*, the layer will add its own learnable
                positional embeddings. Set to *False* if the caller has
                already combined positional information with the token
                embeddings (to avoid double-counting).
            calendar_features: Dictionary of calendar features for time-based embeddings
        
        Returns:
            updated_embeddings: [B, N, D] updated embeddings with residual connection
        """
        B, N, D = embeddings.shape
        P = positions.shape[-1]

        # Optionally incorporate positional information ---------------------
        if add_pos_emb:
            pos_emb = self.pos_embeddings(positions, calendar_features=calendar_features)
            x = embeddings + pos_emb
        else:
            # Caller has already added positional information
            x = embeddings
        
        # Create grid points
        if use_learnable_grid:
            grid_points = torch.clamp(self.learnable_grid, 0, 1)
            grid_points = grid_points.to(embeddings.device)
            grid_points = grid_points.unsqueeze(0).unsqueeze(-1).expand(B, -1, P)
        else:
            grid_points = torch.linspace(0, 1, self.grid_size, device=embeddings.device, dtype=embeddings.dtype)
            grid_points = grid_points.unsqueeze(0).unsqueeze(-1).expand(B, -1, P)
        
        # TFN pipeline: Project -> Evolve -> Sample
        # 1. Project tokens to field
        kernels = self.kernels(grid_points, positions)  # [B, N, M]
        field = torch.einsum('bnm,bnd->bmd', kernels, x)  # [B, M, D]
        
        # 2. Evolve field
        evolved_field = self.evolution(field, grid_points)  # [B, M, D]
        
        # 3. Sample field back to tokens
        updated_embeddings = self._sample_field(evolved_field, grid_points, positions)  # [B, N, D]
        
        # Residual connection and normalization
        updated_embeddings = self.layer_norm1(updated_embeddings + x)
        updated_embeddings = self.dropout(updated_embeddings)
        
        return updated_embeddings
    
    def _sample_field(self, field: torch.Tensor, grid_points: torch.Tensor, 
                     sample_positions: torch.Tensor) -> torch.Tensor:
        """Sample field at token positions using nearest neighbor interpolation."""
        B, M, D = field.shape
        N = sample_positions.shape[1]
        
        # Find nearest grid points
        grid_1d = grid_points[:, :, 0]  # [B, M]
        pos_1d = sample_positions[:, :, 0]  # [B, N]
        
        # Compute distances to all grid points
        diff = pos_1d.unsqueeze(-1) - grid_1d.unsqueeze(1)  # [B, N, M]
        distances = torch.abs(diff)
        
        # Find nearest neighbors
        nearest_indices = torch.argmin(distances, dim=-1)  # [B, N]
        
        # Gather field values at nearest grid points
        batch_indices = torch.arange(B, device=field.device).unsqueeze(1).expand(-1, N)
        sampled_field = field[batch_indices, nearest_indices]  # [B, N, D]
        
        return sampled_field
