"""
Base TFN Layer Components

Clean, reusable model classes for Token Field Network layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math
from typing import Optional, Tuple, Literal

from core.field_projection import FieldProjector
from core.field_evolution import FieldEvolver
from core.field_sampling import FieldSampler


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
    """Trainable field evolution module."""
    
    def __init__(
        self,
        embed_dim: int,
        evolution_type: str = "cnn",
        time_steps: int = 3,
        dropout: float = 0.0,
    ):
        """Create a trainable field evolution module.

        Parameters
        ----------
        embed_dim : int
            Embedding dimension of the field.
        evolution_type : str, default "cnn"
            Evolution operator to use ("cnn" or "pde").
        time_steps : int, default 3
            How many evolution sub-steps to perform.
        dropout : float, default 0.0
            Dropout probability *inside* the evolution operator. This allows
            additional regularisation beyond the layer-level dropout and can
            be tuned from the CLI via the existing ``--dropout`` flag.
        """
        super().__init__()
        self.evolution_type = evolution_type
        self.time_steps = time_steps
        self.embed_dim = embed_dim

        # ---------------------------------------------------------------
        # Robustness: argparse may pass *all* CLI values as strings when no
        # explicit ``type=float`` is set. Convert here to guarantee `float`.
        # ---------------------------------------------------------------
        if isinstance(dropout, str):
            try:
                dropout = float(dropout)
            except ValueError:
                raise ValueError(f"dropout must be float-compatible, got {dropout!r}")

        # Dedicated evolution dropout (Identity if p=0 to avoid overhead)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if evolution_type == "cnn":
            # Learnable CNN evolution
            self.conv_layers = nn.ModuleList([
                nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False)
                for _ in range(time_steps)
            ])
            # Initialize weights
            for conv in self.conv_layers:
                nn.init.normal_(conv.weight, 0, 0.1)

        elif evolution_type in ["pde", "diffusion"]:
            # Learnable diffusion coefficient
            self.alpha = nn.Parameter(torch.tensor(0.1))
            self.dt = nn.Parameter(torch.tensor(0.01))
    
    def forward(self, field: torch.Tensor, grid_points: torch.Tensor) -> torch.Tensor:
        """Evolve field with trainable parameters."""
        if self.evolution_type == "cnn":
            return self._cnn_evolution(field)
        elif self.evolution_type in ["pde", "diffusion"]:
            return self._pde_evolution(field, grid_points)
        else:
            raise ValueError(f"Unknown evolution type: {self.evolution_type}")
    
    def _cnn_evolution(self, field: torch.Tensor) -> torch.Tensor:
        """CNN evolution with cached layers."""
        evolved = field
        for conv in self.conv_layers:
            # [B, M, D] -> [B, D, M] -> [B, D, M] -> [B, M, D]
            evolved = evolved.transpose(1, 2)
            evolved = conv(evolved)
            evolved = evolved.transpose(1, 2)
            evolved = F.relu(evolved)
            # ------------------ internal dropout ------------------
            evolved = self.dropout(evolved)
        return evolved
    
    def _pde_evolution(self, field: torch.Tensor, grid_points: torch.Tensor) -> torch.Tensor:
        """PDE evolution with learnable coefficients."""
        B, M, D = field.shape
        
        # Learnable parameters
        alpha = torch.clamp(self.alpha, min=0.01, max=1.0)
        dt = torch.clamp(self.dt, min=0.001, max=0.1)
        
        evolved = field
        for _ in range(self.time_steps):
            # Compute Laplacian
            laplacian = torch.zeros_like(evolved)
            laplacian[:, 1:-1, :] = (evolved[:, 2:, :] - 2 * evolved[:, 1:-1, :] + evolved[:, :-2, :])
            
            # Update with learnable coefficients
            evolved = evolved + alpha * dt * laplacian
        
        return evolved


class PositionEmbeddings(nn.Module):
    """Learnable position embeddings."""
    
    def __init__(self, embed_dim: int, max_seq_len: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Learnable position embeddings
        self.pos_embeddings = nn.Parameter(torch.randn(max_seq_len, embed_dim))
        nn.init.normal_(self.pos_embeddings, 0, 0.1)
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Convert continuous positions to embeddings."""
        B, N, P = positions.shape
        
        # Scale positions to [0, max_seq_len-1]
        pos_scaled = positions.view(B, N) * (self.max_seq_len - 1)
        pos_indices = torch.clamp(pos_scaled.long(), 0, self.max_seq_len - 1)
        
        # Gather position embeddings
        pos_emb = torch.gather(
            self.pos_embeddings.unsqueeze(0).expand(B, -1, -1),
            1,
            pos_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        )
        
        return pos_emb


class TrainableTFNLayer(nn.Module):
    """Complete trainable TFN layer with all learnable components."""
    
    def __init__(self, embed_dim: int, kernel_type: str = "rbf", evolution_type: str = "cnn",
                 grid_size: int = 100, time_steps: int = 3, max_seq_len: int = 512,
                 dropout: float = 0.1, layer_norm_eps: float = 1e-5):
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
        self.pos_embeddings = PositionEmbeddings(embed_dim, max_seq_len)
        
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
        
        Returns:
            updated_embeddings: [B, N, D] updated embeddings with residual connection
        """
        B, N, D = embeddings.shape
        P = positions.shape[-1]

        # Optionally incorporate positional information ---------------------
        if add_pos_emb:
            pos_emb = self.pos_embeddings(positions)
            x = embeddings + pos_emb
        else:
            # Caller has already added positional information
            x = embeddings
        
        # Create grid points
        if use_learnable_grid:
            grid_points = torch.clamp(self.learnable_grid, 0, 1)
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
