"""
Field projection module for TFN.

This module implements the core field projection mechanism that transforms
discrete token embeddings into continuous fields using kernel-based emission.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Union
from .kernels import KernelBasis, RBFKernel, CompactKernel, FourierKernel, LearnableKernel


class FieldProjector(nn.Module):
    """
    Projects token embeddings into continuous fields using kernels.
    
    This is the core mechanism that transforms discrete token representations
    into continuous spatial fields, enabling the field-based approach of TFN.
    
    Mathematical formulation:
        F(z) = Σᵢ Eᵢ ⊗ Kᵢ(z, μᵢ, θᵢ)
    
    Where:
        - Eᵢ = embedding of token i
        - Kᵢ = kernel of token i  
        - μᵢ = position of token i
        - θᵢ = kernel parameters for token i
        - F(z) = continuous field at position z
    """
    
    def __init__(self, 
                 embed_dim: int,
                 pos_dim: int,
                 kernel_type: str = "rbf",
                 default_kernel_params: Optional[dict] = None):
        """
        Initialize field projector.
        
        Args:
            embed_dim: Dimension of token embeddings
            pos_dim: Dimension of position space
            kernel_type: Type of kernel to use ("rbf", "compact", "fourier", "learnable")
            default_kernel_params: Default parameters for kernel initialization
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.pos_dim = pos_dim
        self.kernel_type = kernel_type
        
        # Initialize kernel
        if default_kernel_params is None:
            default_kernel_params = {}
        
        if kernel_type == "rbf":
            self.kernel = RBFKernel(pos_dim=pos_dim, **default_kernel_params)
        elif kernel_type == "compact":
            self.kernel = CompactKernel(pos_dim=pos_dim, **default_kernel_params)
        elif kernel_type == "fourier":
            self.kernel = FourierKernel(pos_dim=pos_dim, **default_kernel_params)
        elif kernel_type == "learnable":
            self.kernel = LearnableKernel(pos_dim=pos_dim, **default_kernel_params)
        else:
            # Default to RBF kernel for unknown types
            self.kernel = RBFKernel(pos_dim=pos_dim, **default_kernel_params)
    
    def forward(self,
                embeddings: torch.Tensor,      # [B, N, D] token embeddings
                positions: torch.Tensor,       # [B, N, P] token positions
                grid_points: torch.Tensor,     # [M, P] or [B, M, P] grid points
                kernel_params: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Project token embeddings into continuous fields.
        
        Args:
            embeddings: Token embeddings of shape [B, N, D]
            positions: Token positions of shape [B, N, P]
            grid_points: Grid points for field evaluation of shape [M, P] or [B, M, P]
            kernel_params: Kernel parameters of shape [B, N, K] (optional)
            
        Returns:
            Continuous field of shape [B, M, D]
        """
        batch_size, num_tokens, embed_dim = embeddings.shape
        num_grid_points = grid_points.shape[0] if grid_points.dim() == 2 else grid_points.shape[1]
        
        # Ensure grid_points has batch dimension
        if grid_points.dim() == 2:
            grid_points = grid_points.unsqueeze(0).expand(batch_size, -1, -1)  # [M, P] -> [B, M, P]
        
        # Compute kernel values for all token-grid pairs
        # kernel_values: [B, N, M]
        if kernel_params is None:
            # Use default parameters (e.g., sigma=0.2 for RBF)
            if self.kernel_type == "rbf":
                kernel_params = torch.full((batch_size, num_tokens, 1), 0.2, 
                                        device=embeddings.device, dtype=embeddings.dtype)
            else:
                # Default parameters for other kernels
                kernel_params = torch.full((batch_size, num_tokens, 1), 0.2,
                                        device=embeddings.device, dtype=embeddings.dtype)
        
        kernel_values = self.kernel(grid_points, positions, kernel_params)  # [B, N, M]
        
        # Project embeddings to field: E ⊗ K
        # embeddings: [B, N, D] -> [B, N, D, 1]
        # kernel_values: [B, N, 1, M] -> [B, N, 1, M]
        # Result: [B, N, D, M] -> [B, N, M, D]
        embeddings_expanded = embeddings.unsqueeze(-1)  # [B, N, D, 1]
        kernel_values_expanded = kernel_values.unsqueeze(2)  # [B, N, 1, M]
        
        # Compute weighted embeddings: E ⊗ K
        weighted_embeddings = embeddings_expanded * kernel_values_expanded  # [B, N, D, M]
        
        # Aggregate fields from all tokens: Σᵢ Eᵢ ⊗ Kᵢ
        field = weighted_embeddings.sum(dim=1)  # [B, D, M] -> [B, M, D]
        field = field.transpose(1, 2)  # [B, M, D]
        
        return field
    
    def compute_token_influence(self,
                              embeddings: torch.Tensor,      # [B, N, D]
                              positions: torch.Tensor,       # [B, N, P]
                              grid_points: torch.Tensor,     # [M, P] or [B, M, P]
                              kernel_params: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute individual token influence on the field.
        
        This is useful for analysis and debugging.
        
        Args:
            embeddings: Token embeddings of shape [B, N, D]
            positions: Token positions of shape [B, N, P]
            grid_points: Grid points for field evaluation
            kernel_params: Kernel parameters (optional)
            
        Returns:
            Individual token influences of shape [B, N, M, D]
        """
        batch_size, num_tokens, embed_dim = embeddings.shape
        num_grid_points = grid_points.shape[0] if grid_points.dim() == 2 else grid_points.shape[1]
        
        # Ensure grid_points has batch dimension
        if grid_points.dim() == 2:
            grid_points = grid_points.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Compute kernel values
        if kernel_params is None:
            if self.kernel_type == "rbf":
                kernel_params = torch.full((batch_size, num_tokens, 1), 0.2,
                                        device=embeddings.device, dtype=embeddings.dtype)
            else:
                kernel_params = torch.full((batch_size, num_tokens, 1), 0.2,
                                        device=embeddings.device, dtype=embeddings.dtype)
        
        kernel_values = self.kernel(grid_points, positions, kernel_params)  # [B, N, M]
        
        # Compute individual token influences
        embeddings_expanded = embeddings.unsqueeze(-1)  # [B, N, D, 1]
        kernel_values_expanded = kernel_values.unsqueeze(2)  # [B, N, 1, M]
        
        token_influences = embeddings_expanded * kernel_values_expanded  # [B, N, D, M]
        token_influences = token_influences.transpose(2, 3)  # [B, N, M, D]
        
        return token_influences


class UniformFieldGrid(nn.Module):
    """
    Generates uniform grid points for field evaluation.
    
    This is a simple grid generator that creates evenly spaced
    points in the position space.
    """
    
    def __init__(self, pos_dim: int, grid_size: int = 100, bounds: Tuple[float, float] = (0.0, 1.0)):
        """
        Initialize uniform field grid.
        
        Args:
            pos_dim: Dimension of position space
            grid_size: Number of grid points per dimension
            bounds: Bounds for each dimension (min, max)
        """
        super().__init__()
        
        self.pos_dim = pos_dim
        self.grid_size = grid_size
        self.bounds = bounds
        
        # Generate uniform grid
        self.register_buffer('grid_points', self._generate_grid())
    
    def _generate_grid(self) -> torch.Tensor:
        """Generate uniform grid points."""
        if self.pos_dim == 1:
            # 1D grid
            grid = torch.linspace(self.bounds[0], self.bounds[1], self.grid_size)
            return grid.unsqueeze(-1)  # [grid_size, 1]
        else:
            # Multi-dimensional grid
            # Support different grid sizes per dimension if provided as a list
            if isinstance(self.grid_size, int):
                grid_sizes = [self.grid_size] * self.pos_dim
            else:
                grid_sizes = self.grid_size
            
            # Support different bounds per dimension if provided as a list of tuples
            if isinstance(self.bounds, tuple):
                bounds = [self.bounds] * self.pos_dim
            else:
                bounds = self.bounds
            
            # Generate grid for each dimension
            grids = []
            for i in range(self.pos_dim):
                grid_i = torch.linspace(bounds[i][0], bounds[i][1], grid_sizes[i])
                grids.append(grid_i)
            
            # Create meshgrid
            grid_points = torch.meshgrid(*grids, indexing='ij')
            grid_points = torch.stack(grid_points, dim=-1)  # [grid_size, ..., pos_dim]
            
            # Reshape to [total_points, pos_dim]
            total_points = grid_points.shape[0]
            for i in range(1, self.pos_dim):
                total_points *= grid_points.shape[i]
            
            return grid_points.reshape(total_points, self.pos_dim)
    
    def forward(self, batch_size: int = 1) -> torch.Tensor:
        """
        Get grid points for field evaluation.
        
        Args:
            batch_size: Batch size for grid expansion
            
        Returns:
            Grid points of shape [batch_size, num_points, pos_dim]
        """
        if batch_size == 1:
            return self.grid_points.unsqueeze(0)  # [1, num_points, pos_dim]
        else:
            return self.grid_points.unsqueeze(0).expand(batch_size, -1, -1)  # [B, num_points, pos_dim]
    
    @property
    def num_points(self) -> int:
        """Get number of grid points."""
        return self.grid_points.shape[0] 