"""
Kernel system for TFN field emission.

This module provides the mathematical foundation for how tokens emit fields
across the spatial domain. Different kernels encode different physical intuitions.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class KernelBasis(nn.Module, ABC):
    """
    Abstract base class for field emission kernels.
    
    A kernel determines how each token's influence spreads across the spatial domain.
    This is the mathematical foundation of TFN's field-based attention mechanism.
    
    Mathematical formulation:
        K(z, μ, σ) = φ(||z - μ||²/σ²)
    
    where:
        z = grid point position [M, P]
        μ = token position [B, N, P] 
        σ = spread/width parameter [B, N, 1]
        φ = kernel function (Gaussian, RBF, etc.)
    """
    
    def __init__(self, pos_dim: int):
        super().__init__()
        self.pos_dim = pos_dim
        
    @abstractmethod
    def forward(self, 
                z: torch.Tensor,           # [M, P] or [B, M, P] grid points
                mu: torch.Tensor,          # [B, N, P] token positions
                sigma: torch.Tensor) -> torch.Tensor:  # [B, N, 1] spread parameters
        """
        Compute kernel values between grid points and token positions.
        
        Args:
            z: Grid point positions of shape [M, P] or [B, M, P]
            mu: Token positions of shape [B, N, P]
            sigma: Spread parameters of shape [B, N, 1]
            
        Returns:
            Kernel values of shape [B, N, M] representing influence
            of each token on each grid point.
        """
        pass
    
    def compute_influence_matrix(self,
                                z: torch.Tensor,      # [M, P] grid points
                                mu: torch.Tensor,     # [B, N, P] token positions  
                                sigma: torch.Tensor) -> torch.Tensor:  # [B, N, 1] spread
        """
        Compute the complete influence matrix between tokens and grid points.
        
        This is the main interface used by FieldProjector.
        
        Args:
            z: Grid point positions of shape [M, P]
            mu: Token positions of shape [B, N, P]
            sigma: Spread parameters of shape [B, N, 1]
            
        Returns:
            Influence matrix of shape [B, N, M] where [B, i, j] represents
            the influence of token i on grid point j.
        """
        # Ensure z has batch dimension for broadcasting
        if z.dim() == 2:
            z = z.unsqueeze(0)  # [M, P] -> [1, M, P]
        
        # Compute kernel values
        kernel_values = self.forward(z, mu, sigma)  # [B, N, M]
        
        return kernel_values


class RBFKernel(KernelBasis):
    """
    Radial Basis Function (Gaussian) kernel.
    
    This is the most common kernel for TFN, providing smooth, differentiable
    field emission that decays exponentially with distance.
    
    Mathematical formulation:
        K(z, μ, σ) = exp(-||z - μ||²/(2σ²))
    
    Properties:
        - Smooth and infinitely differentiable
        - Infinite support (influences all points)
        - Natural for continuous fields
        - Like Gaussian probability distribution
    """
    
    def __init__(self, pos_dim: int, min_sigma: float = 0.1, max_sigma: float = 10.0):
        super().__init__(pos_dim)
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        
    def forward(self, 
                z: torch.Tensor,           # [M, P] or [B, M, P] grid points
                mu: torch.Tensor,          # [B, N, P] token positions
                sigma: torch.Tensor) -> torch.Tensor:  # [B, N, 1] spread parameters
        """
        Compute RBF kernel values.
        
        Args:
            z: Grid point positions
            mu: Token positions  
            sigma: Spread parameters (clamped to [min_sigma, max_sigma])
            
        Returns:
            RBF kernel values of shape [B, N, M]
        """
        # Ensure z has batch dimension
        if z.dim() == 2:
            z = z.unsqueeze(0)  # [M, P] -> [1, M, P]
        
        # Clamp sigma to valid range
        sigma = torch.clamp(sigma, min=self.min_sigma, max=self.max_sigma)
        
        # Compute squared distances: ||z - μ||²
        # z: [B, M, P], mu: [B, N, P] -> distances: [B, N, M]
        distances_squared = torch.sum((z.unsqueeze(1) - mu.unsqueeze(2))**2, dim=-1)
        
        # Compute RBF kernel: exp(-||z - μ||²/(2σ²))
        # sigma: [B, N, 1] -> [B, N, 1] for broadcasting
        sigma_expanded = sigma.squeeze(-1).unsqueeze(-1)  # [B, N, 1, 1]
        kernel_values = torch.exp(-distances_squared / (2 * sigma_expanded**2))
        
        return kernel_values


class CompactKernel(KernelBasis):
    """
    Compact support kernel with finite influence radius.
    
    This kernel provides localized influence, only affecting nearby grid points.
    This is computationally efficient for sparse interactions.
    
    Mathematical formulation:
        K(z, μ, r) = max(0, 1 - ||z - μ||/r)
    
    Properties:
        - Finite support (only influences nearby points)
        - Computationally efficient
        - Good for sparse interactions
        - Like a "bubble" of influence
    """
    
    def __init__(self, pos_dim: int, min_radius: float = 0.1, max_radius: float = 5.0):
        super().__init__(pos_dim)
        self.min_radius = min_radius
        self.max_radius = max_radius
        
    def forward(self, 
                z: torch.Tensor,           # [M, P] or [B, M, P] grid points
                mu: torch.Tensor,          # [B, N, P] token positions
                radius: torch.Tensor) -> torch.Tensor:  # [B, N, 1] radius parameters
        """
        Compute compact support kernel values.
        
        Args:
            z: Grid point positions
            mu: Token positions
            radius: Radius parameters (clamped to [min_radius, max_radius])
            
        Returns:
            Compact kernel values of shape [B, N, M]
        """
        # Ensure z has batch dimension
        if z.dim() == 2:
            z = z.unsqueeze(0)  # [M, P] -> [1, M, P]
        
        # Clamp radius to valid range
        radius = torch.clamp(radius, min=self.min_radius, max=self.max_radius)
        
        # Compute distances: ||z - μ||
        distances = torch.norm(z.unsqueeze(1) - mu.unsqueeze(2), dim=-1)
        
        # Compute compact kernel: max(0, 1 - ||z - μ||/r)
        # radius: [B, N, 1] -> [B, N, 1] for broadcasting
        radius_expanded = radius.squeeze(-1).unsqueeze(-1)  # [B, N, 1, 1]
        kernel_values = torch.clamp(1 - distances / radius_expanded, min=0)
        
        return kernel_values


class FourierKernel(KernelBasis):
    """
    Fourier (cosine) kernel for oscillatory field emission.
    
    This kernel provides wave-like influence patterns, useful for
    periodic problems or wave phenomena.
    
    Mathematical formulation:
        K(z, μ, ω) = cos(ω||z - μ||)
    
    Properties:
        - Oscillatory behavior
        - Good for wave-like phenomena
        - Natural for periodic problems
        - Like wave interference patterns
    """
    
    def __init__(self, pos_dim: int, min_freq: float = 0.1, max_freq: float = 10.0):
        super().__init__(pos_dim)
        self.min_freq = min_freq
        self.max_freq = max_freq
        
    def forward(self, 
                z: torch.Tensor,           # [M, P] or [B, M, P] grid points
                mu: torch.Tensor,          # [B, N, P] token positions
                freq: torch.Tensor) -> torch.Tensor:  # [B, N, 1] frequency parameters
        """
        Compute Fourier kernel values.
        
        Args:
            z: Grid point positions
            mu: Token positions
            freq: Frequency parameters (clamped to [min_freq, max_freq])
            
        Returns:
            Fourier kernel values of shape [B, N, M]
        """
        # Ensure z has batch dimension
        if z.dim() == 2:
            z = z.unsqueeze(0)  # [M, P] -> [1, M, P]
        
        # Clamp frequency to valid range
        freq = torch.clamp(freq, min=self.min_freq, max=self.max_freq)
        
        # Compute distances: ||z - μ||
        distances = torch.norm(z.unsqueeze(1) - mu.unsqueeze(2), dim=-1)
        
        # Compute Fourier kernel: cos(ω||z - μ||)
        # freq: [B, N, 1] -> [B, N, 1] for broadcasting
        freq_expanded = freq.squeeze(-1).unsqueeze(-1)  # [B, N, 1, 1]
        kernel_values = torch.cos(freq_expanded * distances)
        
        return kernel_values


class LearnableKernel(KernelBasis):
    """
    Learnable kernel that can adapt to data.
    
    This kernel learns the optimal influence pattern from data,
    providing maximum flexibility for different problems.
    
    Mathematical formulation:
        K(z, μ, θ) = φ(||z - μ||, θ) where θ are learnable parameters
    """
    
    def __init__(self, pos_dim: int, hidden_dim: int = 64):
        super().__init__(pos_dim)
        self.hidden_dim = hidden_dim
        
        # Learnable parameters
        self.distance_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, 
                z: torch.Tensor,           # [M, P] or [B, M, P] grid points
                mu: torch.Tensor,          # [B, N, P] token positions
                params: torch.Tensor) -> torch.Tensor:  # [B, N, hidden_dim] learnable params
        """
        Compute learnable kernel values.
        
        Args:
            z: Grid point positions
            mu: Token positions
            params: Learnable parameters (used to condition the network)
            
        Returns:
            Learnable kernel values of shape [B, N, M]
        """
        # Ensure z has batch dimension
        if z.dim() == 2:
            z = z.unsqueeze(0)  # [M, P] -> [1, M, P]
        
        # Compute distances: ||z - μ||
        distances = torch.norm(z.unsqueeze(1) - mu.unsqueeze(2), dim=-1)  # [B, N, M]
        
        # Normalize distances to [0, 1] for the network
        max_distance = torch.max(distances)
        normalized_distances = distances / (max_distance + 1e-8)
        
        # Apply learnable transformation using params to condition the network
        # Reshape for network: [B, N, M] -> [B*N*M, 1]
        distances_flat = normalized_distances.reshape(-1, 1)
        
        # Use params to condition the network (simple approach: add params to distances)
        # params: [B, N, hidden_dim] -> [B*N, hidden_dim]
        params_flat = params.reshape(-1, params.shape[-1])  # [B*N, hidden_dim]
        
        # Expand params to match distances: [B*N, hidden_dim] -> [B*N*M, hidden_dim]
        params_expanded = params_flat.unsqueeze(1).expand(-1, distances.shape[-1], -1)  # [B*N, M, hidden_dim]
        params_expanded = params_expanded.reshape(-1, params.shape[-1])  # [B*N*M, hidden_dim]
        
        # Combine distances with params (simple concatenation)
        # For now, just use the first dimension of params as a scaling factor
        scaling_factor = params_expanded[:, 0:1]  # [B*N*M, 1]
        scaled_distances = distances_flat * (1 + scaling_factor)
        
        # Apply network
        kernel_values_flat = self.distance_net(scaled_distances)  # [B*N*M, 1]
        
        # Reshape back: [B*N*M, 1] -> [B, N, M]
        kernel_values = kernel_values_flat.reshape(distances.shape)
        
        return kernel_values


class KernelFactory:
    """
    Factory for creating different kernel types.
    
    This provides a clean interface for selecting and configuring kernels.
    """
    
    @staticmethod
    def create(kernel_type: str, pos_dim: int, **kwargs) -> KernelBasis:
        """
        Create a kernel of the specified type.
        
        Args:
            kernel_type: Type of kernel ("rbf", "compact", "fourier", "learnable")
            pos_dim: Dimension of position space
            **kwargs: Additional kernel-specific parameters
            
        Returns:
            Kernel instance
        """
        if kernel_type == "rbf":
            return RBFKernel(pos_dim, **kwargs)
        elif kernel_type == "compact":
            return CompactKernel(pos_dim, **kwargs)
        elif kernel_type == "fourier":
            return FourierKernel(pos_dim, **kwargs)
        elif kernel_type == "learnable":
            return LearnableKernel(pos_dim, **kwargs)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    @staticmethod
    def get_available_kernels() -> list:
        """Get list of available kernel types."""
        return ["rbf", "compact", "fourier", "learnable"] 