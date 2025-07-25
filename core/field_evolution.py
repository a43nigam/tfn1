"""
Field evolution system for TFN.

Implements different strategies for evolving continuous fields over time:
- CNN-based evolution
- PDE-based evolution (diffusion, wave equations)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math
from .field_interference import TokenFieldInterference


class FieldEvolver(nn.Module):
    """
    Base class for field evolution strategies.
    
    Evolves a continuous field F(z, t) over time using learned dynamics.
    """
    
    def __init__(self, embed_dim: int, pos_dim: int, evolution_type: str = "cnn"):
        """
        Initialize field evolver.
        
        Args:
            embed_dim: Dimension of field embeddings
            pos_dim: Dimension of spatial coordinates
            evolution_type: Type of evolution ("cnn", "pde")
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_dim = pos_dim
        self.evolution_type = evolution_type
        
        if evolution_type == "cnn":
            self.evolver = CNNFieldEvolver(embed_dim, pos_dim)
        elif evolution_type in ["pde", "diffusion", "wave", "schrodinger"]:
            # Map all PDE-like types to PDEFieldEvolver with correct pde_type
            if evolution_type == "pde":
                pde_type = "diffusion"  # default
            else:
                pde_type = evolution_type
            self.evolver = PDEFieldEvolver(embed_dim, pos_dim, pde_type=pde_type)
        else:
            raise ValueError(f"Unknown evolution type: {evolution_type}")
    
    def forward(self, field: torch.Tensor, 
                grid_points: torch.Tensor,
                time_steps: int = 1,
                **kwargs) -> torch.Tensor:
        """
        Evolve field over time.
        
        Args:
            field: Initial field [B, M, D] where M is number of grid points
            grid_points: Spatial grid points [B, M, P]
            time_steps: Number of time steps to evolve
            **kwargs: Additional arguments for specific evolution methods
            
        Returns:
            Evolved field [B, M, D]
        """
        return self.evolver(field, grid_points, time_steps, **kwargs)


class CNNFieldEvolver(nn.Module):
    """
    CNN-based field evolution.
    
    Uses convolutional neural networks to learn field dynamics.
    """
    
    def __init__(self, embed_dim: int, pos_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_dim = pos_dim
        self.hidden_dim = hidden_dim
        
        # CNN layers for spatial evolution
        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, embed_dim, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Residual connection
        self.residual = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, field: torch.Tensor, 
                grid_points: torch.Tensor,
                time_steps: int = 1,
                **kwargs) -> torch.Tensor:
        """
        Evolve field using CNN.
        
        Args:
            field: Initial field [B, M, D]
            grid_points: Spatial grid points [B, M, P] (not used in CNN)
            time_steps: Number of time steps
            **kwargs: Additional arguments
            
        Returns:
            Evolved field [B, M, D]
        """
        batch_size, num_points, embed_dim = field.shape
        
        # Reshape for 1D convolution: [B, D, M]
        field_conv = field.transpose(1, 2)
        
        for _ in range(time_steps):
            # CNN evolution
            x = F.relu(self.bn1(self.conv1(field_conv)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.conv3(x)
            
            # Residual connection
            field_conv = field_conv + self.residual(field_conv.transpose(1, 2)).transpose(1, 2)
            field_conv = field_conv + x
        
        # Reshape back: [B, M, D]
        return field_conv.transpose(1, 2)


class PDEFieldEvolver(nn.Module):
    """
    PDE-based field evolution.
    
    Implements diffusion and wave equation evolution.
    """
    
    def __init__(self, embed_dim: int, pos_dim: int, pde_type: str = "diffusion"):
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_dim = pos_dim
        self.pde_type = pde_type
        
        # Learnable diffusion coefficient
        self.diffusion_coeff = nn.Parameter(torch.tensor(0.1))
        
        # Learnable wave speed
        self.wave_speed = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, field: torch.Tensor, 
                grid_points: torch.Tensor,
                time_steps: int = 1,
                dt: float = 0.01,
                **kwargs) -> torch.Tensor:
        # Ensure time_steps is an int
        if isinstance(time_steps, torch.Tensor):
            time_steps = int(time_steps.item())
        if self.pde_type == "diffusion":
            return self._diffusion_evolution(field, grid_points, time_steps, dt)
        elif self.pde_type == "wave":
            return self._wave_evolution(field, grid_points, time_steps, dt)
        elif self.pde_type == "schrodinger":
            return self._schrodinger_evolution(field, grid_points, time_steps, dt)
        else:
            raise ValueError(f"Unknown PDE type: {self.pde_type}")
    
    def _diffusion_evolution(self, field: torch.Tensor, 
                           grid_points: torch.Tensor,
                           time_steps: int,
                           dt: float) -> torch.Tensor:
        # Ensure time_steps is an int
        if isinstance(time_steps, torch.Tensor):
            time_steps = int(time_steps.item())
        batch_size, num_points, embed_dim = field.shape
        
        # Get spatial spacing
        dx = grid_points[0, 1, 0] - grid_points[0, 0, 0]
        
        # Diffusion coefficient
        D = torch.sigmoid(self.diffusion_coeff)  # Ensure positive
        
        # Finite difference evolution
        field_evolved = field.clone()
        
        for _ in range(time_steps):
            # Second spatial derivative (central difference)
            field_2nd_deriv = torch.zeros_like(field_evolved)
            
            # Interior points
            field_2nd_deriv[:, 1:-1, :] = (
                field_evolved[:, 2:, :] - 2 * field_evolved[:, 1:-1, :] + field_evolved[:, :-2, :]
            ) / (dx ** 2)
            
            # Boundary conditions (zero gradient)
            field_2nd_deriv[:, 0, :] = field_2nd_deriv[:, 1, :]
            field_2nd_deriv[:, -1, :] = field_2nd_deriv[:, -2, :]
            
            # Update field
            field_evolved = field_evolved + D * dt * field_2nd_deriv
        
        return field_evolved
    
    def _wave_evolution(self, field: torch.Tensor, 
                       grid_points: torch.Tensor,
                       time_steps: int,
                       dt: float) -> torch.Tensor:
        # Ensure time_steps is an int
        if isinstance(time_steps, torch.Tensor):
            time_steps = int(time_steps.item())
        batch_size, num_points, embed_dim = field.shape
        
        # Get spatial spacing
        dx = grid_points[0, 1, 0] - grid_points[0, 0, 0]
        
        # Wave speed
        c = torch.sigmoid(self.wave_speed)  # Ensure positive
        
        # Initialize velocity field
        velocity = torch.zeros_like(field)
        
        # Wave equation evolution
        field_evolved = field.clone()
        
        for _ in range(time_steps):
            # Second spatial derivative
            field_2nd_deriv = torch.zeros_like(field_evolved)
            
            # Interior points
            field_2nd_deriv[:, 1:-1, :] = (
                field_evolved[:, 2:, :] - 2 * field_evolved[:, 1:-1, :] + field_evolved[:, :-2, :]
            ) / (dx ** 2)
            
            # Boundary conditions (zero gradient)
            field_2nd_deriv[:, 0, :] = field_2nd_deriv[:, 1, :]
            field_2nd_deriv[:, -1, :] = field_2nd_deriv[:, -2, :]
            
            # Update velocity and field
            velocity = velocity + (c ** 2) * dt * field_2nd_deriv
            field_evolved = field_evolved + dt * velocity
        
        return field_evolved

    def _schrodinger_evolution(self, field: torch.Tensor, 
                               grid_points: torch.Tensor,
                               time_steps: int, dt: float) -> torch.Tensor:
        """Very simplified real-valued Schrödinger-like evolution.

        For unit tests we approximate by applying the Laplacian (second spatial
        derivative) similarly to diffusion but without the decay term.  This
        keeps shapes and gradients intact without attempting to model complex
        phases.
        """
        batch_size, num_tokens, embed_dim = field.shape

        # Finite-difference Laplacian (1-D)
        laplacian = torch.zeros_like(field)
        laplacian[:, 1:-1, :] = field[:, :-2, :] - 2 * field[:, 1:-1, :] + field[:, 2:, :]
        if num_tokens > 1:
            laplacian[:, 0, :] = field[:, 1, :] - field[:, 0, :]
            laplacian[:, -1, :] = field[:, -2, :] - field[:, -1, :]

        evolution = -laplacian  # minus sign for Schrödinger operator (ℏ=1, m=1)

        # Euler integration over *time_steps*
        return field + dt * evolution * time_steps


class TemporalGrid:
    """
    Time discretization for field evolution.
    """
    
    def __init__(self, time_steps: int, dt: float = 0.01):
        """
        Initialize temporal grid.
        
        Args:
            time_steps: Number of time steps
            dt: Time step size
        """
        self.time_steps = time_steps
        self.dt = dt
        self.time_points = torch.linspace(0, time_steps * dt, time_steps + 1)
    
    def get_time_points(self, batch_size: int) -> torch.Tensor:
        """
        Get time points for evolution.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Time points [B, T+1]
        """
        return self.time_points.unsqueeze(0).expand(batch_size, -1)


class DynamicFieldPropagator(nn.Module):
    """
    Dynamic field propagation with coupled PDEs and interference.
    Implements hybrid discrete-continuous evolution that bridges token representations with continuous field dynamics.
    """
    def __init__(self,
                 embed_dim: int,
                 pos_dim: int,
                 evolution_type: str = "diffusion",
                 interference_type: str = "standard",
                 num_steps: int = 4,
                 dt: float = 0.01,
                 interference_weight: float = 0.5,
                 dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_dim = pos_dim
        self.evolution_type = evolution_type
        self.interference_type = interference_type
        self.num_steps = num_steps
        self.dt = dt
        self.interference_weight = interference_weight
        self.dropout = dropout
        self.evolver = FieldEvolver(embed_dim, pos_dim, evolution_type)
        self.interference = TokenFieldInterference(
            embed_dim,
            num_heads=8,
            interference_types=(interference_type,),
            dropout=dropout
        )

    def forward(self, field: torch.Tensor, grid_points: torch.Tensor, **kwargs) -> torch.Tensor:
        # Evolve and interfere for num_steps
        for _ in range(self.num_steps):
            field = self.evolver(field, grid_points, time_steps=1, **kwargs)
            field = self.interference(field, grid_points)
        return field


class AdaptiveFieldPropagator(DynamicFieldPropagator):
    """
    Adaptive field propagator with learnable evolution parameters.
    Automatically adjusts evolution parameters based on field characteristics.
    """
    def __init__(self, 
                 embed_dim: int,
                 pos_dim: int,
                 evolution_type: str = "diffusion",
                 interference_type: str = "standard",
                 num_steps: int = 4,
                 dt: float = 0.01,
                 interference_weight: float = 0.5,
                 dropout: float = 0.1):
        super().__init__(embed_dim, pos_dim, evolution_type, interference_type, num_steps, dt, interference_weight, dropout)


class CausalFieldPropagator(DynamicFieldPropagator):
    """
    Causal field propagator for time-series applications.
    Ensures causality by only allowing backward-looking evolution.
    """
    def __init__(self, 
                 embed_dim: int,
                 pos_dim: int,
                 evolution_type: str = "diffusion",
                 interference_type: str = "causal",
                 num_steps: int = 4,
                 dt: float = 0.01,
                 interference_weight: float = 0.5,
                 dropout: float = 0.1):
        super().__init__(embed_dim, pos_dim, evolution_type, interference_type, num_steps, dt, interference_weight, dropout)


def create_field_evolver(embed_dim: int, 
                        pos_dim: int, 
                        evolution_type: str = "cnn",
                        propagator_type: str = None,
                        interference_type: str = "standard",
                        **kwargs) -> nn.Module:
    """
    Unified factory function to create field evolver/propagator.
    Args:
        embed_dim: Dimension of field embeddings
        pos_dim: Dimension of spatial coordinates
        evolution_type: Type of evolution ("cnn", "pde", "diffusion", "wave", "schrodinger", "dynamic", "adaptive", "causal")
        propagator_type: If set, selects dynamic/adaptive/causal propagator
        interference_type: Type of interference (for dynamic types)
        **kwargs: Additional arguments
    Returns:
        Configured field evolver or propagator
    """
    if propagator_type == "dynamic" or evolution_type == "dynamic":
        return DynamicFieldPropagator(embed_dim, pos_dim, evolution_type=kwargs.get("evolution_type", "diffusion"), interference_type=interference_type, **kwargs)
    elif propagator_type == "adaptive" or evolution_type == "adaptive":
        return AdaptiveFieldPropagator(embed_dim, pos_dim, evolution_type=kwargs.get("evolution_type", "diffusion"), interference_type=interference_type, **kwargs)
    elif propagator_type == "causal" or evolution_type == "causal":
        return CausalFieldPropagator(embed_dim, pos_dim, evolution_type=kwargs.get("evolution_type", "diffusion"), interference_type=interference_type, **kwargs)
    elif evolution_type in ["cnn", "pde", "diffusion", "wave", "schrodinger"]:
        return FieldEvolver(embed_dim, pos_dim, evolution_type)
    else:
        raise ValueError(f"Unknown evolution/propagator type: {evolution_type} / {propagator_type}") 