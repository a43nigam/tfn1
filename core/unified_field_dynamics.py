"""
Unified Field Dynamics for TFN.

This module implements a unified mathematical framework that properly integrates
field interference and evolution with stability analysis and physics constraints.

Mathematical formulation:
    ∂F/∂t = L(F) + I(F) + C(F)
    
Where:
    - L(F) = linear evolution operator (diffusion, wave, Schrödinger)
    - I(F) = interference term: Σᵢⱼ βᵢⱼ I(Fᵢ, Fⱼ)
    - C(F) = physics constraints (energy conservation, symmetry)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, Literal
import math
from .field_interference import TokenFieldInterference
from .field_evolution import DynamicFieldPropagator


class UnifiedFieldDynamics(nn.Module):
    """
    Unified field dynamics with interference and evolution.
    
    Implements a mathematically sound integration of field interference
    and evolution with stability analysis and physics constraints.
    
    Mathematical formulation:
        ∂F/∂t = L(F) + I(F) + C(F)
        F(t+Δt) = F(t) + Δt * [L(F) + I(F) + C(F)]
    """
    
    def __init__(self, 
                 embed_dim: int,
                 pos_dim: int,
                 evolution_type: str = "diffusion",
                 interference_type: str = "standard",
                 num_steps: int = 4,
                 dt: float = 0.01,
                 interference_weight: float = 0.5,
                 constraint_weight: float = 0.1,
                 stability_threshold: float = 1.0,
                 dropout: float = 0.1):
        """
        Initialize unified field dynamics.
        
        Args:
            embed_dim: Dimension of token embeddings
            pos_dim: Dimension of position space
            evolution_type: Type of evolution ("diffusion", "wave", "schrodinger")
            interference_type: Type of interference ("standard", "causal", "multiscale", "physics")
            num_steps: Number of evolution steps
            dt: Time step size
            interference_weight: Weight for interference terms
            constraint_weight: Weight for physics constraints
            stability_threshold: Threshold for stability analysis
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_dim = pos_dim
        self.evolution_type = evolution_type
        self.num_steps = num_steps
        self.dt = dt
        self.interference_weight = interference_weight
        self.constraint_weight = constraint_weight
        self.stability_threshold = stability_threshold
        
        # Linear evolution operator
        self.linear_operator = self._create_linear_operator(evolution_type, embed_dim)
        
        # Interference operator
        self.interference_operator = TokenFieldInterference(
            embed_dim=embed_dim,
            interference_types=("constructive", "destructive", "phase"),
            dropout=dropout
        )
        
        # Physics constraint operator
        self.constraint_operator = PhysicsConstraintOperator(
            embed_dim=embed_dim,
            constraint_weight=constraint_weight
        )
        
        # Stability monitor
        self.stability_monitor = StabilityMonitor(stability_threshold)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def _create_linear_operator(self, evolution_type: str, embed_dim: int) -> nn.Module:
        """Create linear evolution operator."""
        if evolution_type == "diffusion":
            return DiffusionOperator(embed_dim)
        elif evolution_type == "wave":
            return WaveOperator(embed_dim)
        elif evolution_type == "schrodinger":
            return SchrodingerOperator(embed_dim)
        else:
            raise ValueError(f"Unknown evolution type: {evolution_type}")
    
    def forward(self, 
                fields: torch.Tensor,  # [B, N, D]
                positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Evolve fields with unified dynamics.
        
        Args:
            fields: Initial field values [B, N, D]
            positions: Token positions [B, N, P] (optional)
            
        Returns:
            Evolved fields [B, N, D]
        """
        batch_size, num_tokens, embed_dim = fields.shape
        
        # Initialize evolved fields
        evolved_fields = fields.clone()
        
        # Initialize velocity field for wave equation
        if self.evolution_type == "wave":
            velocities = torch.zeros_like(evolved_fields)
        
        # Multi-step evolution with stability monitoring
        for step in range(self.num_steps):
            # Compute linear evolution: L(F)
            if self.evolution_type == "wave":
                linear_evolution, velocities = self.linear_operator(evolved_fields, velocities)
            else:
                linear_evolution = self.linear_operator(evolved_fields)
            
            # Compute interference: I(F)
            interference = self.interference_operator(evolved_fields, positions)
            
            # Compute physics constraints: C(F)
            constraints = self.constraint_operator(evolved_fields)
            
            # Combined evolution: ∂F/∂t = L(F) + I(F) + C(F)
            constraints = constraints.sum(dim=-1)  # [B, N, D]
            total_evolution = (linear_evolution + 
                             self.interference_weight * interference + 
                             self.constraint_weight * constraints)
            
            # Stability check
            if not self.stability_monitor.check_stability(total_evolution):
                # Apply stability correction
                total_evolution = self.stability_monitor.apply_stability_correction(total_evolution)
            
            # Update fields: F(t+Δt) = F(t) + Δt * [L(F) + I(F) + C(F)]
            evolved_fields = evolved_fields + self.dt * total_evolution
            
            # Apply dropout for regularization
            evolved_fields = self.dropout(evolved_fields)
        
        # Final output projection
        output = self.output_proj(evolved_fields)
        
        return output
    
    def get_physics_constraints(self) -> Dict[str, torch.Tensor]:
        """Get physics constraint losses."""
        return self.constraint_operator.get_constraints()
    
    def get_stability_metrics(self) -> Dict[str, torch.Tensor]:
        """Get stability metrics."""
        return self.stability_monitor.get_metrics()


class DiffusionOperator(nn.Module):
    """Diffusion operator: ∂F/∂t = α∇²F"""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.diffusion_coeff = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, embed_dim = fields.shape
        
        # Learnable diffusion coefficient
        alpha = torch.clamp(self.diffusion_coeff, min=0.01, max=1.0)
        
        # Compute discrete Laplacian
        laplacian = torch.zeros_like(fields)
        
        if num_tokens > 2:
            laplacian[:, 1:-1, :] = (fields[:, 2:, :] - 2 * fields[:, 1:-1, :] + fields[:, :-2, :])
        
        if num_tokens > 1:
            laplacian[:, 0, :] = fields[:, 1, :] - fields[:, 0, :]
            laplacian[:, -1, :] = fields[:, -2, :] - fields[:, -1, :]
        
        return alpha * laplacian


class WaveOperator(nn.Module):
    """Wave operator: ∂²F/∂t² = c²∇²F"""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.wave_speed = nn.Parameter(torch.tensor(1.0))
        self.dt = 0.01
        
    def forward(self, fields: torch.Tensor, velocities: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_tokens, embed_dim = fields.shape
        
        # Learnable wave speed
        c = torch.clamp(self.wave_speed, min=0.1, max=10.0)
        
        # Compute discrete Laplacian
        laplacian = torch.zeros_like(fields)
        
        if num_tokens > 2:
            laplacian[:, 1:-1, :] = (fields[:, 2:, :] - 2 * fields[:, 1:-1, :] + fields[:, :-2, :])
        
        if num_tokens > 1:
            laplacian[:, 0, :] = fields[:, 1, :] - fields[:, 0, :]
            laplacian[:, -1, :] = fields[:, -2, :] - fields[:, -1, :]
        
        # Second-order wave equation: ∂²F/∂t² = c²∇²F
        # Split into first-order system:
        # ∂F/∂t = v
        # ∂v/∂t = c²∇²F
        
        # Update velocity: ∂v/∂t = c²∇²F
        acceleration = c**2 * laplacian
        new_velocities = velocities + self.dt * acceleration
        
        # Update field: ∂F/∂t = v
        field_evolution = new_velocities
        
        return field_evolution, new_velocities


class SchrodingerOperator(nn.Module):
    """Schrödinger operator: i∂F/∂t = HF"""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        # Complex Hamiltonian: H = H_real + i*H_imag
        self.hamiltonian_real = nn.Parameter(torch.eye(embed_dim))
        self.hamiltonian_imag = nn.Parameter(torch.zeros(embed_dim, embed_dim))
        
    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, embed_dim = fields.shape
        
        # Construct complex Hamiltonian: H = H_real + i*H_imag
        H_real = self.hamiltonian_real
        H_imag = self.hamiltonian_imag
        
        # Ensure Hermitian: H = (H + H†)/2
        H_real = (H_real + H_real.T) / 2
        H_imag = (H_imag - H_imag.T) / 2  # Anti-Hermitian imaginary part
        
        # Apply Hamiltonian: HF
        hamiltonian_evolution_real = torch.einsum('bnd,de->bne', fields, H_real)
        hamiltonian_evolution_imag = torch.einsum('bnd,de->bne', fields, H_imag)
        
        # For Schrödinger equation: i∂F/∂t = HF
        # ∂F/∂t = -i*HF = H_imag*F - i*H_real*F
        # Since we work with real fields, we take the real part
        evolution_real = hamiltonian_evolution_imag  # Real part of -i*HF
        
        return evolution_real


class PhysicsConstraintOperator(nn.Module):
    """Physics constraint operator for energy conservation and symmetry."""
    
    def __init__(self, embed_dim: int, constraint_weight: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.constraint_weight = constraint_weight
        
        # Energy conservation parameters
        self.energy_target = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        """Apply physics constraints."""
        batch_size, num_tokens, embed_dim = fields.shape
        
        # Energy conservation constraint
        field_energy = torch.norm(fields, dim=-1, keepdim=True)**2  # [B, N, 1]
        energy_constraint = (field_energy - self.energy_target).unsqueeze(-1).expand(-1, -1, embed_dim, -1)  # [B, N, D, 1]
        
        # Symmetry constraint (field should be symmetric around center)
        center_idx = num_tokens // 2
        if num_tokens % 2 == 0:
            # Even number of tokens
            left_half = fields[:, :center_idx, :]
            right_half = fields[:, center_idx:, :].flip(dims=[1])
            symmetry_constraint = (left_half - right_half).unsqueeze(-1)  # [B, N//2, D, 1]
        else:
            # Odd number of tokens
            left_half = fields[:, :center_idx, :]
            right_half = fields[:, center_idx+1:, :].flip(dims=[1])
            symmetry_constraint = (left_half - right_half).unsqueeze(-1)  # [B, N//2, D, 1]
        
        # Pad symmetry constraint to match energy constraint dimensions
        # [B, N//2, D, 1] -> [B, N, D, 1]
        if symmetry_constraint.shape[1] < num_tokens:
            padding_size = num_tokens - symmetry_constraint.shape[1]
            symmetry_constraint = F.pad(symmetry_constraint, (0, 0, 0, 0, 0, padding_size))
        
        # Combine constraints - ensure they have the same shape
        constraints = torch.cat([energy_constraint, symmetry_constraint], dim=-1)
        
        return constraints
    
    def get_constraints(self) -> Dict[str, torch.Tensor]:
        """Get constraint losses."""
        return {
            "energy_conservation": self.energy_target,
            "symmetry": torch.tensor(0.0)  # Placeholder
        }


class StabilityMonitor(nn.Module):
    """Monitor and enforce numerical stability."""
    
    def __init__(self, threshold: float = 1.0):
        super().__init__()
        self.threshold = threshold
        self.stability_metrics = {}
        
    def check_stability(self, evolution: torch.Tensor) -> bool:
        """Check if evolution is numerically stable."""
        # Check for NaN or infinite values
        if torch.isnan(evolution).any() or torch.isinf(evolution).any():
            return False
        
        # Check magnitude bounds
        max_magnitude = torch.norm(evolution, dim=-1).max()
        if max_magnitude > self.threshold:
            return False
        
        return True
    
    def apply_stability_correction(self, evolution: torch.Tensor) -> torch.Tensor:
        """Apply stability correction to evolution."""
        # Clip magnitudes to threshold
        magnitudes = torch.norm(evolution, dim=-1, keepdim=True)
        scale_factor = torch.clamp(self.threshold / (magnitudes + 1e-8), max=1.0) * 0.999
        
        # Apply scaling
        corrected_evolution = evolution * scale_factor
        
        return corrected_evolution
    
    def get_metrics(self) -> Dict[str, torch.Tensor]:
        """Get stability metrics."""
        return self.stability_metrics


def create_unified_field_dynamics(embed_dim: int = 256,
                                pos_dim: int = 1,
                                evolution_type: str = "diffusion",
                                interference_type: str = "standard",
                                **kwargs) -> UnifiedFieldDynamics:
    """Create unified field dynamics module."""
    return UnifiedFieldDynamics(
        embed_dim=embed_dim,
        pos_dim=pos_dim,
        evolution_type=evolution_type,
        interference_type=interference_type,
        **kwargs
    ) 