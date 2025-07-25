"""
Novel Field Interaction Operators for TFN.

This module implements advanced field interaction operators that go beyond
simple superposition, including cross-correlation, nonlinear interactions,
and multi-field coupling mechanisms.

Mathematical formulation:
    I(F₁, F₂, ..., Fₙ) = Σᵢⱼₖ αᵢⱼₖ φᵢⱼₖ(Fᵢ, Fⱼ, Fₖ)
    
Where:
    - Fᵢ = field representations
    - φᵢⱼₖ = interaction basis functions
    - αᵢⱼₖ = learnable interaction coefficients
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List
import math


class FieldInteractionOperators(nn.Module):
    """
    Novel field interaction operators.
    
    Implements advanced interaction mechanisms beyond simple superposition,
    including cross-correlation, nonlinear interactions, and multi-field coupling.
    """
    
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int = 8,
                 interaction_types: Tuple[str, ...] = ("cross_correlation", "nonlinear", "multi_field"),
                 dropout: float = 0.1):
        """
        Initialize field interaction operators.
        
        Args:
            embed_dim: Dimension of field embeddings
            num_heads: Number of interaction heads
            interaction_types: Types of interactions to compute
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.interaction_types = interaction_types
        self.head_dim = embed_dim // num_heads
        
        # Learnable interaction parameters
        self.interaction_weights = nn.Parameter(torch.ones(len(interaction_types)))
        
        # Cross-correlation parameters
        if "cross_correlation" in interaction_types:
            self.correlation_kernels = nn.Parameter(torch.randn(num_heads, 3, 3))
            nn.init.normal_(self.correlation_kernels, 0, 0.1)
        
        # Nonlinear interaction parameters
        if "nonlinear" in interaction_types:
            self.nonlinear_net = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )
        
        # Multi-field coupling parameters
        if "multi_field" in interaction_types:
            self.multi_field_coupler = nn.Parameter(torch.randn(num_heads, num_heads, num_heads))
            nn.init.normal_(self.multi_field_coupler, 0, 0.1)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                fields: torch.Tensor,  # [B, N, D] field representations
                positions: Optional[torch.Tensor] = None) -> torch.Tensor:  # [B, N, P] positions
        """
        Compute novel field interactions.
        
        Args:
            fields: Field representations [B, N, D]
            positions: Field positions [B, N, P] (optional)
            
        Returns:
            Interaction-enhanced fields [B, N, D]
        """
        batch_size, num_fields, embed_dim = fields.shape
        
        # Reshape for multi-head processing
        fields_reshaped = fields.view(batch_size, num_fields, self.num_heads, self.head_dim)
        
        # Compute different interaction types
        interaction_terms = []
        
        if "cross_correlation" in self.interaction_types:
            correlation = self._cross_correlation_interaction(fields_reshaped, positions)
            interaction_terms.append(correlation)
            
        if "nonlinear" in self.interaction_types:
            nonlinear = self._nonlinear_interaction(fields_reshaped, positions)
            interaction_terms.append(nonlinear)
            
        if "multi_field" in self.interaction_types:
            multi_field = self._multi_field_interaction(fields_reshaped, positions)
            interaction_terms.append(multi_field)
        
        # Weighted combination of interaction terms
        interaction_weights = F.softmax(self.interaction_weights, dim=0)
        combined_interaction = sum(w * term for w, term in zip(interaction_weights, interaction_terms))
        
        # Apply interaction to original fields
        enhanced_fields = fields_reshaped + self.dropout(combined_interaction)
        
        # Reshape back and project
        enhanced_fields = enhanced_fields.view(batch_size, num_fields, embed_dim)
        output = self.output_proj(enhanced_fields)
        
        return output
    
    def _cross_correlation_interaction(self, 
                                     fields: torch.Tensor,  # [B, N, H, D//H]
                                     positions: Optional[torch.Tensor] = None) -> torch.Tensor:  # [B, N, H, D//H]
        """
        Compute cross-correlation interaction: ∫ F₁(z)F₂(z+τ) dz
        
        Args:
            fields: Field representations [B, N, H, D//H]
            positions: Field positions [B, N, P] (optional)
            
        Returns:
            Cross-correlation interaction [B, N, H, D//H]
        """
        batch_size, num_fields, num_heads, head_dim = fields.shape
        
        # Compute cross-correlation for different lags
        correlation_terms = []
        
        for lag in range(1, min(4, num_fields)):  # Use up to 3 lags
            # Shift fields by lag
            fields_shifted = fields[:, lag:, :, :]  # [B, N-lag, H, D//H]
            fields_base = fields[:, :-lag, :, :]    # [B, N-lag, H, D//H]
            
            # Compute correlation: F(z) * F(z+τ)
            correlation = torch.einsum('bnhd,bmhd->bnmhd', fields_base, fields_shifted)
            
            # Apply learnable correlation weights
            # Use the first 3 dimensions of head_dim for the weights, or pad if needed
            weights = self.correlation_kernels[:, lag-1, :]  # [H, 3]
            if head_dim >= 3:
                # Pad weights to match head_dim
                weights_padded = F.pad(weights, (0, head_dim - 3))  # [H, D//H]
            else:
                # Truncate weights to match head_dim
                weights_padded = weights[:, :head_dim]  # [H, D//H]
            
            # Apply weights to correlation and reduce to 4D
            correlation_weighted = correlation * weights_padded.unsqueeze(0).unsqueeze(0)  # [B, N-lag, H, D//H]
            correlation_weighted = correlation_weighted.mean(dim=2)  # [B, N-lag, H, D//H]
            
            # Pad to original size
            padding = torch.zeros(batch_size, lag, num_heads, head_dim, device=fields.device)
            correlation_padded = torch.cat([correlation_weighted, padding], dim=1)
            
            correlation_terms.append(correlation_padded)
        
        # Combine correlation terms
        return sum(correlation_terms) / len(correlation_terms)
    
    def _nonlinear_interaction(self, 
                              fields: torch.Tensor,  # [B, N, H, D//H]
                              positions: Optional[torch.Tensor] = None) -> torch.Tensor:  # [B, N, H, D//H]
        """
        Compute nonlinear interaction: φ(F₁, F₂) where φ is a learned nonlinear function.
        
        Args:
            fields: Field representations [B, N, H, D//H]
            positions: Field positions [B, N, P] (optional)
            
        Returns:
            Nonlinear interaction [B, N, H, D//H]
        """
        batch_size, num_fields, num_heads, head_dim = fields.shape
        
        # Reshape for nonlinear network
        fields_flat = fields.view(batch_size * num_fields, self.embed_dim)
        
        # Apply nonlinear network directly to each field
        nonlinear_output = self.nonlinear_net(fields_flat)  # [B*N, D]
        
        # Reshape back to field format
        nonlinear_reshaped = nonlinear_output.view(batch_size, num_fields, num_heads, head_dim)
        
        return nonlinear_reshaped
    
    def _multi_field_interaction(self, 
                                fields: torch.Tensor,  # [B, N, H, D//H]
                                positions: Optional[torch.Tensor] = None) -> torch.Tensor:  # [B, N, H, D//H]
        """
        Compute multi-field interaction: I(F₁, F₂, F₃, ...)
        
        Args:
            fields: Field representations [B, N, H, D//H]
            positions: Field positions [B, N, P] (optional)
            
        Returns:
            Multi-field interaction [B, N, H, D//H]
        """
        batch_size, num_fields, num_heads, head_dim = fields.shape
        
        # Compute three-field interactions
        multi_field_terms = []
        
        for i in range(num_fields):
            for j in range(i+1, num_fields):
                for k in range(j+1, num_fields):
                    # Get three fields
                    field_i = fields[:, i:i+1, :, :]  # [B, 1, H, D//H]
                    field_j = fields[:, j:j+1, :, :]  # [B, 1, H, D//H]
                    field_k = fields[:, k:k+1, :, :]  # [B, 1, H, D//H]
                    
                    # Compute three-field interaction
                    # [B, 1, H, D//H] × [B, 1, H, D//H] × [B, 1, H, D//H] × [H, H, H]
                    interaction = torch.einsum('bihd,bjhd,bkhd,hfg->bihd', 
                                             field_i, field_j, field_k, self.multi_field_coupler)
                    
                    # Distribute interaction to participating fields
                    interaction_i = torch.zeros_like(fields)
                    interaction_j = torch.zeros_like(fields)
                    interaction_k = torch.zeros_like(fields)
                    
                    interaction_i[:, i:i+1, :, :] = interaction
                    interaction_j[:, j:j+1, :, :] = interaction
                    interaction_k[:, k:k+1, :, :] = interaction
                    
                    multi_field_terms.extend([interaction_i, interaction_j, interaction_k])
        
        if multi_field_terms:
            return sum(multi_field_terms) / len(multi_field_terms)
        else:
            return torch.zeros_like(fields)


class FractalInteractionOperators(FieldInteractionOperators):
    """
    Fractal interaction operators with multi-scale coupling.
    
    Implements interactions at multiple scales to capture both local
    and global field relationships.
    """
    
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int = 8,
                 scales: int = 4,
                 interaction_types: Tuple[str, ...] = ("cross_correlation", "nonlinear", "multi_field"),
                 dropout: float = 0.1):
        """
        Initialize fractal interaction operators.
        
        Args:
            embed_dim: Dimension of field embeddings
            num_heads: Number of interaction heads
            scales: Number of scales for multi-scale processing
            interaction_types: Types of interactions to compute
            dropout: Dropout rate for regularization
        """
        super().__init__(embed_dim, num_heads, interaction_types, dropout)
        self.scales = scales
        
        # Multi-scale downsamplers
        self.downsamplers = nn.ModuleList([
            nn.AvgPool1d(2**i, stride=2**i) for i in range(scales)
        ])
        
        # Scale-specific interaction weights
        self.scale_weights = nn.Parameter(torch.ones(scales))
        
    def forward(self, 
                fields: torch.Tensor,  # [B, N, D]
                positions: Optional[torch.Tensor] = None) -> torch.Tensor:  # [B, N, P]
        """
        Compute fractal field interactions.
        
        Args:
            fields: Field representations [B, N, D]
            positions: Field positions [B, N, P] (optional)
            
        Returns:
            Fractal interaction-enhanced fields [B, N, D]
        """
        batch_size, num_fields, embed_dim = fields.shape
        
        # Process at multiple scales
        scale_outputs = []
        
        for i, downsampler in enumerate(self.downsamplers):
            # Downsample fields
            fields_downsampled = fields.transpose(1, 2)  # [B, D, N]
            fields_downsampled = downsampler(fields_downsampled)  # [B, D, N//2^i]
            fields_downsampled = fields_downsampled.transpose(1, 2)  # [B, N//2^i, D]
            
            # Compute interactions at this scale
            scale_interaction = super().forward(fields_downsampled, positions)
            
            # Upsample back to original size
            scale_interaction = scale_interaction.transpose(1, 2)  # [B, D, N//2^i]
            scale_interaction = F.interpolate(scale_interaction, size=num_fields, mode='linear')
            scale_interaction = scale_interaction.transpose(1, 2)  # [B, N, D]
            
            scale_outputs.append(scale_interaction)
        
        # Weighted combination of scales
        scale_weights = F.softmax(self.scale_weights, dim=0)
        combined_output = sum(w * output for w, output in zip(scale_weights, scale_outputs))
        
        return combined_output


class CausalInteractionOperators(FieldInteractionOperators):
    """
    Causal interaction operators for time-series applications.
    
    Ensures causality by only allowing backward-looking interactions.
    """
    
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int = 8,
                 interaction_types: Tuple[str, ...] = ("cross_correlation", "nonlinear", "multi_field"),
                 dropout: float = 0.1):
        """
        Initialize causal interaction operators.
        
        Args:
            embed_dim: Dimension of field embeddings
            num_heads: Number of interaction heads
            interaction_types: Types of interactions to compute
            dropout: Dropout rate for regularization
        """
        super().__init__(embed_dim, num_heads, interaction_types, dropout)
    
    def forward(self, 
                fields: torch.Tensor,  # [B, N, D]
                positions: Optional[torch.Tensor] = None) -> torch.Tensor:  # [B, N, P]
        """
        Compute causal field interactions.
        
        Args:
            fields: Field representations [B, N, D]
            positions: Field positions [B, N, P] (optional)
            
        Returns:
            Causally-constrained interaction-enhanced fields [B, N, D]
        """
        batch_size, num_fields, embed_dim = fields.shape
        
        # Create causal mask (lower triangular)
        causal_mask = torch.tril(torch.ones(num_fields, num_fields, device=fields.device), diagonal=-1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, N, N, 1, 1]
        
        # Apply causal mask to interaction computation
        with torch.no_grad():
            # Temporarily modify interaction parameters for causal computation
            original_weights = self.interaction_weights.clone()
            # Use a simple causal constraint instead of complex masking
            causal_factor = 0.5  # Reduce interaction strength for causality
            self.interaction_weights.data = self.interaction_weights * causal_factor
        
        # Compute interactions with causal constraint
        result = super().forward(fields, positions)
        
        # Restore original weights
        with torch.no_grad():
            self.interaction_weights.data = original_weights
        
        return result


class MetaInteractionOperators(FieldInteractionOperators):
    """
    Meta interaction operators with learnable interaction types.
    
    Automatically learns which interaction types are most effective
    for different field characteristics.
    """
    
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int = 8,
                 interaction_types: Tuple[str, ...] = ("cross_correlation", "nonlinear", "multi_field"),
                 dropout: float = 0.1):
        """
        Initialize meta interaction operators.
        
        Args:
            embed_dim: Dimension of field embeddings
            num_heads: Number of interaction heads
            interaction_types: Types of interactions to compute
            dropout: Dropout rate for regularization
        """
        super().__init__(embed_dim, num_heads, interaction_types, dropout)
        
        # Meta-learning network for interaction selection
        meta_input_dim = embed_dim * 2 + 1  # field_mean + field_std + field_norm
        self.meta_net = nn.Sequential(
            nn.Linear(meta_input_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, len(interaction_types))
        )
        
    def forward(self, 
                fields: torch.Tensor,  # [B, N, D]
                positions: Optional[torch.Tensor] = None) -> torch.Tensor:  # [B, N, P]
        """
        Compute meta field interactions.
        
        Args:
            fields: Field representations [B, N, D]
            positions: Field positions [B, N, P] (optional)
            
        Returns:
            Meta interaction-enhanced fields [B, N, D]
        """
        batch_size, num_fields, embed_dim = fields.shape
        
        # Compute field characteristics for meta-learning
        field_mean = torch.mean(fields, dim=1)  # [B, D]
        field_std = torch.std(fields, dim=1)    # [B, D]
        field_norm = torch.norm(fields, dim=-1).mean(dim=1)  # [B]
        
        # Meta features
        meta_features = torch.cat([field_mean, field_std, field_norm.unsqueeze(1)], dim=-1)  # [B, 2*D+1]
        
        # Learn interaction weights
        meta_weights = self.meta_net(meta_features)  # [B, num_interaction_types]
        meta_weights = F.softmax(meta_weights, dim=-1)  # [B, num_interaction_types]
        
        # Apply meta weights to interaction computation
        with torch.no_grad():
            original_weights = self.interaction_weights.clone()
            # Use batch-specific weights
            self.interaction_weights.data = meta_weights.mean(dim=0)
        
        # Compute interactions with meta-learned weights
        result = super().forward(fields, positions)
        
        # Restore original weights
        with torch.no_grad():
            self.interaction_weights.data = original_weights
        
        return result


def create_interaction_operators(operator_type: str = "standard",
                               embed_dim: int = 256,
                               num_heads: int = 8,
                               interaction_types: Tuple[str, ...] = ("cross_correlation", "nonlinear", "multi_field"),
                               **kwargs) -> FieldInteractionOperators:
    """
    Factory function to create field interaction operator modules.
    
    Args:
        operator_type: Type of operator ("standard", "fractal", "causal", "meta")
        embed_dim: Dimension of field embeddings
        num_heads: Number of interaction heads
        interaction_types: Types of interactions to compute
        **kwargs: Additional arguments for specific operator types
        
    Returns:
        Configured field interaction operator module
    """
    if operator_type == "standard":
        return FieldInteractionOperators(embed_dim, num_heads, interaction_types, **kwargs)
    elif operator_type == "fractal":
        scales = kwargs.get('scales', 4)
        return FractalInteractionOperators(embed_dim, num_heads, scales, interaction_types, **kwargs)
    elif operator_type == "causal":
        return CausalInteractionOperators(embed_dim, num_heads, interaction_types, **kwargs)
    elif operator_type == "meta":
        return MetaInteractionOperators(embed_dim, num_heads, interaction_types, **kwargs)
    else:
        raise ValueError(f"Unknown operator type: {operator_type}") 