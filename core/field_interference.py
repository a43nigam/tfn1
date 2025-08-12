"""
Token-Centric Field Interference Mechanisms for TFN.

This module implements field interference directly in token space, maintaining
computational efficiency while adding physics-inspired interactions.

Mathematical formulation:
    I(F_i, F_j) = Σ_k α_k φ_k(F_i, F_j)
    
Where:
    - F_i, F_j = token field representations
    - φ_k = interference basis functions
    - α_k = learnable interference coefficients
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math


class NoOpInterference(nn.Module):
    """A no-op module that acts as an additive identity for ablation studies."""
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, token_fields: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Returns a zero tensor with the same shape and device, acting as an
        # additive identity (x + 0 = x).
        return torch.zeros_like(token_fields)


class TokenFieldInterference(nn.Module):
    """
    Token-centric field interference mechanism.
    
    Instead of continuous field interference, this operates directly on token
    field representations, maintaining O(N) complexity while adding physics-inspired
    interactions.
    
    Mathematical formulation:
        I(F_i, F_j) = Σ_k α_k φ_k(F_i, F_j)
        
    Where F_i, F_j are token field representations and φ_k are interference
    basis functions with learnable coefficients α_k.
    """
    
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int = 8,
                 interference_types: Tuple[str, ...] = ("constructive", "destructive", "phase"),
                 dropout: float = 0.1):
        """
        Initialize token field interference.
        
        Args:
            embed_dim: Dimension of token embeddings
            num_heads: Number of interference heads
            interference_types: Types of interference to compute
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.interference_types = interference_types
        self.head_dim = embed_dim // num_heads
        
        # Learnable interference coupling matrix
        self.field_coupler = nn.Parameter(torch.randn(num_heads, num_heads))
        nn.init.normal_(self.field_coupler, 0, 0.1)
        
        # Interference type weights
        self.interference_weights = nn.Parameter(torch.ones(len(interference_types)))
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                token_fields: torch.Tensor,  # [B, N, D] token field representations
                positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute token field interference.
        
        Args:
            token_fields: Token field representations [B, N, D]
            positions: Token positions [B, N, P] (optional, for position-aware interference)
            
        Returns:
            Interference-enhanced token fields [B, N, D]
        """
        batch_size, num_tokens, embed_dim = token_fields.shape
        
        # Reshape for multi-head interference
        # [B, N, D] -> [B, N, H, D//H]
        fields_reshaped = token_fields.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        
        # Compute interference types
        interference_terms = []
        
        if "constructive" in self.interference_types:
            constructive = self._constructive_interference(fields_reshaped)
            interference_terms.append(constructive)
            
        if "destructive" in self.interference_types:
            destructive = self._destructive_interference(fields_reshaped)
            interference_terms.append(destructive)
            
        if "phase" in self.interference_types:
            phase = self._phase_interference(fields_reshaped)
            interference_terms.append(phase)
        
        # Weighted combination of interference terms
        if len(interference_terms) == 0:
            # No valid interference type, return zeros of correct shape
            combined_interference = torch.zeros_like(fields_reshaped)
        else:
            interference_weights = F.softmax(self.interference_weights, dim=0)
            combined_interference = sum(w * term for w, term in zip(interference_weights, interference_terms))
        
        # Apply interference to original fields
        enhanced_fields = fields_reshaped + self.dropout(combined_interference)
        
        # Reshape back and project
        enhanced_fields = enhanced_fields.view(batch_size, num_tokens, embed_dim)
        output = self.output_proj(enhanced_fields)
        
        return output
    
    def _constructive_interference(self, fields: torch.Tensor) -> torch.Tensor:
        """Compute constructive interference efficiently (O(N)) using aggregated sums.

        Instead of explicitly forming all O(N^2) token pairs, we exploit the identity
            Σ_j 2 Re(F_i · F_j) = 2 Re(F_i · Σ_j F_j)
        and incorporate the learnable head–mixing matrix `field_coupler`.
        """
        B, N, H, D_h = fields.shape

        # Aggregate fields across all tokens once: Σ_j F_j
        summed_fields = fields.sum(dim=1)  # [B, H, D_h]

        # Mix heads via the coupler: Σ_k C_{h,k} Σ_j F_j^k
        mixed_sum = torch.einsum('bkd,hk->bhd', summed_fields, self.field_coupler)  # [B, H, D_h]

        # Dot-product between each token F_i^h and the mixed global vector
        interference = 2 * torch.sum(fields * mixed_sum.unsqueeze(1), dim=-1, keepdim=True)  # [B, N, H, 1]

        return interference
    
    def _destructive_interference(self, fields: torch.Tensor) -> torch.Tensor:
        """Compute destructive interference efficiently as the negative of the constructive term."""
        B, N, H, D_h = fields.shape

        summed_fields = fields.sum(dim=1)  # [B, H, D_h]
        mixed_sum = torch.einsum('bkd,hk->bhd', summed_fields, self.field_coupler)  # [B, H, D_h]

        interference = -2 * torch.sum(fields * mixed_sum.unsqueeze(1), dim=-1, keepdim=True)  # [B, N, H, 1]

        return interference
    
    def _phase_interference(self, fields: torch.Tensor) -> torch.Tensor:
        """Compute phase interference (cosine of phase difference) in linear time."""
        B, N, H, D_h = fields.shape

        # Normalise fields to unit magnitude per-token
        norm = torch.norm(fields, dim=-1, keepdim=True) + 1e-8
        unit_fields = fields / norm  # [B, N, H, D_h]

        summed_units = unit_fields.sum(dim=1)  # [B, H, D_h]
        mixed_sum = torch.einsum('bkd,hk->bhd', summed_units, self.field_coupler)  # [B, H, D_h]

        interference = torch.sum(unit_fields * mixed_sum.unsqueeze(1), dim=-1, keepdim=True)  # [B, N, H, 1]

        return interference


class CausalFieldInterference(TokenFieldInterference):
    """
    Causal field interference for time-series applications.
    
    Only allows backward-looking interference to maintain causality.
    """
    
    def forward(self, 
                token_fields: torch.Tensor,  # [B, N, D]
                positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute causal field interference.
        
        Args:
            token_fields: Token field representations [B, N, D]
            positions: Token positions [B, N, P] (optional)
            
        Returns:
            Causally-constrained interference-enhanced fields [B, N, D]
        """
        batch_size, num_tokens, embed_dim = token_fields.shape
        
        # Reshape for multi-head interference
        # [B, N, D] -> [B, N, H, D//H]
        fields_reshaped = token_fields.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        
        # --- EFFICIENT CAUSAL MASKING ---
        # OPTIMIZATION: Replaced iterative Python for-loop with vectorized torch.cumsum
        # 
        # BEFORE (inefficient):
        #   for i in range(num_tokens):
        #       if i > 0:
        #           causal_fields[:, i, :, :] = fields_reshaped[:, :i, :, :].mean(dim=1)
        # 
        # AFTER (efficient):
        #   - Uses torch.cumsum() for parallel computation on GPU
        #   - Single pass through the sequence instead of O(N²) operations
        #   - Leverages PyTorch's optimized tensor operations
        #   - Maintains exact same mathematical result
        #
        # Mathematical equivalence:
        #   - For token i, we need mean(fields[0:i])
        #   - cumsum(fields)[i] = sum(fields[0:i+1])
        #   - mean(fields[0:i]) = cumsum(fields)[i-1] / i
        #   - Shift by 1 position to get causal context (tokens 0 to i-1)
        #
        # 1. Compute the cumulative sum of fields along the sequence dimension.
        # cumsum() is a highly optimized parallel operation.
        cumulative_sum = torch.cumsum(fields_reshaped, dim=1)
        
        # 2. Generate a denominator for the running mean.
        # [1.0, 2.0, 3.0, ..., num_tokens]
        arange_divisor = torch.arange(1, num_tokens + 1, device=token_fields.device, dtype=token_fields.dtype)
        arange_divisor = arange_divisor.view(1, -1, 1, 1)
        
        # 3. Compute the running mean.
        # The mean up to token i is the cumulative sum at i divided by i+1.
        running_mean = cumulative_sum / arange_divisor
        
        # 4. Create the causal context by shifting the running mean.
        # The context for token i should only include tokens 0 to i-1.
        # We shift the running_mean tensor to the right by one position.
        causal_fields = torch.zeros_like(fields_reshaped)
        if num_tokens > 1:
            causal_fields[:, 1:, :, :] = running_mean[:, :-1, :, :]
        # --- END EFFICIENT CAUSAL MASKING ---
        
        # Compute interference types using causal fields
        interference_terms = []
        
        if "constructive" in self.interference_types:
            constructive = self._causal_constructive_interference(fields_reshaped, causal_fields)
            interference_terms.append(constructive)
            
        if "destructive" in self.interference_types:
            destructive = self._causal_destructive_interference(fields_reshaped, causal_fields)
            interference_terms.append(destructive)
            
        if "phase" in self.interference_types:
            phase = self._causal_phase_interference(fields_reshaped, causal_fields)
            interference_terms.append(phase)
        
        # Weighted combination of interference terms
        if len(interference_terms) == 0:
            # No valid interference type, return zeros of correct shape
            combined_interference = torch.zeros_like(fields_reshaped)
        else:
            interference_weights = F.softmax(self.interference_weights, dim=0)
            combined_interference = sum(w * term for w, term in zip(interference_weights, interference_terms))
        
        # Apply interference to original fields
        enhanced_fields = fields_reshaped + self.dropout(combined_interference)
        
        # Reshape back and project
        enhanced_fields = enhanced_fields.view(batch_size, num_tokens, embed_dim)
        output = self.output_proj(enhanced_fields)
        
        return output
    
    def _causal_constructive_interference(self, fields: torch.Tensor, causal_fields: torch.Tensor) -> torch.Tensor:
        """Compute causal constructive interference using only past information."""
        B, N, H, D_h = fields.shape
        
        # Use causal fields (past context) for interference
        mixed_causal = torch.einsum('bnhd,hk->bnkd', causal_fields, self.field_coupler)  # [B, N, H, D_h]
        
        # Compute interference between current token and its causal context
        interference = 2 * torch.sum(fields * mixed_causal, dim=-1, keepdim=True)  # [B, N, H, 1]
        
        return interference
    
    def _causal_destructive_interference(self, fields: torch.Tensor, causal_fields: torch.Tensor) -> torch.Tensor:
        """Compute causal destructive interference using only past information."""
        B, N, H, D_h = fields.shape
        
        # Use causal fields (past context) for interference  
        mixed_causal = torch.einsum('bnhd,hk->bnkd', causal_fields, self.field_coupler)  # [B, N, H, D_h]
        
        # Negative interference for destructive case
        interference = -2 * torch.sum(fields * mixed_causal, dim=-1, keepdim=True)  # [B, N, H, 1]
        
        return interference
    
    def _causal_phase_interference(self, fields: torch.Tensor, causal_fields: torch.Tensor) -> torch.Tensor:
        """Compute causal phase interference using only past information."""
        B, N, H, D_h = fields.shape
        
        # Normalise fields to unit magnitude per-token
        norm = torch.norm(fields, dim=-1, keepdim=True) + 1e-8
        unit_fields = fields / norm  # [B, N, H, D_h]
        
        # Normalise causal fields
        causal_norm = torch.norm(causal_fields, dim=-1, keepdim=True) + 1e-8
        unit_causal_fields = causal_fields / causal_norm  # [B, N, H, D_h]
        
        mixed_causal = torch.einsum('bnhd,hk->bnkd', unit_causal_fields, self.field_coupler)  # [B, N, H, D_h]
        
        interference = torch.sum(unit_fields * mixed_causal, dim=-1, keepdim=True)  # [B, N, H, 1]
        
        return interference


class MultiScaleFieldInterference(TokenFieldInterference):
    """
    Multi-scale field interference with fractal coupling.
    
    Implements interference at multiple scales to capture both local
    and global field interactions.
    """
    
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int = 8,
                 scales: int = 4,
                 interference_types: Tuple[str, ...] = ("constructive", "destructive", "phase"),
                 dropout: float = 0.1):
        """
        Initialize multi-scale field interference.
        
        Args:
            embed_dim: Dimension of token embeddings
            num_heads: Number of interference heads
            scales: Number of scales for multi-scale processing
            interference_types: Types of interference to compute
            dropout: Dropout rate for regularization
        """
        super().__init__(embed_dim, num_heads, interference_types, dropout)
        self.scales = scales
        
        # Multi-scale downsamplers
        self.downsamplers = nn.ModuleList([
            nn.AvgPool1d(2**i, stride=2**i) for i in range(scales)
        ])
        
        # Scale-specific interference weights
        self.scale_weights = nn.Parameter(torch.ones(scales))
        
    def forward(self, 
                token_fields: torch.Tensor,  # [B, N, D]
                positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute multi-scale field interference.
        
        Args:
            token_fields: Token field representations [B, N, D]
            positions: Token positions [B, N, P] (optional)
            
        Returns:
            Multi-scale interference-enhanced fields [B, N, D]
        """
        batch_size, num_tokens, embed_dim = token_fields.shape
        
        # Process at multiple scales
        scale_outputs = []
        
        for i, downsampler in enumerate(self.downsamplers):
            # Downsample fields
            fields_downsampled = token_fields.transpose(1, 2)  # [B, D, N]
            fields_downsampled = downsampler(fields_downsampled)  # [B, D, N//2^i]
            fields_downsampled = fields_downsampled.transpose(1, 2)  # [B, N//2^i, D]
            
            # Compute interference at this scale
            scale_interference = super().forward(fields_downsampled, positions)
            
            # Upsample back to original size
            scale_interference = scale_interference.transpose(1, 2)  # [B, D, N//2^i]
            scale_interference = F.interpolate(scale_interference, size=num_tokens, mode='linear')
            scale_interference = scale_interference.transpose(1, 2)  # [B, N, D]
            
            scale_outputs.append(scale_interference)
        
        # Weighted combination of scales
        scale_weights = F.softmax(self.scale_weights, dim=0)
        combined_output = sum(w * output for w, output in zip(scale_weights, scale_outputs))
        
        return combined_output



def create_field_interference(interference_type: str = "standard",
                            embed_dim: int = 256,
                            num_heads: int = 8,
                            **kwargs) -> nn.Module:
    """
    Factory function to create field interference modules.
    
    Args:
        interference_type: Type of interference ("standard", "causal", "multiscale", "none")
        embed_dim: Dimension of token embeddings
        num_heads: Number of interference heads
        **kwargs: Additional arguments for specific interference types
        
    Returns:
        Configured field interference module
    """
    if interference_type is None or interference_type.lower() == 'none':
        return NoOpInterference()

    if interference_type == "standard":
        return TokenFieldInterference(embed_dim, num_heads, **kwargs)
    elif interference_type == "causal":
        return CausalFieldInterference(embed_dim, num_heads, **kwargs)
    elif interference_type == "multiscale":
        return MultiScaleFieldInterference(embed_dim, num_heads, **kwargs)
    else:
        raise ValueError(f"Unknown interference type: {interference_type}") 