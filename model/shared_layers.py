"""tfn.model.shared_layers
Reusable building blocks shared across baseline classifiers & regressors.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

__all__ = [
    "PositionalEmbeddingStrategy",
    "LearnedPositionalEmbeddings",
    "TimeBasedEmbeddings",
    "SinusoidalEmbeddings",
    "create_positional_embedding_strategy",
    "LinearAttention",
]


class PositionalEmbeddingStrategy(ABC):
    """Abstract base class for positional embedding strategies."""
    
    @abstractmethod
    def __init__(self, max_len: int, embed_dim: int, **kwargs):
        pass
    
    @abstractmethod
    def forward(self, positions: torch.Tensor, **kwargs) -> torch.Tensor:
        pass


class LearnedPositionalEmbeddings(PositionalEmbeddingStrategy):
    """Standard learned absolute positional embeddings.

    Parameters
    ----------
    max_len : int
        Maximum sequence length for which to prepare embeddings.
    embed_dim : int
        Embedding dimensionality.
    """

    def __init__(self, max_len: int, embed_dim: int, **kwargs) -> None:
        super().__init__(max_len, embed_dim, **kwargs)
        self.pos = nn.Embedding(max_len, embed_dim)
        nn.init.normal_(self.pos.weight, mean=0.0, std=0.02)

    def forward(self, positions: torch.Tensor, **kwargs) -> torch.Tensor:  # [L, D]
        # Handle both discrete indices and continuous positions
        if positions.dtype in [torch.long, torch.int32, torch.int64]:
            # Discrete indices - use directly
            if positions.dim() == 2:
                # [B, N] -> use positions directly
                return self.pos(positions)
            else:
                # [N] -> expand to [1, N]
                return self.pos(positions.unsqueeze(0))
        else:
            # Continuous positions - convert to indices
            # This is a fallback for when continuous positions are passed to learned embeddings
            # In practice, use 'continuous' or 'sinusoidal' strategy for continuous positions
            seq_len = positions.shape[1] if positions.dim() > 1 else positions.shape[0]
            idx = torch.arange(seq_len, device=positions.device)
            
            # Ensure the embedding layer is on the same device as the input tensor
            if self.pos.weight.device != positions.device:
                self.pos = self.pos.to(positions.device)
            
            return self.pos(idx)

    def __call__(self, positions: torch.Tensor, **kwargs) -> torch.Tensor:
        """Make the object callable."""
        return self.forward(positions, **kwargs)


class TimeBasedEmbeddings(PositionalEmbeddingStrategy):
    """Time-based positional embeddings using calendar features.
    
    Parameters
    ----------
    max_len : int
        Maximum sequence length for which to prepare embeddings.
    embed_dim : int
        Embedding dimensionality.
    calendar_features : List[str]
        List of calendar features to use ("hour", "day_of_week", "day_of_month", "month", "is_weekend").
    feature_cardinalities : Dict[str, int]
        Cardinality of each calendar feature (e.g., hour: 24, day_of_week: 7).
    """
    
    def __init__(self, max_len: int, embed_dim: int, 
                 calendar_features: Optional[List[str]] = None,
                 feature_cardinalities: Optional[Dict[str, int]] = None,
                 **kwargs) -> None:
        super().__init__(max_len, embed_dim, **kwargs)
        
        self.calendar_features = calendar_features or [
            "hour", "day_of_week", "day_of_month", "month", "is_weekend"
        ]
        
        # Default cardinalities for calendar features
        default_cardinalities = {
            "hour": 24,
            "day_of_week": 7,
            "day_of_month": 31,
            "month": 12,
            "is_weekend": 2,
        }
        
        self.feature_cardinalities = feature_cardinalities or default_cardinalities
        
        # Validate that all requested features have cardinalities
        for feature in self.calendar_features:
            if feature not in self.feature_cardinalities:
                raise ValueError(f"Missing cardinality for calendar feature: {feature}")
        
        # Create embeddings for each calendar feature
        self.feature_embeddings = nn.ModuleDict()
        embed_dim_per_feature = embed_dim // len(self.calendar_features)
        
        for feature in self.calendar_features:
            cardinality = self.feature_cardinalities[feature]
            self.feature_embeddings[feature] = nn.Embedding(cardinality, embed_dim_per_feature)
            nn.init.normal_(self.feature_embeddings[feature].weight, mean=0.0, std=0.02)
        
        # If embed_dim doesn't divide evenly, add a small projection
        if embed_dim % len(self.calendar_features) != 0:
            total_embed_dim = embed_dim_per_feature * len(self.calendar_features)
            self.projection = nn.Linear(total_embed_dim, embed_dim)
        else:
            self.projection = None
    
    def forward(self, positions: torch.Tensor, calendar_features: Optional[Dict[str, torch.Tensor]] = None, **kwargs) -> torch.Tensor:
        """
        Forward pass for time-based embeddings.
        
        Args:
            positions: Token positions [B, N, P] or [N, P]
            calendar_features: Dictionary of calendar features [B, N] or [N]
            
        Returns:
            Time-based embeddings [B, N, D] or [N, D]
        """
        if calendar_features is None:
            # Fallback to learned positional embeddings if no calendar features provided
            seq_len = positions.shape[1] if positions.dim() > 1 else positions.shape[0]
            return self._fallback_embeddings(seq_len, device=positions.device)
        
        # Ensure positions has batch dimension
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)  # [N, P] -> [1, N, P]
        
        batch_size, seq_len, _ = positions.shape
        
        # Initialize embeddings
        embeddings = []
        
        for feature in self.calendar_features:
            if feature in calendar_features:
                feature_values = calendar_features[feature]
                
                # Ensure feature_values has batch dimension
                if feature_values.dim() == 1:
                    feature_values = feature_values.unsqueeze(0)  # [N] -> [1, N]
                
                # Ensure feature_values has correct shape
                if feature_values.shape[1] != seq_len:
                    # Pad or truncate to match sequence length
                    if feature_values.shape[1] < seq_len:
                        # Pad with last value
                        padding = feature_values[:, -1:].repeat(1, seq_len - feature_values.shape[1])
                        feature_values = torch.cat([feature_values, padding], dim=1)
                    else:
                        # Truncate
                        feature_values = feature_values[:, :seq_len]
                
                # Clamp values to valid range for this feature
                cardinality = self.feature_cardinalities[feature]
                feature_values = torch.clamp(feature_values, 0, cardinality - 1)
                
                # Ensure the embedding layer is on the same device as the input
                if self.feature_embeddings[feature].weight.device != positions.device:
                    self.feature_embeddings[feature] = self.feature_embeddings[feature].to(positions.device)
                
                # Get embeddings for this feature
                feature_emb = self.feature_embeddings[feature](feature_values)  # [B, N, embed_dim_per_feature]
                embeddings.append(feature_emb)
            else:
                # Use zero embeddings for missing features
                embed_dim_per_feature = self.feature_embeddings[self.calendar_features[0]].embedding_dim
                zero_emb = torch.zeros(batch_size, seq_len, embed_dim_per_feature, 
                                     device=positions.device, dtype=positions.dtype)
                embeddings.append(zero_emb)
        
        # Concatenate all feature embeddings
        combined_embeddings = torch.cat(embeddings, dim=-1)  # [B, N, total_embed_dim]
        
        # Apply projection if needed
        if self.projection is not None:
            # Ensure the projection layer is on the same device as the input
            if self.projection.weight.device != positions.device:
                self.projection = self.projection.to(positions.device)
            combined_embeddings = self.projection(combined_embeddings)
        
        # Remove batch dimension if input didn't have one
        if positions.shape[0] == 1 and positions.dim() == 3:
            combined_embeddings = combined_embeddings.squeeze(0)  # [1, N, D] -> [N, D]
        
        return combined_embeddings
    
    def _fallback_embeddings(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Fallback to learned positional embeddings."""
        # Create a simple learned embedding as fallback
        fallback_embed = nn.Embedding(seq_len, self.feature_embeddings[self.calendar_features[0]].embedding_dim * len(self.calendar_features))
        fallback_embed = fallback_embed.to(device)
        idx = torch.arange(seq_len, device=device)
        return fallback_embed(idx)

    def __call__(self, positions: torch.Tensor, **kwargs) -> torch.Tensor:
        """Make the object callable."""
        return self.forward(positions, **kwargs)


class SinusoidalEmbeddings(PositionalEmbeddingStrategy):
    """Sinusoidal positional embeddings (Transformer-style).
    
    Parameters
    ----------
    max_len : int
        Maximum sequence length for which to prepare embeddings.
    embed_dim : int
        Embedding dimensionality.
    """
    
    def __init__(self, max_len: int, embed_dim: int, **kwargs) -> None:
        super().__init__(max_len, embed_dim, **kwargs)
        self.max_len = max_len
        self.embed_dim = embed_dim
        
        # Create sinusoidal embeddings
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / embed_dim))
        
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Store as a parameter instead of buffer for compatibility
        self.pe = nn.Parameter(pe, requires_grad=False)
    
    def forward(self, positions: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass for sinusoidal embeddings."""
        seq_len = positions.shape[1] if positions.dim() > 1 else positions.shape[0]
        return self.pe[:seq_len].to(positions.device)

    def __call__(self, positions: torch.Tensor, **kwargs) -> torch.Tensor:
        """Make the object callable."""
        return self.forward(positions, **kwargs)


class ContinuousPositionalEmbeddings(PositionalEmbeddingStrategy):
    """Continuous positional embeddings for spatial coordinates (PDE datasets).
    
    This strategy handles continuous position values by using sinusoidal
    embeddings with the actual position values as input.
    
    Parameters
    ----------
    max_len : int
        Maximum sequence length for which to prepare embeddings.
    embed_dim : int
        Embedding dimensionality.
    """
    
    def __init__(self, max_len: int, embed_dim: int, **kwargs) -> None:
        super().__init__(max_len, embed_dim, **kwargs)
        self.max_len = max_len
        self.embed_dim = embed_dim
    
    def forward(self, positions: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass for continuous positional embeddings.
        
        Args:
            positions: Continuous position values [B, N, P] or [N, P] where P is spatial dimension
            
        Returns:
            Continuous positional embeddings [B, N, D] or [N, D]
        """
        # Ensure positions has the right shape
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)  # [N, P] -> [1, N, P]
        
        batch_size, seq_len, pos_dim = positions.shape
        
        # Create sinusoidal embeddings for each spatial dimension
        embeddings = []
        
        for dim in range(pos_dim):
            # Extract positions for this dimension
            pos_vals = positions[:, :, dim]  # [B, N]
            
            # Create sinusoidal embeddings for this dimension
            # Use different frequencies for different embedding dimensions
            embed_dim_per_pos = self.embed_dim // pos_dim
            if dim == pos_dim - 1:  # Last dimension gets remaining dimensions
                embed_dim_per_pos = self.embed_dim - (pos_dim - 1) * (self.embed_dim // pos_dim)
            
            if embed_dim_per_pos > 0:
                # Create sinusoidal embeddings
                div_term = torch.exp(torch.arange(0, embed_dim_per_pos, 2).float() * 
                                   -(torch.log(torch.tensor(10000.0)) / embed_dim_per_pos))
                div_term = div_term.to(positions.device)
                
                # Expand positions for broadcasting
                pos_expanded = pos_vals.unsqueeze(-1)  # [B, N, 1]
                
                # Create sinusoidal embeddings
                pe = torch.zeros(batch_size, seq_len, embed_dim_per_pos, device=positions.device)
                pe[:, :, 0::2] = torch.sin(pos_expanded * div_term)
                pe[:, :, 1::2] = torch.cos(pos_expanded * div_term)
                
                embeddings.append(pe)
        
        # Concatenate embeddings from all spatial dimensions
        if embeddings:
            combined = torch.cat(embeddings, dim=-1)
            # Ensure we have the right embedding dimension
            if combined.shape[-1] < self.embed_dim:
                # Pad with zeros if needed
                padding = torch.zeros(batch_size, seq_len, self.embed_dim - combined.shape[-1], 
                                    device=positions.device)
                combined = torch.cat([combined, padding], dim=-1)
            elif combined.shape[-1] > self.embed_dim:
                # Truncate if needed
                combined = combined[:, :, :self.embed_dim]
        else:
            # Fallback: create zero embeddings
            combined = torch.zeros(batch_size, seq_len, self.embed_dim, device=positions.device)
        
        return combined

    def __call__(self, positions: torch.Tensor, **kwargs) -> torch.Tensor:
        """Make the object callable."""
        return self.forward(positions, **kwargs)


def create_positional_embedding_strategy(strategy_name: str, max_len: int, embed_dim: int, **kwargs) -> PositionalEmbeddingStrategy:
    """Factory function to create positional embedding strategies."""
    if strategy_name == "learned":
        return LearnedPositionalEmbeddings(max_len, embed_dim, **kwargs)
    elif strategy_name == "time_based":
        return TimeBasedEmbeddings(max_len, embed_dim, **kwargs)
    elif strategy_name == "sinusoidal":
        return SinusoidalEmbeddings(max_len, embed_dim, **kwargs)
    elif strategy_name == "continuous":
        return ContinuousPositionalEmbeddings(max_len, embed_dim, **kwargs)
    else:
        raise ValueError(f"Unknown positional embedding strategy: {strategy_name}")


class LinearAttention(nn.Module):
    """Single-head linear (FAVOR) attention approximation.

    A light-weight replacement for softmax attention that scales **O(L·D)**
    instead of **O(L²)**.
    """

    def __init__(self, embed_dim: int, proj_dim: int = 64) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim

        # Random projection matrices
        self.q_proj = nn.Parameter(torch.randn(embed_dim, proj_dim) / proj_dim**0.5)
        self.k_proj = nn.Parameter(torch.randn(embed_dim, proj_dim) / proj_dim**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, L, D]
        B, L, D = x.shape

        # Project to random features
        q_prime = torch.einsum("bld,de->ble", x, self.q_proj)  # [B, L, proj_dim]
        k_prime = torch.einsum("bld,de->ble", x, self.k_proj)  # [B, L, proj_dim]

        # Apply activation function
        q_prime = torch.exp(q_prime)
        k_prime = torch.exp(k_prime)

        # Compute key-value product
        kv = torch.einsum("ble,bld->bed", k_prime, x)  # [B, proj_dim, D]

        # Attention numerator: [B, L, D]
        num = torch.einsum("blp,bpd->bld", q_prime, kv)
        # Normaliser: [B, L, 1]
        z = 1 / (q_prime.sum(dim=-1, keepdim=True) + 1e-8)
        return num * z  # element-wise broadcast 