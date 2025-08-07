"""
Enhanced TFN Model with Field Interference Mechanisms.

This module integrates the novel field interference mechanisms with the existing
TFN architecture, providing a complete implementation of the token-centric
field interference approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import math

from core.field_projection import FieldProjector
from core.field_evolution import FieldEvolver, DynamicFieldPropagator, create_field_evolver
from core.field_sampling import FieldSampler
from core.field_interference import TokenFieldInterference, create_field_interference
from core.unified_field_dynamics import UnifiedFieldDynamics
from model.shared_layers import create_positional_embedding_strategy  # NEW


class EnhancedTFNLayer(nn.Module):
    """
    Enhanced TFN Layer using Unified Field Dynamics.
    Integrates field projection, unified field dynamics (evolution + interference + constraints), and field sampling.
    """
    def __init__(
                 self,
                 embed_dim: int,
                 pos_dim: int = 1,
                 kernel_type: str = "rbf",
                 evolution_type: str = "diffusion",
                 interference_type: str = "standard",
                 grid_size: int = 100,
                 num_steps: int = 4,
                 dropout: float = 0.1,
                 *,
                 # Modular positional embeddings
                 positional_embedding_strategy: str = "learned",
                 calendar_features: Optional[list[str]] = None,
                 feature_cardinalities: Optional[dict[str, int]] = None,
                 max_seq_len: int = 512,
                 # Misc
                 layer_norm_eps: float = 1e-5,
                 **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_dim = pos_dim
        self.grid_size = grid_size
        # Field projection
        self.field_projector = FieldProjector(
            embed_dim=embed_dim,
            pos_dim=pos_dim,
            kernel_type=kernel_type
        )
        # Unified field dynamics
        self.unified_dynamics = UnifiedFieldDynamics(
            embed_dim=embed_dim,
            pos_dim=pos_dim,
            evolution_type=evolution_type,
            interference_type=interference_type,
            num_steps=num_steps,
            dropout=dropout
        )
        # Field sampling
        self.field_sampler = FieldSampler(mode='linear')
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Position embeddings (now modular via shared factory)
        self.pos_embeddings = create_positional_embedding_strategy(
            positional_embedding_strategy,
            max_len=max_seq_len,
            embed_dim=embed_dim,
            calendar_features=calendar_features,
            feature_cardinalities=feature_cardinalities,
        )
    def forward(self, 
                embeddings: torch.Tensor,      # [B, N, D] token embeddings
                positions: torch.Tensor,       # [B, N, P] token positions
                grid_points: Optional[torch.Tensor] = None,
                add_pos_emb: bool = True) -> torch.Tensor:
        batch_size, num_tokens, embed_dim = embeddings.shape
        if grid_points is None:
            grid_points = self._generate_grid_points(batch_size)
        # Optionally add positional embeddings before projection
        if add_pos_emb and self.pos_embeddings is not None:
            embeddings = embeddings + self.pos_embeddings(positions, calendar_features=None)

        # Step 1: Field Projection
        field = self.field_projector(embeddings, positions, grid_points)  # [B, M, D]
        # Step 2: Unified Field Dynamics (evolution + interference + constraints)
        field_evolved = self.unified_dynamics(field, positions)
        # Step 3: Field Sampling
        enhanced_embeddings = self.field_sampler(field_evolved, grid_points, positions)  # [B, N, D]
        # Residual connection and layer normalization
        enhanced_embeddings = self.layer_norm1(enhanced_embeddings + embeddings)
        # Output projection
        output = self.output_proj(enhanced_embeddings)
        output = self.dropout(output)
        # Final layer normalization
        output = self.layer_norm2(output + enhanced_embeddings)
        return output
    def _generate_grid_points(self, batch_size: int) -> torch.Tensor:
        if self.pos_dim == 1:
            grid = torch.linspace(0.0, 1.0, self.grid_size, device=next(self.parameters()).device)
            grid = grid.unsqueeze(-1)
        else:
            grid_points_per_dim = int(self.grid_size ** (1.0 / self.pos_dim))
            grid_ranges = [torch.linspace(0.0, 1.0, grid_points_per_dim) for _ in range(self.pos_dim)]
            grid = torch.meshgrid(*grid_ranges, indexing='ij')
            grid = torch.stack(grid, dim=-1).reshape(-1, self.pos_dim)
        grid = grid.unsqueeze(0).expand(batch_size, -1, -1)
        return grid
    def get_physics_constraints(self) -> Dict[str, torch.Tensor]:
        return self.unified_dynamics.get_physics_constraints()


class EnhancedTFNModel(nn.Module):
    """
    Complete Enhanced TFN Model with Field Interference.
    
    A full TFN model that integrates all the field interference mechanisms
    for end-to-end training and inference.
    """
    
    def __init__(self, 
                 vocab_size: int,
                 embed_dim: int,
                 num_layers: int,
                 pos_dim: int = 1,
                 kernel_type: str = "rbf",
                 evolution_type: str = "diffusion",
                 interference_type: str = "standard",
                 grid_size: int = 100,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 *,
                 max_seq_len: int = 512,
                 # ADD THESE NEW ARGUMENTS
                 positional_embedding_strategy: str = "learned",
                 calendar_features: Optional[List[str]] = None,
                 feature_cardinalities: Optional[Dict[str, int]] = None):
        """
        Initialize enhanced TFN model.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Dimension of embeddings
            num_layers: Number of TFN layers
            pos_dim: Dimension of position space
            kernel_type: Type of kernel for field projection
            evolution_type: Type of evolution for field evolution
            interference_type: Type of interference
            grid_size: Number of grid points
            num_heads: Number of attention heads
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            positional_embedding_strategy: Strategy for positional embeddings ("learned", "sinusoidal", "calendar", etc.)
            calendar_features: List of calendar features for calendar-based positional embeddings
            feature_cardinalities: Dictionary mapping feature names to their cardinalities for categorical embeddings
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Position embedding
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Enhanced TFN layers
        self.layers = nn.ModuleList([
            EnhancedTFNLayer(
                embed_dim=embed_dim,
                pos_dim=pos_dim,
                kernel_type=kernel_type,
                evolution_type=evolution_type,
                interference_type=interference_type,
                grid_size=grid_size,
                num_steps=4,  # Default for UnifiedFieldDynamics
                dropout=dropout,
                # PASS THE NEW ARGUMENTS DOWN
                positional_embedding_strategy=positional_embedding_strategy,
                calendar_features=calendar_features,
                feature_cardinalities=feature_cardinalities,
                max_seq_len=max_seq_len
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        
        # Layer normalization
        self.final_norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        # Token embedding initialization
        nn.init.normal_(self.token_embedding.weight, 0, 0.02)
        
        # Position embedding initialization
        nn.init.normal_(self.pos_embedding.weight, 0, 0.02)
        
        # Output projection initialization
        nn.init.normal_(self.output_proj.weight, 0, 0.02)
        nn.init.zeros_(self.output_proj.bias)
        
    def forward(self, 
                input_ids: torch.Tensor,  # [B, N] token indices
                positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through enhanced TFN model.
        
        Args:
            input_ids: Input token indices [B, N]
            positions: Token positions [B, N, P] (optional)
            
        Returns:
            Logits for next token prediction [B, N, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        embeddings = self.token_embedding(input_ids)  # [B, N, D]
        
        # Generate position coordinates for TFN layers
        if positions is None:
            # Use normalized sequential positions
            positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.float32)
            positions = positions / (seq_len - 1)  # Normalize to [0, 1]
            positions = positions.unsqueeze(0).expand(batch_size, -1)  # [B, N]
            positions = positions.unsqueeze(-1)  # [B, N, 1]
        
        # Position embeddings for residual connection
        pos_indices = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_embeddings = self.pos_embedding(pos_indices)  # [1, N, D]
        pos_embeddings = pos_embeddings.expand(batch_size, -1, -1)  # [B, N, D]
        
        # Combine token and position embeddings
        x = embeddings + pos_embeddings
        
        # Pass through enhanced TFN layers
        for layer in self.layers:
            x = layer(x, positions)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Output projection
        logits = self.output_proj(x)  # [B, N, vocab_size]
        
        return logits
    
    def get_physics_constraints(self) -> Dict[str, torch.Tensor]:
        """
        Get physics constraint losses from all layers.
        
        Returns:
            Dictionary of constraint losses
        """
        constraints = {}
        
        for i, layer in enumerate(self.layers):
            layer_constraints = layer.get_physics_constraints()
            for key, value in layer_constraints.items():
                constraints[f"layer_{i}_{key}"] = value
        
        return constraints


def create_enhanced_tfn_model(vocab_size: int,
                             embed_dim: int,
                             num_layers: int,
                             interference_type: str = "standard",
                             **kwargs) -> EnhancedTFNModel:
    """
    Factory function to create enhanced TFN models.
    
    Args:
        vocab_size: Size of vocabulary
        embed_dim: Dimension of embeddings
        num_layers: Number of TFN layers
        interference_type: Type of interference mechanism
        **kwargs: Additional arguments
        
    Returns:
        Configured enhanced TFN model
    """
    return EnhancedTFNModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        interference_type=interference_type,
        **kwargs
    )


class EnhancedTFNRegressor(nn.Module):
    """
    Enhanced TFN Model for Regression Tasks.
    
    A regression-compatible version of the enhanced TFN that accepts
    continuous input features instead of discrete tokens.
    """
    
    def __init__(self, 
                 input_dim: int,
                 embed_dim: int,
                 output_dim: int,
                 output_len: int,
                 num_layers: int,
                 pos_dim: int = 1,
                 kernel_type: str = "rbf",
                 evolution_type: str = "diffusion",
                 interference_type: str = "standard",
                 grid_size: int = 100,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 *,
                 max_seq_len: int = 512,
                 # ADD THESE NEW ARGUMENTS
                 positional_embedding_strategy: str = "learned",
                 calendar_features: Optional[List[str]] = None,
                 feature_cardinalities: Optional[Dict[str, int]] = None):
        """
        Initialize enhanced TFN regressor.
        
        Args:
            input_dim: Input feature dimension
            embed_dim: Dimension of embeddings
            output_dim: Output feature dimension
            output_len: Output sequence length
            num_layers: Number of TFN layers
            pos_dim: Dimension of position space
            kernel_type: Type of kernel for field projection
            evolution_type: Type of evolution for field evolution
            interference_type: Type of interference
            grid_size: Number of grid points
            num_heads: Number of attention heads
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            positional_embedding_strategy: Strategy for positional embeddings ("learned", "sinusoidal", "calendar", etc.)
            calendar_features: List of calendar features for calendar-based positional embeddings
            feature_cardinalities: Dictionary mapping feature names to their cardinalities for categorical embeddings
        """
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.output_len = output_len
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # Input projection (for continuous features)
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # Position embedding
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Enhanced TFN layers
        self.layers = nn.ModuleList([
            EnhancedTFNLayer(
                embed_dim=embed_dim,
                pos_dim=pos_dim,
                kernel_type=kernel_type,
                evolution_type=evolution_type,
                interference_type=interference_type,
                grid_size=grid_size,
                num_steps=4,  # Default for UnifiedFieldDynamics
                dropout=dropout,
                # PASS THE NEW ARGUMENTS DOWN
                positional_embedding_strategy=positional_embedding_strategy,
                calendar_features=calendar_features,
                feature_cardinalities=feature_cardinalities,
                max_seq_len=max_seq_len
            )
            for _ in range(num_layers)
        ])
        
        # Output projection for regression
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, output_dim)
        )
        
        # Layer normalization
        self.final_norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        # Input projection initialization
        nn.init.normal_(self.input_proj.weight, 0, 0.02)
        nn.init.zeros_(self.input_proj.bias)
        
        # Position embedding initialization
        nn.init.normal_(self.pos_embedding.weight, 0, 0.02)
        
        # Output projection initialization
        for layer in self.output_proj:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.02)
                nn.init.zeros_(layer.bias)
        
    def forward(self, 
                inputs: torch.Tensor,  # [B, N, input_dim] continuous features
                positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through enhanced TFN regressor.
        
        Args:
            inputs: Input continuous features [B, N, input_dim]
            positions: Token positions [B, N, P] (optional)
            
        Returns:
            Regression outputs [B, output_len, output_dim]
        """
        batch_size, seq_len, input_dim = inputs.shape
        
        # Input projection
        embeddings = self.input_proj(inputs)  # [B, N, embed_dim]
        
        # Generate position coordinates for TFN layers
        if positions is None:
            # Use normalized sequential positions
            positions = torch.arange(seq_len, device=inputs.device, dtype=torch.float32)
            positions = positions / (seq_len - 1)  # Normalize to [0, 1]
            positions = positions.unsqueeze(0).expand(batch_size, -1)  # [B, N]
            positions = positions.unsqueeze(-1)  # [B, N, 1]
        
        # Position embeddings for residual connection
        pos_indices = torch.arange(seq_len, device=inputs.device).unsqueeze(0)
        pos_embeddings = self.pos_embedding(pos_indices)  # [1, N, embed_dim]
        pos_embeddings = pos_embeddings.expand(batch_size, -1, -1)  # [B, N, embed_dim]
        
        # Combine input and position embeddings
        x = embeddings + pos_embeddings
        
        # Pass through enhanced TFN layers
        for layer in self.layers:
            x = layer(x, positions)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Output projection for regression
        # Take the last output_len tokens for prediction
        if seq_len >= self.output_len:
            x = x[:, -self.output_len:, :]  # [B, output_len, embed_dim]
        else:
            # Pad if sequence is shorter than output_len
            padding = torch.zeros(batch_size, self.output_len - seq_len, self.embed_dim, 
                                device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=1)  # [B, output_len, embed_dim]
        
        outputs = self.output_proj(x)  # [B, output_len, output_dim]
        
        return outputs
    
    def get_physics_constraints(self) -> Dict[str, torch.Tensor]:
        """
        Get physics constraint losses from all layers.
        
        Returns:
            Dictionary of constraint losses
        """
        constraints = {}
        
        for i, layer in enumerate(self.layers):
            layer_constraints = layer.get_physics_constraints()
            for key, value in layer_constraints.items():
                constraints[f"layer_{i}_{key}"] = value
        
        return constraints 