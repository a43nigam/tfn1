"""
Unified TFN model for both classification and regression tasks.

This module provides a single, configurable TFN model that can handle
both classification and regression tasks with appropriate input/output heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Literal, Dict, Any, List

from .tfn_enhanced import EnhancedTFNLayer
from model.shared_layers import create_positional_embedding_strategy
from .base_model import BaseSequenceModel

# Type aliases for better readability
TaskType = Literal["classification", "regression"]


class TFN(BaseSequenceModel):
    """A unified TFN model configurable for classification or regression.

    Arguments
    ---------
    task:
        Either "classification" or "regression". Determines the input & head.
    vocab_size:
        Size of token vocabulary **(classification only)**. If *None*, the
        model expects pre-embedded inputs of shape ``[B, N, embed_dim]``.
    input_dim:
        Feature dimension for numeric inputs **(regression only)**. Ignored
        for classification.
    num_classes:
        Output classes (classification).
    output_dim:
        Output feature dimension (regression).
    embed_dim:
        Embedding dimension of TFN layers.
    num_layers:
        Number of stacked TFN layers.
    kernel_type:
        Kernel choice for field projection ("rbf", "compact", "fourier").
    evolution_type:
        Evolution operator ("cnn", "pde", etc.).
    interference_type:
        Field interference type ("standard", "causal", "multi_scale").
    grid_size:
        Spatial grid resolution.
    time_steps:
        Evolution steps per layer.
    dropout:
        Dropout probability used in TFN layers & heads.
    positional_embedding_strategy:
        Positional embedding strategy ("learned", "time_based", "sinusoidal").
    calendar_features:
        List of calendar features for time-based embeddings.
    feature_cardinalities:
        Cardinality of each calendar feature.
    """

    def __init__(
        self,
        task: TaskType,
        *,
        # Classification-specific
        vocab_size: Optional[int] = None,
        num_classes: int = 2,
        # Regression-specific
        input_dim: Optional[int] = None,
        output_dim: int = 1,
        output_len: int = 1,
        # Shared hyper-params
        embed_dim: int = 128,
        num_layers: int = 2,
        kernel_type: str = "rbf",
        evolution_type: str = "cnn",
        interference_type: str = "standard",
        grid_size: int = 100,
        time_steps: int = 3,
        dropout: float = 0.1,
        # Positional encoding range (if positions None)
        pos_min: float = 0.1,
        pos_max: float = 0.9,
        # New parameters for modular embeddings
        positional_embedding_strategy: Optional[str] = None,
        calendar_features: Optional[List[str]] = None,
        feature_cardinalities: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()

        if task not in ("classification", "regression"):
            raise ValueError(f"Unknown task type: {task}")
        self.task = task

        # Set default positional embedding strategy based on task
        if positional_embedding_strategy is None:
            if task == "regression":
                positional_embedding_strategy = "continuous"  # Better for PDE datasets
            else:
                positional_embedding_strategy = "learned"  # Better for classification

        # Validate positional range --------------------------------------
        if not (0.0 <= pos_min < pos_max <= 1.0):
            raise ValueError(f"pos_min and pos_max must satisfy 0<=min<max<=1, got {pos_min}, {pos_max}")

        self.embed_dim = embed_dim
        self.pos_min = pos_min
        self.pos_max = pos_max

        # ------------------------------------------------------------------
        # Input Embedding / Projection
        # ------------------------------------------------------------------
        if task == "classification":
            if vocab_size is None:
                raise ValueError("vocab_size must be provided for classification task")
            self.input_embed = nn.Embedding(vocab_size, embed_dim)
        else:  # regression
            if input_dim is None:
                raise ValueError("input_dim must be provided for regression task")
            self.input_proj = nn.Linear(input_dim, embed_dim)

        # ------------------------------------------------------------------
        # TFN Layers using modern core components
        # ------------------------------------------------------------------
        self.tfn_layers = nn.ModuleList()
        
        for _ in range(num_layers):
            # Use EnhancedTFNLayer which uses modern core components
            layer = EnhancedTFNLayer(
                embed_dim=embed_dim,
                pos_dim=1,  # 1D for time series
                kernel_type=kernel_type,
                evolution_type=evolution_type,
                interference_type=interference_type,
                grid_size=grid_size,
                num_steps=time_steps,
                dropout=dropout,
                positional_embedding_strategy=positional_embedding_strategy,
                calendar_features=calendar_features,
                feature_cardinalities=feature_cardinalities,
                max_seq_len=output_len,
            )
            self.tfn_layers.append(layer)

        # ------------------------------------------------------------------
        # Task Heads
        # ------------------------------------------------------------------
        self.output_len = output_len
        self.output_dim = output_dim
        if task == "classification":
            self.head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, num_classes),
            )
        else:  # regression
            self.head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, output_dim),
            )

        self._init_weights()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, inputs: torch.Tensor, positions: Optional[torch.Tensor] = None, 
                calendar_features: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:  # type: ignore
        """Forward pass.

        Parameters
        ----------
        inputs:
            • Classification – ``[B, N]`` token ids (LongTensor).
            • Regression      – ``[B, N, input_dim]`` numeric features.
        positions:
            Optional positions ``[B, N, 1]``; auto-generated if *None*.
        calendar_features:
            Optional calendar features for time-based embeddings.
        """
        B, N = inputs.shape[:2]

        # ------------------------------------------------------------------
        # Input Embedding / Projection
        # ------------------------------------------------------------------
        if self.task == "classification":
            # Token embedding
            embeddings = self.input_embed(inputs)  # [B, N, embed_dim]
        else:  # regression
            # Linear projection
            embeddings = self.input_proj(inputs)  # [B, N, embed_dim]

        # ------------------------------------------------------------------
        # Position Generation
        # ------------------------------------------------------------------
        if positions is None:
            # Generate normalized sequential positions
            positions = torch.linspace(self.pos_min, self.pos_max, N, device=inputs.device)
            positions = positions.unsqueeze(0).expand(B, -1)  # [B, N]
            positions = positions.unsqueeze(-1)  # [B, N, 1]

        # ------------------------------------------------------------------
        # TFN Layers
        # ------------------------------------------------------------------
        x = embeddings
        for layer in self.tfn_layers:
            # EnhancedTFNLayer uses modern core components
            x = layer(x, positions)

        # ------------------------------------------------------------------
        # Task-Specific Head
        # ------------------------------------------------------------------
        if self.task == "classification":
            # Use first token for classification
            pooled = x[:, 0, :]  # [B, embed_dim]
            out = self.head(pooled)  # [B, num_classes]
        else:  # regression
            # For regression, use the last few tokens for multi-step forecasting
            # Take the last output_len tokens, or all if sequence is shorter
            seq_len = x.size(1)
            if seq_len >= self.output_len:
                # Use last output_len tokens
                pooled = x[:, -self.output_len:, :]  # [B, output_len, embed_dim]
            else:
                # Pad with the last token if sequence is too short
                last_token = x[:, -1:, :]  # [B, 1, embed_dim]
                padding = last_token.repeat(1, self.output_len - seq_len, 1)  # [B, output_len - seq_len, embed_dim]
                pooled = torch.cat([x, padding], dim=1)  # [B, output_len, embed_dim]
            
            # Apply head to each token
            batch_size, output_len, embed_dim = pooled.shape
            pooled_flat = pooled.reshape(-1, embed_dim)  # [B * output_len, embed_dim]
            out_flat = self.head(pooled_flat)  # [B * output_len, output_dim]
            out = out_flat.reshape(batch_size, output_len, self.output_dim)  # [B, output_len, output_dim]

        return out

    def _init_weights(self) -> None:
        """Initialize model weights."""
        # Input embedding/projection initialization
        if self.task == "classification":
            nn.init.normal_(self.input_embed.weight, 0, 0.02)
        else:
            nn.init.normal_(self.input_proj.weight, 0, 0.02)
            nn.init.zeros_(self.input_proj.bias)

        # Head initialization
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


# Backward compatibility alias
UnifiedTFN = TFN 