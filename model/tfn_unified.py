"""
Unified Token Field Network (TFN)
================================
A single, well-parameterized implementation that can be configured for
sequence **classification** or **regression** tasks while re-using the
core TFN layer stack (1-D). 2-D / image TFN remains in `tfn_pytorch.py`.

Usage Example (CLI / script)::

    model = UnifiedTFN(
        task="classification",          # or "regression"
        vocab_size=30522,               # needed for text classification
        num_classes=4,                  # for classification
        embed_dim=256,
        num_layers=4,
        kernel_type="rbf",
        evolution_type="cnn",
    )

Key Design Points
-----------------
• Shared token embedding / input projection depending on `task` type.
• Same stack of `TrainableTFNLayer` (from `tfn.model.tfn_base`).
• Final head switched via `task` (classification vs regression).
• Keeps full CLI configurability: all constructor kwargs are exposed and
  mapped from argparse flags in training scripts.
• Keeps tensor-shape comments and docstrings for clarity.
"""

from __future__ import annotations

from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tfn_base import TrainableTFNLayer
from .tfn_enhanced import EnhancedTFNLayer  # Optional future use

TaskType = Literal["classification", "regression"]


class TFN(nn.Module):
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
    grid_size:
        Spatial grid resolution.
    time_steps:
        Evolution steps per layer.
    dropout:
        Dropout probability used in TFN layers & heads.
    use_enhanced:
        If *True*, uses :class:`EnhancedTFNLayer` instead of
        :class:`TrainableTFNLayer`.
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
        grid_size: int = 100,
        time_steps: int = 3,
        dropout: float = 0.1,
        # Positional encoding range (if positions None)
        pos_min: float = 0.1,
        pos_max: float = 0.9,
        use_enhanced: bool = False,
    ) -> None:
        super().__init__()

        if task not in ("classification", "regression"):
            raise ValueError(f"Unknown task type: {task}")
        self.task = task

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
        # Core TFN Layers (shared)
        # ------------------------------------------------------------------
        layer_cls = EnhancedTFNLayer if use_enhanced else TrainableTFNLayer
        self.tfn_layers = nn.ModuleList(
            [
                layer_cls(
                    embed_dim=embed_dim,
                    kernel_type=kernel_type,
                    evolution_type=evolution_type,
                    grid_size=grid_size,
                    time_steps=time_steps,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

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
                nn.Linear(embed_dim // 2, output_len * output_dim),
            )

        self._init_weights()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, inputs: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:  # type: ignore
        """Forward pass.

        Parameters
        ----------
        inputs:
            • Classification – ``[B, N]`` token ids (LongTensor).
            • Regression      – ``[B, N, input_dim]`` numeric features.
        positions:
            Optional positions ``[B, N, 1]``; auto-generated if *None*.
        """
        B, N = inputs.shape[0], inputs.shape[1]

        # Embedding / projection ------------------------------------------------
        if self.task == "classification":
            x = self.input_embed(inputs)  # [B, N, E]
        else:  # regression
            x = self.input_proj(inputs.float())  # [B, N, E]

        # Position handling -----------------------------------------------------
        if positions is None:
            positions = torch.linspace(self.pos_min, self.pos_max, N, device=inputs.device)
            positions = positions.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1)

        # ------------------------------------------------------------------
        # Add position embeddings ONCE before the TFN stack
        # ------------------------------------------------------------------
        if self.tfn_layers:
            pos_emb = self.tfn_layers[0].pos_embeddings(positions)
            x = x + pos_emb

        # TFN layers ------------------------------------------------------------
        for layer in self.tfn_layers:
            # Skip internal positional addition to avoid double-counting
            x = layer(x, positions, add_pos_emb=False)

        # Global average pooling (sequence → vector) ----------------------------
        pooled = x.mean(dim=1)
        out = self.head(pooled)
        if self.task == "regression":
            out = out.view(out.size(0), self.output_len, self.output_dim)
        return out if self.task == "regression" else out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        if hasattr(self, "input_embed"):
            nn.init.normal_(self.input_embed.weight, 0, 0.1)
        if hasattr(self, "input_proj"):
            nn.init.normal_(self.input_proj.weight, 0, 0.1)
            nn.init.zeros_(self.input_proj.bias)

        for mod in self.head:
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, 0, 0.1)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

    # ------------------------------------------------------------------
    # Convenient factory methods ---------------------------------------
    # ------------------------------------------------------------------
    @classmethod
    def for_classification(
        cls,
        vocab_size: int,
        num_classes: int,
        **kwargs,
    ) -> "TFN":
        """Factory for classification task."""
        return cls(task="classification", vocab_size=vocab_size, num_classes=num_classes, **kwargs)

    @classmethod
    def for_regression(
        cls,
        input_dim: int,
        output_dim: int = 1,
        **kwargs,
    ) -> "TFN":
        """Factory for regression task."""
        return cls(task="regression", input_dim=input_dim, output_dim=output_dim, **kwargs) 

# ------------------------------------------------------------------
# Backward-compatibility alias
# ------------------------------------------------------------------

UnifiedTFN = TFN 