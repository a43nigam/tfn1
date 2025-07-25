from __future__ import annotations

"""seq_baselines.py
Light-weight baseline models for synthetic sequence-to-sequence tasks.

The models share a common interface:
    forward(tokens) -> logits of shape [B, L, vocab_size]

1. TFNSeqModel – wraps a single TrainableTFNLayer (1-D) followed by a linear head.
2. SimpleTransformerSeqModel – thin wrapper around nn.TransformerEncoderLayer
   with learned positional embeddings.
3. SimplePerformerSeqModel – linear-attention approximation implemented with
   random features (FAVOR) *without* external dependencies. Falls back to the
   Transformer baseline if CUDA/random projection is unavailable.

These implementations are intentionally minimal – their goal is to provide fair
references for our 1-D TFN on long-sequence synthetic benchmarks, not to match
state-of-the-art language models.
"""

from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tfn_base import TrainableTFNLayer  # local import

__all__ = [
    "TFNSeqModel",
    "SimpleTransformerSeqModel",
    "SimplePerformerSeqModel",
]

# ---------------------------------------------------------------------------
# Positional embedding helper (shared by Transformer / Performer)
# ---------------------------------------------------------------------------

class LearnedPositionalEmbeddings(nn.Module):
    """Standard learned absolute positional embeddings."""

    def __init__(self, max_len: int, embed_dim: int) -> None:
        super().__init__()
        self.pos = nn.Embedding(max_len, embed_dim)
        nn.init.normal_(self.pos.weight, mean=0.0, std=0.02)

    def forward(self, seq_len: int) -> torch.Tensor:  # [L, D]
        idx = torch.arange(seq_len, device=self.pos.weight.device)
        return self.pos(idx)

# ---------------------------------------------------------------------------
# 1. TFN sequence model
# ---------------------------------------------------------------------------

class TFNSeqModel(nn.Module):
    """Simplified 1-D TFN sequence model for token-level prediction."""

    def __init__(
        self,
        vocab_size: int,
        seq_len: int = 512,
        embed_dim: int = 128,
        grid_size: int = 256,
        kernel_type: str = "rbf",
        evolution_type: str = "cnn",
        time_steps: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.tfn = TrainableTFNLayer(
            embed_dim=embed_dim, 
            grid_size=grid_size,
            kernel_type=kernel_type,
            evolution_type=evolution_type,
            time_steps=time_steps,
            dropout=dropout
        )
        self.out_proj = nn.Linear(embed_dim, vocab_size, bias=False)
        self.max_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, L]
        B, L = x.shape
        pos = torch.linspace(0, 1, L, device=x.device).view(1, L, 1).expand(B, -1, -1)
        h = self.embed(x)
        h = self.tfn(h, pos)  # [B, L, D]
        logits = self.out_proj(h)  # [B, L, V]
        return logits

# ---------------------------------------------------------------------------
# 2. Standard Transformer baseline
# ---------------------------------------------------------------------------

class SimpleTransformerSeqModel(nn.Module):
    """Baseline Transformer encoder for sequence tasks."""

    def __init__(
        self,
        vocab_size: int,
        seq_len: int = 512,
        embed_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = LearnedPositionalEmbeddings(seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, L]
        h = self.embed(x) + self.pos(x.size(1))
        h = self.transformer(h)
        logits = self.out_proj(h)
        return logits

# ---------------------------------------------------------------------------
# 3. Performer-style linear attention baseline
# ---------------------------------------------------------------------------

class LinearAttention(nn.Module):
    """Single-head linear (FAVOR) attention approximation.

    This is **not** a full Performer; it is a lightweight implementation good
    enough for synthetic benchmarks where sequence length is large but
    feature complexity is low.
    """

    def __init__(self, dim: int, proj_dim: int = 64) -> None:
        super().__init__()
        self.dim = dim
        self.proj_dim = proj_dim
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        # Random orthogonal projection matrix – fixed after init
        q = torch.randn(dim, proj_dim)
        q, _ = torch.linalg.qr(q, mode="reduced")
        self.register_buffer("projection", q)  # [D, P]

    @staticmethod
    def _feature_map(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1  # guarantees positivity

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, L, D]
        q, k, v = self.query(x), self.key(x), self.value(x)
        # Project to low-dimensional features
        q_prime = self._feature_map(q @ self.projection)  # [B, L, P]
        k_prime = self._feature_map(k @ self.projection)  # [B, L, P]

        # Compute KV term first: [B, P, D]
        kv = torch.einsum("blp,bld->bpd", k_prime, v)
        # Attention numerator: [B, L, D]
        num = torch.einsum("blp,bpd->bld", q_prime, kv)
        # Normaliser: [B, L, 1]
        z = 1 / (q_prime.sum(dim=-1, keepdim=True) + 1e-8)
        return num * z  # element-wise broadcast

class SimplePerformerSeqModel(nn.Module):
    """Stack of LinearAttention layers mimicking Performer behaviour."""

    def __init__(
        self,
        vocab_size: int,
        seq_len: int = 512,
        embed_dim: int = 128,
        num_layers: int = 2,
        proj_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = LearnedPositionalEmbeddings(seq_len, embed_dim)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    LinearAttention(embed_dim, proj_dim=proj_dim),
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim * 4, embed_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(num_layers)
            ]
        )
        self.out_proj = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, L]
        h = self.embed(x) + self.pos(x.size(1))
        for layer in self.layers:
            residual = h
            h = layer(h)
            h = h + residual
        logits = self.out_proj(h)
        return logits 