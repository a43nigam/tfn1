"""tfn.model.shared_layers
Reusable building blocks shared across baseline classifiers & regressors.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "LearnedPositionalEmbeddings",
    "LinearAttention",
]


class LearnedPositionalEmbeddings(nn.Module):
    """Standard learned absolute positional embeddings.

    Parameters
    ----------
    max_len : int
        Maximum sequence length for which to prepare embeddings.
    embed_dim : int
        Embedding dimensionality.
    """

    def __init__(self, max_len: int, embed_dim: int) -> None:
        super().__init__()
        self.pos = nn.Embedding(max_len, embed_dim)
        nn.init.normal_(self.pos.weight, mean=0.0, std=0.02)

    def forward(self, seq_len: int) -> torch.Tensor:  # [L, D]
        idx = torch.arange(seq_len, device=self.pos.weight.device)
        return self.pos(idx)


class LinearAttention(nn.Module):
    """Single-head linear (FAVOR) attention approximation.

    A light-weight replacement for softmax attention that scales **O(L·D)**
    instead of **O(L²)**.
    """

    def __init__(self, dim: int, proj_dim: int = 64) -> None:
        super().__init__()
        self.dim = dim
        self.proj_dim = proj_dim
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)

        # Random orthogonal projection matrix (fixed)
        q = torch.randn(dim, proj_dim)
        q, _ = torch.linalg.qr(q, mode="reduced")
        self.register_buffer("projection", q)  # [D, P]

    @staticmethod
    def _feature_map(x: torch.Tensor) -> torch.Tensor:
        """Positive feature map ensuring non-negativity (ELU + 1)."""
        return F.elu(x) + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, L, D]
        q, k, v = self.query(x), self.key(x), self.value(x)
        q_prime = self._feature_map(q @ self.projection)  # [B, L, P]
        k_prime = self._feature_map(k @ self.projection)  # [B, L, P]

        kv = torch.einsum("blp,bld->bpd", k_prime, v)  # [B, P, D]
        num = torch.einsum("blp,bpd->bld", q_prime, kv)  # [B, L, D]
        z = 1 / (q_prime.sum(dim=-1, keepdim=True) + 1e-8)  # [B, L, 1]
        return num * z 