from __future__ import annotations

"""
baselines.py
Unified baseline models for 1D sequence tasks (classification and regression).

Includes:
- TransformerBaseline
- PerformerBaseline
- LSTMBaseline
- CNNBaseline

Each class supports both classification and regression via the 'task' parameter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from .shared_layers import LearnedPositionalEmbeddings, LinearAttention


class TransformerBaseline(nn.Module):
    """
    Transformer encoder for sequence classification or regression.
    Args:
        task: 'classification' or 'regression'
        vocab_size: Required for classification (input: token indices)
        input_dim: Required for regression (input: continuous features)
        embed_dim: Embedding dimension
        output_dim: Output dimension for regression
        num_classes: Number of classes for classification
        seq_len: Sequence length
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    def __init__(
        self,
        task: str,
        vocab_size: Optional[int] = None,
        input_dim: Optional[int] = None,
        embed_dim: int = 128,
        output_dim: int = 1,
        output_len: int = 1,
        num_classes: int = 2,
        seq_len: int = 512,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.task = task
        self.embed_dim = embed_dim
        self.output_len = output_len
        self.output_dim = output_dim
        self.pos = LearnedPositionalEmbeddings(seq_len, embed_dim)
        if task == "classification":
            assert vocab_size is not None, "vocab_size required for classification"
            self.input = nn.Embedding(vocab_size, embed_dim)
        else:
            assert input_dim is not None, "input_dim required for regression"
            self.input = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        if task == "classification":
            self.head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, num_classes)
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, output_dim)
            )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L] (classification) or [B, L, input_dim] (regression)"""
        h = self.input(x) + self.pos(min(x.size(1), self.pos.pos.num_embeddings))
        h = self.transformer(h)
        if self.task == "classification":
            pooled = h[:, 0, :]  # [B, embed_dim] - use first token for classification
            out = self.head(pooled)
        else:
            # For regression, use the last few tokens for multi-step forecasting
            # Take the last output_len tokens, or all if sequence is shorter
            seq_len = h.size(1)
            if seq_len >= self.output_len:
                # Use last output_len tokens
                pooled = h[:, -self.output_len:, :]  # [B, output_len, embed_dim]
            else:
                # Pad with the last token if sequence is too short
                last_token = h[:, -1:, :]  # [B, 1, embed_dim]
                padding = last_token.repeat(1, self.output_len - seq_len, 1)  # [B, output_len - seq_len, embed_dim]
                pooled = torch.cat([h, padding], dim=1)  # [B, output_len, embed_dim]
            
            # Apply head to each token
            batch_size, output_len, embed_dim = pooled.shape
            pooled_flat = pooled.reshape(-1, embed_dim)  # [B * output_len, embed_dim]
            out_flat = self.head(pooled_flat)  # [B * output_len, output_dim]
            out = out_flat.reshape(batch_size, output_len, self.output_dim)  # [B, output_len, output_dim]
        return out


class PerformerBaseline(nn.Module):
    """
    Performer-style linear attention for sequence classification or regression.
    Args are the same as TransformerBaseline, with proj_dim for Performer.
    """
    def __init__(
        self,
        task: str,
        vocab_size: Optional[int] = None,
        input_dim: Optional[int] = None,
        embed_dim: int = 128,
        output_dim: int = 1,
        output_len: int = 1,
        num_classes: int = 2,
        seq_len: int = 512,
        num_layers: int = 2,
        proj_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.task = task
        self.output_len = output_len
        self.output_dim = output_dim
        self.pos = LearnedPositionalEmbeddings(seq_len, embed_dim)
        if task == "classification":
            assert vocab_size is not None
            self.input = nn.Embedding(vocab_size, embed_dim)
        else:
            assert input_dim is not None
            self.input = nn.Linear(input_dim, embed_dim)
        self.layers = nn.ModuleList([
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
        ])
        if task == "classification":
            self.head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, num_classes)
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, output_dim)
            )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input(x) + self.pos(min(x.size(1), self.pos.pos.num_embeddings))
        for layer in self.layers:
            h = layer(h)
        if self.task == "classification":
            pooled = h.mean(dim=1)  # [B, embed_dim]
            out = self.head(pooled)
        else:
            # For regression, use the last few tokens for multi-step forecasting
            seq_len = h.size(1)
            if seq_len >= self.output_len:
                pooled = h[:, -self.output_len:, :]  # [B, output_len, embed_dim]
            else:
                last_token = h[:, -1:, :]  # [B, 1, embed_dim]
                padding = last_token.repeat(1, self.output_len - seq_len, 1)  # [B, output_len - seq_len, embed_dim]
                pooled = torch.cat([h, padding], dim=1)  # [B, output_len, embed_dim]
            
            # Apply head to each token
            batch_size, output_len, embed_dim = pooled.shape
            pooled_flat = pooled.reshape(-1, embed_dim)  # [B * output_len, embed_dim]
            out_flat = self.head(pooled_flat)  # [B * output_len, output_dim]
            out = out_flat.reshape(batch_size, output_len, self.output_dim)  # [B, output_len, output_dim]
        return out


class LSTMBaseline(nn.Module):
    """
    LSTM for sequence classification or regression.
    Args are the same as TransformerBaseline, with hidden_dim and bidirectional.
    """
    def __init__(
        self,
        task: str,
        vocab_size: Optional[int] = None,
        input_dim: Optional[int] = None,
        embed_dim: int = 128,
        output_dim: int = 1,
        output_len: int = 1,
        num_classes: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.task = task
        self.output_len = output_len
        self.output_dim = output_dim
        if task == "classification":
            assert vocab_size is not None
            self.input = nn.Embedding(vocab_size, embed_dim)
        else:
            assert input_dim is not None
            self.input = nn.Linear(input_dim, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        if task == "classification":
            self.head = nn.Sequential(
                nn.Linear(lstm_output_dim, lstm_output_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(lstm_output_dim // 2, num_classes)
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(lstm_output_dim, lstm_output_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(lstm_output_dim // 2, output_dim)
            )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input(x)
        lstm_out, (hidden, cell) = self.lstm(h)
        if self.task == "classification":
            if self.lstm.bidirectional:
                last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
            else:
                last_hidden = hidden[-1]
            out = self.head(last_hidden)
        else:
            # For regression, use the last few outputs for multi-step forecasting
            seq_len = lstm_out.size(1)
            if seq_len >= self.output_len:
                pooled = lstm_out[:, -self.output_len:, :]  # [B, output_len, hidden_dim]
            else:
                last_output = lstm_out[:, -1:, :]  # [B, 1, hidden_dim]
                padding = last_output.repeat(1, self.output_len - seq_len, 1)  # [B, output_len - seq_len, hidden_dim]
                pooled = torch.cat([lstm_out, padding], dim=1)  # [B, output_len, hidden_dim]
            
            # Apply head to each output
            batch_size, output_len, hidden_dim = pooled.shape
            pooled_flat = pooled.reshape(-1, hidden_dim)  # [B * output_len, hidden_dim]
            out_flat = self.head(pooled_flat)  # [B * output_len, output_dim]
            out = out_flat.reshape(batch_size, output_len, self.output_dim)  # [B, output_len, output_dim]
        return out


class CNNBaseline(nn.Module):
    """
    CNN for sequence classification or regression.
    Args are the same as TransformerBaseline, with num_filters and filter_sizes.
    """
    def __init__(
        self,
        task: str,
        vocab_size: Optional[int] = None,
        input_dim: Optional[int] = None,
        embed_dim: int = 128,
        output_dim: int = 1,
        output_len: int = 1,
        num_classes: int = 2,
        num_filters: int = 128,
        filter_sizes: List[int] = [2, 3, 4],  # Smaller default filter sizes
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.task = task
        self.output_len = output_len
        self.output_dim = output_dim
        if task == "classification":
            assert vocab_size is not None
            self.input = nn.Embedding(vocab_size, embed_dim)
        else:
            assert input_dim is not None
            self.input = nn.Linear(input_dim, embed_dim)
        
        # Adaptive filter sizes based on typical sequence length
        # Use smaller filters for shorter sequences
        if hasattr(self, 'seq_len') and self.seq_len < 10:
            filter_sizes = [2, 3]  # Smaller filters for short sequences
        elif hasattr(self, 'seq_len') and self.seq_len < 20:
            filter_sizes = [2, 3, 4]  # Medium filters for medium sequences
        # Otherwise use the provided filter_sizes
        
        # Additional safety check for very short sequences
        if len(filter_sizes) > 0 and max(filter_sizes) > 4:
            # If we have very short sequences, use even smaller filters
            filter_sizes = [2, 3]  # Force small filters for safety
        
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k)
            for k in filter_sizes
        ])
        total_filters = num_filters * len(filter_sizes)
        if task == "classification":
            self.head = nn.Sequential(
                nn.Linear(total_filters, total_filters // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(total_filters // 2, num_classes)
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(total_filters, total_filters // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(total_filters // 2, output_dim)
            )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input(x)  # [B, L, embed_dim]
        h = h.transpose(1, 2)  # [B, embed_dim, L]
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(h))
            pooled = F.max_pool1d(conv_out, conv_out.size(2))
            conv_outputs.append(pooled.squeeze(2))
        concatenated = torch.cat(conv_outputs, dim=1)
        
        if self.task == "classification":
            out = self.head(concatenated)
        else:
            # For regression, we need to generate multiple outputs
            # Since CNN outputs a single vector, we'll repeat it for multi-step forecasting
            batch_size = concatenated.size(0)
            # Repeat the output for each time step
            repeated = concatenated.unsqueeze(1).repeat(1, self.output_len, 1)  # [B, output_len, features]
            batch_size, output_len, features = repeated.shape
            repeated_flat = repeated.reshape(-1, features)  # [B * output_len, features]
            out_flat = self.head(repeated_flat)  # [B * output_len, output_dim]
            out = out_flat.reshape(batch_size, output_len, self.output_dim)  # [B, output_len, output_dim]
        return out 