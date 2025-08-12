from __future__ import annotations

"""
baselines.py
Unified baseline models for 1D sequence tasks (classification and regression).

Includes:
- TransformerBaseline
- PerformerBaseline  
- LSTMBaseline
- CNNBaseline
- RoBERTaBaseline (SOTA for NLP)
- InformerBaseline (SOTA for time series)
- TCNBaseline (Temporal Convolutional Networks)
- LinearAttentionBaseline (Efficient attention)

Each class supports both classification and regression via the 'task' parameter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
import math

from .shared_layers import LinearAttention, create_positional_embedding_strategy
from .base_model import BaseSequenceModel


class TransformerBaseline(BaseSequenceModel):
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
        positional_embedding_strategy: Strategy for positional embeddings
        calendar_features: Calendar features for time-based embeddings
        feature_cardinalities: Cardinality of each calendar feature
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
        positional_embedding_strategy: str = "auto",
        calendar_features: Optional[List[str]] = None,
        feature_cardinalities: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        self.task = task
        self.embed_dim = embed_dim
        self.output_len = output_len
        self.output_dim = output_dim
        self.seq_len = seq_len
        
        # Use the factory pattern for positional embeddings
        self.pos_embedding = create_positional_embedding_strategy(
            strategy_name=positional_embedding_strategy,
            max_len=seq_len,
            embed_dim=embed_dim,
            calendar_features=calendar_features,
            feature_cardinalities=feature_cardinalities,
        )
        
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
            
    def forward(self, 
                inputs: torch.Tensor, 
                positions: Optional[torch.Tensor] = None,
                calendar_features: Optional[Dict[str, torch.Tensor]] = None,
                **kwargs) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor. Shape: [B, L] (classification) or [B, L, input_dim] (regression)
        positions : Optional[torch.Tensor], default=None
            Optional positional information. If None, uses sequential positions.
        calendar_features : Optional[Dict[str, torch.Tensor]], default=None
            Calendar features for time-based positional embeddings.
        **kwargs
            Additional keyword arguments.
            
        Returns
        -------
        torch.Tensor
            Model output.
        """
        # Input embedding/projection
        h = self.input(inputs)  # [B, L, embed_dim]
        
        # Generate positions if not provided
        if positions is None:
            seq_len = inputs.size(1)
            positions = torch.arange(seq_len, device=inputs.device).unsqueeze(0)  # [1, L]
        
        # Add positional embeddings using the factory strategy
        pos_emb = self.pos_embedding(positions, calendar_features=calendar_features)
        h = h + pos_emb
        
        # Transformer layers
        h = self.transformer(h)
        
        if self.task == "classification":
            # FIXED: Use mean pooling instead of first token for more robust classification
            # The first token is not a special [CLS] token in our data loaders
            pooled = h.mean(dim=1)  # [B, embed_dim] - mean pooling over sequence
            out = self.head(pooled)
        else:
            # For regression, use the last few tokens for multi-step forecasting
            if self.output_len == 1:
                out = self.head(h[:, -1, :])  # [B, output_dim]
            else:
                # Multi-step forecasting: use last output_len tokens
                last_tokens = h[:, -self.output_len:, :]  # [B, output_len, embed_dim]
                out = self.head(last_tokens)  # [B, output_len, output_dim]
        
        return out


class PerformerBaseline(BaseSequenceModel):
    """Performer with linear attention for efficient sequence modeling."""
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
        positional_embedding_strategy: str = "auto",
        calendar_features: Optional[List[str]] = None,
        feature_cardinalities: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        self.task = task
        self.embed_dim = embed_dim
        self.output_len = output_len
        self.output_dim = output_dim
        self.seq_len = seq_len
        
        # Use the factory pattern for positional embeddings
        self.pos_embedding = create_positional_embedding_strategy(
            strategy_name=positional_embedding_strategy,
            max_len=seq_len,
            embed_dim=embed_dim,
            calendar_features=calendar_features,
            feature_cardinalities=feature_cardinalities,
        )
        
        if task == "classification":
            assert vocab_size is not None, "vocab_size required for classification"
            self.input = nn.Embedding(vocab_size, embed_dim)
        else:
            assert input_dim is not None, "input_dim required for regression"
            self.input = nn.Linear(input_dim, embed_dim)
        
        # Linear attention layers
        self.layers = nn.ModuleList([
            LinearAttention(embed_dim, proj_dim) 
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
    
    def forward(self, 
                inputs: torch.Tensor, 
                positions: Optional[torch.Tensor] = None,
                calendar_features: Optional[Dict[str, torch.Tensor]] = None,
                **kwargs) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor. Shape: [B, L] (classification) or [B, L, input_dim] (regression)
        positions : Optional[torch.Tensor], default=None
            Optional positional information. If None, uses sequential positions.
        calendar_features : Optional[Dict[str, torch.Tensor]], default=None
            Calendar features for time-based positional embeddings.
        **kwargs
            Additional keyword arguments.
            
        Returns
        -------
        torch.Tensor
            Model output.
        """
        # Input embedding/projection
        h = self.input(inputs)  # [B, L, embed_dim]
        
        # Generate positions if not provided
        if positions is None:
            seq_len = inputs.size(1)
            positions = torch.arange(seq_len, device=inputs.device).unsqueeze(0)  # [1, L]
        
        # Add positional embeddings using the factory strategy
        pos_emb = self.pos_embedding(positions, calendar_features=calendar_features)
        h = h + pos_emb
        
        # Linear attention layers
        for layer in self.layers:
            h = layer(h)
        
        if self.task == "classification":
            # FIXED: Use mean pooling instead of first token for more robust classification
            # The first token is not a special [CLS] token in our data loaders
            pooled = h.mean(dim=1)  # [B, embed_dim] - mean pooling over sequence
            out = self.head(pooled)
        else:
            # For regression, use the last few tokens for multi-step forecasting
            if self.output_len == 1:
                out = self.head(h[:, -1, :])  # [B, output_dim]
            else:
                # Multi-step forecasting: use last output_len tokens
                last_tokens = h[:, -self.output_len:, :]  # [B, output_len, embed_dim]
                out = self.head(last_tokens)  # [B, output_len, output_dim]
        
        return out


class RoBERTaBaseline(nn.Module):
    """
    RoBERTa-style model with improved training techniques.
    SOTA baseline for NLP tasks.
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
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.task = task
        self.embed_dim = embed_dim
        self.output_len = output_len
        self.output_dim = output_dim
        
        if task == "classification":
            assert vocab_size is not None, "vocab_size required for classification"
            self.input = nn.Embedding(vocab_size, embed_dim)
        else:
            assert input_dim is not None, "input_dim required for regression"
            self.input = nn.Linear(input_dim, embed_dim)
        
        # RoBERTa-style positional embeddings
        self.pos = create_positional_embedding_strategy(
            strategy_name="learned",
            max_len=seq_len,
            embed_dim=embed_dim
        )
        
        # Layer normalization for embeddings
        self.embed_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.embed_dropout = nn.Dropout(dropout)
        
        # Transformer layers with improved architecture
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm for better training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Task-specific heads
        if task == "classification":
            self.head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, num_classes)
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, output_dim)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L] (classification) or [B, L, input_dim] (regression)"""
        # Embedding + positional + normalization
        h = self.input(x)
        
        # Generate positions for positional embeddings
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        positions = positions.unsqueeze(0).expand(x.size(0), -1)  # [B, L]
        
        h = h + self.pos(positions)
        h = self.embed_norm(h)
        h = self.embed_dropout(h)
        
        # Transformer layers
        h = self.transformer(h)
        
        if self.task == "classification":
            # FIXED: Use mean pooling instead of first token for more robust classification
            # The first token is not a special [CLS] token in our data loaders
            pooled = h.mean(dim=1)  # [B, embed_dim] - mean pooling over sequence
            out = self.head(pooled)
        else:
            # Multi-step forecasting
            seq_len = h.size(1)
            if seq_len >= self.output_len:
                pooled = h[:, -self.output_len:, :]
            else:
                last_token = h[:, -1:, :]
                padding = last_token.repeat(1, self.output_len - seq_len, 1)
                pooled = torch.cat([h, padding], dim=1)
            
            batch_size, output_len, embed_dim = pooled.shape
            pooled_flat = pooled.reshape(-1, embed_dim)
            out_flat = self.head(pooled_flat)
            out = out_flat.reshape(batch_size, output_len, self.output_dim)
        return out


class InformerBaseline(nn.Module):
    """
    Informer model for efficient long sequence time series forecasting.
    SOTA baseline for time series tasks.
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
        num_heads: int = 8,
        dropout: float = 0.1,
        factor: int = 5,  # ProbAttention factor
    ) -> None:
        super().__init__()
        self.task = task
        self.embed_dim = embed_dim
        self.output_len = output_len
        self.output_dim = output_dim
        self.factor = factor
        
        if task == "classification":
            assert vocab_size is not None, "vocab_size required for classification"
            self.input = nn.Embedding(vocab_size, embed_dim)
        else:
            assert input_dim is not None, "input_dim required for regression"
            self.input = nn.Linear(input_dim, embed_dim)
        
        # Informer-style positional encoding
        self.pos = create_positional_embedding_strategy(
            strategy_name="learned",
            max_len=seq_len,
            embed_dim=embed_dim
        )
        
        # ProbAttention layers (simplified version)
        self.layers = nn.ModuleList([
            ProbAttentionLayer(embed_dim, num_heads, dropout, factor)
            for _ in range(num_layers)
        ])
        
        # Task-specific heads
        if task == "classification":
            self.head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, num_classes)
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, output_dim)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L] (classification) or [B, L, input_dim] (regression)"""
        # Generate positions for positional embeddings
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        positions = positions.unsqueeze(0).expand(x.size(0), -1)  # [B, L]
        
        h = self.input(x) + self.pos(positions)
        
        for layer in self.layers:
            h = layer(h)
        
        if self.task == "classification":
            # FIXED: Use mean pooling instead of first token for more robust classification
            # The first token is not a special [CLS] token in our data loaders
            pooled = h.mean(dim=1)  # [B, embed_dim] - mean pooling over sequence
            out = self.head(pooled)
        else:
            # Multi-step forecasting
            seq_len = h.size(1)
            if seq_len >= self.output_len:
                pooled = h[:, -self.output_len:, :]
            else:
                last_token = h[:, -1:, :]
                padding = last_token.repeat(1, self.output_len - seq_len, 1)
                pooled = torch.cat([h, padding], dim=1)
            
            batch_size, output_len, embed_dim = pooled.shape
            pooled_flat = pooled.reshape(-1, embed_dim)
            out_flat = self.head(pooled_flat)
            out = out_flat.reshape(batch_size, output_len, self.output_dim)
        return out


class ProbAttentionLayer(nn.Module):
    """Simplified ProbAttention layer for Informer."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float, factor: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.factor = factor
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        
        # Project queries, keys, values
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Standard attention (simplified for now)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)
        
        # Residual connection and normalization
        out = self.norm(x + out)
        return out


class TCNBaseline(nn.Module):
    """
    Temporal Convolutional Network baseline.
    Efficient for time series tasks.
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
        num_channels: List[int] = [64, 128, 256],
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.task = task
        self.embed_dim = embed_dim
        self.output_len = output_len
        self.output_dim = output_dim
        
        if task == "classification":
            assert vocab_size is not None, "vocab_size required for classification"
            self.input = nn.Embedding(vocab_size, embed_dim)
        else:
            assert input_dim is not None, "input_dim required for regression"
            self.input = nn.Linear(input_dim, embed_dim)
        
        # TCN layers
        layers = []
        in_channels = embed_dim
        for out_channels in num_channels:
            layers.append(
                TCNBlock(in_channels, out_channels, kernel_size, dropout)
            )
            in_channels = out_channels
        
        self.tcn = nn.Sequential(*layers)
        
        # Task-specific heads
        if task == "classification":
            self.head = nn.Sequential(
                nn.Linear(in_channels, in_channels // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(in_channels // 2, num_classes)
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(in_channels, in_channels // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(in_channels // 2, output_dim)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L] (classification) or [B, L, input_dim] (regression)"""
        h = self.input(x)  # [B, L, embed_dim]
        h = h.transpose(1, 2)  # [B, embed_dim, L] for TCN
        
        h = self.tcn(h)
        
        if self.task == "classification":
            # Global average pooling
            pooled = h.mean(dim=2)  # [B, channels]
            out = self.head(pooled)
        else:
            # Multi-step forecasting
            h = h.transpose(1, 2)  # [B, L, channels]
            seq_len = h.size(1)
            if seq_len >= self.output_len:
                pooled = h[:, -self.output_len:, :]
            else:
                last_token = h[:, -1:, :]
                padding = last_token.repeat(1, self.output_len - seq_len, 1)
                pooled = torch.cat([h, padding], dim=1)
            
            batch_size, output_len, channels = pooled.shape
            pooled_flat = pooled.reshape(-1, channels)
            out_flat = self.head(pooled_flat)
            out = out_flat.reshape(batch_size, output_len, self.output_dim)
        return out


class TCNBlock(nn.Module):
    """Temporal Convolutional Network block."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size-1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size-1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)
        
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        return out + residual


class LSTMBaseline(nn.Module):
    """
    LSTM baseline for sequence modeling.
    Args:
        task: 'classification' or 'regression'
        vocab_size: Required for classification (input: token indices)
        input_dim: Required for regression (input: continuous features)
        embed_dim: Embedding dimension
        output_dim: Output dimension for regression
        num_classes: Number of classes for classification
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        bidirectional: Whether to use bidirectional LSTM
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
        self.embed_dim = embed_dim
        self.output_len = output_len
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        
        if task == "classification":
            assert vocab_size is not None, "vocab_size required for classification"
            self.input = nn.Embedding(vocab_size, embed_dim)
        else:
            assert input_dim is not None, "input_dim required for regression"
            self.input = nn.Linear(input_dim, embed_dim)
        
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
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
        """x: [B, L] (classification) or [B, L, input_dim] (regression)"""
        h = self.input(x)  # [B, L, embed_dim]
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(h)  # [B, L, hidden_dim*2]
        
        if self.task == "classification":
            # Use last hidden state for classification
            if self.bidirectional:
                # Concatenate forward and backward hidden states
                last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [B, hidden_dim*2]
            else:
                last_hidden = hidden[-1]  # [B, hidden_dim]
            out = self.head(last_hidden)
        else:
            # Multi-step forecasting using last few outputs
            seq_len = lstm_out.size(1)
            if seq_len >= self.output_len:
                pooled = lstm_out[:, -self.output_len:, :]  # [B, output_len, hidden_dim*2]
            else:
                last_output = lstm_out[:, -1:, :]  # [B, 1, hidden_dim*2]
                padding = last_output.repeat(1, self.output_len - seq_len, 1)  # [B, output_len - seq_len, hidden_dim*2]
                pooled = torch.cat([lstm_out, padding], dim=1)  # [B, output_len, hidden_dim*2]
            
            # Apply head to each token
            batch_size, output_len, hidden_dim = pooled.shape
            pooled_flat = pooled.reshape(-1, hidden_dim)  # [B * output_len, hidden_dim*2]
            out_flat = self.head(pooled_flat)  # [B * output_len, output_dim]
            out = out_flat.reshape(batch_size, output_len, self.output_dim)  # [B, output_len, output_dim]
        return out


class CNNBaseline(nn.Module):
    """
    CNN baseline for sequence modeling.
    Args:
        task: 'classification' or 'regression'
        vocab_size: Required for classification (input: token indices)
        input_dim: Required for regression (input: continuous features)
        embed_dim: Embedding dimension
        output_dim: Output dimension for regression
        num_classes: Number of classes for classification
        num_filters: Number of filters per layer
        filter_sizes: List of filter sizes
        dropout: Dropout rate
        seq_len: Maximum sequence length (for adaptive pooling)
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
        seq_len: Optional[int] = None,  # NEW: allows adaptive filters
    ) -> None:
        super().__init__()
        self.task = task
        self.embed_dim = embed_dim
        self.output_len = output_len
        self.output_dim = output_dim
        self.filter_sizes = filter_sizes
        
        if task == "classification":
            assert vocab_size is not None, "vocab_size required for classification"
            self.input = nn.Embedding(vocab_size, embed_dim)
        else:
            assert input_dim is not None, "input_dim required for regression"
            self.input = nn.Linear(input_dim, embed_dim)
        
        # Convolutional layers for each filter size
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k-1)
            for k in filter_sizes
        ])
        
        # Adaptive pooling
        if seq_len is not None:
            self.pool = nn.AdaptiveAvgPool1d(1)
        else:
            self.pool = nn.AdaptiveMaxPool1d(1)
        
        # Calculate output dimension after convolutions
        conv_output_dim = num_filters * len(filter_sizes)
        
        if task == "classification":
            self.head = nn.Sequential(
                nn.Linear(conv_output_dim, conv_output_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(conv_output_dim // 2, num_classes)
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(conv_output_dim, conv_output_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(conv_output_dim // 2, output_dim)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L] (classification) or [B, L, input_dim] (regression)"""
        h = self.input(x)  # [B, L, embed_dim]
        h = h.transpose(1, 2)  # [B, embed_dim, L] for convolutions
        
        # Apply convolutions
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(h))  # [B, num_filters, L]
            conv_outputs.append(conv_out)
        
        # Concatenate outputs from different filter sizes
        h = torch.cat(conv_outputs, dim=1)  # [B, num_filters*len(filter_sizes), L]
        
        if self.task == "classification":
            # Global pooling for classification
            h = self.pool(h).squeeze(-1)  # [B, num_filters*len(filter_sizes)]
            out = self.head(h)
        else:
            # Multi-step forecasting
            h = h.transpose(1, 2)  # [B, L, num_filters*len(filter_sizes)]
            seq_len = h.size(1)
            if seq_len >= self.output_len:
                pooled = h[:, -self.output_len:, :]  # [B, output_len, num_filters*len(filter_sizes)]
            else:
                last_token = h[:, -1:, :]  # [B, 1, num_filters*len(filter_sizes)]
                padding = last_token.repeat(1, self.output_len - seq_len, 1)  # [B, output_len - seq_len, num_filters*len(filter_sizes)]
                pooled = torch.cat([h, padding], dim=1)  # [B, output_len, num_filters*len(filter_sizes)]
            
            # Apply head to each token
            batch_size, output_len, features = pooled.shape
            pooled_flat = pooled.reshape(-1, features)  # [B * output_len, num_filters*len(filter_sizes)]
            out_flat = self.head(pooled_flat)  # [B * output_len, output_dim]
            out = out_flat.reshape(batch_size, output_len, self.output_dim)  # [B, output_len, output_dim]
        return out 