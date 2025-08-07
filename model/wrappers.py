"""
Model wrappers for normalization and preprocessing.

This module provides wrapper models that handle data preprocessing
while keeping the core training pipeline unchanged.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class RevIN(nn.Module):
    """
    Reversible Instance Normalization module.
    
    This implements the RevIN approach from the paper:
    "Reversible Instance Normalization for Accurate Time-Series Forecasting"
    
    Args:
        num_features: Number of features to normalize
        eps: Small value to prevent division by zero
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        
        # Learnable affine parameters
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))
        
        # Statistics storage
        self.mean = None
        self.stdev = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Forward pass with normalization or denormalization.
        
        Args:
            x: Input tensor of shape [B, N, F] or [B, F]
            mode: Either 'norm' for normalization or 'denorm' for denormalization
            
        Returns:
            Normalized or denormalized tensor
        """
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'norm' or 'denorm'")
        return x

    def _get_statistics(self, x: torch.Tensor) -> None:
        """Compute and store mean and standard deviation."""
        # Handle both [B, N, F] and [B, F] shapes
        if x.dim() == 3:
            # [B, N, F] -> compute stats across time dimension
            self.mean = torch.mean(x, dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        else:
            # [B, F] -> compute stats across batch dimension
            self.mean = torch.mean(x, dim=0, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=0, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the input tensor."""
        if self.mean is None or self.stdev is None:
            raise RuntimeError("Statistics not computed. Call forward with mode='norm' first.")
        
        # Normalize: (x - mean) / std
        x = x - self.mean
        x = x / self.stdev
        
        # Apply affine transformation: weight * x + bias
        if x.dim() == 3:
            # [B, N, F] -> apply to feature dimension
            x = x * self.affine_weight.unsqueeze(0).unsqueeze(0)
            x = x + self.affine_bias.unsqueeze(0).unsqueeze(0)
        else:
            # [B, F] -> apply to feature dimension
            x = x * self.affine_weight.unsqueeze(0)
            x = x + self.affine_bias.unsqueeze(0)
        
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize the input tensor."""
        if self.mean is None or self.stdev is None:
            raise RuntimeError("Statistics not computed. Call forward with mode='norm' first.")
        
        # Reverse affine transformation: (x - bias) / weight
        if x.dim() == 3:
            # [B, N, F] -> apply to feature dimension
            x = x - self.affine_bias.unsqueeze(0).unsqueeze(0)
            x = x / (self.affine_weight.unsqueeze(0).unsqueeze(0) + self.eps)
        else:
            # [B, F] -> apply to feature dimension
            x = x - self.affine_bias.unsqueeze(0)
            x = x / (self.affine_weight.unsqueeze(0) + self.eps)
        
        # Denormalize: x * std + mean
        x = x * self.stdev
        x = x + self.mean
        
        return x


class RevinModel(nn.Module):
    """
    A wrapper model that applies RevIN to a base model.
    
    This wrapper handles the normalization and denormalization automatically,
    so the base model always works with normalized data, and the output
    is automatically denormalized back to the original scale.
    
    Args:
        base_model: The model to wrap (e.g., EnhancedTFNRegressor)
        num_features: Number of features for RevIN
    """
    def __init__(self, base_model: nn.Module, num_features: int):
        super().__init__()
        self.base_model = base_model
        self.revin = RevIN(num_features)
        
        # Store the original model's name for logging
        self.name = f"RevinModel({base_model.__class__.__name__})"

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass with automatic normalization and denormalization.
        
        Args:
            x: Input tensor
            **kwargs: Additional arguments passed to the base model
            
        Returns:
            Denormalized output from the base model
        """
        # 1. Normalize the input
        x_normalized = self.revin(x, 'norm')
        
        # 2. Pass the normalized input to the base model
        output_normalized = self.base_model(x_normalized, **kwargs)
        
        # 3. Denormalize the output
        output_denormalized = self.revin(output_normalized, 'denorm')
        
        return output_denormalized

    def get_physics_constraints(self) -> Dict[str, torch.Tensor]:
        """Get physics constraints from the base model if available."""
        if hasattr(self.base_model, 'get_physics_constraints'):
            return self.base_model.get_physics_constraints()
        return {}


# Factory function for easy model wrapping
def create_revin_wrapper(base_model: nn.Module, num_features: int) -> RevinModel:
    """
    Create a RevIN wrapper around a base model.
    
    Args:
        base_model: The model to wrap
        num_features: Number of features for RevIN
        
    Returns:
        RevinModel wrapper
    """
    return RevinModel(base_model, num_features) 