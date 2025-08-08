"""
Model wrappers for normalization and preprocessing.

This module provides wrapper models that handle data preprocessing
while keeping the core training pipeline unchanged.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple


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


class PARN(nn.Module):
    """
    Physics-Aware Reversible Normalization (PARN) module.
    
    Decouples location and scale normalization and can feed preserved
    statistics back to the model as conditioning features.
    
    Args:
        num_features: Number of features to normalize.
        mode: Normalization mode ('location', 'scale', 'full').
        eps: Small value to prevent division by zero.
    """
    def __init__(self, num_features: int, mode: str = 'location', eps: float = 1e-5):
        super().__init__()
        if mode not in ['location', 'scale', 'full']:
            raise ValueError(f"Invalid PARN mode: {mode}")
        self.num_features = num_features
        self.mode = mode
        self.eps = eps
        
        # Learnable affine parameters, same as RevIN
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))
        
        # Statistics storage
        self.mean = None
        self.stdev = None

    def forward(self, x: torch.Tensor, operation: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for normalization or denormalization.
        
        Args:
            x: Input tensor of shape [B, N, F].
            operation: 'norm' for normalization or 'denorm' for denormalization.
            
        Returns:
            For 'norm': A tuple of (normalized_tensor, stats_dict).
            For 'denorm': A tuple of (denormalized_tensor, empty_dict).
        """
        if operation == 'norm':
            self._get_statistics(x)
            x_norm, stats = self._normalize(x)
            return x_norm, stats
        elif operation == 'denorm':
            x_denorm = self._denormalize(x)
            return x_denorm, {}  # Return empty dict for consistent interface
        else:
            raise ValueError(f"Unknown operation: {operation}")

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

    def _normalize(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Normalize the input tensor and return preserved statistics."""
        if self.mean is None or self.stdev is None:
            raise RuntimeError("Statistics not computed. Call forward with operation='norm' first.")
        
        stats = {}
        
        if self.mode == 'location':
            # Remove location (mean) only
            x = x - self.mean
            stats['scale'] = self.stdev  # Preserve scale
        elif self.mode == 'scale':
            # Remove scale (std) only
            x = x / self.stdev
            stats['location'] = self.mean  # Preserve location
        elif self.mode == 'full':
            # Remove both location and scale
            x = (x - self.mean) / self.stdev
            stats['location'] = self.mean
            stats['scale'] = self.stdev
        
        # Apply affine transformation
        if x.dim() == 3:
            # [B, N, F] -> apply to feature dimension
            x = x * self.affine_weight.unsqueeze(0).unsqueeze(0)
            x = x + self.affine_bias.unsqueeze(0).unsqueeze(0)
        else:
            # [B, F] -> apply to feature dimension
            x = x * self.affine_weight.unsqueeze(0)
            x = x + self.affine_bias.unsqueeze(0)
        
        return x, stats

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize the input tensor."""
        if self.mean is None or self.stdev is None:
            raise RuntimeError("Statistics not computed. Call forward with operation='norm' first.")
        
        # Reverse affine transformation
        if x.dim() == 3:
            # [B, N, F] -> apply to feature dimension
            x = x - self.affine_bias.unsqueeze(0).unsqueeze(0)
            x = x / (self.affine_weight.unsqueeze(0).unsqueeze(0) + self.eps)
        else:
            # [B, F] -> apply to feature dimension
            x = x - self.affine_bias.unsqueeze(0)
            x = x / (self.affine_weight.unsqueeze(0) + self.eps)
        
        # Denormalize based on the original mode
        if self.mode == 'location':
            x = x + self.mean
        elif self.mode == 'scale':
            x = x * self.stdev
        elif self.mode == 'full':
            x = x * self.stdev + self.mean
        
        return x


class PARNModel(nn.Module):
    """
    A wrapper model that applies PARN to a base model. It augments the
    input with the preserved statistics before passing to the base model.
    
    Args:
        base_model: The model to wrap (e.g., EnhancedTFNRegressor)
        num_features: Number of features for PARN
        mode: PARN normalization mode ('location', 'scale', 'full')
    """
    def __init__(self, base_model: nn.Module, num_features: int, mode: str = 'location'):
        super().__init__()
        self.base_model = base_model
        self.parn = PARN(num_features, mode=mode)
        self.mode = mode
        self.name = f"PARNModel({base_model.__class__.__name__}, mode='{mode}')"

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass with automatic normalization and denormalization.
        
        Args:
            x: Input tensor
            **kwargs: Additional arguments passed to the base model
            
        Returns:
            Denormalized output from the base model
        """
        # 1. Normalize the input and get preserved stats
        x_norm, stats = self.parn(x, operation='norm')
        
        # 2. Augment the input with the preserved stats
        augmented_input = [x_norm]
        if 'location' in stats:
            # Expand mean to match sequence length and append as a feature
            loc_stat = stats['location'].expand_as(x_norm)
            augmented_input.append(loc_stat)
        if 'scale' in stats:
            # Expand std to match sequence length and append as a feature
            scale_stat = stats['scale'].expand_as(x_norm)
            augmented_input.append(scale_stat)
            
        # Only concatenate if we have additional statistics
        if len(augmented_input) > 1:
            x_augmented = torch.cat(augmented_input, dim=-1)
        else:
            x_augmented = x_norm
        
        # 3. Pass the augmented input to the base model
        output_augmented = self.base_model(x_augmented, **kwargs)
        
        # 4. The model only predicts the core features, not the stats.
        # Slice the output to get the core prediction.
        output_normalized = output_augmented[:, :, :self.parn.num_features]
        
        # 5. Denormalize the output
        output_denormalized, _ = self.parn(output_normalized, operation='denorm')
        
        return output_denormalized



    def get_physics_constraints(self) -> Dict[str, torch.Tensor]:
        """Get physics constraints from the base model if available."""
        if hasattr(self.base_model, 'get_physics_constraints'):
            return self.base_model.get_physics_constraints()
        return {}


# Factory functions for easy model wrapping
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


def create_parn_wrapper(base_model: nn.Module, num_features: int, mode: str = 'location') -> PARNModel:
    """
    Create a PARN wrapper around a base model.
    
    Args:
        base_model: The model to wrap
        num_features: Number of features for PARN
        mode: PARN normalization mode ('location', 'scale', 'full')
        
    Returns:
        PARNModel wrapper
    """
    return PARNModel(base_model, num_features, mode) 