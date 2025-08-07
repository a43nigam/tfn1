from __future__ import annotations

import torch
import torch.nn as nn
import warnings
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union
from src import metrics

class TaskStrategy(ABC):
    """Abstract base class for task-specific training logic."""

    @abstractmethod
    def get_criterion(self) -> nn.Module:
        """Returns the appropriate loss function for the task."""
        pass

    @abstractmethod
    def process_forward_pass(self, model: nn.Module, x: Union[torch.Tensor, Tuple[torch.Tensor, ...]], y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes the forward pass through the model and returns logits/predictions and loss.
        
        Args:
            model: The neural network model
            x: Input tensor or tuple of tensors (e.g., input_ids and attention_mask)
            y: Target tensor
            
        Returns:
            Tuple of (logits/predictions, loss)
        """
        pass

    @abstractmethod
    def calculate_metrics(self, logits: torch.Tensor, targets: torch.Tensor, scaler: Optional[Any] = None, **kwargs) -> Dict[str, float]:
        """
        Calculates and returns a dictionary of relevant metrics.
        
        Args:
            logits: Model output logits/predictions
            targets: Ground truth targets
            scaler: Optional scaler for denormalization (used in regression tasks)
            
        Returns:
            Dictionary mapping metric names to values
        """
        pass


class ClassificationStrategy(TaskStrategy):
    """Strategy for classification and NER tasks."""
    
    def get_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def process_forward_pass(self, model: nn.Module, x: Union[torch.Tensor, Tuple[torch.Tensor, ...]], y: torch.Tensor, positions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Handle input format (single tensor or tuple with attention mask)
        if isinstance(x, tuple):
            if positions is not None:
                logits = model(x[0], positions=positions)
            else:
                logits = model(x[0])
        else:
            if positions is not None:
                logits = model(x, positions=positions)
            else:
                logits = model(x)
        
        # Handle sequence-level tasks where we need to average over sequence dimension
        if logits.dim() == 3 and logits.size(1) > 1:
            logits = logits.mean(dim=1)
        
        loss = self.get_criterion()(logits, y)
        return logits, loss

    def calculate_metrics(self, logits: torch.Tensor, targets: torch.Tensor, scaler: Optional[Any] = None, **kwargs) -> Dict[str, float]:
        preds = torch.argmax(logits, dim=-1)
        correct = (preds == targets).float().sum().item()
        total = targets.numel()
        accuracy = correct / total if total > 0 else 0.0
        return {"acc": accuracy}


class RegressionStrategy(TaskStrategy):
    """Strategy for regression and time-series tasks."""
    
    def __init__(self, scaler: Optional[Any] = None, target_col_idx: int = 0):
        """
        Initialize the regression strategy.
        
        Args:
            scaler: Optional scaler object for denormalization
            target_col_idx: Index of the target column for denormalization
        """
        self.scaler = scaler
        self.target_col_idx = target_col_idx
        
        # Raise a warning at initialization if the scaler is missing
        if self.scaler is None:
            warnings.warn(
                "Scaler not provided to RegressionStrategy. Metrics will be on a normalized scale.",
                UserWarning,
                stacklevel=2
            )
    
    def get_criterion(self) -> nn.Module:
        return nn.MSELoss()

    def process_forward_pass(self, model: nn.Module, x: Union[torch.Tensor, Tuple[torch.Tensor, ...]], y: torch.Tensor, positions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pass positions to the model if available
        if positions is not None:
            preds = model(x, positions=positions)
        else:
            preds = model(x)
        
        # Handle shape mismatches between model output and targets
        # Model output: [B, output_len, output_dim]
        # Target shape: [B, seq_len, input_dim] or [B, seq_len]
        
        # Ensure consistent shapes for loss and metric calculation
        if y.dim() == 3:
            # Target is [B, seq_len, input_dim]
            if preds.shape[1] != y.shape[1] or preds.shape[2] != y.shape[2]:
                # Reshape predictions to match target shape
                if preds.shape[1] > y.shape[1]:
                    # Truncate predictions to match target length
                    preds = preds[:, :y.shape[1], :]
                elif preds.shape[1] < y.shape[1]:
                    # Pad predictions to match target length
                    padding = preds[:, -1:, :].repeat(1, y.shape[1] - preds.shape[1], 1)
                    preds = torch.cat([preds, padding], dim=1)
                
                # Handle output dimension mismatch
                if preds.shape[2] != y.shape[2]:
                    if preds.shape[2] > y.shape[2]:
                        preds = preds[:, :, :y.shape[2]]
                    else:
                        padding = torch.zeros(preds.shape[0], preds.shape[1], y.shape[2] - preds.shape[2], device=preds.device)
                        preds = torch.cat([preds, padding], dim=2)
        
        # Flatten for loss calculation (consistent approach)
        y_flat = y.view(y.size(0), -1)
        preds_flat = preds.view(preds.size(0), -1)
        
        loss = self.get_criterion()(preds_flat, y_flat)
        return preds, loss

    def calculate_metrics(self, preds: torch.Tensor, targets: torch.Tensor, **kwargs) -> Dict[str, float]:
        """
        Calculate regression metrics with robust denormalization for any scaler type.
        
        Args:
            preds: Model predictions
            targets: Ground truth targets
            **kwargs: Additional arguments (ignored, scaler is stored in instance)
            
        Returns:
            Dictionary containing MSE and MAE metrics
        """
        from data.timeseries_loader import InstanceNormalizer, FeatureWiseNormalizer

        y_flat = targets.view(targets.size(0), -1)
        preds_flat = preds.view(preds.size(0), -1)
        
        # Default to calculating on the normalized scale
        mse_val = metrics.mse(preds_flat, y_flat)
        mae_val = metrics.mae(preds_flat, y_flat)

        # Attempt to de-normalize only if a compatible scaler is present
        if self.scaler is not None:
            try:
                if hasattr(self.scaler, 'mean_'): # Handles GlobalStandardScaler
                    mean = self.scaler.mean_[self.target_col_idx]
                    std = self.scaler.scale_[self.target_col_idx]
                elif hasattr(self.scaler, 'feature_means'): # Handles FeatureWiseNormalizer
                    mean = self.scaler.feature_means[self.target_col_idx]
                    std = self.scaler.feature_stds[self.target_col_idx]
                else: # Handles InstanceNormalizer or unknown scalers
                    mean, std = None, None

                if mean is not None and std is not None:
                    preds_denorm = preds_flat * std + mean
                    y_denorm = y_flat * std + mean
                    mse_val = metrics.mse(preds_denorm, y_denorm)
                    mae_val = metrics.mae(preds_denorm, y_denorm)
                else:
                    warnings.warn(
                        "Calculating regression metrics on a normalized scale. De-normalization not supported for this scaler.",
                        UserWarning, stacklevel=2
                    )
            except Exception as e:
                warnings.warn(f"Failed to de-normalize metrics due to an error: {e}", UserWarning, stacklevel=2)

        return {"mse": mse_val, "mae": mae_val}


class PDEStrategy(TaskStrategy):
    """Strategy for physics-informed neural networks and PDE solving tasks.
    
    This strategy is specifically designed for PDE datasets and uses the relative L2 error
    metric, which is the standard metric for evaluating PDE solvers in the literature.
    """
    
    def __init__(self, scaler: Optional[Any] = None, target_col_idx: int = 0):
        """
        Initialize the PDE strategy.
        
        Args:
            scaler: Optional scaler object for denormalization
            target_col_idx: Index of the target column for denormalization
        """
        self.scaler = scaler
        self.target_col_idx = target_col_idx
        
        # Raise a warning at initialization if the scaler is missing
        if self.scaler is None:
            warnings.warn(
                "Scaler not provided to PDEStrategy. Metrics will be on a normalized scale.",
                UserWarning,
                stacklevel=2
            )
    
    def get_criterion(self) -> nn.Module:
        """Use MSE loss for PDE solving tasks."""
        return nn.MSELoss()

    def process_forward_pass(self, model: nn.Module, x: Union[torch.Tensor, Tuple[torch.Tensor, ...]], y: torch.Tensor, positions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process forward pass with explicit position support for PDE datasets."""
        # Pass positions to the model if available
        if positions is not None:
            preds = model(x, positions=positions)
        else:
            preds = model(x)
        
        # Handle shape mismatches between model output and targets
        # Model output: [B, output_len, output_dim]
        # Target shape: [B, seq_len, input_dim] or [B, seq_len]
        
        # Ensure consistent shapes for loss and metric calculation
        if y.dim() == 3:
            # Target is [B, seq_len, input_dim]
            if preds.shape[1] != y.shape[1] or preds.shape[2] != y.shape[2]:
                # Reshape predictions to match target shape
                if preds.shape[1] > y.shape[1]:
                    # Truncate predictions to match target length
                    preds = preds[:, :y.shape[1], :]
                elif preds.shape[1] < y.shape[1]:
                    # Pad predictions to match target length
                    padding = preds[:, -1:, :].repeat(1, y.shape[1] - preds.shape[1], 1)
                    preds = torch.cat([preds, padding], dim=1)
                
                # Handle output dimension mismatch
                if preds.shape[2] != y.shape[2]:
                    if preds.shape[2] > y.shape[2]:
                        preds = preds[:, :, :y.shape[2]]
                    else:
                        padding = torch.zeros(preds.shape[0], preds.shape[1], y.shape[2] - preds.shape[2], device=preds.device)
                        preds = torch.cat([preds, padding], dim=2)
        
        # Flatten for loss calculation (consistent approach)
        y_flat = y.view(y.size(0), -1)
        preds_flat = preds.view(preds.size(0), -1)
        
        loss = self.get_criterion()(preds_flat, y_flat)
        return preds, loss

    def calculate_metrics(self, preds: torch.Tensor, targets: torch.Tensor, **kwargs) -> Dict[str, float]:
        """
        Calculate PDE-specific metrics with robust denormalization for any scaler type.
        
        The primary metric is the relative L2 error, which is the standard metric
        for evaluating PDE solvers in the literature.
        
        Args:
            preds: Model predictions
            targets: Ground truth targets
            **kwargs: Additional arguments (ignored, scaler is stored in instance)
            
        Returns:
            Dictionary containing relative L2 error, MSE, and MAE metrics
        """
        from data.timeseries_loader import InstanceNormalizer, FeatureWiseNormalizer

        y_flat = targets.view(targets.size(0), -1)
        preds_flat = preds.view(preds.size(0), -1)
        
        # Default to calculating on the normalized scale
        relative_l2_val = metrics.relative_l2_error(preds_flat, y_flat)
        mse_val = metrics.mse(preds_flat, y_flat)
        mae_val = metrics.mae(preds_flat, y_flat)

        # Attempt to de-normalize only if a compatible scaler is present
        if self.scaler is not None:
            try:
                if hasattr(self.scaler, 'mean_'): # Handles GlobalStandardScaler
                    mean = self.scaler.mean_[self.target_col_idx]
                    std = self.scaler.scale_[self.target_col_idx]
                elif hasattr(self.scaler, 'feature_means'): # Handles FeatureWiseNormalizer
                    mean = self.scaler.feature_means[self.target_col_idx]
                    std = self.scaler.feature_stds[self.target_col_idx]
                else: # Handles InstanceNormalizer or unknown scalers
                    mean, std = None, None

                if mean is not None and std is not None:
                    preds_denorm = preds_flat * std + mean
                    y_denorm = y_flat * std + mean
                    relative_l2_val = metrics.relative_l2_error(preds_denorm, y_denorm)
                    mse_val = metrics.mse(preds_denorm, y_denorm)
                    mae_val = metrics.mae(preds_denorm, y_denorm)
                else:
                    warnings.warn(
                        "Calculating PDE metrics on a normalized scale. De-normalization not supported for this scaler.",
                        UserWarning, stacklevel=2
                    )
            except Exception as e:
                warnings.warn(f"Failed to de-normalize metrics due to an error: {e}", UserWarning, stacklevel=2)
            
        return {
            "relative_l2": relative_l2_val,
            "mse": mse_val, 
            "mae": mae_val
        }


class LanguageModelingStrategy(TaskStrategy):
    """Strategy for language modeling tasks (next token prediction)."""
    
    def get_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def process_forward_pass(self, model: nn.Module, x: Union[torch.Tensor, Tuple[torch.Tensor, ...]], y: torch.Tensor, positions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # For LM, x may be a tuple (input_ids, attention_mask)
        if isinstance(x, tuple):
            if positions is not None:
                logits = model(x[0], positions=positions)
            else:
                logits = model(x[0])
        else:
            if positions is not None:
                logits = model(x, positions=positions)
            else:
                logits = model(x)
        
        # logits: [B, L, vocab_size], y: [B, L]
        # Flatten for loss calculation
        logits_flat = logits.view(-1, logits.size(-1))
        y_flat = y.view(-1)
        
        loss = self.get_criterion()(logits_flat, y_flat)
        return logits, loss

    def calculate_metrics(self, logits: torch.Tensor, targets: torch.Tensor, scaler: Optional[Any] = None, **kwargs) -> Dict[str, float]:
        # For language modeling - next token prediction accuracy
        # logits should be in original shape [B, L, vocab_size], targets [B, L]
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        
        # Ignore padding tokens (targets == -100)
        valid_mask = targets_flat != -100
        if valid_mask.sum() == 0:
            return {"acc": 0.0}
        
        preds = torch.argmax(logits_flat, dim=-1)
        correct = (preds == targets_flat).float()
        # Only count valid tokens (not padding)
        correct = correct[valid_mask].sum().item()
        total = valid_mask.sum().item()
        accuracy = correct / total if total > 0 else 0.0
        return {"acc": accuracy} 