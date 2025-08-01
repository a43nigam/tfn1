"""
Base model interface for all sequence models.

This module defines the abstract base class that all sequence models must inherit from,
ensuring a consistent interface across different architectures.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Optional, Any


class BaseSequenceModel(nn.Module, ABC):
    """Abstract base class for all sequence models.
    
    This class enforces a consistent interface for all sequence models,
    making them truly interchangeable in the training pipeline.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """Initialize the model.
        
        All models must accept the same core parameters:
        - task: 'classification' or 'regression'
        - embed_dim: Embedding dimension
        - positional_embedding_strategy: Strategy for positional embeddings
        - max_seq_len: Maximum sequence length
        """
        super().__init__()

    @abstractmethod
    def forward(self, 
                inputs: torch.Tensor, 
                positions: Optional[torch.Tensor] = None,
                calendar_features: Optional[Dict[str, torch.Tensor]] = None,
                **kwargs) -> torch.Tensor:
        """
        Defines the forward pass for any sequence model.
        
        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor. Shape depends on task:
            - Classification: [batch_size, seq_len] (token indices)
            - Regression: [batch_size, seq_len, input_dim] (feature vectors)
        positions : Optional[torch.Tensor], default=None
            Optional positional information. Shape: [batch_size, seq_len, pos_dim]
            If None, the model should generate positions automatically.
        calendar_features : Optional[Dict[str, torch.Tensor]], default=None
            Optional calendar features for time-based positional embeddings.
            Keys are feature names (e.g., "hour", "day_of_week"), values are tensors
            of shape [batch_size, seq_len].
        **kwargs
            Additional keyword arguments for model-specific parameters.
            
        Returns
        -------
        torch.Tensor
            Model output. Shape depends on task:
            - Classification: [batch_size, num_classes]
            - Regression: [batch_size, output_len, output_dim]
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """
        Return model information for logging and debugging.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing model metadata like parameter count,
            architecture details, etc.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_type": self.__class__.__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
        }

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 