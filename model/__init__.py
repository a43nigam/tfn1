"""
TFN Model Package

Clean, reusable model classes for Token Field Networks.
"""

from .tfn_base import (
    LearnableKernels,
    TrainableEvolution,
    PositionEmbeddings,
    TrainableTFNLayer
)

from .tfn_unified import TFN, UnifiedTFN  # UnifiedTFN is now alias of TFN
from .tfn_pytorch import ImageTFN

# ---------------------------------------------------------------------------
# Backward-compatibility aliases (deprecated)
# ---------------------------------------------------------------------------

# Legacy aliases for backward compatibility
TFNClassifier = TFN
TFNRegressor = TFN

__all__ = [
    # Base components
    'LearnableKernels',
    'TrainableEvolution', 
    'PositionEmbeddings',
    'TrainableTFNLayer',
    
    # Unified model
    'TFN', 'UnifiedTFN', 'TFNClassifier', 'TFNRegressor',
    'ImageTFN',
]
