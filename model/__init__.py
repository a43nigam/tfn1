"""
TFN Model Package

Clean, reusable model classes for Token Field Networks.
"""

from .tfn_base import (
    LearnableKernels,
    TrainableEvolution,
    TrainableTFNLayer
)

from .tfn_unified import TFN, UnifiedTFN  # UnifiedTFN is now alias of TFN
from .tfn_pytorch import ImageTFN

__all__ = [
    'LearnableKernels',
    'TrainableEvolution',
    'TrainableTFNLayer',
    'TFN', 'UnifiedTFN', 'ImageTFN',
]
