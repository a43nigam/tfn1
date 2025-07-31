"""
TFN Model Package

Clean, reusable model classes for Token Field Networks.
"""

from .tfn_unified import TFN, UnifiedTFN  # UnifiedTFN is now alias of TFN
from .tfn_enhanced import EnhancedTFNModel, EnhancedTFNRegressor
from .tfn_pytorch import ImageTFN

__all__ = [
    'TFN', 'UnifiedTFN', 'EnhancedTFNModel', 'EnhancedTFNRegressor', 'ImageTFN',
]
