"""
Core TFN components.

This module contains the fundamental building blocks of the Token Field Network:
- Kernel system for field emission
- Field projection for continuous representation
- Field evolution for temporal dynamics
- Field interference for token interactions
- Unified field dynamics for integrated evolution
- Grid generation for spatial discretization
"""

from .kernels import (
    KernelBasis, RBFKernel, CompactKernel, FourierKernel, 
    KernelFactory
)
from .field_projection import FieldProjector, UniformFieldGrid
from .field_evolution import (
    FieldEvolver, CNNFieldEvolver, 
    PDEFieldEvolver, TemporalGrid, create_field_evolver
)
from .field_interference import (
    TokenFieldInterference, CausalFieldInterference, 
    MultiScaleFieldInterference,
    create_field_interference
)
from .unified_field_dynamics import (
    UnifiedFieldDynamics, DiffusionOperator, WaveOperator,
    SchrodingerOperator, StabilityMonitor
)
from .field_sampling import FieldSampler

__all__ = [
    # Kernel system
    'KernelBasis', 'RBFKernel', 'CompactKernel', 'FourierKernel', 
    'KernelFactory',
    # Field projection
    'FieldProjector', 'UniformFieldGrid',
    # Field evolution
    'FieldEvolver', 'CNNFieldEvolver', 
    'PDEFieldEvolver', 'TemporalGrid', 'create_field_evolver'
] 