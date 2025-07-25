"""
Essential utility functions for TFN kernel testing.

Provides shape validation, gradient testing, and numerical stability checks
that are used for testing the kernel system.
"""

from typing import Tuple
import torch
import torch.nn as nn


def validate_shapes(tensor: torch.Tensor, 
                   expected_shape: Tuple[int, ...], 
                   name: str) -> None:
    """
    Validate tensor shapes with clear error messages.
    
    Args:
        tensor: Tensor to validate
        expected_shape: Expected shape tuple
        name: Name of the tensor for error messages
        
    Raises:
        ValueError: If tensor shape doesn't match expected shape
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor, got {type(tensor)}")
    
    if tensor.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, "
                       f"got {tensor.shape}")


def check_gradient_flow(test_module: nn.Module, 
                      input_tensor: torch.Tensor,
                      atol: float = 1e-6,
                      rtol: float = 1e-5) -> bool:
    """
    Test gradient flow through a module using torch.autograd.gradcheck.
    
    Args:
        test_module: Module to test
        input_tensor: Input tensor for testing
        atol: Absolute tolerance for gradient check
        rtol: Relative tolerance for gradient check
        
    Returns:
        True if gradient check passes, False otherwise
    """
    try:
        # Ensure input requires gradients
        if not input_tensor.requires_grad:
            input_tensor = input_tensor.detach().requires_grad_(True)
        
        # Define a simple loss function for testing
        def test_function(x):
            output = test_module(x)
            return output.sum()
        
        # Run gradient check
        result = torch.autograd.gradcheck(
            test_function, 
            input_tensor, 
            atol=atol, 
            rtol=rtol,
            raise_exception=False
        )
        
        return result
        
    except Exception as e:
        print(f"Gradient test failed with exception: {e}")
        return False


def check_numerical_stability(tensor: torch.Tensor, 
                             name: str,
                             check_inf: bool = True,
                             check_nan: bool = True) -> None:
    """
    Check for numerical stability issues in a tensor.
    
    Args:
        tensor: Tensor to check
        name: Name of the tensor for error messages
        check_inf: Whether to check for infinite values
        check_nan: Whether to check for NaN values
        
    Raises:
        ValueError: If numerical issues are detected
    """
    if check_nan and torch.isnan(tensor).any():
        raise ValueError(f"{name} contains NaN values")
    
    if check_inf and torch.isinf(tensor).any():
        raise ValueError(f"{name} contains infinite values")
    
    # Check for extremely large values that might indicate instability
    if torch.abs(tensor).max() > 1e6:
        print(f"Warning: {name} contains very large values (max: {torch.abs(tensor).max():.2e})")


def create_test_tensor(shape: Tuple[int, ...], 
                      device: torch.device = torch.device('cpu'),
                      requires_grad: bool = True) -> torch.Tensor:
    """
    Create a test tensor with given shape for testing modules.
    
    Args:
        shape: Desired tensor shape
        device: Device to create tensor on
        requires_grad: Whether tensor requires gradients
        
    Returns:
        Test tensor with small random values
    """
    return torch.randn(shape, device=device, requires_grad=requires_grad) * 0.1


def log_tensor_info(tensor: torch.Tensor, name: str) -> None:
    """
    Log information about a tensor for debugging.
    
    Args:
        tensor: Tensor to log information about
        name: Name of the tensor
    """
    print(f"{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print(f"  Requires grad: {tensor.requires_grad}")
    print(f"  Min: {tensor.min().item():.6f}")
    print(f"  Max: {tensor.max().item():.6f}")
    print(f"  Mean: {tensor.mean().item():.6f}")
    print(f"  Std: {tensor.std().item():.6f}")
    print() 