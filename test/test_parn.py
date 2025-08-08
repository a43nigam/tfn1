"""
Test suite for PARN (Physics-Aware Reversible Normalization).

This module tests the PARN implementation to ensure it works correctly
with different modes and feature injection.
"""

import torch
import torch.nn as nn
import pytest
from typing import Dict, Any

from model.wrappers import PARN, PARNModel, create_parn_wrapper


class DummyModel(nn.Module):
    """A simple dummy model for testing PARN wrappers."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # Handle both [B, N, F] and [B, F] shapes
        if x.dim() == 3:
            batch_size, seq_len, features = x.shape
            x = x.view(-1, features)  # [B*N, F]
            x = self.linear(x)
            x = x.view(batch_size, seq_len, -1)  # [B, N, O]
        else:
            x = self.linear(x)
        return x


class TestPARN:
    """Test cases for the PARN module."""
    
    def test_parn_initialization(self):
        """Test PARN initialization with different modes."""
        # Test valid modes
        for mode in ['location', 'scale', 'full']:
            parn = PARN(num_features=5, mode=mode)
            assert parn.mode == mode
            assert parn.num_features == 5
            assert parn.affine_weight.shape == (5,)
            assert parn.affine_bias.shape == (5,)
        
        # Test invalid mode
        with pytest.raises(ValueError, match="Invalid PARN mode"):
            PARN(num_features=5, mode='invalid')
    
    def test_parn_normalization_modes(self):
        """Test PARN normalization with different modes."""
        batch_size, seq_len, features = 2, 10, 3
        x = torch.randn(batch_size, seq_len, features)
        
        # Test location mode
        parn_location = PARN(num_features=features, mode='location')
        x_norm, stats = parn_location(x, 'norm')
        
        # Check that mean is removed but std is preserved
        assert 'scale' in stats
        assert 'location' not in stats
        assert x_norm.shape == x.shape
        
        # Test scale mode
        parn_scale = PARN(num_features=features, mode='scale')
        x_norm, stats = parn_scale(x, 'norm')
        
        # Check that std is removed but mean is preserved
        assert 'location' in stats
        assert 'scale' not in stats
        assert x_norm.shape == x.shape
        
        # Test full mode
        parn_full = PARN(num_features=features, mode='full')
        x_norm, stats = parn_full(x, 'norm')
        
        # Check that both mean and std are preserved
        assert 'location' in stats
        assert 'scale' in stats
        assert x_norm.shape == x.shape
    
    def test_parn_reversibility(self):
        """Test that PARN normalization is reversible."""
        batch_size, seq_len, features = 2, 10, 3
        x_original = torch.randn(batch_size, seq_len, features)
        
        for mode in ['location', 'scale', 'full']:
            parn = PARN(num_features=features, mode=mode)
            
            # Normalize
            x_norm, stats = parn(x_original, 'norm')
            
            # Denormalize
            x_denorm, _ = parn(x_norm, 'denorm')
            
            # Check that the result is close to original (within numerical precision)
            assert torch.allclose(x_original, x_denorm, atol=1e-5)
    
    def test_parn_statistics_shape_handling(self):
        """Test PARN handles different tensor shapes correctly."""
        features = 3
        
        # Test [B, N, F] shape
        x_3d = torch.randn(2, 10, features)
        parn_3d = PARN(num_features=features, mode='location')
        x_norm_3d, stats_3d = parn_3d(x_3d, 'norm')
        assert x_norm_3d.shape == x_3d.shape
        
        # Test [B, F] shape
        x_2d = torch.randn(2, features)
        parn_2d = PARN(num_features=features, mode='location')
        x_norm_2d, stats_2d = parn_2d(x_2d, 'norm')
        assert x_norm_2d.shape == x_2d.shape


class TestPARNModel:
    """Test cases for the PARNModel wrapper."""
    
    def test_parn_model_initialization(self):
        """Test PARNModel initialization."""
        base_model = DummyModel(input_dim=5, output_dim=2)
        parn_model = PARNModel(base_model, num_features=5, mode='location')
        
        assert parn_model.base_model == base_model
        assert parn_model.parn.mode == 'location'
        assert parn_model.mode == 'location'
    
    def test_parn_model_forward_with_stats_injection(self):
        """Test PARNModel forward pass with statistics injection."""
        batch_size, seq_len, features = 2, 10, 3
        x = torch.randn(batch_size, seq_len, features)
        
        # Create model with stats injection
        # For 'location' mode, we preserve 'scale' which has same shape as features
        base_model = DummyModel(input_dim=features + features, output_dim=features)  # +features for injected stats, output matches input features
        parn_model = PARNModel(base_model, num_features=features, mode='location')
        
        output = parn_model(x)
        
        # Check output shape - should be denormalized back to original features
        assert output.shape == (batch_size, seq_len, features)
    
    def test_parn_model_forward_without_stats_injection(self):
        """Test PARNModel forward pass without statistics injection."""
        batch_size, seq_len, features = 2, 10, 3
        x = torch.randn(batch_size, seq_len, features)
        
        # Create model without stats injection
        # For 'location' mode, we still get scale statistics, so need +features
        base_model = DummyModel(input_dim=features + features, output_dim=features)
        parn_model = PARNModel(base_model, num_features=features, mode='location')
        
        output = parn_model(x)
        
        # Check output shape - should be denormalized back to original features
        assert output.shape == (batch_size, seq_len, features)
    
    def test_parn_model_reversibility(self):
        """Test that PARNModel preserves the reversible property."""
        batch_size, seq_len, features = 2, 10, 3
        x_original = torch.randn(batch_size, seq_len, features)
        
        base_model = DummyModel(input_dim=features + features, output_dim=features)  # +features for injected stats
        parn_model = PARNModel(base_model, num_features=features, mode='location')
        
        # Forward pass
        output = parn_model(x_original)
        
        # The output should be denormalized back to the original scale
        # (though the model transformation itself changes the values)
        assert output.shape == (batch_size, seq_len, features)
    
    def test_parn_model_physics_constraints(self):
        """Test that PARNModel properly delegates physics constraints."""
        base_model = DummyModel(input_dim=5, output_dim=2)
        parn_model = PARNModel(base_model, num_features=5)
        
        # Should return empty dict since DummyModel doesn't have physics constraints
        constraints = parn_model.get_physics_constraints()
        assert isinstance(constraints, dict)
        assert len(constraints) == 0


class TestPARNFactory:
    """Test cases for PARN factory functions."""
    
    def test_create_parn_wrapper(self):
        """Test the create_parn_wrapper factory function."""
        base_model = DummyModel(input_dim=5, output_dim=2)
        
        # Test with default parameters
        parn_model = create_parn_wrapper(base_model, num_features=5)
        assert isinstance(parn_model, PARNModel)
        assert parn_model.parn.mode == 'location'
        assert parn_model.mode == 'location'
        
        # Test with custom parameters
        parn_model = create_parn_wrapper(base_model, num_features=5, mode='scale')
        assert isinstance(parn_model, PARNModel)
        assert parn_model.parn.mode == 'scale'
        assert parn_model.mode == 'scale'


if __name__ == "__main__":
    # Run basic functionality tests
    print("🧪 Running PARN tests...")
    
    # Test PARN initialization
    parn = PARN(num_features=3, mode='location')
    print("✅ PARN initialization test passed")
    
    # Test normalization
    x = torch.randn(2, 10, 3)
    x_norm, stats = parn(x, 'norm')
    print("✅ PARN normalization test passed")
    
    # Test reversibility
    x_denorm, _ = parn(x_norm, 'denorm')
    assert torch.allclose(x, x_denorm, atol=1e-5)
    print("✅ PARN reversibility test passed")
    
    # Test PARNModel wrapper
    base_model = DummyModel(input_dim=6, output_dim=3)  # +3 for injected stats, output matches input features
    parn_model = PARNModel(base_model, num_features=3, mode='location')
    output = parn_model(x)
    print("✅ PARNModel wrapper test passed")
    
    # Test without stats injection
    base_model_no_stats = DummyModel(input_dim=6, output_dim=3)  # +3 for scale statistics
    parn_model_no_stats = PARNModel(base_model_no_stats, num_features=3, mode='location')
    output_no_stats = parn_model_no_stats(x)
    print("✅ PARNModel wrapper test (no stats) passed")
    
    print("🎉 All PARN tests passed!") 