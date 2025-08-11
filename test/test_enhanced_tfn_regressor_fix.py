"""
Test that EnhancedTFNRegressor properly accepts and uses the new parameters.

This test verifies that the fixes for projector_type, proj_dim, and 
positional_embedding_strategy are working correctly.
"""

import torch
import torch.nn as nn
import sys
import os

# Add the model directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.tfn_enhanced import EnhancedTFNRegressor


def test_enhanced_tfn_regressor_new_params():
    """Test that EnhancedTFNRegressor accepts the new parameters."""
    print("Testing EnhancedTFNRegressor with new parameters...")
    
    # Test with standard projector
    regressor_standard = EnhancedTFNRegressor(
        input_dim=64,
        embed_dim=128,
        output_dim=32,
        output_len=10,
        num_layers=2,
        grid_size=100,
        projector_type='standard',
        positional_embedding_strategy='learned'
    )
    print("✓ Standard projector regressor created successfully")
    
    # Test with low-rank projector
    regressor_low_rank = EnhancedTFNRegressor(
        input_dim=64,
        embed_dim=128,
        output_dim=32,
        output_len=10,
        num_layers=2,
        grid_size=100,
        projector_type='low_rank',
        proj_dim=32,
        positional_embedding_strategy='sinusoidal'
    )
    print("✓ Low-rank projector regressor created successfully")
    
    # Test with different positional embedding strategies
    regressor_learned = EnhancedTFNRegressor(
        input_dim=64,
        embed_dim=128,
        output_dim=32,
        output_len=10,
        num_layers=2,
        grid_size=100,
        projector_type='low_rank',
        proj_dim=32,
        positional_embedding_strategy='learned'
    )
    print("✓ Learned positional embedding regressor created successfully")
    
    # Verify that all layers have the correct projector types
    for i, layer in enumerate(regressor_standard.layers):
        assert layer.projector_type == 'standard'
        print(f"  ✓ Layer {i} has standard projector")
    
    for i, layer in enumerate(regressor_low_rank.layers):
        assert layer.projector_type == 'low_rank'
        assert layer.proj_dim == 32
        print(f"  ✓ Layer {i} has low-rank projector with proj_dim={layer.proj_dim}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 16
    
    inputs = torch.randn(batch_size, seq_len, 64)
    
    # Test standard projector forward pass
    output_standard = regressor_standard(inputs)
    print(f"✓ Standard projector forward pass: {output_standard.shape}")
    
    # Test low-rank projector forward pass
    output_low_rank = regressor_low_rank(inputs)
    print(f"✓ Low-rank projector forward pass: {output_low_rank.shape}")
    
    # Both should produce the same output shape
    assert output_standard.shape == output_low_rank.shape
    assert output_standard.shape == (batch_size, 10, 32)
    
    print("✓ All EnhancedTFNRegressor tests passed!")


def test_positional_embedding_strategies():
    """Test different positional embedding strategies."""
    print("\nTesting different positional embedding strategies...")
    
    strategies = ['learned', 'sinusoidal']
    
    for strategy in strategies:
        try:
            regressor = EnhancedTFNRegressor(
                input_dim=32,
                embed_dim=64,
                output_dim=16,
                output_len=5,
                num_layers=1,
                grid_size=50,
                projector_type='low_rank',
                proj_dim=16,
                positional_embedding_strategy=strategy
            )
            print(f"✓ {strategy} positional embedding strategy works")
            
            # Test forward pass
            inputs = torch.randn(1, 8, 32)
            output = regressor(inputs)
            assert output.shape == (1, 5, 16)
            
        except Exception as e:
            print(f"❌ {strategy} positional embedding strategy failed: {e}")
            raise


def test_parameter_validation():
    """Test that invalid parameters are properly rejected."""
    print("\nTesting parameter validation...")
    
    # Test invalid projector type
    try:
        regressor = EnhancedTFNRegressor(
            input_dim=32,
            embed_dim=64,
            output_dim=16,
            output_len=5,
            num_layers=1,
            grid_size=50,
            projector_type='invalid_type'
        )
        assert False, "Should have raised ValueError for invalid projector_type"
    except ValueError as e:
        assert "Unknown projector_type: invalid_type" in str(e)
        print("✓ Invalid projector type correctly rejected")
    
    # Test missing proj_dim for low-rank projector
    try:
        regressor = EnhancedTFNRegressor(
            input_dim=32,
            embed_dim=64,
            output_dim=16,
            output_len=5,
            num_layers=1,
            grid_size=50,
            projector_type='low_rank'
            # Missing proj_dim should use default
        )
        print("✓ Low-rank projector with default proj_dim works")
        
        # Verify default value
        for layer in regressor.layers:
            assert layer.proj_dim == 64  # Default value
            print(f"  ✓ Default proj_dim={layer.proj_dim}")
            
    except Exception as e:
        print(f"❌ Low-rank projector with default proj_dim failed: {e}")
        raise
    
    print("✓ All parameter validation tests passed!")


def test_backward_compatibility():
    """Test that existing code continues to work."""
    print("\nTesting backward compatibility...")
    
    # Test with minimal parameters (should use defaults)
    regressor_minimal = EnhancedTFNRegressor(
        input_dim=32,
        embed_dim=64,
        output_dim=16,
        output_len=5,
        num_layers=1
    )
    print("✓ Minimal parameter regressor created successfully")
    
    # Verify default values
    for layer in regressor_minimal.layers:
        assert layer.projector_type == 'standard'  # Default
        assert layer.proj_dim == 64  # Default
        print(f"  ✓ Default projector_type={layer.projector_type}, proj_dim={layer.proj_dim}")
    
    # Test forward pass
    inputs = torch.randn(1, 8, 32)
    output = regressor_minimal(inputs)
    assert output.shape == (1, 5, 16)
    print("✓ Minimal parameter regressor forward pass works")
    
    print("✓ Backward compatibility tests passed!")


def main():
    """Run all tests."""
    print("Enhanced TFN Regressor Parameter Fix Tests")
    print("=" * 60)
    
    try:
        test_enhanced_tfn_regressor_new_params()
        test_positional_embedding_strategies()
        test_parameter_validation()
        test_backward_compatibility()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed successfully!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main() 