"""
Test integration of LowRankFieldProjector into EnhancedTFN models.

This module tests that the new projector_type parameter works correctly
across all EnhancedTFN model classes.
"""

import torch
import torch.nn as nn
import sys
import os

# Add the model directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.tfn_enhanced import (
    EnhancedTFNLayer, 
    EnhancedTFNModel, 
    EnhancedTFNRegressor,
    create_enhanced_tfn_model
)


def test_enhanced_tfn_layer_projector_types():
    """Test that EnhancedTFNLayer can use both projector types."""
    print("Testing EnhancedTFNLayer projector types...")
    
    # Test parameters
    embed_dim = 128
    pos_dim = 1
    grid_size = 100
    proj_dim = 32
    
    # Test standard projector
    layer_standard = EnhancedTFNLayer(
        embed_dim=embed_dim,
        pos_dim=pos_dim,
        grid_size=grid_size,
        projector_type='standard'
    )
    print("✓ Standard projector created successfully")
    
    # Test low-rank projector
    layer_low_rank = EnhancedTFNLayer(
        embed_dim=embed_dim,
        pos_dim=pos_dim,
        grid_size=grid_size,
        projector_type='low_rank',
        proj_dim=proj_dim
    )
    print("✓ Low-rank projector created successfully")
    
    # Verify projector types
    assert layer_standard.projector_type == 'standard'
    assert layer_low_rank.projector_type == 'low_rank'
    assert layer_low_rank.proj_dim == proj_dim
    
    # Test forward pass with both
    batch_size = 2
    num_tokens = 16
    
    embeddings = torch.randn(batch_size, num_tokens, embed_dim)
    positions = torch.randn(batch_size, num_tokens, pos_dim)
    
    # Standard projector forward pass
    output_standard = layer_standard(embeddings, positions)
    print(f"✓ Standard projector forward pass: {output_standard.shape}")
    
    # Low-rank projector forward pass
    output_low_rank = layer_low_rank(embeddings, positions)
    print(f"✓ Low-rank projector forward pass: {output_low_rank.shape}")
    
    # Both should produce the same output shape
    assert output_standard.shape == output_low_rank.shape
    assert output_standard.shape == (batch_size, num_tokens, embed_dim)
    
    print("✓ All EnhancedTFNLayer tests passed!")


def test_enhanced_tfn_model_projector_types():
    """Test that EnhancedTFNModel can use both projector types."""
    print("\nTesting EnhancedTFNModel projector types...")
    
    # Test parameters
    vocab_size = 1000
    embed_dim = 128
    num_layers = 2
    grid_size = 100
    proj_dim = 32
    
    # Test standard projector
    model_standard = EnhancedTFNModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        grid_size=grid_size,
        projector_type='standard'
    )
    print("✓ Standard projector model created successfully")
    
    # Test low-rank projector
    model_low_rank = EnhancedTFNModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        grid_size=grid_size,
        projector_type='low_rank',
        proj_dim=proj_dim
    )
    print("✓ Low-rank projector model created successfully")
    
    # Verify all layers have correct projector types
    for i, layer in enumerate(model_standard.layers):
        assert layer.projector_type == 'standard'
    
    for i, layer in enumerate(model_low_rank.layers):
        assert layer.projector_type == 'low_rank'
        assert layer.proj_dim == proj_dim
    
    # Test forward pass with both
    batch_size = 2
    seq_len = 16
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Standard projector forward pass
    output_standard = model_standard(input_ids)
    print(f"✓ Standard projector model forward pass: {output_standard.shape}")
    
    # Low-rank projector forward pass
    output_low_rank = model_low_rank(input_ids)
    print(f"✓ Low-rank projector model forward pass: {output_low_rank.shape}")
    
    # Both should produce the same output shape
    assert output_standard.shape == output_low_rank.shape
    assert output_standard.shape == (batch_size, seq_len, vocab_size)
    
    print("✓ All EnhancedTFNModel tests passed!")


def test_enhanced_tfn_regressor_projector_types():
    """Test that EnhancedTFNRegressor can use both projector types."""
    print("\nTesting EnhancedTFNRegressor projector types...")
    
    # Test parameters
    input_dim = 64
    embed_dim = 128
    output_dim = 32
    output_len = 10
    num_layers = 2
    grid_size = 100
    proj_dim = 32
    
    # Test standard projector
    regressor_standard = EnhancedTFNRegressor(
        input_dim=input_dim,
        embed_dim=embed_dim,
        output_dim=output_dim,
        output_len=output_len,
        num_layers=num_layers,
        grid_size=grid_size,
        projector_type='standard'
    )
    print("✓ Standard projector regressor created successfully")
    
    # Test low-rank projector
    regressor_low_rank = EnhancedTFNRegressor(
        input_dim=input_dim,
        embed_dim=embed_dim,
        output_dim=output_dim,
        output_len=output_len,
        num_layers=num_layers,
        grid_size=grid_size,
        projector_type='low_rank',
        proj_dim=proj_dim
    )
    print("✓ Low-rank projector regressor created successfully")
    
    # Verify all layers have correct projector types
    for i, layer in enumerate(regressor_standard.layers):
        assert layer.projector_type == 'standard'
    
    for i, layer in enumerate(regressor_low_rank.layers):
        assert layer.projector_type == 'low_rank'
        assert layer.proj_dim == proj_dim
    
    # Test forward pass with both
    batch_size = 2
    seq_len = 16
    
    inputs = torch.randn(batch_size, seq_len, input_dim)
    
    # Standard projector forward pass
    output_standard = regressor_standard(inputs)
    print(f"✓ Standard projector regressor forward pass: {output_standard.shape}")
    
    # Low-rank projector forward pass
    output_low_rank = regressor_low_rank(inputs)
    print(f"✓ Low-rank projector regressor forward pass: {output_low_rank.shape}")
    
    # Both should produce the same output shape
    assert output_standard.shape == output_low_rank.shape
    assert output_standard.shape == (batch_size, output_len, output_dim)
    
    print("✓ All EnhancedTFNRegressor tests passed!")


def test_create_enhanced_tfn_model():
    """Test the factory function with different projector types."""
    print("\nTesting create_enhanced_tfn_model factory function...")
    
    # Test parameters
    vocab_size = 1000
    embed_dim = 128
    num_layers = 2
    grid_size = 100
    proj_dim = 32
    
    # Test standard projector
    model_standard = create_enhanced_tfn_model(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        grid_size=grid_size,
        projector_type='standard'
    )
    print("✓ Factory function created standard projector model")
    
    # Test low-rank projector
    model_low_rank = create_enhanced_tfn_model(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        grid_size=grid_size,
        projector_type='low_rank',
        proj_dim=proj_dim
    )
    print("✓ Factory function created low-rank projector model")
    
    # Verify projector types
    for layer in model_standard.layers:
        assert layer.projector_type == 'standard'
    
    for layer in model_low_rank.layers:
        assert layer.projector_type == 'low_rank'
        assert layer.proj_dim == proj_dim
    
    print("✓ Factory function tests passed!")


def test_invalid_projector_type():
    """Test that invalid projector types raise appropriate errors."""
    print("\nTesting invalid projector type handling...")
    
    try:
        layer = EnhancedTFNLayer(
            embed_dim=128,
            pos_dim=2,
            grid_size=100,
            projector_type='invalid_type'
        )
        assert False, "Should have raised ValueError for invalid projector_type"
    except ValueError as e:
        assert "Unknown projector_type: invalid_type" in str(e)
        print("✓ Invalid projector type correctly rejected")
    
    print("✓ Invalid projector type handling tests passed!")


def main():
    """Run all integration tests."""
    print("Enhanced TFN Low-Rank Projector Integration Tests")
    print("=" * 60)
    
    try:
        test_enhanced_tfn_layer_projector_types()
        test_enhanced_tfn_model_projector_types()
        test_enhanced_tfn_regressor_projector_types()
        test_create_enhanced_tfn_model()
        test_invalid_projector_type()
        
        print("\n" + "=" * 60)
        print("🎉 All integration tests passed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main() 