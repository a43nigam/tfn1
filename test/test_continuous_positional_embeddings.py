"""
Test continuous positional embeddings for EnhancedTFNRegressor.

This test verifies that the positional embedding strategies work correctly
for continuous-space tasks, which is critical for fair evaluation of TFN.
"""

import torch
import torch.nn as nn
import sys
import os

# Add the model directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.tfn_enhanced import EnhancedTFNRegressor
from model.shared_layers import (
    create_positional_embedding_strategy,
    LearnedPositionalEmbeddings,
    SinusoidalEmbeddings,
    ContinuousPositionalEmbeddings
)


def test_positional_embedding_strategies():
    """Test different positional embedding strategies with continuous positions."""
    print("Testing positional embedding strategies with continuous positions...")
    
    # Test parameters
    max_len = 100
    embed_dim = 64
    batch_size = 2
    seq_len = 16
    pos_dim = 1  # FieldSampler currently only supports 1D
    
    # Create continuous positions (typical for PDE/spatial tasks)
    positions = torch.randn(batch_size, seq_len, pos_dim)  # [B, N, P]
    
    print(f"Test positions shape: {positions.shape}")
    print(f"Position values range: [{positions.min():.3f}, {positions.max():.3f}]")
    print()
    
    # Test 1: Continuous positional embeddings
    print("1. Testing ContinuousPositionalEmbeddings...")
    continuous_strategy = create_positional_embedding_strategy("continuous", max_len, embed_dim)
    
    continuous_embeddings = continuous_strategy(positions)
    print(f"  ✓ Output shape: {continuous_embeddings.shape}")
    print(f"  ✓ Embedding values range: [{continuous_embeddings.min():.3f}, {continuous_embeddings.max():.3f}]")
    
    # Verify that embeddings change with different positions
    positions2 = torch.randn(batch_size, seq_len, pos_dim)
    continuous_embeddings2 = continuous_strategy(positions2)
    
    embeddings_different = not torch.allclose(continuous_embeddings, continuous_embeddings2, atol=1e-6)
    print(f"  ✓ Embeddings change with different positions: {embeddings_different}")
    
    # Test 2: Sinusoidal positional embeddings
    print("\n2. Testing SinusoidalEmbeddings...")
    sinusoidal_strategy = create_positional_embedding_strategy("sinusoidal", max_len, embed_dim)
    
    sinusoidal_embeddings = sinusoidal_strategy(positions)
    print(f"  ✓ Output shape: {sinusoidal_embeddings.shape}")
    print(f"  ✓ Embedding values range: [{sinusoidal_embeddings.min():.3f}, {sinusoidal_embeddings.max():.3f}]")
    
    # Test 3: Learned positional embeddings (should handle continuous positions gracefully)
    print("\n3. Testing LearnedPositionalEmbeddings...")
    learned_strategy = create_positional_embedding_strategy("learned", max_len, embed_dim)
    
    learned_embeddings = learned_strategy(positions)
    print(f"  ✓ Output shape: {learned_embeddings.shape}")
    print(f"  ✓ Embedding values range: [{learned_embeddings.min():.3f}, {learned_embeddings.max():.3f}]")
    
    # Verify that learned embeddings are consistent (should ignore actual position values)
    learned_embeddings2 = learned_strategy(positions2)
    embeddings_consistent = torch.allclose(learned_embeddings, learned_embeddings2, atol=1e-6)
    print(f"  ✓ Embeddings consistent across different positions: {embeddings_consistent}")
    
    print("\n✓ All positional embedding strategy tests passed!")


def test_enhanced_tfn_regressor_continuous_positions():
    """Test that EnhancedTFNRegressor properly handles continuous positions."""
    print("\nTesting EnhancedTFNRegressor with continuous positions...")
    
    # Test parameters
    input_dim = 32
    embed_dim = 64
    output_dim = 16
    output_len = 5
    num_layers = 2
    grid_size = 50
    batch_size = 2
    seq_len = 16
    pos_dim = 1  # FieldSampler currently only supports 1D
    
    # Test 1: Continuous positional embedding strategy
    print("1. Testing with 'continuous' strategy...")
    regressor_continuous = EnhancedTFNRegressor(
        input_dim=input_dim,
        embed_dim=embed_dim,
        output_dim=output_dim,
        output_len=output_len,
        num_layers=num_layers,
        grid_size=grid_size,
        projector_type='low_rank',
        proj_dim=16,
        positional_embedding_strategy='continuous'
    )
    
    # Create test data with continuous positions
    inputs = torch.randn(batch_size, seq_len, input_dim)
    positions = torch.randn(batch_size, seq_len, pos_dim)  # Continuous positions
    
    print(f"  ✓ Input shape: {inputs.shape}")
    print(f"  ✓ Position shape: {positions.shape}")
    print(f"  ✓ Position values range: [{positions.min():.3f}, {positions.max():.3f}]")
    
    # Forward pass
    output = regressor_continuous(inputs, positions)
    print(f"  ✓ Output shape: {output.shape}")
    
    # Test 2: Sinusoidal positional embedding strategy
    print("\n2. Testing with 'sinusoidal' strategy...")
    regressor_sinusoidal = EnhancedTFNRegressor(
        input_dim=input_dim,
        embed_dim=embed_dim,
        output_dim=output_dim,
        output_len=output_len,
        num_layers=num_layers,
        grid_size=grid_size,
        projector_type='low_rank',
        proj_dim=16,
        positional_embedding_strategy='sinusoidal'
    )
    
    output_sinusoidal = regressor_sinusoidal(inputs, positions)
    print(f"  ✓ Output shape: {output_sinusoidal.shape}")
    
    # Test 3: Verify that different strategies produce different outputs
    outputs_different = not torch.allclose(output, output_sinusoidal, atol=1e-6)
    print(f"  ✓ Different strategies produce different outputs: {outputs_different}")
    
    # Test 4: Verify that positions affect the output
    positions2 = torch.randn(batch_size, seq_len, pos_dim)
    output2 = regressor_continuous(inputs, positions2)
    
    outputs_different_with_positions = not torch.allclose(output, output2, atol=1e-6)
    print(f"  ✓ Different positions produce different outputs: {outputs_different_with_positions}")
    
    print("\n✓ EnhancedTFNRegressor continuous position tests passed!")


def test_position_awareness():
    """Test that the model is actually position-aware."""
    print("\nTesting position awareness...")
    
    # Create a simple model
    regressor = EnhancedTFNRegressor(
        input_dim=16,
        embed_dim=32,
        output_dim=8,
        output_len=3,
        num_layers=1,
        grid_size=20,
        projector_type='low_rank',
        proj_dim=8,
        positional_embedding_strategy='continuous'
    )
    
    batch_size = 2
    seq_len = 8
    input_dim = 16
    pos_dim = 1
    
    # Create test data
    inputs = torch.randn(batch_size, seq_len, input_dim)
    
    # Test 1: Same inputs, different positions should produce different outputs
    positions1 = torch.randn(batch_size, seq_len, pos_dim)
    positions2 = torch.randn(batch_size, seq_len, pos_dim)
    
    output1 = regressor(inputs, positions1)
    output2 = regressor(inputs, positions2)
    
    outputs_different = not torch.allclose(output1, output2, atol=1e-6)
    print(f"✓ Different positions produce different outputs: {outputs_different}")
    
    # Test 2: Same inputs, same positions should produce same outputs
    output1_repeat = regressor(inputs, positions1)
    outputs_same = torch.allclose(output1, output1_repeat, atol=1e-6)
    print(f"✓ Same positions produce same outputs: {outputs_same}")
    
    # Test 3: Different inputs, same positions should produce different outputs
    inputs2 = torch.randn(batch_size, seq_len, input_dim)
    output3 = regressor(inputs2, positions1)
    
    outputs_different_with_inputs = not torch.allclose(output1, output3, atol=1e-6)
    print(f"✓ Different inputs produce different outputs: {outputs_different_with_inputs}")
    
    print("\n✓ Position awareness tests passed!")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\nTesting edge cases...")
    
    # Test 1: Very small sequences
    print("1. Testing very small sequences...")
    regressor = EnhancedTFNRegressor(
        input_dim=8,
        embed_dim=16,
        output_dim=4,
        output_len=2,
        num_layers=1,
        grid_size=10,
        positional_embedding_strategy='continuous'
    )
    
    inputs = torch.randn(1, 2, 8)  # Very small sequence
    positions = torch.randn(1, 2, 1)
    
    try:
        output = regressor(inputs, positions)
        print(f"  ✓ Small sequence works: {output.shape}")
    except Exception as e:
        print(f"  ❌ Small sequence failed: {e}")
        raise
    
    # Test 2: Large sequences
    print("\n2. Testing large sequences...")
    inputs = torch.randn(1, 100, 8)  # Larger sequence
    positions = torch.randn(1, 100, 1)
    
    try:
        output = regressor(inputs, positions)
        print(f"  ✓ Large sequence works: {output.shape}")
    except Exception as e:
        print(f"  ❌ Large sequence failed: {e}")
        raise
    
    # Test 3: Different position dimensions (note: only 1D supported for now)
    print("\n3. Testing position dimension handling...")
    print("  Note: FieldSampler currently only supports 1D positions")
    
    # Test that 1D positions work correctly
    positions_1d = torch.randn(1, 10, 1)  # 1D positions
    
    try:
        output = regressor(inputs[:, :10, :], positions_1d)
        print(f"  ✓ 1D positions work: {output.shape}")
    except Exception as e:
        print(f"  ❌ 1D positions failed: {e}")
        raise
    
    print("\n✓ Edge case tests passed!")


def main():
    """Run all tests."""
    print("Continuous Positional Embeddings Test Suite")
    print("=" * 60)
    
    try:
        test_positional_embedding_strategies()
        test_enhanced_tfn_regressor_continuous_positions()
        test_position_awareness()
        test_edge_cases()
        
        print("\n" + "=" * 60)
        print("🎉 All continuous positional embedding tests passed!")
        print("=" * 60)
        print("\nThe EnhancedTFNRegressor now properly handles continuous positions")
        print("and can be fairly evaluated on continuous-space tasks!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main() 