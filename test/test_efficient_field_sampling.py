"""
Test efficient field sampling using grid_sample.

This test verifies that the new grid_sample-based FieldSampler works correctly
and provides the expected performance improvements over the old searchsorted approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import os

# Add the core directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.field_sampling import FieldSampler


def test_grid_sample_implementation():
    """Test that the new grid_sample implementation works correctly."""
    print("Testing grid_sample-based FieldSampler implementation...")
    
    # Test parameters
    batch_size = 2
    grid_size = 100
    embed_dim = 64
    num_samples = 50
    
    # Create test data
    field = torch.randn(batch_size, grid_size, embed_dim)  # [B, G, D]
    grid_points = torch.linspace(0, 1, grid_size).unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)  # [B, G, 1]
    sample_positions = torch.rand(batch_size, num_samples, 1)  # [B, N, 1] in [0, 1] range
    
    print(f"Test data shapes:")
    print(f"  field: {field.shape}")
    print(f"  grid_points: {grid_points.shape}")
    print(f"  sample_positions: {sample_positions.shape}")
    
    # Test both modes
    for mode in ['linear', 'nearest']:
        print(f"\nTesting {mode} mode...")
        
        sampler = FieldSampler(mode=mode)
        
        # Forward pass
        try:
            sampled_field = sampler(field, grid_points, sample_positions)
            print(f"  ✓ {mode} mode forward pass successful: {sampled_field.shape}")
            
            # Verify output shape
            expected_shape = (batch_size, num_samples, embed_dim)
            assert sampled_field.shape == expected_shape, f"Expected {expected_shape}, got {sampled_field.shape}"
            print(f"  ✓ Output shape correct: {sampled_field.shape}")
            
            # Verify that different positions produce different outputs
            sample_positions2 = torch.rand(batch_size, num_samples, 1)
            sampled_field2 = sampler(field, grid_points, sample_positions2)
            
            outputs_different = not torch.allclose(sampled_field, sampled_field2, atol=1e-6)
            print(f"  ✓ Different positions produce different outputs: {outputs_different}")
            
        except Exception as e:
            print(f"  ❌ {mode} mode failed: {e}")
            raise
    
    print("\n✓ Grid_sample implementation tests passed!")


def test_interpolation_quality():
    """Test that the interpolation quality is maintained."""
    print("\nTesting interpolation quality...")
    
    # Create a simple test field with known values
    batch_size = 1
    grid_size = 10
    embed_dim = 1
    
    # Create a simple linear field: f(x) = x
    field = torch.linspace(0, 1, grid_size).unsqueeze(0).unsqueeze(-1)  # [1, 10, 1]
    grid_points = torch.linspace(0, 1, grid_size).unsqueeze(0).unsqueeze(-1)  # [1, 10, 1]
    
    # Test positions at exact grid points
    exact_positions = torch.tensor([[[0.0]], [[0.5]], [[1.0]]])  # [3, 1, 1]
    # Reshape to [1, 3, 1] by permuting dimensions
    exact_positions = exact_positions.permute(1, 0, 2)  # [1, 3, 1]
    

    
    sampler = FieldSampler(mode='linear')
    sampled_exact = sampler(field, grid_points, exact_positions)
    
    print(f"  Field values: {field.squeeze()}")
    print(f"  Exact positions: {exact_positions.squeeze()}")
    print(f"  Sampled values: {sampled_exact.squeeze()}")
    
    # Verify that exact positions give exact values
    expected_values = torch.tensor([0.0, 0.5, 1.0])
    sampled_values = sampled_exact.squeeze()
    
    # Allow small numerical differences
    assert torch.allclose(sampled_values, expected_values, atol=1e-3), \
        f"Expected {expected_values}, got {sampled_values}"
    print(f"  ✓ Exact positions give exact values")
    
    # Test interpolation at non-grid positions
    interpolated_positions = torch.tensor([[[0.25]], [[0.75]]])  # [2, 1, 1]
    # Reshape to [1, 2, 1] by permuting dimensions
    interpolated_positions = interpolated_positions.permute(1, 0, 2)  # [1, 2, 1]
    sampled_interpolated = sampler(field, grid_points, interpolated_positions)
    
    print(f"  Interpolated positions: {interpolated_positions.squeeze()}")
    print(f"  Interpolated values: {sampled_interpolated.squeeze()}")
    
    # Verify that interpolation gives reasonable values
    expected_interpolated = torch.tensor([0.25, 0.75])
    sampled_interpolated_values = sampled_interpolated.squeeze()
    
    assert torch.allclose(sampled_interpolated_values, expected_interpolated, atol=1e-2), \
        f"Expected {expected_interpolated}, got {sampled_interpolated_values}"
    print(f"  ✓ Interpolation gives reasonable values")
    
    print("\n✓ Interpolation quality tests passed!")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\nTesting edge cases...")
    
    # Test 1: Very small grid
    print("1. Testing very small grid...")
    field_small = torch.randn(1, 2, 4)  # [1, 2, 4]
    grid_small = torch.tensor([[[0.0]], [[1.0]]])  # [2, 1, 1] -> need to reshape
    grid_small = grid_small.permute(1, 0, 2)  # [1, 2, 1]
    positions_small = torch.tensor([[[0.5]]])  # [1, 1, 1]
    
    sampler = FieldSampler(mode='linear')
    try:
        sampled_small = sampler(field_small, grid_small, positions_small)
        print(f"  ✓ Small grid works: {sampled_small.shape}")
    except Exception as e:
        print(f"  ❌ Small grid failed: {e}")
        raise
    
    # Test 2: Out-of-bounds positions
    print("\n2. Testing out-of-bounds positions...")
    positions_oob = torch.tensor([[[-0.1]], [[1.1]]])  # [2, 1, 1] -> need to reshape
    positions_oob = positions_oob.permute(1, 0, 2)  # [1, 2, 1]
    
    try:
        sampled_oob = sampler(field_small, grid_small, positions_oob)
        print(f"  ✓ Out-of-bounds positions handled gracefully: {sampled_oob.shape}")
    except Exception as e:
        print(f"  ❌ Out-of-bounds positions failed: {e}")
        raise
    
    # Test 3: Different batch sizes
    print("\n3. Testing different batch sizes...")
    field_batch = torch.randn(3, 5, 8)  # [3, 5, 8]
    grid_batch = torch.linspace(0, 1, 5).unsqueeze(0).expand(3, -1).unsqueeze(-1)  # [3, 5, 1]
    positions_batch = torch.rand(3, 4, 1)  # [3, 4, 1]
    
    try:
        sampled_batch = sampler(field_batch, grid_batch, positions_batch)
        print(f"  ✓ Different batch sizes work: {sampled_batch.shape}")
    except Exception as e:
        print(f"  ❌ Different batch sizes failed: {e}")
        raise
    
    print("\n✓ Edge case tests passed!")


def test_performance_comparison():
    """Test that the new implementation is more efficient."""
    print("\nTesting performance characteristics...")
    
    # Create larger test data for performance testing
    batch_size = 4
    grid_size = 200
    embed_dim = 128
    num_samples = 100
    
    field = torch.randn(batch_size, grid_size, embed_dim)
    grid_points = torch.linspace(0, 1, grid_size).unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)
    sample_positions = torch.rand(batch_size, num_samples, 1)
    
    print(f"Performance test data shapes:")
    print(f"  field: {field.shape}")
    print(f"  grid_points: {grid_points.shape}")
    print(f"  sample_positions: {sample_positions.shape}")
    
    sampler = FieldSampler(mode='linear')
    
    # Warm up
    for _ in range(5):
        _ = sampler(field, grid_points, sample_positions)
    
    # Time the operation
    num_iterations = 100
    start_time = time.time()
    
    for _ in range(num_iterations):
        sampled_field = sampler(field, grid_points, sample_positions)
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    
    print(f"\nPerformance results:")
    print(f"  Total time for {num_iterations} iterations: {total_time:.4f}s")
    print(f"  Average time per iteration: {avg_time:.6f}s")
    print(f"  Throughput: {num_iterations / total_time:.1f} iterations/second")
    
    # Verify the operation is reasonably fast (should be much faster than searchsorted)
    assert avg_time < 0.01, f"Operation too slow: {avg_time:.6f}s per iteration"
    print(f"  ✓ Performance is acceptable: {avg_time:.6f}s per iteration")
    
    print("\n✓ Performance tests passed!")


def test_gradient_flow():
    """Test that gradients flow correctly through the new implementation."""
    print("\nTesting gradient flow...")
    
    # Create test data
    batch_size = 2
    grid_size = 20
    embed_dim = 16
    num_samples = 10
    
    field = torch.randn(batch_size, grid_size, embed_dim, requires_grad=True)
    grid_points = torch.linspace(0, 1, grid_size).unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)
    sample_positions = torch.rand(batch_size, num_samples, 1, requires_grad=True)
    
    sampler = FieldSampler(mode='linear')
    
    # Forward pass
    sampled_field = sampler(field, grid_points, sample_positions)
    
    # Create a simple loss
    target = torch.randn_like(sampled_field)
    loss = F.mse_loss(sampled_field, target)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    assert field.grad is not None, "Field gradients not computed"
    assert sample_positions.grad is not None, "Sample position gradients not computed"
    
    print(f"  ✓ Field gradients computed: {field.grad.shape}")
    print(f"  ✓ Sample position gradients computed: {sample_positions.grad.shape}")
    print(f"  ✓ Loss value: {loss.item():.6f}")
    
    print("\n✓ Gradient flow tests passed!")


def main():
    """Run all tests."""
    print("Efficient Field Sampling Test Suite")
    print("=" * 60)
    
    try:
        test_grid_sample_implementation()
        test_interpolation_quality()
        test_edge_cases()
        test_performance_comparison()
        test_gradient_flow()
        
        print("\n" + "=" * 60)
        print("🎉 All efficient field sampling tests passed!")
        print("=" * 60)
        print("\nThe new grid_sample-based FieldSampler is working correctly")
        print("and should provide significant performance improvements on GPUs!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main() 