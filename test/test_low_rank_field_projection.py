"""
Test suite for LowRankFieldProjector class.

This module tests the memory-efficient low-rank field projection implementation
to ensure it works correctly and provides the expected memory savings.
"""

import torch
import torch.nn as nn
import pytest
import sys
import os

# Add the core directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.field_projection import LowRankFieldProjector, FieldProjector
from core.kernels import RBFKernel


class TestLowRankFieldProjector:
    """Test cases for LowRankFieldProjector class."""
    
    @pytest.fixture
    def basic_params(self):
        """Basic parameters for testing."""
        return {
            'embed_dim': 128,
            'pos_dim': 2,
            'kernel_type': 'rbf',
            'proj_dim': 32
        }
    
    @pytest.fixture
    def sample_data(self, basic_params):
        """Sample data for testing."""
        batch_size = 2
        num_tokens = 8
        grid_size = 16
        
        torch.manual_seed(42)
        
        embeddings = torch.randn(batch_size, num_tokens, basic_params['embed_dim'])
        positions = torch.randn(batch_size, num_tokens, basic_params['pos_dim'])
        grid_points = torch.randn(batch_size, grid_size, basic_params['pos_dim'])
        
        return embeddings, positions, grid_points
    
    def test_initialization(self, basic_params):
        """Test that LowRankFieldProjector initializes correctly."""
        projector = LowRankFieldProjector(**basic_params)
        
        assert projector.embed_dim == basic_params['embed_dim']
        assert projector.pos_dim == basic_params['pos_dim']
        assert projector.proj_dim == basic_params['proj_dim']
        assert projector.kernel_type == basic_params['kernel_type']
        
        # Check that projection layers are created
        assert isinstance(projector.embedding_projector, nn.Linear)
        assert isinstance(projector.field_upsampler, nn.Linear)
        
        # Check layer dimensions
        assert projector.embedding_projector.in_features == basic_params['embed_dim']
        assert projector.embedding_projector.out_features == basic_params['proj_dim']
        assert projector.field_upsampler.in_features == basic_params['proj_dim']
        assert projector.field_upsampler.out_features == basic_params['embed_dim']
    
    def test_forward_pass(self, basic_params, sample_data):
        """Test the forward pass of LowRankFieldProjector."""
        projector = LowRankFieldProjector(**basic_params)
        embeddings, positions, grid_points = sample_data
        
        # Forward pass
        output = projector(embeddings, positions, grid_points)
        
        # Check output shape
        expected_shape = (embeddings.shape[0], grid_points.shape[1], basic_params['embed_dim'])
        assert output.shape == expected_shape
        
        # Check that output is not all zeros (basic functionality check)
        assert not torch.allclose(output, torch.zeros_like(output))
    
    def test_memory_savings_calculation(self, basic_params):
        """Test memory savings calculation."""
        projector = LowRankFieldProjector(**basic_params)
        
        batch_size = 2
        num_tokens = 8
        grid_size = 16
        
        memory_info = projector.get_memory_savings(batch_size, num_tokens, grid_size)
        
        # Check that all required keys are present
        required_keys = ['standard_memory', 'low_rank_memory', 'memory_savings', 
                        'savings_ratio', 'compression_factor']
        for key in required_keys:
            assert key in memory_info
        
        # Check that memory savings are positive
        assert memory_info['memory_savings'] > 0
        assert memory_info['savings_ratio'] > 0
        assert memory_info['compression_factor'] > 1
        
        # Verify calculations
        expected_standard = batch_size * num_tokens * basic_params['embed_dim'] * grid_size
        assert memory_info['standard_memory'] == expected_standard
        
        expected_low_rank = (batch_size * num_tokens * basic_params['proj_dim'] + 
                           batch_size * grid_size * basic_params['proj_dim'] + 
                           batch_size * grid_size * basic_params['embed_dim'])
        assert memory_info['low_rank_memory'] == expected_low_rank
    
    def test_token_influence_computation(self, basic_params, sample_data):
        """Test individual token influence computation."""
        projector = LowRankFieldProjector(**basic_params)
        embeddings, positions, grid_points = sample_data
        
        # Compute token influences
        token_influences = projector.compute_token_influence(
            embeddings, positions, grid_points
        )
        
        # Check output shape: [B, N, M, D]
        expected_shape = (embeddings.shape[0], embeddings.shape[1], 
                         grid_points.shape[1], basic_params['embed_dim'])
        assert token_influences.shape == expected_shape
        
        # Check that influences are not all zeros
        assert not torch.allclose(token_influences, torch.zeros_like(token_influences))
    
    def test_different_kernel_types(self):
        """Test initialization with different kernel types."""
        kernel_types = ['rbf', 'compact', 'fourier', 'multi_frequency_fourier']
        
        for kernel_type in kernel_types:
            try:
                projector = LowRankFieldProjector(
                    embed_dim=64,
                    pos_dim=2,
                    kernel_type=kernel_type,
                    proj_dim=16
                )
                assert projector.kernel_type == kernel_type
            except Exception as e:
                pytest.fail(f"Failed to initialize with kernel type '{kernel_type}': {e}")
    
    def test_data_dependent_kernels(self):
        """Test initialization with data-dependent kernel types."""
        kernel_types = ['data_dependent_rbf', 'data_dependent_compact', 'film_learnable']
        
        for kernel_type in kernel_types:
            try:
                projector = LowRankFieldProjector(
                    embed_dim=64,
                    pos_dim=2,
                    kernel_type=kernel_type,
                    proj_dim=16
                )
                assert projector.kernel_type == kernel_type
            except Exception as e:
                pytest.fail(f"Failed to initialize with data-dependent kernel type '{kernel_type}': {e}")
    
    def test_grid_points_batch_dimension_handling(self, basic_params, sample_data):
        """Test handling of grid points with and without batch dimension."""
        projector = LowRankFieldProjector(**basic_params)
        embeddings, positions, _ = sample_data
        
        # Test with grid points without batch dimension
        grid_size = 16  # Use local variable instead of basic_params
        grid_points_no_batch = torch.randn(grid_size, basic_params['pos_dim'])
        
        # Test with grid points with batch dimension
        grid_points_with_batch = torch.randn(embeddings.shape[0], grid_size, basic_params['pos_dim'])
        
        # Both should work
        output_no_batch = projector(embeddings, positions, grid_points_no_batch)
        output_with_batch = projector(embeddings, positions, grid_points_with_batch)
        
        # Check shapes
        assert output_no_batch.shape == (embeddings.shape[0], grid_size, basic_params['embed_dim'])
        assert output_with_batch.shape == (embeddings.shape[0], grid_size, basic_params['embed_dim'])
        
        # Check that outputs are not all zeros (basic functionality)
        assert not torch.allclose(output_no_batch, torch.zeros_like(output_no_batch))
        assert not torch.allclose(output_with_batch, torch.zeros_like(output_with_batch))
        
        # Check that outputs have reasonable values (not NaN or inf)
        assert not torch.isnan(output_no_batch).any()
        assert not torch.isnan(output_with_batch).any()
        assert not torch.isinf(output_no_batch).any()
        assert not torch.isinf(output_with_batch).any()
    
    def test_gradient_flow(self, basic_params, sample_data):
        """Test that gradients flow through the network."""
        projector = LowRankFieldProjector(**basic_params)
        embeddings, positions, grid_points = sample_data
        
        # Enable gradient computation
        embeddings.requires_grad_(True)
        positions.requires_grad_(True)
        
        # Forward pass
        output = projector(embeddings, positions, grid_points)
        
        # Compute loss and backward pass
        loss = output.sum()
        loss.backward()
        
        # Check that gradients are computed
        assert embeddings.grad is not None
        assert positions.grad is not None
        
        # Check that gradients are not all zeros
        assert not torch.allclose(embeddings.grad, torch.zeros_like(embeddings.grad))
        assert not torch.allclose(positions.grad, torch.zeros_like(positions.grad))
    
    def test_device_consistency(self, basic_params, sample_data):
        """Test that the projector works on different devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        projector = LowRankFieldProjector(**basic_params)
        embeddings, positions, grid_points = sample_data
        
        # Move to CUDA
        projector = projector.cuda()
        embeddings_cuda = embeddings.cuda()
        positions_cuda = positions.cuda()
        grid_points_cuda = grid_points.cuda()
        
        # Forward pass on CUDA
        output_cuda = projector(embeddings_cuda, positions_cuda, grid_points_cuda)
        
        # Check output device
        assert output_cuda.device.type == 'cuda'
        assert output_cuda.shape == (embeddings.shape[0], grid_points.shape[1], basic_params['embed_dim'])


if __name__ == "__main__":
    # Run basic tests if executed directly
    print("Testing LowRankFieldProjector...")
    
    # Define test parameters directly
    basic_params = {
        'embed_dim': 128,
        'pos_dim': 2,
        'grid_size': 100,
        'kernel_type': 'rbf',
        'proj_dim': 32
    }
    
    # Generate sample data
    torch.manual_seed(42)
    batch_size = 4
    num_tokens = 16
    embeddings = torch.randn(batch_size, num_tokens, basic_params['embed_dim'])
    positions = torch.randn(batch_size, num_tokens, basic_params['pos_dim'])
    grid_points = torch.randn(batch_size, basic_params['grid_size'], basic_params['pos_dim'])
    
    # Test initialization
    projector = LowRankFieldProjector(**basic_params)
    print("✓ Initialization successful")
    
    # Test forward pass
    output = projector(embeddings, positions, grid_points)
    print(f"✓ Forward pass successful, output shape: {output.shape}")
    
    # Test memory savings
    memory_info = projector.get_memory_savings(4, 16, 100)
    print(f"✓ Memory savings calculated:")
    print(f"  - Standard memory: {memory_info['standard_memory']:,}")
    print(f"  - Low-rank memory: {memory_info['low_rank_memory']:,}")
    print(f"  - Memory savings: {memory_info['memory_savings']:,}")
    print(f"  - Compression factor: {memory_info['compression_factor']:.2f}x")
    
    # Test token influence computation
    token_influences = projector.compute_token_influence(embeddings, positions, grid_points)
    print(f"✓ Token influence computation successful, shape: {token_influences.shape}")
    
    print("\nAll tests passed! 🎉") 