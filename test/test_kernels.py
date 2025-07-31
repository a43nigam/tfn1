"""
Unit tests for kernel mathematical functions.

Tests the core mathematical properties and correctness of kernel implementations.
"""

import torch
import pytest
import math
from core.kernels import RBFKernel, CompactKernel, FourierKernel, KernelFactory


class TestRBFKernel:
    """Test RBF kernel mathematical properties."""
    
    def test_rbf_kernel_shape(self):
        """Test RBF kernel output shape."""
        kernel = RBFKernel(pos_dim=1)
        z = torch.randn(10, 1)  # [M, P]
        mu = torch.randn(2, 3, 1)  # [B, N, P]
        sigma = torch.ones(2, 3, 1) * 0.5  # [B, N, 1]
        
        result = kernel(z, mu, sigma)
        assert result.shape == (2, 3, 10)  # [B, N, M]
    
    def test_rbf_kernel_gaussian_property(self):
        """Test that RBF kernel satisfies Gaussian properties."""
        kernel = RBFKernel(pos_dim=1, min_sigma=0.1, max_sigma=2.0)
        
        # Test at the center point
        mu = torch.zeros(1, 1, 1)  # Center at origin
        sigma = torch.ones(1, 1, 1) * 0.5
        z_center = torch.zeros(1, 1)  # Same as mu
        
        result = kernel(z_center, mu, sigma)
        # At center, RBF should be 1.0 (exp(0))
        assert torch.allclose(result, torch.ones(1, 1, 1))
    
    def test_rbf_kernel_decay_property(self):
        """Test that RBF kernel decays with distance."""
        kernel = RBFKernel(pos_dim=1)
        
        mu = torch.zeros(1, 1, 1)  # Center at origin
        sigma = torch.ones(1, 1, 1) * 0.5
        
        # Test points at increasing distances
        z_near = torch.tensor([[0.1]])  # Close to center
        z_far = torch.tensor([[1.0]])   # Far from center
        
        result_near = kernel(z_near, mu, sigma)
        result_far = kernel(z_far, mu, sigma)
        
        # Far point should have lower value than near point
        assert result_far < result_near
    
    def test_rbf_kernel_sigma_clamping(self):
        """Test that sigma is properly clamped."""
        kernel = RBFKernel(pos_dim=1, min_sigma=0.1, max_sigma=2.0)
        
        mu = torch.zeros(1, 1, 1)
        z = torch.zeros(1, 1)
        
        # Test with sigma below minimum
        sigma_low = torch.ones(1, 1, 1) * 0.01  # Below min_sigma
        result_low = kernel(z, mu, sigma_low)
        
        # Test with sigma above maximum
        sigma_high = torch.ones(1, 1, 1) * 10.0  # Above max_sigma
        result_high = kernel(z, mu, sigma_high)
        
        # Should not crash and should produce valid results
        assert torch.isfinite(result_low).all()
        assert torch.isfinite(result_high).all()


class TestCompactKernel:
    """Test compact kernel mathematical properties."""
    
    def test_compact_kernel_shape(self):
        """Test compact kernel output shape."""
        kernel = CompactKernel(pos_dim=1)
        z = torch.randn(10, 1)  # [M, P]
        mu = torch.randn(2, 3, 1)  # [B, N, P]
        radius = torch.ones(2, 3, 1) * 0.5  # [B, N, 1]
        
        result = kernel(z, mu, radius)
        assert result.shape == (2, 3, 10)  # [B, N, M]
    
    def test_compact_kernel_support(self):
        """Test that compact kernel has finite support."""
        kernel = CompactKernel(pos_dim=1)
        
        mu = torch.zeros(1, 1, 1)  # Center at origin
        radius = torch.ones(1, 1, 1) * 0.5
        
        # Test point within radius
        z_inside = torch.tensor([[0.3]])  # Within radius
        result_inside = kernel(z_inside, mu, radius)
        assert result_inside > 0
        
        # Test point outside radius
        z_outside = torch.tensor([[1.0]])  # Outside radius
        result_outside = kernel(z_outside, mu, radius)
        assert result_outside == 0
    
    def test_compact_kernel_linearity(self):
        """Test compact kernel linear decay property."""
        kernel = CompactKernel(pos_dim=1)
        
        mu = torch.zeros(1, 1, 1)  # Center at origin
        radius = torch.ones(1, 1, 1) * 1.0
        
        # Test at center (distance = 0)
        z_center = torch.zeros(1, 1)
        result_center = kernel(z_center, mu, radius)
        assert torch.allclose(result_center, torch.ones(1, 1, 1))
        
        # Test at half radius (distance = 0.5)
        z_half = torch.tensor([[0.5]])
        result_half = kernel(z_half, mu, radius)
        expected_half = torch.tensor([[[0.5]]])  # 1 - 0.5/1.0
        assert torch.allclose(result_half, expected_half)


class TestFourierKernel:
    """Test Fourier kernel mathematical properties."""
    
    def test_fourier_kernel_shape(self):
        """Test Fourier kernel output shape."""
        kernel = FourierKernel(pos_dim=1)
        z = torch.randn(10, 1)  # [M, P]
        mu = torch.randn(2, 3, 1)  # [B, N, P]
        freq = torch.ones(2, 3, 1) * 2.0  # [B, N, 1]
        
        result = kernel(z, mu, freq)
        assert result.shape == (2, 3, 10)  # [B, N, M]
    
    def test_fourier_kernel_oscillatory(self):
        """Test that Fourier kernel produces oscillatory behavior."""
        kernel = FourierKernel(pos_dim=1)
        
        mu = torch.zeros(1, 1, 1)  # Center at origin
        freq = torch.ones(1, 1, 1) * math.pi  # Frequency = π
        
        # Test at distance π/freq = 1 (should be cos(π) = -1)
        z_pi = torch.tensor([[1.0]])
        result_pi = kernel(z_pi, mu, freq)
        expected_pi = torch.tensor([[[-1.0]]])
        assert torch.allclose(result_pi, expected_pi, atol=1e-6)
        
        # Test at distance 2π/freq = 2 (should be cos(2π) = 1)
        z_2pi = torch.tensor([[2.0]])
        result_2pi = kernel(z_2pi, mu, freq)
        expected_2pi = torch.tensor([[[1.0]]])
        assert torch.allclose(result_2pi, expected_2pi, atol=1e-6)
    
    def test_fourier_kernel_frequency_clamping(self):
        """Test that frequency is properly clamped."""
        kernel = FourierKernel(pos_dim=1, min_freq=0.1, max_freq=10.0)
        
        mu = torch.zeros(1, 1, 1)
        z = torch.zeros(1, 1)
        
        # Test with frequency below minimum
        freq_low = torch.ones(1, 1, 1) * 0.01  # Below min_freq
        result_low = kernel(z, mu, freq_low)
        
        # Test with frequency above maximum
        freq_high = torch.ones(1, 1, 1) * 100.0  # Above max_freq
        result_high = kernel(z, mu, freq_high)
        
        # Should not crash and should produce valid results
        assert torch.isfinite(result_low).all()
        assert torch.isfinite(result_high).all()


class TestKernelFactory:
    """Test kernel factory functionality."""
    
    def test_kernel_factory_creation(self):
        """Test that kernel factory creates correct kernel types."""
        pos_dim = 2
        
        # Test RBF kernel creation
        rbf_kernel = KernelFactory.create("rbf", pos_dim)
        assert isinstance(rbf_kernel, RBFKernel)
        assert rbf_kernel.pos_dim == pos_dim
        
        # Test compact kernel creation
        compact_kernel = KernelFactory.create("compact", pos_dim)
        assert isinstance(compact_kernel, CompactKernel)
        assert compact_kernel.pos_dim == pos_dim
        
        # Test Fourier kernel creation
        fourier_kernel = KernelFactory.create("fourier", pos_dim)
        assert isinstance(fourier_kernel, FourierKernel)
        assert fourier_kernel.pos_dim == pos_dim
    
    def test_kernel_factory_invalid_type(self):
        """Test that kernel factory raises error for invalid types."""
        with pytest.raises(ValueError):
            KernelFactory.create("invalid_kernel", 1)
    
    def test_kernel_factory_with_params(self):
        """Test kernel factory with additional parameters."""
        rbf_kernel = KernelFactory.create("rbf", 1, min_sigma=0.5, max_sigma=5.0)
        assert rbf_kernel.min_sigma == 0.5
        assert rbf_kernel.max_sigma == 5.0
    
    def test_available_kernels(self):
        """Test that available kernels list is correct."""
        available = KernelFactory.get_available_kernels()
        expected = ["rbf", "compact", "fourier", 
                   "data_dependent_rbf", "data_dependent_compact", 
                   "multi_frequency_fourier", "film_learnable"]
        assert available == expected


class TestKernelNumericalStability:
    """Test numerical stability of kernels."""
    
    def test_kernels_with_extreme_values(self):
        """Test kernels handle extreme input values gracefully."""
        kernels = [
            ("rbf", RBFKernel(1)),
            ("compact", CompactKernel(1)),
            ("fourier", FourierKernel(1))
        ]
        
        # Extreme positions
        z_extreme = torch.tensor([[1e6], [-1e6]])  # Very large/small positions
        mu_extreme = torch.tensor([[[0.0]], [[1e6]]])  # [B=2, N=1, P=1]
        
        for name, kernel in kernels:
            if name == "rbf":
                params = torch.ones(2, 1, 1) * 0.5  # sigma
            elif name == "compact":
                params = torch.ones(2, 1, 1) * 1e3  # large radius
            else:  # fourier
                params = torch.ones(2, 1, 1) * 0.1  # small frequency
            
            result = kernel(z_extreme, mu_extreme, params)
            
            # Results should be finite (not NaN or inf)
            assert torch.isfinite(result).all(), f"{name} kernel failed with extreme values"
            
            # Results should be in reasonable range
            assert (result >= -1e10).all() and (result <= 1e10).all(), f"{name} kernel values too extreme"


if __name__ == "__main__":
    pytest.main([__file__]) 