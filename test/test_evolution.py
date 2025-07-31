"""
Unit tests for field evolution mathematical functions.

Tests the core mathematical properties and correctness of evolution implementations.
"""

import torch
import pytest
import math
from core.field_evolution import (
    PDEFieldEvolver, CNNFieldEvolver, ModernizedCNNFieldEvolver,
    SpatiallyVaryingPDEFieldEvolver, FieldEvolver
)


class TestPDEFieldEvolver:
    """Test PDE-based field evolution mathematical properties."""
    
    def test_diffusion_evolution_shape(self):
        """Test diffusion evolution output shape."""
        evolver = PDEFieldEvolver(embed_dim=32, pos_dim=1, pde_type="diffusion")
        
        field = torch.randn(2, 10, 32)  # [B, M, D]
        grid_points = torch.linspace(0, 1, 10).unsqueeze(0).unsqueeze(-1).expand(2, -1, 1)  # [B, M, P]
        
        result = evolver(field, grid_points, time_steps=3)
        assert result.shape == field.shape
    
    def test_wave_evolution_shape(self):
        """Test wave evolution output shape."""
        evolver = PDEFieldEvolver(embed_dim=32, pos_dim=1, pde_type="wave")
        
        field = torch.randn(2, 10, 32)  # [B, M, D]
        grid_points = torch.linspace(0, 1, 10).unsqueeze(0).unsqueeze(-1).expand(2, -1, 1)  # [B, M, P]
        
        result = evolver(field, grid_points, time_steps=3)
        assert result.shape == field.shape
    
    def test_schrodinger_evolution_shape(self):
        """Test Schrödinger evolution output shape."""
        evolver = PDEFieldEvolver(embed_dim=32, pos_dim=1, pde_type="schrodinger")
        
        field = torch.randn(2, 10, 32)  # [B, M, D]
        grid_points = torch.linspace(0, 1, 10).unsqueeze(0).unsqueeze(-1).expand(2, -1, 1)  # [B, M, P]
        
        result = evolver(field, grid_points, time_steps=3)
        assert result.shape == field.shape
    
    def test_diffusion_smoothing_property(self):
        """Test that diffusion evolution smooths sharp features."""
        evolver = PDEFieldEvolver(embed_dim=1, pos_dim=1, pde_type="diffusion")
        
        # Create a sharp spike in the middle
        field = torch.zeros(1, 11, 1)  # [B=1, M=11, D=1]
        field[0, 5, 0] = 10.0  # Sharp spike at center
        
        grid_points = torch.linspace(0, 1, 11).unsqueeze(0).unsqueeze(-1)  # [1, 11, 1]
        
        # Evolve for several time steps
        result = evolver(field, grid_points, time_steps=10, dt=0.01)
        
        # After diffusion, the spike should be smoothed out
        # The maximum should be reduced and spread to neighbors
        assert result[0, 5, 0] < field[0, 5, 0]  # Peak reduced
        assert result[0, 4, 0] > 0  # Spread to neighbors
        assert result[0, 6, 0] > 0  # Spread to neighbors
    
    def test_diffusion_conservation_property(self):
        """Test that diffusion conserves total field energy (approximately)."""
        evolver = PDEFieldEvolver(embed_dim=1, pos_dim=1, pde_type="diffusion")
        
        # Create initial field with some energy
        field = torch.randn(1, 20, 1) * 0.1  # Small values to avoid boundary effects
        grid_points = torch.linspace(0, 1, 20).unsqueeze(0).unsqueeze(-1)
        
        initial_energy = torch.sum(field**2)
        
        # Evolve for a few steps
        result = evolver(field, grid_points, time_steps=5, dt=0.001)  # Small dt
        final_energy = torch.sum(result**2)
        
        # Energy should be approximately conserved (with some numerical error)
        energy_ratio = final_energy / initial_energy
        assert 0.8 < energy_ratio < 1.2  # Allow for some numerical diffusion
    
    def test_wave_oscillation_property(self):
        """Test that wave equation produces oscillatory behavior."""
        evolver = PDEFieldEvolver(embed_dim=1, pos_dim=1, pde_type="wave")
        
        # Create initial displacement
        field = torch.zeros(1, 21, 1)
        field[0, 10, 0] = 1.0  # Initial displacement at center
        
        grid_points = torch.linspace(0, 1, 21).unsqueeze(0).unsqueeze(-1)
        
        # Evolve and check for oscillatory behavior
        result_t1 = evolver(field, grid_points, time_steps=10, dt=0.01)
        result_t2 = evolver(result_t1, grid_points, time_steps=10, dt=0.01)
        
        # Wave should have propagated from center
        center_val_t0 = field[0, 10, 0]
        center_val_t1 = result_t1[0, 10, 0]
        center_val_t2 = result_t2[0, 10, 0]
        
        # Values should change over time (oscillation)
        assert not torch.allclose(center_val_t0, center_val_t1, atol=1e-3)
        assert not torch.allclose(center_val_t1, center_val_t2, atol=1e-3)
    
    def test_pde_numerical_stability(self):
        """Test that PDE evolution remains numerically stable."""
        evolvers = [
            PDEFieldEvolver(embed_dim=16, pos_dim=1, pde_type="diffusion"),
            PDEFieldEvolver(embed_dim=16, pos_dim=1, pde_type="wave"),
            PDEFieldEvolver(embed_dim=16, pos_dim=1, pde_type="schrodinger")
        ]
        
        # Test with various field configurations
        test_fields = [
            torch.randn(1, 15, 16) * 0.1,  # Small random field
            torch.zeros(1, 15, 16),        # Zero field
            torch.ones(1, 15, 16),         # Constant field
        ]
        
        grid_points = torch.linspace(0, 1, 15).unsqueeze(0).unsqueeze(-1)
        
        for evolver in evolvers:
            for field in test_fields:
                result = evolver(field, grid_points, time_steps=5, dt=0.01)
                
                # Results should be finite
                assert torch.isfinite(result).all()
                
                # Results should not explode
                assert torch.abs(result).max() < 1e3


class TestCNNFieldEvolver:
    """Test CNN-based field evolution properties."""
    
    def test_cnn_evolution_shape(self):
        """Test CNN evolution output shape."""
        evolver = CNNFieldEvolver(embed_dim=64, pos_dim=1)
        
        field = torch.randn(2, 20, 64)  # [B, M, D]
        grid_points = torch.linspace(0, 1, 20).unsqueeze(0).unsqueeze(-1).expand(2, -1, 1)  # [B, M, P]
        
        result = evolver(field, grid_points, time_steps=3)
        assert result.shape == field.shape
    
    def test_cnn_evolution_residual_connection(self):
        """Test that CNN evolution includes residual connections."""
        evolver = CNNFieldEvolver(embed_dim=32, pos_dim=1)
        
        # Use a simple field
        field = torch.ones(1, 10, 32) * 0.5
        grid_points = torch.linspace(0, 1, 10).unsqueeze(0).unsqueeze(-1)
        
        # Single step evolution
        result = evolver(field, grid_points, time_steps=1)
        
        # With residual connections, result should not be too different from input
        difference = torch.abs(result - field).mean()
        assert difference < 2.0  # Should not change drastically in one step
    
    def test_cnn_evolution_nonlinearity(self):
        """Test that CNN evolution introduces nonlinearity."""
        evolver = CNNFieldEvolver(embed_dim=32, pos_dim=1)
        
        field1 = torch.zeros(1, 10, 32)
        field2 = torch.ones(1, 10, 32) * 0.1
        field_sum = field1 + field2
        
        grid_points = torch.linspace(0, 1, 10).unsqueeze(0).unsqueeze(-1)
        
        result1 = evolver(field1, grid_points, time_steps=1)
        result2 = evolver(field2, grid_points, time_steps=1)
        result_sum = evolver(field_sum, grid_points, time_steps=1)
        
        # Due to nonlinearity: f(a + b) ≠ f(a) + f(b)
        linear_sum = result1 + result2
        assert not torch.allclose(result_sum, linear_sum, atol=1e-3)


class TestModernizedCNNFieldEvolver:
    """Test modernized CNN evolution properties."""
    
    def test_modernized_cnn_shape(self):
        """Test modernized CNN evolution output shape."""
        evolver = ModernizedCNNFieldEvolver(
            embed_dim=64, pos_dim=1, 
            kernel_sizes=[3, 5, 7]
        )
        
        field = torch.randn(2, 20, 64)
        grid_points = torch.linspace(0, 1, 20).unsqueeze(0).unsqueeze(-1).expand(2, -1, 1)
        
        result = evolver(field, grid_points, time_steps=2)
        assert result.shape == field.shape
    
    def test_modernized_cnn_multi_scale(self):
        """Test that modernized CNN uses multiple kernel sizes."""
        # Test with different kernel sizes
        evolver1 = ModernizedCNNFieldEvolver(embed_dim=32, pos_dim=1, kernel_sizes=[3])
        evolver2 = ModernizedCNNFieldEvolver(embed_dim=32, pos_dim=1, kernel_sizes=[3, 5, 7])
        
        field = torch.randn(1, 15, 32)
        grid_points = torch.linspace(0, 1, 15).unsqueeze(0).unsqueeze(-1)
        
        result1 = evolver1(field, grid_points, time_steps=1)
        result2 = evolver2(field, grid_points, time_steps=1)
        
        # Multi-scale should produce different results
        assert not torch.allclose(result1, result2, atol=1e-3)


class TestSpatiallyVaryingPDEFieldEvolver:
    """Test spatially-varying PDE evolution properties."""
    
    def test_spatially_varying_shape(self):
        """Test spatially-varying PDE evolution output shape."""
        evolver = SpatiallyVaryingPDEFieldEvolver(
            embed_dim=32, pos_dim=1, pde_type="diffusion"
        )
        
        field = torch.randn(2, 15, 32)
        grid_points = torch.linspace(0, 1, 15).unsqueeze(0).unsqueeze(-1).expand(2, -1, 1)
        
        result = evolver(field, grid_points, time_steps=3)
        assert result.shape == field.shape
    
    def test_spatially_varying_adaptivity(self):
        """Test that spatially-varying PDE adapts to local field properties."""
        evolver = SpatiallyVaryingPDEFieldEvolver(
            embed_dim=8, pos_dim=1, pde_type="diffusion"
        )
        
        # Create field with varying magnitudes
        field = torch.zeros(1, 20, 8)
        field[0, 5:10, :] = 1.0   # High magnitude region
        field[0, 15:, :] = 0.1    # Low magnitude region
        
        grid_points = torch.linspace(0, 1, 20).unsqueeze(0).unsqueeze(-1)
        
        result = evolver(field, grid_points, time_steps=5, dt=0.01)
        
        # The evolution should adapt to different regions differently
        assert torch.isfinite(result).all()
        # Different regions should evolve differently
        high_region_change = torch.abs(result[0, 5:10, :] - field[0, 5:10, :]).mean()
        low_region_change = torch.abs(result[0, 15:, :] - field[0, 15:, :]).mean()
        
        # Changes should be different (adaptive behavior)
        assert not torch.allclose(torch.tensor(high_region_change), torch.tensor(low_region_change), atol=1e-4)


class TestFieldEvolverFactory:
    """Test FieldEvolver factory functionality."""
    
    def test_field_evolver_creation(self):
        """Test that FieldEvolver creates correct evolver types."""
        embed_dim, pos_dim = 32, 1
        
        # Test CNN evolver
        cnn_evolver = FieldEvolver(embed_dim, pos_dim, "cnn")
        assert hasattr(cnn_evolver, 'evolver')
        
        # Test modernized CNN evolver
        modern_evolver = FieldEvolver(embed_dim, pos_dim, "modernized_cnn")
        assert hasattr(modern_evolver, 'evolver')
        
        # Test PDE evolver
        pde_evolver = FieldEvolver(embed_dim, pos_dim, "diffusion")
        assert hasattr(pde_evolver, 'evolver')
        
        # Test spatially varying PDE evolver
        sv_evolver = FieldEvolver(embed_dim, pos_dim, "spatially_varying_pde")
        assert hasattr(sv_evolver, 'evolver')
    
    def test_field_evolver_invalid_type(self):
        """Test that FieldEvolver raises error for invalid types."""
        with pytest.raises(ValueError):
            FieldEvolver(32, 1, "invalid_evolution_type")
    
    def test_field_evolver_forward_consistency(self):
        """Test that all evolver types produce consistent outputs."""
        embed_dim, pos_dim = 16, 1
        field = torch.randn(1, 10, embed_dim)
        grid_points = torch.linspace(0, 1, 10).unsqueeze(0).unsqueeze(-1)
        
        evolution_types = ["cnn", "modernized_cnn", "diffusion", "wave", "spatially_varying_pde"]
        
        for evo_type in evolution_types:
            evolver = FieldEvolver(embed_dim, pos_dim, evo_type)
            result = evolver(field, grid_points, time_steps=2)
            
            # All should preserve shape
            assert result.shape == field.shape
            
            # All should produce finite results
            assert torch.isfinite(result).all()


class TestEvolutionNumericalStability:
    """Test numerical stability of evolution methods."""
    
    def test_evolution_with_extreme_fields(self):
        """Test evolution methods with extreme field values."""
        evolvers = [
            ("cnn", CNNFieldEvolver(16, 1)),
            ("pde_diffusion", PDEFieldEvolver(16, 1, "diffusion")),
            ("pde_wave", PDEFieldEvolver(16, 1, "wave")),
            ("modernized_cnn", ModernizedCNNFieldEvolver(16, 1))
        ]
        
        # Test extreme field values
        extreme_fields = [
            torch.ones(1, 10, 16) * 1e3,   # Large positive values
            torch.ones(1, 10, 16) * -1e3,  # Large negative values
            torch.zeros(1, 10, 16),        # Zero field
        ]
        
        grid_points = torch.linspace(0, 1, 10).unsqueeze(0).unsqueeze(-1)
        
        for name, evolver in evolvers:
            for i, field in enumerate(extreme_fields):
                try:
                    result = evolver(field, grid_points, time_steps=2)
                    
                    # Should produce finite results
                    assert torch.isfinite(result).all(), f"{name} failed with extreme field {i}"
                    
                    # Should not explode beyond reasonable bounds
                    assert torch.abs(result).max() < 1e6, f"{name} exploded with extreme field {i}"
                    
                except Exception as e:
                    pytest.fail(f"{name} crashed with extreme field {i}: {e}")


if __name__ == "__main__":
    pytest.main([__file__]) 