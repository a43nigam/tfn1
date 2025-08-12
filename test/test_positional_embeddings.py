"""
Test positional embedding strategies to ensure the critical bug is fixed.

This test verifies that:
1. LearnedPositionalEmbeddings raises an error for continuous positions (instead of silently ignoring them)
2. AutoPositionalEmbeddingStrategy automatically selects the correct strategy
3. Continuous positions are handled correctly by appropriate strategies
"""

import torch
import pytest
from model.shared_layers import (
    LearnedPositionalEmbeddings,
    ContinuousPositionalEmbeddings,
    SinusoidalEmbeddings,
    AutoPositionalEmbeddingStrategy,
    create_positional_embedding_strategy
)


class TestPositionalEmbeddings:
    """Test suite for positional embedding strategies."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.max_len = 100
        self.embed_dim = 64
        self.batch_size = 4
        self.seq_len = 20
        
    def test_learned_embeddings_discrete_positions(self):
        """Test that LearnedPositionalEmbeddings works with discrete positions."""
        strategy = LearnedPositionalEmbeddings(self.max_len, self.embed_dim)
        
        # Discrete positions should work
        discrete_positions = torch.randint(0, self.max_len, (self.batch_size, self.seq_len))
        embeddings = strategy(discrete_positions)
        
        assert embeddings.shape == (self.batch_size, self.seq_len, self.embed_dim)
        assert embeddings.dtype == torch.float32
        
    def test_learned_embeddings_continuous_positions_error(self):
        """Test that LearnedPositionalEmbeddings raises error for continuous positions."""
        strategy = LearnedPositionalEmbeddings(self.max_len, self.embed_dim)
        
        # Continuous positions should raise an error
        continuous_positions = torch.rand(self.batch_size, self.seq_len)
        
        with pytest.raises(ValueError, match="LearnedPositionalEmbeddings received continuous positions"):
            strategy(continuous_positions)
    
    def test_continuous_embeddings_continuous_positions(self):
        """Test that ContinuousPositionalEmbeddings works with continuous positions."""
        strategy = ContinuousPositionalEmbeddings(self.max_len, self.embed_dim)
        
        # 1D continuous positions - should be [B, N, 1] for spatial dimension
        continuous_positions_1d = torch.rand(self.batch_size, self.seq_len, 1)
        embeddings_1d = strategy(continuous_positions_1d)
        
        assert embeddings_1d.shape == (self.batch_size, self.seq_len, self.embed_dim)
        assert embeddings_1d.dtype == torch.float32
        
        # Multi-dimensional continuous positions - should be [B, N, 2] for 2D spatial coordinates
        continuous_positions_2d = torch.rand(self.batch_size, self.seq_len, 2)
        embeddings_2d = strategy(continuous_positions_2d)
        
        assert embeddings_2d.shape == (self.batch_size, self.seq_len, self.embed_dim)
        assert embeddings_2d.dtype == torch.float32
    
    def test_sinusoidal_embeddings_continuous_positions(self):
        """Test that SinusoidalEmbeddings works with continuous positions."""
        strategy = SinusoidalEmbeddings(self.max_len, self.embed_dim)
        
        # Continuous positions should work
        continuous_positions = torch.rand(self.batch_size, self.seq_len)
        embeddings = strategy(continuous_positions)
        
        assert embeddings.shape == (self.batch_size, self.seq_len, self.embed_dim)
        assert embeddings.dtype == torch.float32
    
    def test_auto_strategy_discrete_positions(self):
        """Test that AutoPositionalEmbeddingStrategy selects 'learned' for discrete positions."""
        strategy = AutoPositionalEmbeddingStrategy(self.max_len, self.embed_dim)
        
        # Discrete positions should use learned strategy
        discrete_positions = torch.randint(0, self.max_len, (self.batch_size, self.seq_len))
        embeddings = strategy(discrete_positions)
        
        assert embeddings.shape == (self.batch_size, self.seq_len, self.embed_dim)
        assert strategy._last_strategy == 'learned'
    
    def test_auto_strategy_continuous_positions_1d(self):
        """Test that AutoPositionalEmbeddingStrategy selects 'sinusoidal' for 1D continuous positions."""
        strategy = AutoPositionalEmbeddingStrategy(self.max_len, self.embed_dim)
        
        # 1D continuous positions should use sinusoidal strategy
        continuous_positions = torch.rand(self.batch_size, self.seq_len)
        embeddings = strategy(continuous_positions)
        
        assert embeddings.shape == (self.batch_size, self.seq_len, self.embed_dim)
        assert strategy._last_strategy == 'sinusoidal'
    
    def test_auto_strategy_continuous_positions_2d(self):
        """Test that AutoPositionalEmbeddingStrategy selects 'continuous' for multi-dimensional continuous positions."""
        strategy = AutoPositionalEmbeddingStrategy(self.max_len, self.embed_dim)
        
        # Multi-dimensional continuous positions should use continuous strategy
        continuous_positions = torch.rand(self.batch_size, self.seq_len, 2)
        embeddings = strategy(continuous_positions)
        
        assert embeddings.shape == (self.batch_size, self.seq_len, self.embed_dim)
        assert strategy._last_strategy == 'continuous'
    
    def test_factory_auto_strategy(self):
        """Test that the factory function creates AutoPositionalEmbeddingStrategy correctly."""
        strategy = create_positional_embedding_strategy("auto", self.max_len, self.embed_dim)
        
        assert isinstance(strategy, AutoPositionalEmbeddingStrategy)
        
        # Test that it works with continuous positions
        continuous_positions = torch.rand(self.batch_size, self.seq_len)
        embeddings = strategy(continuous_positions)
        
        assert embeddings.shape == (self.batch_size, self.seq_len, self.embed_dim)
    
    def test_factory_learned_strategy(self):
        """Test that the factory function creates LearnedPositionalEmbeddings correctly."""
        strategy = create_positional_embedding_strategy("learned", self.max_len, self.embed_dim)
        
        assert isinstance(strategy, LearnedPositionalEmbeddings)
        
        # Test that it works with discrete positions
        discrete_positions = torch.randint(0, self.max_len, (self.batch_size, self.seq_len))
        embeddings = strategy(discrete_positions)
        
        assert embeddings.shape == (self.batch_size, self.seq_len, self.embed_dim)
    
    def test_factory_continuous_strategy(self):
        """Test that the factory function creates ContinuousPositionalEmbeddings correctly."""
        strategy = create_positional_embedding_strategy("continuous", self.max_len, self.embed_dim)
        
        assert isinstance(strategy, ContinuousPositionalEmbeddings)
        
        # Test that it works with continuous positions
        continuous_positions = torch.rand(self.batch_size, self.seq_len, 1)  # [B, N, 1] for 1D spatial
        embeddings = strategy(continuous_positions)
        
        assert embeddings.shape == (self.batch_size, self.seq_len, self.embed_dim)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 