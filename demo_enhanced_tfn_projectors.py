#!/usr/bin/env python3
"""
Demonstration script for Enhanced TFN models with different projector types.

This script shows how to use both the standard FieldProjector and the new
LowRankFieldProjector in EnhancedTFN models, comparing their performance
and memory usage.
"""

import torch
import torch.nn as nn
import time
import psutil
import os
from model.tfn_enhanced import (
    EnhancedTFNLayer, 
    EnhancedTFNModel, 
    EnhancedTFNRegressor,
    create_enhanced_tfn_model
)


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def benchmark_projector_types():
    """Benchmark both projector types in EnhancedTFNLayer."""
    print("Enhanced TFN Projector Type Benchmark")
    print("=" * 50)
    
    # Test parameters
    embed_dim = 256
    pos_dim = 1
    grid_size = 200
    proj_dim = 64
    batch_size = 2
    num_tokens = 32
    
    print(f"Test Configuration:")
    print(f"  - Embedding dimension: {embed_dim}")
    print(f"  - Position dimension: {pos_dim}")
    print(f"  - Grid size: {grid_size}")
    print(f"  - Projection dimension: {proj_dim}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Number of tokens: {num_tokens}")
    print()
    
    # Generate test data
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    embeddings = torch.randn(batch_size, num_tokens, embed_dim, device=device)
    positions = torch.randn(batch_size, num_tokens, pos_dim, device=device)
    
    print(f"Test data generated on device: {device}")
    print()
    
    # Test standard projector
    print("Testing Standard FieldProjector...")
    layer_standard = EnhancedTFNLayer(
        embed_dim=embed_dim,
        pos_dim=pos_dim,
        grid_size=grid_size,
        projector_type='standard'
    ).to(device)
    
    start_memory = get_memory_usage()
    start_time = time.time()
    
    with torch.no_grad():
        output_standard = layer_standard(embeddings, positions)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    standard_time = time.time() - start_time
    end_memory = get_memory_usage()
    standard_memory_used = end_memory - start_memory
    
    print(f"  ✓ Output shape: {output_standard.shape}")
    print(f"  ✓ Time: {standard_time:.4f}s")
    print(f"  ✓ Memory used: {standard_memory_used:.2f} MB")
    print()
    
    # Test low-rank projector
    print("Testing LowRankFieldProjector...")
    layer_low_rank = EnhancedTFNLayer(
        embed_dim=embed_dim,
        pos_dim=pos_dim,
        grid_size=grid_size,
        projector_type='low_rank',
        proj_dim=proj_dim
    ).to(device)
    
    start_memory = get_memory_usage()
    start_time = time.time()
    
    with torch.no_grad():
        output_low_rank = layer_low_rank(embeddings, positions)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    low_rank_time = time.time() - start_time
    end_memory = get_memory_usage()
    low_rank_memory_used = end_memory - start_memory
    
    print(f"  ✓ Output shape: {output_low_rank.shape}")
    print(f"  ✓ Time: {low_rank_time:.4f}s")
    print(f"  ✓ Memory used: {low_rank_memory_used:.2f} MB")
    print()
    
    # Results comparison
    print("Results Comparison")
    print("=" * 50)
    print(f"Output similarity: {torch.allclose(output_standard, output_low_rank, atol=1e-3)}")
    print(f"Speed improvement: {standard_time / low_rank_time:.2f}x")
    print(f"Memory improvement: {standard_memory_used / low_rank_memory_used:.2f}x")
    
    # Calculate theoretical memory savings
    theoretical_savings = layer_low_rank.field_projector.get_memory_savings(
        batch_size, num_tokens, grid_size
    )
    
    print(f"\nTheoretical Memory Analysis:")
    print(f"  - Standard approach: {theoretical_savings['standard_memory']:,} elements")
    print(f"  - Low-rank approach: {theoretical_savings['low_rank_memory']:,} elements")
    print(f"  - Compression factor: {theoretical_savings['compression_factor']:.2f}x")
    print(f"  - Savings ratio: {theoretical_savings['savings_ratio']:.2%}")
    
    return {
        'standard_time': standard_time,
        'low_rank_time': low_rank_time,
        'standard_memory': standard_memory_used,
        'low_rank_memory': low_rank_memory_used,
        'theoretical_savings': theoretical_savings
    }


def demonstrate_model_creation():
    """Demonstrate creating different types of models with different projectors."""
    print("\n" + "=" * 60)
    print("Enhanced TFN Model Creation Examples")
    print("=" * 60)
    
    # Example 1: Language model with standard projector
    print("Example 1: Language Model with Standard Projector")
    print("-" * 50)
    
    model_standard = EnhancedTFNModel(
        vocab_size=50000,
        embed_dim=512,
        num_layers=6,
        grid_size=100,
        projector_type='standard'
    )
    
    print(f"  ✓ Model created with {len(model_standard.layers)} layers")
    print(f"  ✓ First layer projector type: {model_standard.layers[0].projector_type}")
    print(f"  ✓ Total parameters: {sum(p.numel() for p in model_standard.parameters()):,}")
    
    # Example 2: Language model with low-rank projector
    print("\nExample 2: Language Model with Low-Rank Projector")
    print("-" * 50)
    
    model_low_rank = EnhancedTFNModel(
        vocab_size=50000,
        embed_dim=512,
        num_layers=6,
        grid_size=100,
        projector_type='low_rank',
        proj_dim=64
    )
    
    print(f"  ✓ Model created with {len(model_low_rank.layers)} layers")
    print(f"  ✓ First layer projector type: {model_low_rank.layers[0].projector_type}")
    print(f"  ✓ First layer projection dimension: {model_low_rank.layers[0].proj_dim}")
    print(f"  ✓ Total parameters: {sum(p.numel() for p in model_low_rank.parameters()):,}")
    
    # Example 3: Regression model with low-rank projector
    print("\nExample 3: Regression Model with Low-Rank Projector")
    print("-" * 50)
    
    regressor = EnhancedTFNRegressor(
        input_dim=128,
        embed_dim=256,
        output_dim=64,
        output_len=20,
        num_layers=4,
        grid_size=150,
        projector_type='low_rank',
        proj_dim=48
    )
    
    print(f"  ✓ Regressor created with {len(regressor.layers)} layers")
    print(f"  ✓ First layer projector type: {regressor.layers[0].projector_type}")
    print(f"  ✓ First layer projection dimension: {regressor.layers[0].proj_dim}")
    print(f"  ✓ Total parameters: {sum(p.numel() for p in regressor.parameters()):,}")
    
    # Example 4: Using factory function
    print("\nExample 4: Using Factory Function")
    print("-" * 50)
    
    factory_model = create_enhanced_tfn_model(
        vocab_size=10000,
        embed_dim=256,
        num_layers=3,
        grid_size=80,
        projector_type='low_rank',
        proj_dim=32
    )
    
    print(f"  ✓ Factory model created with {len(factory_model.layers)} layers")
    print(f"  ✓ First layer projector type: {factory_model.layers[0].projector_type}")
    print(f"  ✓ First layer projection dimension: {factory_model.layers[0].proj_dim}")


def demonstrate_usage_patterns():
    """Demonstrate common usage patterns for different projector types."""
    print("\n" + "=" * 60)
    print("Usage Pattern Examples")
    print("=" * 60)
    
    print("1. Memory-Constrained Environments (use low-rank):")
    print("   model = EnhancedTFNModel(")
    print("       vocab_size=50000,")
    print("       embed_dim=1024,")
    print("       num_layers=12,")
    print("       grid_size=500,")
    print("       projector_type='low_rank',")
    print("       proj_dim=128")
    print("   )")
    print()
    
    print("2. High-Accuracy Requirements (use standard):")
    print("   model = EnhancedTFNModel(")
    print("       vocab_size=50000,")
    print("       embed_dim=512,")
    print("       num_layers=6,")
    print("       grid_size=100,")
    print("       projector_type='standard'")
    print("   )")
    print()
    
    print("3. Balanced Approach (use low-rank with moderate projection):")
    print("   model = EnhancedTFNModel(")
    print("       vocab_size=50000,")
    print("       embed_dim=768,")
    print("       num_layers=8,")
    print("       grid_size=200,")
    print("       projector_type='low_rank',")
    print("       proj_dim=96")
    print("   )")
    print()
    
    print("4. Regression Tasks with Low-Rank:")
    print("   regressor = EnhancedTFNRegressor(")
    print("       input_dim=256,")
    print("       embed_dim=512,")
    print("       output_dim=128,")
    print("       output_len=50,")
    print("       num_layers=6,")
    print("       grid_size=300,")
    print("       projector_type='low_rank',")
    print("       proj_dim=64")
    print("   )")


def main():
    """Main demonstration function."""
    print("Enhanced TFN Projector Types Demonstration")
    print("=" * 60)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("CUDA not available, using CPU")
    
    print()
    
    try:
        # Run benchmark
        benchmark_results = benchmark_projector_types()
        
        # Demonstrate model creation
        demonstrate_model_creation()
        
        # Show usage patterns
        demonstrate_usage_patterns()
        
        print("\n" + "=" * 60)
        print("🎉 Demonstration completed successfully!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demonstration failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main() 