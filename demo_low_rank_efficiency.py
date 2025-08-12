#!/usr/bin/env python3
"""
Demonstration script comparing LowRankFieldProjector with standard FieldProjector.

This script showcases the memory efficiency and performance benefits of the
low-rank approximation approach for field projection in Token Field Networks.
"""

import torch
import torch.nn as nn
import time
import psutil
import os
from core.field_projection import LowRankFieldProjector, FieldProjector


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def benchmark_projectors(batch_size=2, num_tokens=64, embed_dim=256, pos_dim=2, grid_size=200, proj_dim=32):
    """
    Benchmark both field projectors and compare their performance.
    
    Args:
        batch_size: Batch size for testing
        num_tokens: Number of tokens per sequence
        embed_dim: Embedding dimension
        pos_dim: Position space dimension
        grid_size: Number of grid points
        proj_dim: Projection dimension for low-rank approach
    """
    print(f"Benchmarking Field Projectors")
    print(f"=============================")
    print(f"Batch size: {batch_size}")
    print(f"Tokens per sequence: {num_tokens}")
    print(f"Embedding dimension: {embed_dim}")
    print(f"Position dimension: {pos_dim}")
    print(f"Grid size: {grid_size}")
    print(f"Projection dimension: {proj_dim}")
    print()
    
    # Generate test data
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    embeddings = torch.randn(batch_size, num_tokens, embed_dim, device=device)
    positions = torch.randn(batch_size, num_tokens, pos_dim, device=device)
    grid_points = torch.randn(batch_size, grid_size, pos_dim, device=device)
    
    print(f"Test data generated on device: {device}")
    print(f"Input shapes:")
    print(f"  - Embeddings: {embeddings.shape}")
    print(f"  - Positions: {positions.shape}")
    print(f"  - Grid points: {grid_points.shape}")
    print()
    
    # Initialize projectors
    standard_projector = FieldProjector(embed_dim, pos_dim, 'rbf').to(device)
    low_rank_projector = LowRankFieldProjector(embed_dim, pos_dim, 'rbf', proj_dim).to(device)
    
    # Warm up (run once to initialize CUDA if available)
    if device.type == 'cuda':
        with torch.no_grad():
            _ = standard_projector(embeddings, positions, grid_points)
            _ = low_rank_projector(embeddings, positions, grid_points)
        torch.cuda.synchronize()
    
    # Benchmark standard projector
    print("Testing Standard FieldProjector...")
    start_memory = get_memory_usage()
    
    start_time = time.time()
    with torch.no_grad():
        standard_output = standard_projector(embeddings, positions, grid_points)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    standard_time = time.time() - start_time
    
    end_memory = get_memory_usage()
    standard_memory_used = end_memory - start_memory
    
    print(f"  ✓ Output shape: {standard_output.shape}")
    print(f"  ✓ Time: {standard_time:.4f}s")
    print(f"  ✓ Memory used: {standard_memory_used:.2f} MB")
    print()
    
    # Benchmark low-rank projector
    print("Testing LowRankFieldProjector...")
    start_memory = get_memory_usage()
    
    start_time = time.time()
    with torch.no_grad():
        low_rank_output = low_rank_projector(embeddings, positions, grid_points)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    low_rank_time = time.time() - start_time
    
    end_memory = get_memory_usage()
    low_rank_memory_used = end_memory - start_memory
    
    print(f"  ✓ Output shape: {low_rank_output.shape}")
    print(f"  ✓ Time: {low_rank_time:.4f}s")
    print(f"  ✓ Memory used: {low_rank_memory_used:.2f} MB")
    print()
    
    # Calculate theoretical memory savings
    theoretical_savings = low_rank_projector.get_memory_savings(batch_size, num_tokens, grid_size)
    
    # Compare outputs
    output_diff = torch.abs(standard_output - low_rank_output).mean().item()
    output_similarity = 1 - output_diff / (torch.abs(standard_output).mean().item() + 1e-8)
    
    # Results summary
    print("Results Summary")
    print("===============")
    print(f"Output similarity: {output_similarity:.4f} (1.0 = identical)")
    print(f"Speed improvement: {standard_time / low_rank_time:.2f}x")
    print(f"Memory improvement: {standard_memory_used / low_rank_memory_used:.2f}x")
    print()
    
    print("Theoretical Memory Analysis")
    print("===========================")
    print(f"Standard approach memory: {theoretical_savings['standard_memory']:,} elements")
    print(f"Low-rank approach memory: {theoretical_savings['low_rank_memory']:,} elements")
    print(f"Memory savings: {theoretical_savings['memory_savings']:,} elements")
    print(f"Compression factor: {theoretical_savings['compression_factor']:.2f}x")
    print(f"Savings ratio: {theoretical_savings['savings_ratio']:.2%}")
    
    return {
        'standard_time': standard_time,
        'low_rank_time': low_rank_time,
        'standard_memory': standard_memory_used,
        'low_rank_memory': low_rank_memory_used,
        'output_similarity': output_similarity,
        'theoretical_savings': theoretical_savings
    }


def demonstrate_scalability():
    """Demonstrate how memory savings scale with different problem sizes."""
    print("\n" + "="*60)
    print("Scalability Analysis")
    print("="*60)
    
    # Test different problem sizes
    test_cases = [
        {'batch_size': 1, 'num_tokens': 32, 'embed_dim': 128, 'grid_size': 100},
        {'batch_size': 2, 'num_tokens': 64, 'embed_dim': 256, 'grid_size': 200},
        {'batch_size': 4, 'num_tokens': 128, 'embed_dim': 512, 'grid_size': 400},
        {'batch_size': 8, 'num_tokens': 256, 'embed_dim': 1024, 'grid_size': 800},
    ]
    
    proj_dim = 64  # Fixed projection dimension
    
    print(f"{'Problem Size':<20} {'Standard Memory':<15} {'Low-Rank Memory':<15} {'Savings':<10}")
    print("-" * 70)
    
    for case in test_cases:
        # Calculate theoretical memory usage
        standard_memory = case['batch_size'] * case['num_tokens'] * case['embed_dim'] * case['grid_size']
        low_rank_memory = (case['batch_size'] * case['num_tokens'] * proj_dim * 2 + 
                          case['batch_size'] * proj_dim + 
                          case['batch_size'] * case['embed_dim'])
        
        savings = standard_memory - low_rank_memory
        savings_ratio = savings / standard_memory
        
        problem_size = f"B{case['batch_size']}×N{case['num_tokens']}×D{case['embed_dim']}×M{case['grid_size']}"
        
        print(f"{problem_size:<20} {standard_memory:>12,} {low_rank_memory:>12,} {savings_ratio:>8.1%}")


def main():
    """Main demonstration function."""
    print("Token Field Network - Low-Rank Field Projection Demo")
    print("=" * 60)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("CUDA not available, using CPU")
    
    print()
    
    # Run main benchmark
    results = benchmark_projectors(
        batch_size=2,
        num_tokens=64,
        embed_dim=256,
        pos_dim=2,
        grid_size=200,
        proj_dim=32
    )
    
    # Demonstrate scalability
    demonstrate_scalability()
    
    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main() 