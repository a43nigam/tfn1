"""
Grid utilities for TFN layers.

Provides auto-grid-size heuristics and grid management utilities.
"""

import torch
import math
from typing import Tuple, Optional, Dict, Any


def compute_auto_grid_size(
    seq_len: int,
    embed_dim: int,
    target_flops: Optional[float] = None,
    memory_constraint: Optional[float] = None,
    min_grid_size: int = 32,
    max_grid_size: int = 512,
    heuristic: str = "sqrt"
) -> int:
    """Compute optimal grid size based on sequence length and constraints.
    
    Args:
        seq_len: Length of input sequence
        embed_dim: Embedding dimension
        target_flops: Target FLOPs per token (if specified)
        memory_constraint: Memory constraint in MB (if specified)
        min_grid_size: Minimum grid size
        max_grid_size: Maximum grid size
        heuristic: Heuristic to use ("sqrt", "linear", "log", "adaptive")
    
    Returns:
        Optimal grid size
    """
    
    if heuristic == "sqrt":
        # Square root heuristic: grid_size = sqrt(seq_len)
        grid_size = int(math.sqrt(seq_len))
    
    elif heuristic == "linear":
        # Linear heuristic: grid_size = seq_len / 8
        grid_size = max(seq_len // 8, min_grid_size)
    
    elif heuristic == "log":
        # Logarithmic heuristic: grid_size = log2(seq_len) * 32
        grid_size = int(math.log2(seq_len) * 32)
    
    elif heuristic == "adaptive":
        # Adaptive heuristic based on sequence length
        if seq_len <= 256:
            grid_size = seq_len // 4
        elif seq_len <= 1024:
            grid_size = seq_len // 8
        elif seq_len <= 4096:
            grid_size = seq_len // 16
        else:
            grid_size = seq_len // 32
    
    else:
        raise ValueError(f"Unknown heuristic: {heuristic}")
    
    # Apply constraints
    grid_size = max(min_grid_size, min(max_grid_size, grid_size))
    
    # Apply FLOPs constraint if specified
    if target_flops is not None:
        current_flops = seq_len * grid_size * embed_dim
        if current_flops > target_flops:
            # Reduce grid size to meet FLOPs target
            grid_size = max(min_grid_size, int(target_flops / (seq_len * embed_dim)))
    
    # Apply memory constraint if specified
    if memory_constraint is not None:
        # Estimate memory usage: batch_size * seq_len * grid_size * embed_dim * 4 bytes
        estimated_memory = seq_len * grid_size * embed_dim * 4 / (1024 * 1024)  # MB
        if estimated_memory > memory_constraint:
            # Reduce grid size to meet memory constraint
            grid_size = max(min_grid_size, int(memory_constraint * 1024 * 1024 / (seq_len * embed_dim * 4)))
    
    return grid_size


def compute_grid_size_per_layer(
    seq_len: int,
    num_layers: int,
    embed_dim: int,
    strategy: str = "constant",
    min_grid_size: int = 32,
    max_grid_size: int = 512
) -> list[int]:
    """Compute grid sizes for each layer in a multi-layer model.
    
    Args:
        seq_len: Length of input sequence
        num_layers: Number of layers
        embed_dim: Embedding dimension
        strategy: Strategy for grid size progression ("constant", "decreasing", "increasing")
        min_grid_size: Minimum grid size
        max_grid_size: Maximum grid size
    
    Returns:
        List of grid sizes for each layer
    """
    
    if strategy == "constant":
        # Same grid size for all layers
        base_grid_size = compute_auto_grid_size(seq_len, embed_dim)
        return [base_grid_size] * num_layers
    
    elif strategy == "decreasing":
        # Decreasing grid size (coarser at depth)
        base_grid_size = compute_auto_grid_size(seq_len, embed_dim)
        grid_sizes = []
        for layer in range(num_layers):
            # Reduce grid size by factor of 2 every 2 layers
            factor = 2 ** (layer // 2)
            grid_size = max(min_grid_size, base_grid_size // factor)
            grid_sizes.append(grid_size)
        return grid_sizes
    
    elif strategy == "increasing":
        # Increasing grid size (finer at depth)
        base_grid_size = compute_auto_grid_size(seq_len, embed_dim)
        grid_sizes = []
        for layer in range(num_layers):
            # Increase grid size by factor of 2 every 2 layers
            factor = 2 ** (layer // 2)
            grid_size = min(max_grid_size, base_grid_size * factor)
            grid_sizes.append(grid_size)
        return grid_sizes
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def estimate_memory_usage(
    batch_size: int,
    seq_len: int,
    grid_size: int,
    embed_dim: int,
    num_layers: int = 1,
    dtype: torch.dtype = torch.float32
) -> Dict[str, float]:
    """Estimate memory usage for TFN model.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        grid_size: Grid size
        embed_dim: Embedding dimension
        num_layers: Number of layers
        dtype: Data type
    
    Returns:
        Dictionary with memory estimates in MB
    """
    
    # Bytes per element
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    
    # Field projection: [B, N, D] -> [B, M, D]
    field_projection_memory = batch_size * seq_len * grid_size * embed_dim * bytes_per_element
    
    # Field evolution: [B, M, D] for each layer
    field_evolution_memory = batch_size * grid_size * embed_dim * num_layers * bytes_per_element
    
    # Field sampling: [B, M, D] -> [B, N, D]
    field_sampling_memory = batch_size * grid_size * seq_len * embed_dim * bytes_per_element
    
    # Total memory
    total_memory = field_projection_memory + field_evolution_memory + field_sampling_memory
    
    return {
        "field_projection_mb": field_projection_memory / (1024 * 1024),
        "field_evolution_mb": field_evolution_memory / (1024 * 1024),
        "field_sampling_mb": field_sampling_memory / (1024 * 1024),
        "total_memory_mb": total_memory / (1024 * 1024)
    }


def estimate_flops(
    seq_len: int,
    grid_size: int,
    embed_dim: int,
    num_layers: int = 1,
    kernel_type: str = "rbf"
) -> Dict[str, float]:
    """Estimate FLOPs for TFN model.
    
    Args:
        seq_len: Sequence length
        grid_size: Grid size
        embed_dim: Embedding dimension
        num_layers: Number of layers
        kernel_type: Type of kernel used
    
    Returns:
        Dictionary with FLOP estimates
    """
    
    # Field projection FLOPs
    if kernel_type == "rbf":
        # RBF kernel: distance computation + exponential
        projection_flops = seq_len * grid_size * (3 + 1)  # 3 for distance, 1 for exp
    elif kernel_type == "compact":
        # Compact kernel: distance computation + threshold
        projection_flops = seq_len * grid_size * 3
    else:
        # Default estimate
        projection_flops = seq_len * grid_size * 4
    
    # Field evolution FLOPs (CNN)
    evolution_flops = grid_size * embed_dim * 3 * num_layers  # 3x3 conv
    
    # Field sampling FLOPs
    sampling_flops = seq_len * grid_size * embed_dim
    
    # Total FLOPs
    total_flops = projection_flops + evolution_flops + sampling_flops
    
    return {
        "projection_flops": projection_flops,
        "evolution_flops": evolution_flops,
        "sampling_flops": sampling_flops,
        "total_flops": total_flops,
        "flops_per_token": total_flops / seq_len
    }


def optimize_grid_size(
    seq_len: int,
    embed_dim: int,
    target_memory_mb: Optional[float] = None,
    target_flops_per_token: Optional[float] = None,
    min_grid_size: int = 32,
    max_grid_size: int = 512
) -> Tuple[int, Dict[str, Any]]:
    """Optimize grid size based on constraints.
    
    Args:
        seq_len: Sequence length
        embed_dim: Embedding dimension
        target_memory_mb: Target memory usage in MB
        target_flops_per_token: Target FLOPs per token
        min_grid_size: Minimum grid size
        max_grid_size: Maximum grid size
    
    Returns:
        (optimal_grid_size, optimization_info)
    """
    
    # Try different grid sizes
    candidates = range(min_grid_size, max_grid_size + 1, 8)
    best_grid_size = min_grid_size
    best_score = float('inf')
    optimization_info = {}
    
    for grid_size in candidates:
        # Check memory constraint
        memory_info = estimate_memory_usage(1, seq_len, grid_size, embed_dim)
        memory_mb = memory_info["total_memory_mb"]
        
        if target_memory_mb and memory_mb > target_memory_mb:
            continue
        
        # Check FLOPs constraint
        flops_info = estimate_flops(seq_len, grid_size, embed_dim)
        flops_per_token = flops_info["flops_per_token"]
        
        if target_flops_per_token and flops_per_token > target_flops_per_token:
            continue
        
        # Compute score (lower is better)
        score = 0
        if target_memory_mb:
            score += (memory_mb / target_memory_mb) ** 2
        if target_flops_per_token:
            score += (flops_per_token / target_flops_per_token) ** 2
        
        if score < best_score:
            best_score = score
            best_grid_size = grid_size
            optimization_info = {
                "memory_mb": memory_mb,
                "flops_per_token": flops_per_token,
                "score": score
            }
    
    return best_grid_size, optimization_info


# ---------------------------------------------------------------------------
# Grid size presets for common scenarios
# ---------------------------------------------------------------------------

GRID_SIZE_PRESETS = {
    "short_text": {
        "seq_len_range": (64, 256),
        "grid_size": lambda seq_len: max(32, seq_len // 4),
        "heuristic": "linear"
    },
    "medium_text": {
        "seq_len_range": (256, 1024),
        "grid_size": lambda seq_len: max(64, seq_len // 8),
        "heuristic": "sqrt"
    },
    "long_text": {
        "seq_len_range": (1024, 4096),
        "grid_size": lambda seq_len: max(128, seq_len // 16),
        "heuristic": "adaptive"
    },
    "very_long_text": {
        "seq_len_range": (4096, 8192),
        "grid_size": lambda seq_len: max(256, seq_len // 32),
        "heuristic": "log"
    }
}


def get_grid_size_preset(seq_len: int) -> Tuple[int, str]:
    """Get grid size preset based on sequence length.
    
    Args:
        seq_len: Sequence length
    
    Returns:
        (grid_size, heuristic_name)
    """
    
    for preset_name, preset in GRID_SIZE_PRESETS.items():
        min_len, max_len = preset["seq_len_range"]
        if min_len <= seq_len <= max_len:
            grid_size = preset["grid_size"](seq_len)
            heuristic = preset["heuristic"]
            return grid_size, heuristic
    
    # Default for very long sequences
    return max(512, seq_len // 64), "log"


if __name__ == "__main__":
    # Test grid size utilities
    seq_lens = [128, 512, 1024, 4096, 8192]
    
    print("Grid size presets:")
    for seq_len in seq_lens:
        grid_size, heuristic = get_grid_size_preset(seq_len)
        print(f"seq_len={seq_len}: grid_size={grid_size}, heuristic={heuristic}")
    
    print("\nMemory estimates (batch_size=1, embed_dim=256):")
    for seq_len in seq_lens:
        grid_size, _ = get_grid_size_preset(seq_len)
        memory_info = estimate_memory_usage(1, seq_len, grid_size, 256)
        print(f"seq_len={seq_len}, grid_size={grid_size}: {memory_info['total_memory_mb']:.1f} MB")
    
    print("\nFLOPs estimates (embed_dim=256):")
    for seq_len in seq_lens:
        grid_size, _ = get_grid_size_preset(seq_len)
        flops_info = estimate_flops(seq_len, grid_size, 256)
        print(f"seq_len={seq_len}, grid_size={grid_size}: {flops_info['flops_per_token']:.0f} FLOPs/token") 