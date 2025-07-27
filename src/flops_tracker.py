"""
FLOPS Tracking System for TFN Training

Provides real-time FLOPS measurement and tracking during training runs
and hyperparameter sweeps.
"""

import torch
import torch.nn as nn
import time
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
import math


class FLOPSProfiler:
    """Profiles FLOPS usage during model forward passes."""
    
    def __init__(self):
        self.total_flops = 0
        self.total_forward_passes = 0
        self.flops_history = []
        self.timing_history = []
        
    def reset(self):
        """Reset profiler state."""
        self.total_flops = 0
        self.total_forward_passes = 0
        self.flops_history = []
        self.timing_history = []
    
    def record_forward_pass(self, flops: int, time_taken: float):
        """Record a forward pass with its FLOPS and timing."""
        self.total_flops += flops
        self.total_forward_passes += 1
        self.flops_history.append(flops)
        self.timing_history.append(time_taken)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get FLOPS statistics."""
        if not self.flops_history:
            return {
                "total_flops": 0,
                "avg_flops_per_pass": 0,
                "total_forward_passes": 0,
                "avg_time_per_pass": 0,
                "flops_per_second": 0
            }
        
        avg_flops = sum(self.flops_history) / len(self.flops_history)
        avg_time = sum(self.timing_history) / len(self.timing_history)
        flops_per_second = avg_flops / avg_time if avg_time > 0 else 0
        
        return {
            "total_flops": self.total_flops,
            "avg_flops_per_pass": avg_flops,
            "total_forward_passes": self.total_forward_passes,
            "avg_time_per_pass": avg_time,
            "flops_per_second": flops_per_second,
            "min_flops": min(self.flops_history),
            "max_flops": max(self.flops_history),
            "std_flops": torch.std(torch.tensor(self.flops_history, dtype=torch.float)).item() if self.flops_history else 0
        }


class TFNFLOPSEstimator:
    """Estimates FLOPS for TFN models based on architecture parameters."""
    
    @staticmethod
    def estimate_tfn_flops(
        batch_size: int,
        seq_len: int,
        embed_dim: int,
        grid_size: int,
        num_layers: int,
        kernel_type: str = "rbf",
        evolution_type: str = "cnn"
    ) -> Dict[str, int]:
        """Estimate FLOPS for TFN model forward pass."""
        
        # Field projection FLOPS
        if kernel_type == "rbf":
            # Distance computation: 3 ops per grid point
            # Exponential: 1 op per grid point
            projection_flops = batch_size * seq_len * grid_size * (3 + 1)
        elif kernel_type == "compact":
            # Distance computation: 3 ops per grid point
            projection_flops = batch_size * seq_len * grid_size * 3
        elif kernel_type == "fourier":
            # Distance + cosine: 4 ops per grid point
            projection_flops = batch_size * seq_len * grid_size * 4
        else:
            # Default estimate
            projection_flops = batch_size * seq_len * grid_size * 4
        
        # Field evolution FLOPS
        if evolution_type == "cnn":
            # 3x3 convolution per layer
            evolution_flops = batch_size * grid_size * embed_dim * 3 * num_layers
        elif evolution_type == "pde":
            # Laplacian computation + update
            evolution_flops = batch_size * grid_size * embed_dim * 5 * num_layers
        else:
            # Default estimate
            evolution_flops = batch_size * grid_size * embed_dim * 4 * num_layers
        
        # Field sampling FLOPS
        sampling_flops = batch_size * seq_len * grid_size * embed_dim
        
        # Embedding and output projection FLOPS
        embedding_flops = batch_size * seq_len * embed_dim
        output_projection_flops = batch_size * seq_len * embed_dim * 2  # Assuming 2-layer head
        
        total_flops = (projection_flops + evolution_flops + sampling_flops + 
                      embedding_flops + output_projection_flops)
        
        return {
            "projection_flops": projection_flops,
            "evolution_flops": evolution_flops,
            "sampling_flops": sampling_flops,
            "embedding_flops": embedding_flops,
            "output_projection_flops": output_projection_flops,
            "total_flops": total_flops,
            "flops_per_token": total_flops // (batch_size * seq_len)
        }
    
    @staticmethod
    def estimate_baseline_flops(
        batch_size: int,
        seq_len: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        model_type: str
    ) -> Dict[str, int]:
        """Estimate FLOPS for baseline models (Transformer, etc.)."""
        
        if model_type == "transformer":
            # Self-attention: O(N²D) per layer
            attention_flops = batch_size * num_layers * seq_len * seq_len * embed_dim * 4
            # FFN: O(ND²) per layer
            ffn_flops = batch_size * num_layers * seq_len * embed_dim * embed_dim * 2
            total_flops = attention_flops + ffn_flops
        elif model_type == "lstm":
            # LSTM: O(ND²) per layer
            total_flops = batch_size * num_layers * seq_len * embed_dim * embed_dim * 4
        elif model_type == "cnn":
            # CNN: O(NDK) where K is kernel size
            kernel_size = 3
            total_flops = batch_size * num_layers * seq_len * embed_dim * kernel_size
        else:
            # Default estimate
            total_flops = batch_size * seq_len * embed_dim * num_layers * 10
        
        return {
            "total_flops": total_flops,
            "flops_per_token": total_flops // (batch_size * seq_len)
        }


class FLOPSWrapper(nn.Module):
    """Wrapper to measure FLOPS during model forward passes."""
    
    def __init__(self, model: nn.Module, estimator: TFNFLOPSEstimator):
        super().__init__()
        self.model = model
        self.estimator = estimator
        self.profiler = FLOPSProfiler()
        self.model_params = {}
        
    def set_model_params(self, **kwargs):
        """Set model parameters for FLOPS estimation."""
        self.model_params = kwargs
    
    def forward(self, *args, **kwargs):
        """Forward pass with FLOPS measurement."""
        start_time = time.time()
        
        # Estimate FLOPS based on input shape and model params
        if hasattr(self.model, 'forward'):
            # Get input shape for estimation
            if args:
                batch_size = args[0].shape[0] if hasattr(args[0], 'shape') else 1
                seq_len = args[0].shape[1] if hasattr(args[0], 'shape') and len(args[0].shape) > 1 else 1
            else:
                batch_size, seq_len = 1, 1
            
            # Estimate FLOPS
            if hasattr(self.model, 'embed_dim'):
                embed_dim = self.model.embed_dim
                grid_size = getattr(self.model, 'grid_size', 100)
                num_layers = getattr(self.model, 'num_layers', 1)
                kernel_type = getattr(self.model, 'kernel_type', 'rbf')
                evolution_type = getattr(self.model, 'evolution_type', 'cnn')
                
                flops_estimate = self.estimator.estimate_tfn_flops(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    embed_dim=embed_dim,
                    grid_size=grid_size,
                    num_layers=num_layers,
                    kernel_type=kernel_type,
                    evolution_type=evolution_type
                )
            else:
                # Baseline model estimation
                embed_dim = getattr(self.model, 'embed_dim', 128)
                num_layers = getattr(self.model, 'num_layers', 1)
                num_heads = getattr(self.model, 'num_heads', 8)
                model_type = getattr(self.model, 'model_type', 'transformer')
                
                flops_estimate = self.estimator.estimate_baseline_flops(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    embed_dim=embed_dim,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    model_type=model_type
                )
            
            estimated_flops = flops_estimate['total_flops']
        else:
            estimated_flops = 0
        
        # Run actual forward pass
        output = self.model(*args, **kwargs)
        
        # Record timing and FLOPS
        time_taken = time.time() - start_time
        self.profiler.record_forward_pass(estimated_flops, time_taken)
        
        return output
    
    def get_flops_stats(self) -> Dict[str, Any]:
        """Get FLOPS statistics."""
        return self.profiler.get_stats()


def create_flops_tracker(model: nn.Module) -> FLOPSWrapper:
    """Create a FLOPS tracking wrapper for a model."""
    estimator = TFNFLOPSEstimator()
    return FLOPSWrapper(model, estimator)


@contextmanager
def flops_profiling(model: nn.Module):
    """Context manager for FLOPS profiling."""
    tracker = create_flops_tracker(model)
    try:
        yield tracker
    finally:
        pass  # Cleanup if needed


def log_flops_stats(stats: Dict[str, Any], prefix: str = ""):
    """Log FLOPS statistics."""
    print(f"{prefix}FLOPS Stats:")
    print(f"{prefix}  Total FLOPS: {stats['total_flops']:,}")
    print(f"{prefix}  Avg FLOPS per pass: {stats['avg_flops_per_pass']:,.0f}")
    print(f"{prefix}  Forward passes: {stats['total_forward_passes']}")
    print(f"{prefix}  Avg time per pass: {stats['avg_time_per_pass']:.4f}s")
    print(f"{prefix}  FLOPS per second: {stats['flops_per_second']:,.0f}")
    if stats['total_forward_passes'] > 1:
        print(f"{prefix}  FLOPS std: {stats['std_flops']:,.0f}") 