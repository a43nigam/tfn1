"""
Memory monitoring utilities for Colab/Kaggle environments.
"""

import psutil
import torch
import gc
from typing import Dict, Any

def get_memory_info() -> Dict[str, Any]:
    """Get current memory usage information."""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
        'percent': process.memory_percent(),
        'available_mb': psutil.virtual_memory().available / 1024 / 1024,
        'total_mb': psutil.virtual_memory().total / 1024 / 1024,
    }

def print_memory_usage(label: str = "Current"):
    """Print current memory usage."""
    info = get_memory_info()
    print(f"💾 {label} Memory Usage:")
    print(f"   RSS: {info['rss_mb']:.1f} MB")
    print(f"   VMS: {info['vms_mb']:.1f} MB")
    print(f"   Process: {info['percent']:.1f}%")
    print(f"   Available: {info['available_mb']:.1f} MB / {info['total_mb']:.1f} MB")

def clear_memory():
    """Clear memory by running garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("🧹 Memory cleared")

def check_memory_safety(required_mb: float = 1000) -> bool:
    """Check if there's enough memory available."""
    info = get_memory_info()
    return info['available_mb'] > required_mb

class MemoryMonitor:
    """Context manager for monitoring memory usage."""
    
    def __init__(self, label: str = "Operation"):
        self.label = label
        self.start_info = None
    
    def __enter__(self):
        self.start_info = get_memory_info()
        print(f"🚀 Starting {self.label}")
        print_memory_usage(f"{self.label} Start")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_info = get_memory_info()
        print(f"✅ Completed {self.label}")
        print_memory_usage(f"{self.label} End")
        
        if self.start_info:
            rss_diff = end_info['rss_mb'] - self.start_info['rss_mb']
            print(f"📊 Memory change: {rss_diff:+.1f} MB RSS")
        
        if not check_memory_safety():
            print("⚠️  Warning: Low memory available!")
            clear_memory()

def estimate_memory_usage(batch_size: int, seq_len: int, vocab_size: int, embed_dim: int) -> float:
    """Estimate memory usage for a model configuration."""
    # Rough estimation in MB
    embedding_mb = (vocab_size * embed_dim * 4) / 1024 / 1024  # float32
    batch_mb = (batch_size * seq_len * embed_dim * 4) / 1024 / 1024
    gradients_mb = batch_mb * 2  # Rough estimate for gradients
    total_mb = embedding_mb + batch_mb + gradients_mb
    
    return total_mb

def suggest_batch_size(seq_len: int, vocab_size: int, embed_dim: int, target_memory_mb: float = 1000) -> int:
    """Suggest a safe batch size based on memory constraints."""
    # Start with a conservative estimate
    batch_size = 1
    
    while True:
        estimated_memory = estimate_memory_usage(batch_size, seq_len, vocab_size, embed_dim)
        if estimated_memory > target_memory_mb:
            return max(1, batch_size - 1)
        batch_size += 1
        
        # Safety check
        if batch_size > 100:
            return 16  # Default fallback 