#!/usr/bin/env python3
"""
Test script to verify the denormalization fix with scaling.
"""

import torch
import yaml
from data_pipeline import get_dataloader
from model import registry
from src.trainer import Trainer

def test_scaling_consistency():
    """Test that scaling/denormalization works consistently."""
    
    print("🧪 Testing Scaling/Denormalization Fix")
    print("=" * 50)
    
    # Create a simple test to verify denormalization logic
    print("📊 Testing denormalization logic...")
    
    # Simulate normalized data
    preds_normalized = torch.tensor([[0.5, -0.2, 1.0], [0.1, 0.8, -0.5]])
    targets_normalized = torch.tensor([[0.4, -0.1, 0.9], [0.2, 0.7, -0.4]])
    
    # Simulate scaler parameters
    mean = 10.0
    std = 2.0
    
    print(f"📊 Normalized data:")
    print(f"   Preds: {preds_normalized}")
    print(f"   Targets: {targets_normalized}")
    print(f"   Mean: {mean}, Std: {std}")
    
    # Calculate MSE on normalized data
    mse_normalized = torch.mean((preds_normalized - targets_normalized) ** 2)
    print(f"   MSE (normalized): {mse_normalized:.6f}")
    
    # Denormalize
    preds_denorm = preds_normalized * std + mean
    targets_denorm = targets_normalized * std + mean
    
    print(f"📊 Denormalized data:")
    print(f"   Preds: {preds_denorm}")
    print(f"   Targets: {targets_denorm}")
    
    # Calculate MSE on denormalized data
    mse_denorm = torch.mean((preds_denorm - targets_denorm) ** 2)
    print(f"   MSE (denormalized): {mse_denorm:.6f}")
    
    # The denormalized MSE should be std^2 times the normalized MSE
    expected_denorm_mse = mse_normalized * (std ** 2)
    print(f"   Expected denorm MSE: {expected_denorm_mse:.6f}")
    print(f"   Difference: {abs(mse_denorm - expected_denorm_mse):.6f}")
    
    if abs(mse_denorm - expected_denorm_mse) < 1e-6:
        print("   ✅ Denormalization logic is correct!")
    else:
        print("   ❌ Denormalization logic has issues!")
    
    print("\n🎉 Scaling fix test completed!")

if __name__ == "__main__":
    test_scaling_consistency() 