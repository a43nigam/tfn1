#!/usr/bin/env python3
"""
Test script for the modular implementation of normalization strategies and time-based embeddings.
"""

import torch
import yaml
import sys
import os
from typing import Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.timeseries_loader import ETTDataset, create_normalization_strategy
from model.shared_layers import create_positional_embedding_strategy
from model.tfn_unified import TFN


def test_normalization_strategies():
    """Test that all normalization strategies work correctly."""
    print("Testing normalization strategies...")
    
    # Create dummy data
    data = torch.randn(100, 7)  # 100 samples, 7 features
    
    # Test global normalization
    global_normalizer = create_normalization_strategy("global")
    global_normalizer.fit(data.numpy())
    transformed_global = global_normalizer.transform(data.numpy())
    print(f"Global normalization - mean: {transformed_global.mean():.4f}, std: {transformed_global.std():.4f}")
    
    # Test instance normalization
    instance_normalizer = create_normalization_strategy("instance")
    transformed_instance = instance_normalizer.transform(data.numpy())
    print(f"Instance normalization - mean: {transformed_instance.mean():.4f}, std: {transformed_instance.std():.4f}")
    
    # Test feature-wise normalization
    feature_normalizer = create_normalization_strategy("feature_wise")
    feature_normalizer.fit(data.numpy())
    transformed_feature = feature_normalizer.transform(data.numpy())
    print(f"Feature-wise normalization - mean: {transformed_feature.mean():.4f}, std: {transformed_feature.std():.4f}")
    
    print("✅ Normalization strategies test passed!")
    return True


def test_positional_embedding_strategies():
    """Test that all positional embedding strategies work correctly."""
    print("Testing positional embedding strategies...")
    
    max_len = 100
    embed_dim = 128
    batch_size = 4
    seq_len = 50
    
    # Create dummy positions
    positions = torch.randn(batch_size, seq_len, 1)
    
    # Test learned embeddings
    learned_emb = create_positional_embedding_strategy("learned", max_len, embed_dim)
    learned_output = learned_emb(positions)
    print(f"Learned embeddings shape: {learned_output.shape}")
    
    # Test sinusoidal embeddings
    sinusoidal_emb = create_positional_embedding_strategy("sinusoidal", max_len, embed_dim)
    sinusoidal_output = sinusoidal_emb(positions)
    print(f"Sinusoidal embeddings shape: {sinusoidal_output.shape}")
    
    # Test time-based embeddings
    calendar_features = {
        "hour": torch.randint(0, 24, (batch_size, seq_len)),
        "day_of_week": torch.randint(0, 7, (batch_size, seq_len)),
        "day_of_month": torch.randint(1, 32, (batch_size, seq_len)),
        "month": torch.randint(1, 13, (batch_size, seq_len)),
        "is_weekend": torch.randint(0, 2, (batch_size, seq_len)),
    }
    
    time_based_emb = create_positional_embedding_strategy("time_based", max_len, embed_dim)
    time_based_output = time_based_emb(positions, calendar_features=calendar_features)
    print(f"Time-based embeddings shape: {time_based_output.shape}")
    
    print("✅ Positional embedding strategies test passed!")
    return True


def test_ett_dataset_with_new_params():
    """Test ETT dataset with new normalization parameters."""
    print("Testing ETT dataset with new parameters...")
    
    try:
        # Test with instance normalization
        train_ds, val_ds, test_ds = ETTDataset.get_splits(
            csv_path="data/ETTh1.csv",
            input_len=96,
            output_len=24,
            normalization_strategy="instance",
            instance_normalize=True
        )
        
        # Get a sample
        sample = train_ds[0]
        print(f"ETT sample shapes - input: {sample['input'].shape}, target: {sample['target'].shape}")
        
        # Check that calendar features are available
        if hasattr(train_ds, 'calendar_features') and train_ds.calendar_features is not None:
            print(f"Calendar features available: {list(train_ds.calendar_features.keys())}")
        
        print("✅ ETT dataset test passed!")
        
    except Exception as e:
        print(f"❌ ETT dataset test failed: {e}")
        return False
    
    return True


def test_tfn_model_with_new_params():
    """Test TFN model with new positional embedding parameters."""
    print("Testing TFN model with new parameters...")
    
    try:
        # Create model with time-based embeddings
        model = TFN(
            task="regression",
            input_dim=7,
            output_dim=1,
            output_len=24,
            embed_dim=128,
            num_layers=2,
            kernel_type="rbf",
            evolution_type="cnn",
            grid_size=100,
            time_steps=3,
            dropout=0.1,
            positional_embedding_strategy="time_based",
            calendar_features=["hour", "day_of_week", "day_of_month", "month", "is_weekend"]
        )
        
        # Create dummy input
        batch_size = 4
        seq_len = 96
        x = torch.randn(batch_size, seq_len, 7)
        
        # Create dummy calendar features
        calendar_features = {
            "hour": torch.randint(0, 24, (batch_size, seq_len)),
            "day_of_week": torch.randint(0, 7, (batch_size, seq_len)),
            "day_of_month": torch.randint(1, 32, (batch_size, seq_len)),
            "month": torch.randint(1, 13, (batch_size, seq_len)),
            "is_weekend": torch.randint(0, 2, (batch_size, seq_len)),
        }
        
        # Forward pass
        output = model(x, calendar_features=calendar_features)
        print(f"TFN model output shape: {output.shape}")
        
        print("✅ TFN model test passed!")
        
    except Exception as e:
        print(f"❌ TFN model test failed: {e}")
        return False
    
    return True


def test_config_files():
    """Test that the new config files can be loaded."""
    print("Testing config files...")
    
    config_files = [
        "configs/ett_instance_normalization.yaml",
        "configs/ett_time_based_embeddings.yaml",
        "configs/ett_combined_improvements.yaml"
    ]
    
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            print(f"✅ Successfully loaded {config_file}")
            
            # Check for new parameters
            if 'data' in config:
                data_cfg = config['data']
                if 'normalization_strategy' in data_cfg:
                    print(f"  - Normalization strategy: {data_cfg['normalization_strategy']}")
                if 'instance_normalize' in data_cfg:
                    print(f"  - Instance normalize: {data_cfg['instance_normalize']}")
            
            if 'model' in config:
                model_cfg = config['model']
                if 'positional_embedding_strategy' in model_cfg:
                    print(f"  - Positional embedding strategy: {model_cfg['positional_embedding_strategy']}")
                if 'calendar_features' in model_cfg:
                    print(f"  - Calendar features: {model_cfg['calendar_features']}")
                    
        except Exception as e:
            print(f"❌ Failed to load {config_file}: {e}")
            return False
    
    return True


def main():
    """Run all tests."""
    print("🧪 Testing modular implementation...")
    print("=" * 50)
    
    tests = [
        test_normalization_strategies,
        test_positional_embedding_strategies,
        test_ett_dataset_with_new_params,
        test_tfn_model_with_new_params,
        test_config_files
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The modular implementation is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 