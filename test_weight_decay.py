#!/usr/bin/env python3
"""
Test script to verify weight decay functionality in training and hyperparameter search.
"""

import os
import sys
from hyperparameter_search import HyperparameterSearch, parse_param_sweep

def test_weight_decay_hyperparameter_search():
    """Test weight decay in hyperparameter search."""
    
    # Test configuration with weight decay
    config = {
        'task': 'classification',
        'device': 'cpu',
        'epochs': 3,  # Short for testing
        'learning_rate': 1e-3,
        'weight_decay': 0.01,  # Test weight decay
        'batch_size': 32,
        'data': {
            'dataset_name': 'synthetic',
            'dataset_size': 100,
            'seq_len': 10,
            'task': 'classification'
        },
        'model': {
            'vocab_size': 100,
            'num_classes': 2,
            'embed_dim': 64,
            'num_layers': 1,
            'kernel_type': 'rbf',
            'evolution_type': 'cnn',
            'grid_size': 50,
            'dropout': 0.1
        },
        'training': {
            'batch_size': 32,
            'lr': 1e-3,
            'weight_decay': 0.01,
            'epochs': 3
        }
    }
    
    # Parameter sweep including weight decay
    param_sweep = {
        'embed_dim': [64],
        'num_layers': [1],
        'kernel_type': ['rbf'],
        'weight_decay': [0.0, 0.01, 0.1]  # Test different weight decay values
    }
    
    print("Testing weight decay in hyperparameter search...")
    print(f"Parameter sweep: {param_sweep}")
    
    # Create search
    search = HyperparameterSearch(
        models=['tfn_classifier'],
        param_sweep=param_sweep,
        config=config,
        output_dir='./test_weight_decay_output',
        patience=2,
        min_epochs=1,
        seed=42
    )
    
    # Run a single trial to test
    try:
        result = search._run_trial("test_001", "tfn_classifier", {"embed_dim": 64, "num_layers": 1, "kernel_type": "rbf", "weight_decay": 0.01})
        print(f"✅ Weight decay test completed successfully!")
        print(f"   Best val_loss: {result.best_val_loss:.4f}")
        print(f"   Best val_accuracy: {result.best_val_accuracy:.4f}")
        return True
    except Exception as e:
        print(f"❌ Weight decay test failed: {str(e)}")
        return False

def test_weight_decay_cli():
    """Test weight decay CLI parsing."""
    
    # Test parameter sweep string with weight decay
    param_sweep_str = "embed_dim:64,128 weight_decay:0.0,0.01,0.1"
    param_sweep = parse_param_sweep(param_sweep_str)
    
    print(f"Testing weight decay CLI parsing...")
    print(f"Input: {param_sweep_str}")
    print(f"Parsed: {param_sweep}")
    
    expected = {
        'embed_dim': [64, 128],
        'weight_decay': [0.0, 0.01, 0.1]
    }
    
    if param_sweep == expected:
        print("✅ Weight decay CLI parsing works correctly!")
        return True
    else:
        print(f"❌ Weight decay CLI parsing failed!")
        print(f"Expected: {expected}")
        print(f"Got: {param_sweep}")
        return False

if __name__ == "__main__":
    print("Testing weight decay functionality...")
    print("=" * 50)
    
    # Test CLI parsing
    cli_success = test_weight_decay_cli()
    print()
    
    # Test hyperparameter search
    search_success = test_weight_decay_hyperparameter_search()
    print()
    
    if cli_success and search_success:
        print("🎉 All weight decay tests passed!")
        sys.exit(0)
    else:
        print("💥 Some weight decay tests failed!")
        sys.exit(1) 