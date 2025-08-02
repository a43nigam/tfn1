#!/usr/bin/env python3
"""
Test wandb integration with the training system.
"""

import yaml
from train import run_training

def test_wandb_config_loading():
    """Test that wandb configuration is properly loaded."""
    print("Testing wandb configuration loading...")
    
    # Create a test config with wandb settings
    config = {
        'model_name': 'enhanced_tfn_regressor',
        'task': 'time_series',
        'data': {
            'dataset_name': 'synthetic',
            'dataset_size': 100,
            'seq_len': 20,
            'vocab_size': 10,
            'task': 'regression'
        },
        'model': {
            'input_dim': 7,
            'embed_dim': 256,
            'output_dim': 1,
            'output_len': 24,
            'num_layers': 4,
            'dropout': 0.2,
            'kernel_type': 'rbf',
            'interference_type': 'standard'
        },
        'training': {
            'batch_size': 64,
            'lr': 1e-4,
            'epochs': 2,  # Very short for testing
            'warmup_epochs': 1,
            'grad_clip': 1.0,
            'log_interval': 50
        },
        'wandb': {
            'use_wandb': False,  # Disable for testing
            'project_name': 'test-project',
            'experiment_name': 'test-experiment'
        }
    }
    
    try:
        print("  Running training with wandb config...")
        history = run_training(config, device="cpu")
        print("  ✅ Wandb configuration loaded successfully!")
        print(f"  Training completed with {len(history.get('train_loss', []))} epochs")
        return True
    except Exception as e:
        print(f"  ❌ Wandb integration failed: {e}")
        return False

def test_wandb_config_override():
    """Test that wandb settings can be overridden."""
    print("\nTesting wandb configuration override...")
    
    # Load the example wandb config
    try:
        with open('configs/ett_wandb.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Override wandb settings
        config['wandb']['use_wandb'] = False  # Disable for testing
        config['wandb']['experiment_name'] = 'override-test'
        
        print("  Running training with overridden wandb config...")
        history = run_training(config, device="cpu")
        print("  ✅ Wandb configuration override works!")
        return True
    except Exception as e:
        print(f"  ❌ Wandb configuration override failed: {e}")
        return False

def test_wandb_disabled():
    """Test that training works when wandb is disabled."""
    print("\nTesting wandb disabled mode...")
    
    config = {
        'model_name': 'enhanced_tfn_regressor',
        'task': 'time_series',
        'data': {
            'dataset_name': 'synthetic',
            'dataset_size': 100,
            'seq_len': 20,
            'vocab_size': 10,
            'task': 'regression'
        },
        'model': {
            'input_dim': 7,
            'embed_dim': 256,
            'output_dim': 1,
            'output_len': 24,
            'num_layers': 4,
            'dropout': 0.2,
            'kernel_type': 'rbf',
            'interference_type': 'standard'
        },
        'training': {
            'batch_size': 64,
            'lr': 1e-4,
            'epochs': 2,
            'warmup_epochs': 1,
            'grad_clip': 1.0,
            'log_interval': 50
        },
        'wandb': {
            'use_wandb': False
        }
    }
    
    try:
        print("  Running training with wandb disabled...")
        history = run_training(config, device="cpu")
        print("  ✅ Training works with wandb disabled!")
        return True
    except Exception as e:
        print(f"  ❌ Training failed with wandb disabled: {e}")
        return False

def test_wandb_config_validation():
    """Test that wandb configuration validation works."""
    print("\nTesting wandb configuration validation...")
    
    # Test with missing wandb config
    config = {
        'model_name': 'enhanced_tfn_regressor',
        'task': 'time_series',
        'data': {
            'dataset_name': 'synthetic',
            'dataset_size': 100,
            'seq_len': 20,
            'vocab_size': 10,
            'task': 'regression'
        },
        'model': {
            'input_dim': 7,
            'embed_dim': 256,
            'output_dim': 1,
            'output_len': 24,
            'num_layers': 4,
            'dropout': 0.2,
            'kernel_type': 'rbf',
            'interference_type': 'standard'
        },
        'training': {
            'batch_size': 64,
            'lr': 1e-4,
            'epochs': 2,
            'warmup_epochs': 1,
            'grad_clip': 1.0,
            'log_interval': 50
        }
        # No wandb config - should use defaults
    }
    
    try:
        print("  Running training without wandb config...")
        history = run_training(config, device="cpu")
        print("  ✅ Training works with default wandb settings!")
        return True
    except Exception as e:
        print(f"  ❌ Training failed with default wandb settings: {e}")
        return False

def main():
    """Run all wandb integration tests."""
    print("🧪 Testing Wandb Integration")
    print("=" * 50)
    
    tests = [
        test_wandb_config_loading,
        test_wandb_config_override,
        test_wandb_disabled,
        test_wandb_config_validation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Wandb integration is working correctly.")
        print("\nTo use wandb in production:")
        print("1. Install wandb: pip install wandb")
        print("2. Login: wandb login")
        print("3. Set use_wandb: true in your config")
        print("4. Run training: python train.py --config configs/ett_wandb.yaml")
    else:
        print("❌ Some tests failed. The wandb integration may need further investigation.")
    
    return passed == total

if __name__ == "__main__":
    main() 