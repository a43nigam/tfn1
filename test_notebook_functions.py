#!/usr/bin/env python3
"""
Simple test to verify notebook-friendly functions work correctly.
"""

import yaml
from train import run_training
from hyperparameter_search import run_search

def test_training_function():
    """Test the run_training function."""
    print("Testing run_training function...")
    
    # Create a simple config for testing
    config = {
        'model_name': 'enhanced_tfn_regressor',
        'task': 'time_series',
        'data': {
            'dataset_name': 'ett',
            'csv_path': 'data/ETTh1.csv',
            'input_len': 96,
            'output_len': 24,
            'input_dim': 7,
            'output_dim': 1
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
        }
    }
    
    try:
        print("  Running training...")
        history = run_training(config, device="cpu")
        print("  ✅ Training completed successfully!")
        print(f"  Final training loss: {history.get('train_loss', [])[-1] if history.get('train_loss') else 'N/A'}")
        return True
    except Exception as e:
        print(f"  ❌ Training failed: {e}")
        return False

def test_search_function():
    """Test the run_search function."""
    print("\nTesting run_search function...")
    
    # Create a simple search config for testing
    search_config = {
        'model_name': 'enhanced_tfn_regressor',
        'task': 'time_series',
        'data': {
            'dataset_name': 'ett',
            'csv_path': 'data/ETTh1.csv',
            'input_len': 96,
            'output_len': 24,
            'input_dim': 7,
            'output_dim': 1
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
        'search_space': {
            'models': ['enhanced_tfn_regressor'],
            'params': {
                'model.embed_dim': {
                    'values': [128, 256]
                },
                'training.lr': {
                    'values': [1e-4, 1e-3]
                }
            }
        },
        'patience': 1,
        'min_epochs': 1,
        'seed': 42
    }
    
    try:
        print("  Running hyperparameter search...")
        results = run_search(
            config=search_config,
            output_dir="./test_search_results",
            device="cpu",
            seed=42
        )
        print("  ✅ Search completed successfully!")
        print(f"  Total trials: {results['total_trials']}")
        print(f"  Output directory: {results['output_dir']}")
        return True
    except Exception as e:
        print(f"  ❌ Search failed: {e}")
        return False

def test_function_imports():
    """Test that the functions can be imported correctly."""
    print("\nTesting function imports...")
    
    try:
        from train import run_training
        from hyperparameter_search import run_search
        print("  ✅ Functions imported successfully!")
        return True
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Notebook-Friendly Functions")
    print("=" * 50)
    
    tests = [
        test_function_imports,
        test_training_function,
        test_search_function
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
        print("✅ All tests passed! Notebook-friendly functions are working correctly.")
    else:
        print("❌ Some tests failed. The functions may need further investigation.")
    
    return passed == total

if __name__ == "__main__":
    main() 