#!/usr/bin/env python3
"""
Comprehensive test script to verify the hyperparameter search fix.
This script tests various scenarios to ensure the fix works correctly.
"""

import yaml
import tempfile
import os
from hyperparameter_search import HyperparameterSearch

def test_classification_model():
    """Test with a classification model."""
    print("\n=== Testing Classification Model ===")
    
    config = {
        'model_name': 'enhanced_tfn_classifier',
        'task': 'classification',
        'data': {
            'dataset_name': 'synthetic_copy',  # Use available dataset
            'input_len': 128,
            'output_len': 1,
            'vocab_size': 10000,
            'num_classes': 2
        },
        'model': {
            'vocab_size': 10000,
            'embed_dim': 256,
            'num_classes': 2,
            'num_layers': 4,
            'dropout': 0.2,
            'kernel_type': 'rbf',
            'interference_type': 'standard'
        },
        'training': {
            'batch_size': 32,
            'lr': 1e-4,
            'epochs': 2,
            'warmup_epochs': 1
        },
        'search_space': {
            'models': ['enhanced_tfn_classifier'],
            'params': {
                'model.embed_dim': [128, 256],
                'model.num_layers': [2, 4],
                'training.lr': [1e-4, 1e-3]
            }
        },
        'patience': 2,
        'min_epochs': 1,
        'seed': 42
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            search = HyperparameterSearch(
                config=config,
                output_dir=temp_dir,
                patience=2,
                min_epochs=1,
                seed=42
            )
            
            # Test first trial
            first_params = search.search_space.get_trial_params(0)
            trial = search._run_trial("test_classification_001", search.models[0], first_params)
            
            print(f"✓ Classification trial completed: {trial.status}")
            print(f"✓ Final accuracy: {trial.final_val_accuracy:.4f}")
            return True
            
        except Exception as e:
            print(f"❌ Classification test failed: {e}")
            return False

def test_regression_model():
    """Test with a regression model."""
    print("\n=== Testing Regression Model ===")
    
    config = {
        'model_name': 'enhanced_tfn_regressor',
        'task': 'regression',
        'data': {
            'dataset_name': 'ett',
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
            'epochs': 2,
            'warmup_epochs': 1
        },
        'search_space': {
            'models': ['enhanced_tfn_regressor'],
            'params': {
                'model.embed_dim': [128, 256],
                'model.kernel_type': ['rbf', 'compact'],
                'training.lr': [1e-4, 1e-3]
            }
        },
        'patience': 2,
        'min_epochs': 1,
        'seed': 42
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            search = HyperparameterSearch(
                config=config,
                output_dir=temp_dir,
                patience=2,
                min_epochs=1,
                seed=42
            )
            
            # Test first trial
            first_params = search.search_space.get_trial_params(0)
            trial = search._run_trial("test_regression_001", search.models[0], first_params)
            
            print(f"✓ Regression trial completed: {trial.status}")
            print(f"✓ Final MSE: {trial.final_val_mse:.4f}")
            return True
            
        except Exception as e:
            print(f"❌ Regression test failed: {e}")
            return False

def test_multiple_models():
    """Test with multiple models in the search space."""
    print("\n=== Testing Multiple Models ===")
    
    config = {
        'model_name': 'enhanced_tfn_regressor',
        'task': 'regression',
        'data': {
            'dataset_name': 'ett',
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
            'evolution_type': 'cnn',  # Required for tfn_regressor
            'interference_type': 'standard'
        },
        'training': {
            'batch_size': 64,
            'lr': 1e-4,
            'epochs': 2,
            'warmup_epochs': 1
        },
        'search_space': {
            'models': ['enhanced_tfn_regressor', 'tfn_regressor'],
            'params': {
                'model.embed_dim': [128, 256],
                'training.lr': [1e-4, 1e-3]
            }
        },
        'patience': 2,
        'min_epochs': 1,
        'seed': 42
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            search = HyperparameterSearch(
                config=config,
                output_dir=temp_dir,
                patience=2,
                min_epochs=1,
                seed=42
            )
            
            print(f"✓ Search space has {len(search.search_space)} combinations")
            print(f"✓ Testing {len(search.models)} models: {search.models}")
            
            # Test first trial for each model
            for i, model_name in enumerate(search.models):
                first_params = search.search_space.get_trial_params(0)
                trial = search._run_trial(f"test_multi_{i+1:03d}", model_name, first_params)
                print(f"✓ {model_name} trial completed: {trial.status}")
            
            return True
            
        except Exception as e:
            print(f"❌ Multiple models test failed: {e}")
            return False

def test_parameter_validation():
    """Test that missing parameters are properly detected."""
    print("\n=== Testing Parameter Validation ===")
    
    # Create a config with missing required parameters
    config = {
        'model_name': 'enhanced_tfn_regressor',
        'task': 'regression',
        'data': {
            'dataset_name': 'ett',
            'input_len': 96,
            'output_len': 24
        },
        'model': {
            # Missing input_dim, output_dim, output_len
            'embed_dim': 256,
            'num_layers': 4,
            'dropout': 0.2,
            'kernel_type': 'rbf',
            'interference_type': 'standard'
        },
        'training': {
            'batch_size': 64,
            'lr': 1e-4,
            'epochs': 2
        },
        'search_space': {
            'models': ['enhanced_tfn_regressor'],
            'params': {
                'model.embed_dim': [128, 256]
            }
        },
        'patience': 2,
        'min_epochs': 1,
        'seed': 42
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            search = HyperparameterSearch(
                config=config,
                output_dir=temp_dir,
                patience=2,
                min_epochs=1,
                seed=42
            )
            
            # This should fail with a clear error message
            first_params = search.search_space.get_trial_params(0)
            trial = search._run_trial("test_validation_001", search.models[0], first_params)
            
            print("❌ Expected validation error but trial succeeded")
            return False
            
        except ValueError as e:
            print(f"✓ Parameter validation caught missing parameters: {str(e)[:100]}...")
            return True
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False

def main():
    """Run all tests."""
    print("Running comprehensive hyperparameter search tests...")
    
    tests = [
        test_classification_model,
        test_regression_model,
        test_multiple_models,
        test_parameter_validation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n{'='*60}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! The hyperparameter search fix is working correctly.")
    else:
        print("❌ Some tests failed. The fix may need further investigation.")
    
    return passed == total

if __name__ == "__main__":
    main() 