"""
Test script for hyperparameter search system.

This script runs a quick test of the hyperparameter search with minimal
parameters to verify everything works correctly.
"""

import os
import tempfile
import shutil
from hyperparameter_search import HyperparameterSearch, parse_param_sweep


def test_hyperparameter_search():
    """Test the hyperparameter search system with minimal parameters."""
    
    # Create temporary output directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Minimal config for testing
        config = {
            'task': 'classification',
            'device': 'cpu',
            'epochs': 3,  # Very short for testing
            'learning_rate': 1e-3,
            'weight_decay': 0.0,
            'batch_size': 32,
            'data': {
                'dataset_name': 'synthetic',
                'dataset_size': 100,  # Small dataset for testing
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
                'batch_size': 32
            }
        }
        
        # Minimal parameter sweep for testing
        param_sweep = {
            'embed_dim': [64, 128],
            'num_layers': [1, 2],
            'kernel_type': ['rbf', 'compact']
        }
        
        # Models to test
        models = ['tfn_classifier']  # Just test one model for speed
        
        print("Testing hyperparameter search system...")
        print(f"Output directory: {temp_dir}")
        print(f"Models: {models}")
        print(f"Parameter sweep: {param_sweep}")
        print("-" * 50)
        
        # Create and run search
        search = HyperparameterSearch(
            models=models,
            param_sweep=param_sweep,
            config=config,
            output_dir=temp_dir,
            patience=2,
            min_epochs=1,
            seed=42
        )
        

        
        search.run_search()
        
        # Check that results were created
        assert os.path.exists(os.path.join(temp_dir, "search_config.json")), "Search config not created"
        assert os.path.exists(os.path.join(temp_dir, "summary.json")), "Summary not created"
        assert os.path.exists(os.path.join(temp_dir, "trials")), "Trials directory not created"
        
        # Check that trials were created
        trial_files = [f for f in os.listdir(os.path.join(temp_dir, "trials")) if f.endswith('.json')]
        expected_trials = len(models) * len(param_sweep['embed_dim']) * len(param_sweep['num_layers']) * len(param_sweep['kernel_type'])
        assert len(trial_files) == expected_trials, f"Expected {expected_trials} trials, got {len(trial_files)}"
        
        print("✅ Hyperparameter search test completed successfully!")
        print(f"Created {len(trial_files)} trial files")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        raise
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_param_sweep_parsing():
    """Test parameter sweep string parsing."""
    
    # Test cases
    test_cases = [
        ("embed_dim:128,256 num_layers:2,4", {
            'embed_dim': [128, 256],
            'num_layers': [2, 4]
        }),
        ("kernel_type:rbf,compact,linear", {
            'kernel_type': ['rbf', 'compact', 'linear']
        }),
        ("learning_rate:1e-3,1e-4 dropout:0.1,0.2", {
            'learning_rate': [0.001, 0.0001],
            'dropout': [0.1, 0.2]
        })
    ]
    
    for param_str, expected in test_cases:
        result = parse_param_sweep(param_str)
        assert result == expected, f"Expected {expected}, got {result}"
    
    print("✅ Parameter sweep parsing test passed!")


if __name__ == "__main__":
    print("Running hyperparameter search tests...")
    
    # Test parameter sweep parsing
    test_param_sweep_parsing()
    
    # Test full hyperparameter search (only if not in CI)
    if os.getenv('CI') != 'true':
        test_hyperparameter_search()
    else:
        print("Skipping full search test in CI environment")
    
    print("All tests passed!") 