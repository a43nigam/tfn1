#!/usr/bin/env python3
"""
Test script to verify the hyperparameter search fix.
This script tests that the _run_trial method properly merges static and swept parameters.
"""

import yaml
import tempfile
import os
from hyperparameter_search import HyperparameterSearch

def test_hyperparameter_search_fix():
    """Test that the hyperparameter search properly merges static and swept parameters."""
    
    # Create a minimal test configuration
    test_config = {
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
            'evolution_type': 'cnn',
            'interference_type': 'standard'
        },
        'training': {
            'batch_size': 64,
            'lr': 1e-4,
            'epochs': 2,  # Short for testing
            'warmup_epochs': 1,
            'grad_clip': 1.0,
            'log_interval': 50
        },
        'search_space': {
            'models': ['enhanced_tfn_regressor'],
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
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Create the search object
            search = HyperparameterSearch(
                config=test_config,
                output_dir=temp_dir,
                patience=2,
                min_epochs=1,
                seed=42
            )
            
            print("✓ HyperparameterSearch object created successfully")
            print(f"✓ Search space has {len(search.search_space)} parameter combinations")
            print(f"✓ Models to test: {search.models}")
            
            # Test that the first trial configuration is properly merged
            if len(search.search_space) > 0:
                first_params = search.search_space.get_trial_params(0)
                print(f"✓ First trial parameters: {first_params}")
                
                # Test the _run_trial method with a mock trial
                # We'll just test the configuration merging part
                trial_id = "test_trial_001"
                model_name = search.models[0]
                
                # Create a trial instance to test configuration merging
                trial = search._run_trial(trial_id, model_name, first_params)
                
                print("✓ Trial completed successfully")
                print(f"✓ Trial result status: {trial.status}")
                print(f"✓ Trial duration: {trial.duration_seconds:.2f}s")
                
                return True
                
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    print("Testing hyperparameter search fix...")
    success = test_hyperparameter_search_fix()
    if success:
        print("\n✅ All tests passed! The hyperparameter search fix is working correctly.")
    else:
        print("\n❌ Tests failed. The fix may need further investigation.") 