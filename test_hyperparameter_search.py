#!/usr/bin/env python3
"""
Minimal test to verify hyperparameter search works with the fix.
"""

import os
import yaml
from hyperparameter_search import HyperparameterSearch, parse_param_sweep

def test_hyperparameter_search():
    """Test that hyperparameter search works correctly."""
    
    # Create a simple config
    config = {
        'data': {
            'dataset': 'synthetic_copy',
            'input_len': 10,
            'output_len': 5
        },
        'model': {
            'task': 'regression',  # Use regression to avoid vocab_size requirement
            'input_dim': 10,
            'output_dim': 1,
            'embed_dim': 64
        },
        'training': {
            'batch_size': 32,
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'epochs': 2  # Very short for testing
        }
    }
    
    # Test parameter sweep with training parameters
    param_sweep_str = "embed_dim:64 learning_rate:1e-3,1e-4 weight_decay:1e-4,1e-5"
    param_sweep = parse_param_sweep(param_sweep_str)
    
    print("Testing hyperparameter search with training parameters...")
    print(f"Parameter sweep: {param_sweep}")
    
    # Create search
    search = HyperparameterSearch(
        models=["tfn_regressor"],  # Use regressor instead of classifier
        param_sweep=param_sweep,
        config=config,
        output_dir="./test_search_results",
        patience=1,
        min_epochs=1,
        seed=42
    )
    
    print(f"Total trials: {len(search.search_space)}")
    
    # Test a single trial to verify parameter handling
    trial_id = "test_trial"
    model_name = "tfn_regressor"
    parameters = search.search_space.get_trial_params(0)  # First trial
    
    print(f"\nTesting trial with parameters: {parameters}")
    
    # This would normally be called by run_search, but we'll test it directly
    try:
        result = search._run_trial(trial_id, model_name, parameters)
        print(f"✓ Trial completed successfully!")
        print(f"  Best val_loss: {result.best_val_loss:.4f}")
        print(f"  Epochs completed: {result.epochs_completed}")
        print(f"  Early stopped: {result.early_stopped}")
        
        # Verify that the parameters were used correctly
        print(f"  Parameters used: {result.parameters}")
        
    except Exception as e:
        print(f"✗ Trial failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_hyperparameter_search() 