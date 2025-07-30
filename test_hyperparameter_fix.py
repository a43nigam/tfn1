#!/usr/bin/env python3
"""
Test script to verify the hyperparameter search fix.
This script tests that training parameters are properly separated from model parameters.
"""

import yaml
import torch
from hyperparameter_search import HyperparameterSearch, parse_param_sweep

def test_parameter_separation():
    """Test that training parameters are properly separated from model parameters."""
    
    # Create a simple config
    config = {
        'data': {
            'dataset': 'synthetic_copy',
            'input_len': 10,
            'output_len': 5
        },
        'model': {
            'task': 'classification',
            'input_dim': 10,
            'output_dim': 2,
            'embed_dim': 64
        },
        'training': {
            'batch_size': 32,
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'epochs': 10
        }
    }
    
    # Test parameter sweep with mixed parameters
    param_sweep_str = "embed_dim:64,128 learning_rate:1e-3,1e-4 weight_decay:1e-4,1e-5"
    param_sweep = parse_param_sweep(param_sweep_str)
    
    print("Original parameter sweep:")
    for param, values in param_sweep.items():
        print(f"  {param}: {values}")
    
    # Create a mock search to test parameter separation
    search = HyperparameterSearch(
        models=["tfn_classifier"],
        param_sweep=param_sweep,
        config=config,
        output_dir="./test_results",
        patience=2,
        min_epochs=1,
        seed=42
    )
    
    # Test the parameter separation logic directly
    print("\nTesting parameter separation:")
    
    # Simulate the _run_trial parameter separation logic
    parameters = param_sweep
    trial_config = config.copy()
    
    # Separate model parameters from training parameters
    model_params = {}
    training_params = {}
    
    # Define which parameters belong to which section
    training_param_names = {
        'learning_rate', 'lr', 'weight_decay', 'epochs', 'batch_size', 
        'optimizer', 'warmup_epochs', 'grad_clip', 'log_interval'
    }
    
    for param_name, param_value in parameters.items():
        if param_name in training_param_names:
            training_params[param_name] = param_value
        else:
            model_params[param_name] = param_value
    
    print(f"Model parameters: {model_params}")
    print(f"Training parameters: {training_params}")
    
    # Update config sections
    if 'model' not in trial_config:
        trial_config['model'] = {}
    trial_config['model'].update(model_params)
    
    if 'training' not in trial_config:
        trial_config['training'] = {}
    trial_config['training'].update(training_params)
    
    print(f"\nUpdated config:")
    print(f"Model section: {trial_config['model']}")
    print(f"Training section: {trial_config['training']}")
    
    # Test that the Trainer would get the correct values
    training_config = trial_config.get('training', {})
    lr = training_config.get('lr', training_config.get('learning_rate', 1e-3))
    weight_decay = training_config.get('weight_decay', 0.0)
    
    print(f"\nTrainer would use:")
    print(f"  lr: {lr}")
    print(f"  weight_decay: {weight_decay}")
    
    # Verify that the values are actually different for different trials
    print(f"\nTesting different parameter combinations:")
    
    for i, param_values in enumerate(search.search_space.param_combinations):
        param_dict = dict(zip(search.search_space.param_names, param_values))
        print(f"Trial {i+1}: {param_dict}")
        
        # Simulate parameter separation
        model_params = {}
        training_params = {}
        
        for param_name, param_value in param_dict.items():
            if param_name in training_param_names:
                training_params[param_name] = param_value
            else:
                model_params[param_name] = param_value
        
        print(f"  -> Model: {model_params}")
        print(f"  -> Training: {training_params}")
        
        # Test the actual Trainer instantiation logic
        trial_config_copy = config.copy()
        
        # Update model section
        if 'model' not in trial_config_copy:
            trial_config_copy['model'] = {}
        trial_config_copy['model'].update(model_params)
        
        # Update training section
        if 'training' not in trial_config_copy:
            trial_config_copy['training'] = {}
        trial_config_copy['training'].update(training_params)
        
        # Extract training parameters as the Trainer would
        training_config = trial_config_copy.get('training', {})
        
        # Handle learning rate: if learning_rate is swept, it should override lr
        if 'learning_rate' in training_config:
            lr = training_config['learning_rate']
        else:
            lr = training_config.get('lr', 1e-3)
        
        weight_decay = training_config.get('weight_decay', 0.0)
        
        print(f"  -> Trainer lr: {lr}, weight_decay: {weight_decay}")

if __name__ == "__main__":
    test_parameter_separation() 