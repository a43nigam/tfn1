#!/usr/bin/env python3
"""
Example usage of the hyperparameter search system.

This script demonstrates how to use the hyperparameter search system
with different models and parameter configurations.
"""

import os
import yaml
from hyperparameter_search import HyperparameterSearch, parse_param_sweep

def example_basic_search():
    """Basic hyperparameter search example."""
    
    # Define the parameter sweep
    param_sweep_str = "embed_dim:64,128 num_layers:1,2 kernel_type:rbf,compact"
    param_sweep = parse_param_sweep(param_sweep_str)
    
    # Base configuration
    config = {
        'task': 'classification',
        'device': 'cpu',
        'epochs': 10,
        'learning_rate': 1e-3,
        'weight_decay': 0.0,
        'batch_size': 32,
        'data': {
            'dataset_name': 'synthetic',
            'dataset_size': 1000,
            'seq_len': 20,
            'task': 'classification'
        },
        'model': {
            'vocab_size': 100,
            'num_classes': 2,
            'dropout': 0.1
        }
    }
    
    # Models to search
    models = ['tfn_classifier', 'enhanced_tfn_classifier']
    
    # Run search
    search = HyperparameterSearch(
        models=models,
        param_sweep=param_sweep,
        config=config,
        output_dir='./search_results',
        patience=3,
        min_epochs=2,
        seed=42
    )
    
    search.run_search()

def example_advanced_search():
    """Advanced hyperparameter search with more parameters."""
    
    # More comprehensive parameter sweep
    param_sweep_str = """
    embed_dim:64,128,256 
    num_layers:1,2,3 
    kernel_type:rbf,compact 
    dropout:0.1,0.2
    learning_rate:1e-3,1e-4
    """
    param_sweep = parse_param_sweep(param_sweep_str)
    
    # Advanced configuration
    config = {
        'task': 'classification',
        'device': 'cpu',
        'epochs': 15,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'batch_size': 64,
        'data': {
            'dataset_name': 'synthetic',
            'dataset_size': 2000,
            'seq_len': 50,
            'task': 'classification'
        },
        'model': {
            'vocab_size': 200,
            'num_classes': 2,
            'dropout': 0.1
        }
    }
    
    # Only search TFN for this example
    models = ['tfn_classifier']
    
    # Run search with more patience
    search = HyperparameterSearch(
        models=models,
        param_sweep=param_sweep,
        config=config,
        output_dir='./advanced_search_results',
        patience=5,
        min_epochs=3,
        seed=42
    )
    
    search.run_search()

def example_from_yaml():
    """Example using YAML configuration file."""
    
    # Create a YAML config file
    yaml_config = {
        'task': 'classification',
        'device': 'cpu',
        'epochs': 8,
        'learning_rate': 1e-3,
        'weight_decay': 0.0,
        'batch_size': 32,
        'data': {
            'dataset_name': 'synthetic',
            'dataset_size': 500,
            'seq_len': 15,
            'task': 'classification'
        },
        'model': {
            'vocab_size': 100,
            'num_classes': 2,
            'dropout': 0.1
        }
    }
    
    # Save config to file
    with open('search_config.yaml', 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False)
    
    # Load config from file
    with open('search_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Parameter sweep
    param_sweep = {
        'embed_dim': [64, 128],
        'num_layers': [1, 2],
        'kernel_type': ['rbf', 'compact']
    }
    
    # Run search
    search = HyperparameterSearch(
        models=['tfn_classifier'],
        param_sweep=param_sweep,
        config=config,
        output_dir='./yaml_search_results',
        patience=2,
        min_epochs=1,
        seed=42
    )
    
    search.run_search()
    
    # Clean up
    os.remove('search_config.yaml')

if __name__ == "__main__":
    print("Hyperparameter Search Examples")
    print("=" * 40)
    
    print("\n1. Basic Search:")
    print("- Models: TFN and Enhanced TFN")
    print("- Parameters: embed_dim, num_layers, kernel_type")
    print("- Duration: ~2-3 minutes")
    example_basic_search()
    
    print("\n2. Advanced Search:")
    print("- Model: TFN only")
    print("- Parameters: embed_dim, num_layers, kernel_type, dropout, learning_rate")
    print("- Duration: ~5-10 minutes")
    example_advanced_search()
    
    print("\n3. YAML Configuration Search:")
    print("- Model: TFN only")
    print("- Configuration loaded from YAML file")
    print("- Duration: ~1-2 minutes")
    example_from_yaml()
    
    print("\nAll examples completed!")
    print("Check the output directories for detailed results.") 