#!/usr/bin/env python3
"""
Kaggle-compatible hyperparameter search example.

This script automatically detects if running in Kaggle and uses
the appropriate writable directory for results.
"""

import os
import sys
from hyperparameter_search import HyperparameterSearch, parse_param_sweep

def get_kaggle_output_dir():
    """Get the appropriate output directory for Kaggle environment."""
    if os.path.exists('/kaggle/working'):
        return '/kaggle/working/hyperparameter_search_results'
    else:
        return './search_results'

def main():
    """Run hyperparameter search with Kaggle-compatible settings."""
    
    # Determine output directory
    output_dir = get_kaggle_output_dir()
    print(f"Using output directory: {output_dir}")
    
    # Simple config for testing
    config = {
        'task': 'classification',
        'device': 'cpu',
        'epochs': 5,  # Short for testing
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
        }
    }
    
    # Small parameter sweep for testing
    param_sweep = {
        'embed_dim': [64, 128],
        'num_layers': [1, 2],
        'kernel_type': ['rbf', 'compact']
    }
    
    # Models to test
    models = ['tfn_classifier']  # Just test one model for speed
    
    print("Starting Kaggle-compatible hyperparameter search...")
    print(f"Models: {models}")
    print(f"Parameter sweep: {param_sweep}")
    print(f"Total trials: {len(models) * len(param_sweep['embed_dim']) * len(param_sweep['num_layers']) * len(param_sweep['kernel_type'])}")
    print("-" * 50)
    
    # Create and run search
    search = HyperparameterSearch(
        models=models,
        param_sweep=param_sweep,
        config=config,
        output_dir=output_dir,
        patience=2,
        min_epochs=1,
        seed=42
    )
    
    search.run_search()
    
    print(f"\nSearch completed! Results saved to: {output_dir}")
    
    # List created files
    if os.path.exists(output_dir):
        print("\nCreated files:")
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                print(f"  {file_path}")

if __name__ == "__main__":
    main() 