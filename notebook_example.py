#!/usr/bin/env python3
"""
Notebook-Friendly Training and Hyperparameter Search Example

This script demonstrates how to use the new notebook-friendly functions
for both training and hyperparameter search. This makes it much easier
to work interactively in Jupyter notebooks.

Example usage in a notebook:

# Training
from train import run_training
import yaml

# Load and modify config
with open('configs/ett.yaml', 'r') as f:
    config = yaml.safe_load(f)

config['model']['embed_dim'] = 512
config['training']['epochs'] = 100

# Run training directly
history = run_training(config)

# Hyperparameter Search
from hyperparameter_search import run_search

# Load search config
with open('configs/searches/ett_regression_search.yaml', 'r') as f:
    search_config = yaml.safe_load(f)

# Modify search parameters
search_config['search_space']['params']['model.embed_dim']['values'] = [128, 256]

# Run search
results = run_search(search_config)
"""

import yaml
import torch
from train import run_training
from hyperparameter_search import run_search

def example_training():
    """Example of using run_training function."""
    print("=== Training Example ===")
    
    # Load base configuration
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
            'epochs': 5,  # Short for example
            'warmup_epochs': 1,
            'grad_clip': 1.0,
            'log_interval': 50
        }
    }
    
    print("Running training with configuration:")
    print(f"  Model: {config['model_name']}")
    print(f"  Embed dim: {config['model']['embed_dim']}")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Learning rate: {config['training']['lr']}")
    
    try:
        # Run training
        history = run_training(config, device="cpu")
        print("✅ Training completed successfully!")
        print(f"  Final training loss: {history.get('train_loss', [])[-1] if history.get('train_loss') else 'N/A'}")
        print(f"  Final validation loss: {history.get('val_loss', [])[-1] if history.get('val_loss') else 'N/A'}")
        return history
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

def example_hyperparameter_search():
    """Example of using run_search function."""
    print("\n=== Hyperparameter Search Example ===")
    
    # Create search configuration
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
            'epochs': 3,  # Very short for example
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
                'model.num_layers': {
                    'values': [2, 4]
                },
                'training.lr': {
                    'logspace': [1e-4, 1e-3],
                    'steps': 2
                }
            }
        },
        'patience': 2,
        'min_epochs': 1,
        'seed': 42
    }
    
    print("Running hyperparameter search with configuration:")
    print(f"  Models: {search_config['search_space']['models']}")
    print(f"  Parameters: {list(search_config['search_space']['params'].keys())}")
    print(f"  Total trials: {len(search_config['search_space']['models'])} × {2*2*2} = 8")
    
    try:
        # Run search
        results = run_search(
            config=search_config,
            output_dir="./example_search_results",
            device="cpu",
            seed=42
        )
        print("✅ Hyperparameter search completed successfully!")
        print(f"  Output directory: {results['output_dir']}")
        print(f"  Total trials: {results['total_trials']}")
        return results
    except Exception as e:
        print(f"❌ Hyperparameter search failed: {e}")
        return None

def example_config_modification():
    """Example of modifying configurations programmatically."""
    print("\n=== Configuration Modification Example ===")
    
    # Load base config
    base_config = {
        'model_name': 'enhanced_tfn_regressor',
        'task': 'time_series',
        'data': {
            'dataset_name': 'ett',
            'csv_path': 'data/ETTh1.csv',
            'input_len': 96,
            'output_len': 24
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
            'epochs': 10,
            'warmup_epochs': 2
        }
    }
    
    # Experiment 1: Different embedding dimensions
    print("Experiment 1: Testing different embedding dimensions")
    for embed_dim in [128, 256, 512]:
        config = base_config.copy()
        config['model']['embed_dim'] = embed_dim
        config['training']['epochs'] = 2  # Short for example
        
        print(f"  Testing embed_dim={embed_dim}")
        try:
            history = run_training(config, device="cpu")
            final_loss = history.get('val_loss', [])[-1] if history.get('val_loss') else 'N/A'
            print(f"    Final val_loss: {final_loss}")
        except Exception as e:
            print(f"    Failed: {e}")
    
    # Experiment 2: Different learning rates
    print("\nExperiment 2: Testing different learning rates")
    for lr in [1e-5, 1e-4, 1e-3]:
        config = base_config.copy()
        config['training']['lr'] = lr
        config['training']['epochs'] = 2  # Short for example
        
        print(f"  Testing lr={lr}")
        try:
            history = run_training(config, device="cpu")
            final_loss = history.get('val_loss', [])[-1] if history.get('val_loss') else 'N/A'
            print(f"    Final val_loss: {final_loss}")
        except Exception as e:
            print(f"    Failed: {e}")

def example_yaml_loading():
    """Example of loading and modifying YAML configurations."""
    print("\n=== YAML Configuration Example ===")
    
    # Create a sample YAML config
    sample_config = {
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
            'epochs': 3,
            'warmup_epochs': 1
        }
    }
    
    # Save to temporary file
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_config, f)
        temp_file = f.name
    
    try:
        # Load and modify config
        with open(temp_file, 'r') as f:
            config = yaml.safe_load(f)
        
        print("Loaded configuration from YAML:")
        print(f"  Model: {config['model_name']}")
        print(f"  Embed dim: {config['model']['embed_dim']}")
        print(f"  Epochs: {config['training']['epochs']}")
        
        # Modify configuration
        config['model']['embed_dim'] = 512
        config['training']['epochs'] = 5
        config['training']['lr'] = 1e-3
        
        print("\nModified configuration:")
        print(f"  Embed dim: {config['model']['embed_dim']}")
        print(f"  Epochs: {config['training']['epochs']}")
        print(f"  Learning rate: {config['training']['lr']}")
        
        # Run training with modified config
        print("\nRunning training with modified config...")
        history = run_training(config, device="cpu")
        print("✅ Training completed successfully!")
        
    finally:
        # Clean up temporary file
        os.unlink(temp_file)

def main():
    """Run all examples."""
    print("🚀 Notebook-Friendly Training and Hyperparameter Search Examples")
    print("=" * 70)
    
    # Example 1: Basic training
    example_training()
    
    # Example 2: Hyperparameter search
    example_hyperparameter_search()
    
    # Example 3: Configuration modification
    example_config_modification()
    
    # Example 4: YAML loading and modification
    example_yaml_loading()
    
    print("\n" + "=" * 70)
    print("✅ All examples completed!")
    print("\nKey benefits of the notebook-friendly approach:")
    print("  • Direct function calls instead of shell commands")
    print("  • Easy configuration modification")
    print("  • Better debugging and inspection")
    print("  • Natural integration with Jupyter notebooks")

if __name__ == "__main__":
    main() 