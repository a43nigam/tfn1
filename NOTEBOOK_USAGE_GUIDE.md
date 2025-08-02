# Notebook-Friendly Usage Guide

## Overview

The training and hyperparameter search systems have been refactored to be notebook-friendly, making it much easier to work interactively in Jupyter notebooks. Instead of constructing shell commands, you can now directly call functions with Python dictionaries.

## Key Functions

### Training: `run_training(config, device="auto")`

```python
from train import run_training
import yaml

# Load configuration
with open('configs/ett.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Modify configuration
config['model']['embed_dim'] = 512
config['training']['epochs'] = 100

# Run training
history = run_training(config, device="auto")
```

### Hyperparameter Search: `run_search(config, output_dir=None, device="auto", seed=None)`

```python
from hyperparameter_search import run_search
import yaml

# Load search configuration
with open('configs/searches/ett_regression_search.yaml', 'r') as f:
    search_config = yaml.safe_load(f)

# Modify search parameters
search_config['search_space']['params']['model.embed_dim']['values'] = [128, 256]

# Run search
results = run_search(search_config, output_dir="./my_search_results")
```

## Benefits

### 1. **Direct Function Calls**
Instead of shell commands:
```python
# Before (clumsy)
!python train.py --config configs/ett.yaml --set model.embed_dim=512 training.epochs=100

# After (clean)
history = run_training(config)
```

### 2. **Easy Configuration Modification**
```python
# Load and modify
config = yaml.safe_load(open('configs/ett.yaml'))
config['model']['embed_dim'] = 512
config['training']['epochs'] = 100
config['training']['lr'] = 1e-3

# Run with modified config
history = run_training(config)
```

### 3. **Better Debugging**
```python
# Inspect configuration before training
print(f"Model: {config['model_name']}")
print(f"Embed dim: {config['model']['embed_dim']}")
print(f"Epochs: {config['training']['epochs']}")

# Run training
history = run_training(config)

# Inspect results
print(f"Final loss: {history['val_loss'][-1]}")
```

### 4. **Iterative Experimentation**
```python
# Test different configurations
for embed_dim in [128, 256, 512]:
    for lr in [1e-4, 1e-3]:
        config['model']['embed_dim'] = embed_dim
        config['training']['lr'] = lr
        
        print(f"Testing embed_dim={embed_dim}, lr={lr}")
        history = run_training(config, device="cpu")
        
        final_loss = history['val_loss'][-1]
        print(f"  Final loss: {final_loss}")
```

## Complete Examples

### Example 1: Basic Training

```python
from train import run_training
import yaml

# Load configuration
with open('configs/ett.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Modify for quick testing
config['training']['epochs'] = 5
config['model']['embed_dim'] = 256

# Run training
history = run_training(config, device="cpu")

# Inspect results
print(f"Training completed!")
print(f"Final training loss: {history['train_loss'][-1]}")
print(f"Final validation loss: {history['val_loss'][-1]}")
```

### Example 2: Hyperparameter Search

```python
from hyperparameter_search import run_search
import yaml

# Load search configuration
with open('configs/searches/ett_regression_search.yaml', 'r') as f:
    search_config = yaml.safe_load(f)

# Modify for quick testing
search_config['training']['epochs'] = 3
search_config['search_space']['params']['model.embed_dim']['values'] = [128, 256]

# Run search
results = run_search(
    config=search_config,
    output_dir="./quick_search",
    device="cpu"
)

print(f"Search completed!")
print(f"Total trials: {results['total_trials']}")
print(f"Output directory: {results['output_dir']}")
```

### Example 3: Configuration Experiments

```python
from train import run_training
import yaml

# Load base configuration
with open('configs/ett.yaml', 'r') as f:
    base_config = yaml.safe_load(f)

# Experiment with different architectures
experiments = [
    {'embed_dim': 128, 'num_layers': 2},
    {'embed_dim': 256, 'num_layers': 4},
    {'embed_dim': 512, 'num_layers': 6},
]

results = {}

for i, params in enumerate(experiments):
    config = base_config.copy()
    config['model'].update(params)
    config['training']['epochs'] = 3  # Short for testing
    
    print(f"Experiment {i+1}: {params}")
    try:
        history = run_training(config, device="cpu")
        final_loss = history['val_loss'][-1]
        results[f"exp_{i+1}"] = {
            'params': params,
            'final_loss': final_loss
        }
        print(f"  Final loss: {final_loss}")
    except Exception as e:
        print(f"  Failed: {e}")

# Find best configuration
best_exp = min(results.keys(), key=lambda k: results[k]['final_loss'])
print(f"\nBest configuration: {results[best_exp]['params']}")
```

### Example 4: Learning Rate Sweep

```python
from train import run_training
import yaml
import numpy as np

# Load configuration
with open('configs/ett.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Learning rates to test
learning_rates = np.logspace(-5, -2, 5)  # [1e-5, 1e-4, 1e-3, 1e-2]

lr_results = {}

for lr in learning_rates:
    config['training']['lr'] = lr
    config['training']['epochs'] = 3  # Short for testing
    
    print(f"Testing lr={lr:.1e}")
    try:
        history = run_training(config, device="cpu")
        final_loss = history['val_loss'][-1]
        lr_results[lr] = final_loss
        print(f"  Final loss: {final_loss}")
    except Exception as e:
        print(f"  Failed: {e}")

# Find best learning rate
best_lr = min(lr_results.keys(), key=lambda lr: lr_results[lr])
print(f"\nBest learning rate: {best_lr:.1e}")
```

## Advanced Usage

### Custom Configuration Creation

```python
# Create configuration programmatically
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
        'epochs': 10,
        'warmup_epochs': 2,
        'grad_clip': 1.0,
        'log_interval': 50
    }
}

# Run training
history = run_training(config)
```

### Batch Experimentation

```python
import itertools
from train import run_training

# Define parameter ranges
param_ranges = {
    'embed_dim': [128, 256, 512],
    'num_layers': [2, 4, 6],
    'lr': [1e-4, 1e-3]
}

# Generate all combinations
param_names = list(param_ranges.keys())
param_values = list(param_ranges.values())
combinations = list(itertools.product(*param_values))

# Load base config
with open('configs/ett.yaml', 'r') as f:
    base_config = yaml.safe_load(f)

# Run experiments
results = {}

for i, combination in enumerate(combinations):
    config = base_config.copy()
    
    # Update config with current combination
    for param_name, param_value in zip(param_names, combination):
        if param_name == 'lr':
            config['training']['lr'] = param_value
        else:
            config['model'][param_name] = param_value
    
    config['training']['epochs'] = 3  # Short for testing
    
    print(f"Experiment {i+1}/{len(combinations)}: {dict(zip(param_names, combination))}")
    
    try:
        history = run_training(config, device="cpu")
        final_loss = history['val_loss'][-1]
        results[i] = {
            'params': dict(zip(param_names, combination)),
            'final_loss': final_loss
        }
        print(f"  Final loss: {final_loss}")
    except Exception as e:
        print(f"  Failed: {e}")

# Find best configuration
best_exp = min(results.keys(), key=lambda k: results[k]['final_loss'])
print(f"\nBest configuration: {results[best_exp]['params']}")
```

## Migration from CLI

### Before (CLI approach)
```bash
# Training
python train.py --config configs/ett.yaml --set model.embed_dim=512 training.epochs=100

# Hyperparameter search
python hyperparameter_search.py --config configs/searches/ett_regression_search.yaml --output_dir ./results
```

### After (Notebook approach)
```python
# Training
from train import run_training
import yaml

with open('configs/ett.yaml', 'r') as f:
    config = yaml.safe_load(f)

config['model']['embed_dim'] = 512
config['training']['epochs'] = 100

history = run_training(config)

# Hyperparameter search
from hyperparameter_search import run_search

with open('configs/searches/ett_regression_search.yaml', 'r') as f:
    search_config = yaml.safe_load(f)

results = run_search(search_config, output_dir="./results")
```

## Tips and Best Practices

### 1. **Use Short Epochs for Testing**
```python
config['training']['epochs'] = 3  # Quick testing
```

### 2. **Use CPU for Quick Experiments**
```python
history = run_training(config, device="cpu")
```

### 3. **Save Configurations**
```python
import yaml

# Save modified config
with open('my_experiment_config.yaml', 'w') as f:
    yaml.dump(config, f)
```

### 4. **Handle Exceptions Gracefully**
```python
try:
    history = run_training(config)
    print(f"Success! Final loss: {history['val_loss'][-1]}")
except Exception as e:
    print(f"Training failed: {e}")
```

### 5. **Compare Results**
```python
# Store results for comparison
experiment_results = {}

for exp_name, exp_config in experiments.items():
    try:
        history = run_training(exp_config)
        experiment_results[exp_name] = {
            'final_loss': history['val_loss'][-1],
            'history': history
        }
    except Exception as e:
        print(f"Experiment {exp_name} failed: {e}")

# Compare results
for exp_name, result in experiment_results.items():
    print(f"{exp_name}: {result['final_loss']}")
```

## Conclusion

The notebook-friendly approach makes it much easier to:
- **Iterate quickly** on experiments
- **Debug configurations** easily
- **Compare results** programmatically
- **Integrate with other tools** in the notebook environment

This approach is particularly valuable for research and development where you need to experiment with different configurations and analyze results interactively. 