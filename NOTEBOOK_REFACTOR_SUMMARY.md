# Notebook-Friendly Refactoring Summary

## Problem Solved

The CLI-based approach was not ideal for the iterative nature of research in notebooks. Running experiments required constructing shell commands (`!python train.py ...`), which was clumsy for passing complex parameters and made debugging difficult.

## Solution Implemented

### 1. **Refactored Core Logic into Functions**

**Training Function: `run_training(config, device="auto")`**
```python
from train import run_training
import yaml

# Load and modify config
with open('configs/ett.yaml', 'r') as f:
    config = yaml.safe_load(f)

config['model']['embed_dim'] = 512
config['training']['epochs'] = 100

# Run training directly
history = run_training(config)
```

**Hyperparameter Search Function: `run_search(config, output_dir=None, device="auto", seed=None)`**
```python
from hyperparameter_search import run_search
import yaml

# Load search config
with open('configs/searches/ett_regression_search.yaml', 'r') as f:
    search_config = yaml.safe_load(f)

# Modify search parameters
search_config['search_space']['params']['model.embed_dim']['values'] = [128, 256]

# Run search
results = run_search(search_config)
```

### 2. **Maintained CLI Compatibility**

The `main()` functions in both files still work exactly as before:
- `train.py` - CLI interface unchanged
- `hyperparameter_search.py` - CLI interface unchanged

### 3. **Enhanced Functionality**

Both functions return useful information:
- **`run_training()`** returns training history dictionary
- **`run_search()`** returns search summary with metadata

## Benefits Achieved

### ✅ **Direct Function Calls**
```python
# Before (clumsy)
!python train.py --config configs/ett.yaml --set model.embed_dim=512 training.epochs=100

# After (clean)
history = run_training(config)
```

### ✅ **Easy Configuration Modification**
```python
# Load and modify
config = yaml.safe_load(open('configs/ett.yaml'))
config['model']['embed_dim'] = 512
config['training']['epochs'] = 100
config['training']['lr'] = 1e-3

# Run with modified config
history = run_training(config)
```

### ✅ **Better Debugging**
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

### ✅ **Iterative Experimentation**
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

## Files Modified

### Core Training Files
- **`train.py`** - Added `run_training()` function, refactored `main()`
- **`hyperparameter_search.py`** - Added `run_search()` function, refactored `main()`

### Documentation and Examples
- **`notebook_example.py`** - Comprehensive examples demonstrating usage
- **`NOTEBOOK_USAGE_GUIDE.md`** - Complete usage guide with examples
- **`NOTEBOOK_REFACTOR_SUMMARY.md`** - This summary document

## Usage Examples

### Basic Training
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

### Hyperparameter Search
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

### Configuration Experiments
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

## Testing Results

✅ **Training Function Tested Successfully**
- Configuration loading and modification works
- Training runs without errors
- History dictionary returned correctly

✅ **Hyperparameter Search Function Tested Successfully**
- Search configuration loading works
- Parameter modification works
- Search execution works correctly

✅ **CLI Compatibility Maintained**
- Original command-line interfaces still work
- No breaking changes to existing workflows

## Migration Guide

### From CLI to Notebook

**Before (CLI approach):**
```bash
# Training
python train.py --config configs/ett.yaml --set model.embed_dim=512 training.epochs=100

# Hyperparameter search
python hyperparameter_search.py --config configs/searches/ett_regression_search.yaml --output_dir ./results
```

**After (Notebook approach):**
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

## Best Practices

### 1. **Use Short Epochs for Testing**
```python
config['training']['epochs'] = 3  # Quick testing
```

### 2. **Use CPU for Quick Experiments**
```python
history = run_training(config, device="cpu")
```

### 3. **Handle Exceptions Gracefully**
```python
try:
    history = run_training(config)
    print(f"Success! Final loss: {history['val_loss'][-1]}")
except Exception as e:
    print(f"Training failed: {e}")
```

### 4. **Save Configurations**
```python
import yaml

# Save modified config
with open('my_experiment_config.yaml', 'w') as f:
    yaml.dump(config, f)
```

## Conclusion

The notebook-friendly refactoring successfully addresses the usability issues in notebook environments while maintaining full CLI compatibility. The new approach provides:

- **🔧 Direct function calls** instead of shell commands
- **📝 Easy configuration modification** with Python dictionaries
- **🐛 Better debugging** with direct access to configurations and results
- **🔄 Iterative experimentation** with programmatic parameter sweeps
- **📊 Natural integration** with Jupyter notebooks and other Python tools

This makes the system much more suitable for research and development workflows where you need to experiment with different configurations and analyze results interactively. 