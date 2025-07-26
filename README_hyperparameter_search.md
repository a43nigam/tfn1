# Hyperparameter Search System

A comprehensive hyperparameter optimization system for Token Field Networks (TFN) that can sweep through different models and parameters with early stopping and detailed result logging.

## Features

- **Multi-Model Support**: Search across TFN and Enhanced TFN models
- **Flexible Parameter Sweeps**: Configure any model parameters for optimization
- **Early Stopping**: Automatically stop trials when training stagnates
- **Detailed Logging**: JSON results with timestamps, metrics, and parameters
- **Kernel Type Support**: Includes RBF and compact kernel options
- **Comprehensive Metrics**: Tracks loss, accuracy, MSE, and MAE
- **Reproducible Results**: Configurable random seeds

## Quick Start

### Basic Usage

```python
from hyperparameter_search import HyperparameterSearch, parse_param_sweep

# Define parameter sweep
param_sweep_str = "embed_dim:64,128 num_layers:1,2 kernel_type:rbf,compact"
param_sweep = parse_param_sweep(param_sweep_str)

# Base configuration
config = {
    'task': 'classification',
    'device': 'cpu',
    'epochs': 10,
    'learning_rate': 1e-3,
    'batch_size': 32,
    'data': {
        'dataset_name': 'synthetic',
        'dataset_size': 1000,
        'seq_len': 20,
        'task': 'classification'
    },
    'model': {
        'vocab_size': 100,
        'num_classes': 2
    }
}

# Run search
search = HyperparameterSearch(
    models=['tfn_classifier', 'enhanced_tfn_classifier'],
    param_sweep=param_sweep,
    config=config,
    output_dir='./search_results',
    patience=3,
    min_epochs=2,
    seed=42
)

search.run_search()
```

### Command Line Usage

```bash
# Basic search
python hyperparameter_search.py \
    --models tfn_classifier enhanced_tfn_classifier \
    --param_sweep "embed_dim:64,128 num_layers:1,2 kernel_type:rbf,compact" \
    --epochs 10 \
    --patience 3 \
    --output_dir ./search_results

# Advanced search with more parameters
python hyperparameter_search.py \
    --models tfn_classifier \
    --param_sweep "embed_dim:64,128,256 num_layers:1,2,3 kernel_type:rbf,compact dropout:0.1,0.2 learning_rate:1e-3,1e-4" \
    --epochs 15 \
    --patience 5 \
    --min_epochs 3 \
    --output_dir ./advanced_search_results
```

## Configuration

### Parameter Sweep Format

The parameter sweep can be specified as a string with space-separated parameters:

```
"param1:value1,value2 param2:value1,value2,value3"
```

Examples:
- `"embed_dim:64,128 num_layers:1,2"`
- `"kernel_type:rbf,compact dropout:0.1,0.2"`
- `"learning_rate:1e-3,1e-4 weight_decay:0.0,1e-4"`

### Supported Parameters

| Parameter | Description | Values |
|-----------|-------------|---------|
| `embed_dim` | Embedding dimension | 64, 128, 256, 512 |
| `num_layers` | Number of TFN layers | 1, 2, 3, 4, 6 |
| `kernel_type` | Field kernel type | `rbf`, `compact` |
| `dropout` | Dropout rate | 0.0, 0.1, 0.2, 0.3 |
| `learning_rate` | Learning rate | 1e-4, 1e-3, 1e-2 |
| `weight_decay` | Weight decay | 0.0, 1e-4, 1e-3 |

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `models` | List of models to search | `['tfn_classifier']` |
| `param_sweep` | Parameter combinations | `{}` |
| `config` | Base configuration | `{}` |
| `output_dir` | Results directory | `./search_results` |
| `patience` | Early stopping patience | `5` |
| `min_epochs` | Minimum epochs before stopping | `2` |
| `seed` | Random seed | `42` |

## Output Format

### Directory Structure

```
search_results/
├── search_summary.json          # Overall search summary
├── trial_001_tfn_classifier.json    # Individual trial results
├── trial_002_tfn_classifier.json
├── trial_003_enhanced_tfn_classifier.json
└── ...
```

### Trial Results Format

Each trial produces a JSON file with:

```json
{
  "trial_id": "trial_001",
  "model_name": "tfn_classifier",
  "parameters": {
    "embed_dim": 64,
    "num_layers": 1,
    "kernel_type": "rbf"
  },
  "start_time": "2024-01-15T10:30:00",
  "end_time": "2024-01-15T10:32:30",
  "duration_seconds": 150.5,
  "epochs_completed": 8,
  "early_stopped": true,
  "early_stop_reason": "Validation loss stagnated for 3 epochs",
  "best_metrics": {
    "val_loss": 0.6710,
    "val_accuracy": 0.5547,
    "val_mse": null,
    "val_mae": null
  },
  "epoch_history": [
    {
      "epoch": 1,
      "timestamp": "2024-01-15T10:30:15",
      "train_loss": 0.8234,
      "train_accuracy": 0.4219,
      "val_loss": 0.7456,
      "val_accuracy": 0.4688
    },
    {
      "epoch": 2,
      "timestamp": "2024-01-15T10:30:45",
      "train_loss": 0.7123,
      "train_accuracy": 0.5312,
      "val_loss": 0.6987,
      "val_accuracy": 0.5156
    }
  ]
}
```

### Search Summary Format

```json
{
  "search_id": "search_20240115_103000",
  "start_time": "2024-01-15T10:30:00",
  "end_time": "2024-01-15T10:45:30",
  "total_duration_seconds": 930.5,
  "models_searched": ["tfn_classifier", "enhanced_tfn_classifier"],
  "total_trials": 16,
  "completed_trials": 16,
  "failed_trials": 0,
  "best_trial": {
    "trial_id": "trial_005",
    "model_name": "tfn_classifier",
    "parameters": {
      "embed_dim": 128,
      "num_layers": 1,
      "kernel_type": "rbf"
    },
    "best_val_loss": 0.6065,
    "best_val_accuracy": 0.7422
  },
  "trial_summaries": [
    {
      "trial_id": "trial_001",
      "model_name": "tfn_classifier",
      "best_val_loss": 0.6710,
      "best_val_accuracy": 0.5547,
      "epochs_completed": 8,
      "early_stopped": true
    }
  ]
}
```

## Examples

### Example 1: Basic TFN Search

```python
# Search TFN with basic parameters
param_sweep = parse_param_sweep("embed_dim:64,128 num_layers:1,2 kernel_type:rbf,compact")

search = HyperparameterSearch(
    models=['tfn_classifier'],
    param_sweep=param_sweep,
    config=base_config,
    output_dir='./tfn_search',
    patience=3,
    min_epochs=2
)

search.run_search()
```

### Example 2: Multi-Model Comparison

```python
# Compare TFN and Enhanced TFN
param_sweep = parse_param_sweep("embed_dim:128,256 num_layers:2,3 kernel_type:rbf")

search = HyperparameterSearch(
    models=['tfn_classifier', 'enhanced_tfn_classifier'],
    param_sweep=param_sweep,
    config=base_config,
    output_dir='./model_comparison',
    patience=5,
    min_epochs=3
)

search.run_search()
```

### Example 3: Comprehensive Search

```python
# Search with many parameters
param_sweep = parse_param_sweep("""
    embed_dim:64,128,256 
    num_layers:1,2,3 
    kernel_type:rbf,compact 
    dropout:0.1,0.2
    learning_rate:1e-3,1e-4
""")

search = HyperparameterSearch(
    models=['tfn_classifier'],
    param_sweep=param_sweep,
    config=base_config,
    output_dir='./comprehensive_search',
    patience=5,
    min_epochs=3
)

search.run_search()
```

## Advanced Features

### Early Stopping

The system automatically stops training when:
- Validation loss doesn't improve for `patience` epochs
- Minimum epochs (`min_epochs`) have been completed

### Stagnation Detection

Training is considered stagnant when:
- Validation loss doesn't decrease by more than 1e-4 for `patience` consecutive epochs
- The model has trained for at least `min_epochs` epochs

### Result Analysis

After running a search, you can analyze results:

```python
import json

# Load search summary
with open('search_results/search_summary.json', 'r') as f:
    summary = json.load(f)

# Find best trial
best_trial = summary['best_trial']
print(f"Best trial: {best_trial['trial_id']}")
print(f"Best val_loss: {best_trial['best_val_loss']}")
print(f"Best val_accuracy: {best_trial['best_val_accuracy']}")

# Load detailed trial results
with open(f"search_results/{best_trial['trial_id']}_{best_trial['model_name']}.json", 'r') as f:
    trial_results = json.load(f)

# Analyze training history
for epoch in trial_results['epoch_history']:
    print(f"Epoch {epoch['epoch']}: val_loss={epoch['val_loss']:.4f}, val_acc={epoch['val_accuracy']:.4f}")
```

## Tips and Best Practices

1. **Start Small**: Begin with a small parameter sweep to test the system
2. **Use Early Stopping**: Set reasonable `patience` values to avoid wasting time
3. **Monitor Resources**: Large searches can be computationally expensive
4. **Check Results**: Always verify the output files are created correctly
5. **Reproducibility**: Use fixed seeds for reproducible results

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or model parameters
2. **Slow Training**: Use smaller datasets for testing
3. **No Results**: Check that models are registered correctly
4. **Early Stopping Too Soon**: Increase `min_epochs` or `patience`

### Debug Mode

Enable debug output by modifying the search code:

```python
# Add debug prints in hyperparameter_search.py
print(f"Debug: Processing trial {trial_id}")
print(f"Debug: Parameters: {parameters}")
```

## Integration with Existing Code

The hyperparameter search system integrates seamlessly with the existing TFN codebase:

- Uses the same model registry (`model/registry.py`)
- Compatible with existing data loaders (`data_pipeline.py`)
- Works with the existing trainer (`src/trainer.py`)
- Supports all existing model configurations

## Future Enhancements

Potential improvements:
- Parallel trial execution
- Bayesian optimization
- Hyperparameter importance analysis
- Automated result visualization
- Integration with experiment tracking tools 