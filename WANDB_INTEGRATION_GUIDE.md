# Weights & Biases Integration Guide

## Overview

The Token Field Network (TFN) training system now supports seamless integration with Weights & Biases (wandb) for experiment tracking, visualization, and collaboration.

## Installation

First, install wandb:

```bash
pip install wandb
```

Then, login to your wandb account:

```bash
wandb login
```

## Configuration

### Basic Wandb Configuration

Add a `wandb` section to your YAML configuration file:

```yaml
# configs/your_experiment.yaml
model_name: enhanced_tfn_regressor
task: time_series

data:
  dataset_name: ett
  csv_path: data/ETTh1.csv
  input_len: 96
  output_len: 24

model:
  input_dim: 7
  embed_dim: 256
  output_dim: 1
  output_len: 24
  num_layers: 4
  dropout: 0.2
  kernel_type: rbf

training:
  batch_size: 64
  lr: 1e-4
  epochs: 10
  warmup_epochs: 2
  grad_clip: 1.0
  log_interval: 50

# Weights & Biases Configuration
wandb:
  use_wandb: true
  project_name: "tfn-experiments"
  experiment_name: "my-experiment-v1"
```

### Advanced Wandb Configuration

You can add more wandb-specific settings:

```yaml
wandb:
  use_wandb: true
  project_name: "tfn-time-series"
  experiment_name: "ett-forecasting-v2"
  tags: ["time-series", "forecasting", "tfn"]
  notes: "Experiment with ETT dataset using enhanced TFN"
  group: "baseline-experiments"
  # Additional wandb config can be added here
  config:
    model_type: "enhanced_tfn"
    dataset: "ett"
    task: "forecasting"
```

## Usage

### Command Line Usage

```bash
# Basic training with wandb
python train.py --config configs/ett_wandb.yaml

# Override wandb settings via command line
python train.py --config configs/ett_wandb.yaml --set wandb.experiment_name="custom-experiment"

# Disable wandb for a run
python train.py --config configs/ett_wandb.yaml --set wandb.use_wandb=false
```

### Notebook Usage

```python
from train import run_training
import yaml

# Load configuration
with open('configs/ett_wandb.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Modify wandb settings
config['wandb']['experiment_name'] = 'notebook-experiment'
config['wandb']['use_wandb'] = True

# Run training with wandb
history = run_training(config)
```

### Hyperparameter Search with Wandb

```python
from hyperparameter_search import run_search
import yaml

# Load search configuration
with open('configs/searches/ett_regression_search.yaml', 'r') as f:
    search_config = yaml.safe_load(f)

# Add wandb configuration
search_config['wandb'] = {
    'use_wandb': True,
    'project_name': 'tfn-hyperparameter-search',
    'experiment_name': 'ett-search-v1'
}

# Run hyperparameter search with wandb
results = run_search(search_config)
```

## What Gets Tracked

### Automatic Tracking

The system automatically tracks:

1. **Training Metrics**:
   - Training/validation loss
   - Training/validation accuracy (for classification)
   - Training/validation MSE/MAE (for regression)
   - Learning rate over time

2. **Model Information**:
   - Model architecture
   - Number of parameters
   - Model hyperparameters

3. **Training Configuration**:
   - All training hyperparameters
   - Dataset information
   - Device information

4. **System Information**:
   - GPU/CPU usage
   - Memory usage
   - Training time

### Custom Tracking

You can add custom metrics by modifying the `TaskStrategy` classes in `src/task_strategies.py`:

```python
# In src/task_strategies.py
def calculate_metrics(self, outputs, targets, loss):
    metrics = super().calculate_metrics(outputs, targets, loss)
    
    # Add custom metrics
    metrics['custom_metric'] = self.calculate_custom_metric(outputs, targets)
    
    return metrics
```

## Wandb Dashboard Features

### 1. **Experiment Comparison**

Compare different experiments side by side:

```python
# In your notebook
import wandb

# Compare experiments
api = wandb.Api()
runs = api.runs("your-username/tfn-experiments")
for run in runs:
    print(f"{run.name}: {run.summary.get('val_loss', 'N/A')}")
```

### 2. **Hyperparameter Analysis**

Analyze the impact of hyperparameters:

```python
# Create hyperparameter sweep
sweep_config = {
    'method': 'grid',
    'parameters': {
        'model.embed_dim': {'values': [128, 256, 512]},
        'training.lr': {'values': [1e-4, 1e-3, 1e-2]},
        'model.num_layers': {'values': [2, 4, 6]}
    }
}
```

### 3. **Custom Plots**

Create custom visualizations:

```python
# In your training code
import wandb

# Log custom plots
wandb.log({
    "attention_weights": wandb.Image(attention_plot),
    "predictions": wandb.plot.line_series(
        xs=list(range(len(predictions))),
        ys=[predictions, targets],
        keys=["Predicted", "Actual"],
        title="Predictions vs Targets"
    )
})
```

## Best Practices

### 1. **Experiment Naming**

Use descriptive experiment names:

```yaml
wandb:
  experiment_name: "ett-forecasting-embed256-layers4-lr1e4"
```

### 2. **Project Organization**

Organize experiments by project:

```yaml
# Time series experiments
wandb:
  project_name: "tfn-time-series"

# NLP experiments  
wandb:
  project_name: "tfn-nlp"

# Vision experiments
wandb:
  project_name: "tfn-vision"
```

### 3. **Tags for Organization**

Use tags to categorize experiments:

```yaml
wandb:
  tags: ["time-series", "forecasting", "tfn", "baseline"]
```

### 4. **Notes for Documentation**

Add detailed notes for important experiments:

```yaml
wandb:
  notes: |
    Baseline experiment with ETT dataset.
    - Model: Enhanced TFN Regressor
    - Embed dim: 256
    - Layers: 4
    - Learning rate: 1e-4
    - Expected to establish baseline performance
```

## Troubleshooting

### Common Issues

1. **Wandb Not Available**
   ```
   Warning: wandb not available. Install with 'pip install wandb' for experiment tracking.
   ```
   Solution: Install wandb with `pip install wandb`

2. **Authentication Issues**
   ```
   wandb: ERROR Not logged in
   ```
   Solution: Run `wandb login` and follow the prompts

3. **Project Not Found**
   ```
   wandb: ERROR Project not found
   ```
   Solution: Create the project in your wandb dashboard or use an existing project name

### Debugging

Enable debug mode for wandb:

```python
import wandb
wandb.init(mode="disabled")  # Disable wandb for debugging
```

## Example Configurations

### Time Series Forecasting

```yaml
# configs/ett_wandb.yaml
model_name: enhanced_tfn_regressor
task: time_series

data:
  dataset_name: ett
  csv_path: data/ETTh1.csv
  input_len: 96
  output_len: 24

model:
  input_dim: 7
  embed_dim: 256
  output_dim: 1
  output_len: 24
  num_layers: 4
  dropout: 0.2
  kernel_type: rbf

training:
  batch_size: 64
  lr: 1e-4
  epochs: 10
  warmup_epochs: 2
  grad_clip: 1.0
  log_interval: 50

wandb:
  use_wandb: true
  project_name: "tfn-time-series"
  experiment_name: "ett-forecasting-v1"
  tags: ["time-series", "forecasting", "tfn"]
  notes: "Baseline experiment with ETT dataset"
```

### NLP Classification

```yaml
# configs/glue_wandb.yaml
model_name: enhanced_tfn_classifier
task: classification

data:
  dataset_name: glue
  task: sst2
  max_length: 512
  tokenizer_name: bert-base-uncased

model:
  vocab_size: 30522
  embed_dim: 256
  num_classes: 2
  num_layers: 4
  dropout: 0.2
  kernel_type: rbf

training:
  batch_size: 32
  lr: 2e-5
  epochs: 5
  warmup_epochs: 1
  grad_clip: 1.0
  log_interval: 100

wandb:
  use_wandb: true
  project_name: "tfn-nlp"
  experiment_name: "sst2-classification-v1"
  tags: ["nlp", "classification", "tfn", "glue"]
  notes: "SST-2 sentiment classification with TFN"
```

## Integration with Hyperparameter Search

The wandb integration works seamlessly with hyperparameter search:

```yaml
# configs/searches/ett_search_wandb.yaml
model_name: enhanced_tfn_regressor
task: time_series

data:
  dataset_name: ett
  csv_path: data/ETTh1.csv
  input_len: 96
  output_len: 24

model:
  input_dim: 7
  embed_dim: 256
  output_dim: 1
  output_len: 24
  num_layers: 4
  dropout: 0.2
  kernel_type: rbf

training:
  batch_size: 64
  lr: 1e-4
  epochs: 5
  warmup_epochs: 1
  grad_clip: 1.0
  log_interval: 50

wandb:
  use_wandb: true
  project_name: "tfn-hyperparameter-search"
  experiment_name: "ett-search-v1"

search_space:
  models: ["enhanced_tfn_regressor"]
  params:
    model.embed_dim:
      values: [128, 256, 512]
    training.lr:
      values: [1e-4, 1e-3]
    model.num_layers:
      values: [2, 4, 6]

patience: 2
min_epochs: 3
seed: 42
```

This integration provides comprehensive experiment tracking, making it easy to analyze results, compare experiments, and collaborate with team members. 

# W&B Integration Guide for Hyperparameter Search

This guide explains how to use Weights & Biases (W&B) with the Token Field Network hyperparameter search system.

## Overview

The hyperparameter search system now supports W&B integration for experiment tracking. Each trial in the search will be logged as a separate W&B run, allowing you to:

- Track metrics across all trials
- Compare different hyperparameter combinations
- Visualize training curves
- Organize experiments by project

## Setup

### 1. Install W&B
```bash
pip install wandb
```

### 2. Login to W&B
```bash
wandb login
```

## Configuration

### Enable W&B in Search Config

Add a `wandb` section to your search configuration:

```yaml
# Weights & Biases Configuration
wandb:
  use_wandb: true  # Set to true to enable logging
  project_name: "tfn-hyperparameter-search"  # Your W&B project name
```

### Example Configuration

```yaml
# Example: configs/searches/ett_regression_search_scoped.yaml
model_name: "enhanced_tfn_regressor"

data:
  dataset_name: "ett"
  csv_path: "data/ETTh1.csv"
  input_len: 96
  output_len: 24

training:
  epochs: 30
  patience: 5
  batch_size: 32

search_space:
  models: ["enhanced_tfn_regressor", "tfn_regressor"]
  params:
    model:
      embed_dim: [128, 256, 512]
      kernel_type: ["rbf", "compact"]
    training:
      lr: [0.001, 0.01]

# W&B Configuration
wandb:
  use_wandb: true
  project_name: "tfn-ett-regression-search"
```

## Usage

### Run Hyperparameter Search with W&B

```bash
python hyperparameter_search.py --config configs/searches/ett_regression_search_scoped.yaml
```

### What Gets Logged

For each trial, W&B will log:

- **Hyperparameters**: All swept parameters (embed_dim, lr, etc.)
- **Metrics**: Training and validation metrics for each epoch
- **System Info**: GPU usage, memory, etc.
- **Model Info**: Parameter count, model architecture
- **Training History**: Complete training curves

### Trial Naming

Each trial is automatically named as:
```
{model_name}_{trial_id}
```

Example: `enhanced_tfn_regressor_trial_001`

## Features

### 1. Automatic Trial Tracking
- Each trial becomes a separate W&B run
- Unique experiment names prevent conflicts
- All hyperparameters are logged automatically

### 2. Metric Logging
- Training and validation losses
- Task-specific metrics (accuracy for classification, MSE/MAE for regression)
- Learning rate schedules
- Early stopping information

### 3. Experiment Organization
- All trials from one search are grouped under the same project
- Easy comparison between different hyperparameter combinations
- Filtering and sorting by metrics

### 4. Robust Error Handling
- Failed trials are logged with error messages
- Search continues even if individual trials fail
- W&B runs are properly closed even on errors

## Example Workflow

### 1. Create Search Config
```yaml
# my_search.yaml
model_name: "tfn_classifier"
data:
  dataset: "sst2"
model:
  vocab_size: 10000
  embed_dim: 128
  num_classes: 2
training:
  epochs: 10
  lr: 0.001
search_space:
  models: ["tfn_classifier", "enhanced_tfn_classifier"]
  params:
    model.embed_dim: [128, 256]
    training.lr: [0.001, 0.01]
wandb:
  use_wandb: true
  project_name: "tfn-sentiment-analysis"
```

### 2. Run Search
```bash
python hyperparameter_search.py --config my_search.yaml
```

### 3. View Results in W&B
- Open your W&B dashboard
- Navigate to the "tfn-sentiment-analysis" project
- Compare runs and find the best hyperparameters

## Troubleshooting

### W&B Not Logging
1. Check that `use_wandb: true` is set in your config
2. Verify you're logged in: `wandb login`
3. Check the console output for W&B status messages

### Trial Naming Conflicts
- Each trial gets a unique ID automatically
- No manual intervention needed

### Memory Issues
- W&B logging is lightweight
- If you experience memory issues, try reducing batch size or search space

## Best Practices

### 1. Project Naming
Use descriptive project names:
- `tfn-{dataset}-{task}-search`
- Example: `tfn-ett-regression-search`

### 2. Search Space Design
- Start with a small search space for testing
- Use logarithmic scales for learning rates
- Include a reasonable number of trials

### 3. Monitoring
- Monitor W&B dashboard during search
- Check for failed trials
- Use W&B's comparison features to analyze results

### 4. Resource Management
- Set appropriate `patience` and `min_epochs`
- Use early stopping to save compute
- Monitor GPU memory usage

## Advanced Features

### Custom Experiment Names
You can modify the experiment naming in `hyperparameter_search.py`:

```python
# In _run_trial method
experiment_name = f"{model_name}_{trial_id}_{parameters_hash}"
```

### Additional W&B Parameters
The Trainer supports additional W&B parameters that can be extended:

```python
trainer = Trainer(
    # ... other parameters ...
    use_wandb=use_wandb,
    project_name=project_name,
    experiment_name=experiment_name,
    # Additional parameters can be added here
)
```

## Integration with Existing Workflows

The W&B integration is designed to work seamlessly with existing training workflows:

- Same configuration format as `train.py`
- Compatible with all existing search configurations
- No changes needed to model or data loading code
- Backward compatible (W&B disabled by default)

## Support

For issues with W&B integration:

1. Check the console output for W&B status messages
2. Verify your W&B login status
3. Test with a minimal search configuration
4. Check the W&B dashboard for run status

The integration is designed to be robust and continue working even if W&B is not available or configured. 