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