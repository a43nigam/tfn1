# Token Field Network Development Guide

This comprehensive guide covers all aspects of developing and using the Token Field Network (TFN) system.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration](#configuration)
3. [Training](#training)
4. [Hyperparameter Search](#hyperparameter-search)
5. [Data Loading](#data-loading)
6. [Model Architecture](#model-architecture)
7. [Normalization Strategies](#normalization-strategies)
8. [Checkpoint Management](#checkpoint-management)
9. [Weights & Biases Integration](#weights--biases-integration)
10. [PDE Benchmarking](#pde-benchmarking)
11. [Notebook Usage](#notebook-usage)
12. [Troubleshooting](#troubleshooting)

## Quick Start

### Basic Training

```bash
# Train on ETT time series dataset
python train.py --config configs/ett.yaml

# Train on Burgers' equation
python train.py --config configs/burgers.yaml

# Train with custom parameters
python train.py --config configs/ett.yaml --set model.embed_dim=512 training.epochs=100
```

### Notebook Usage

```python
from train import run_training
import yaml

# Load and modify configuration
with open('configs/ett.yaml', 'r') as f:
    config = yaml.safe_load(f)

config['model']['embed_dim'] = 512
config['training']['epochs'] = 100

# Run training
history = run_training(config, device="auto")
```

## Configuration

### Configuration Structure

```yaml
model_name: enhanced_tfn_regressor
task: time_series

data:
  dataset_name: ett
  csv_path: data/ETTh1.csv
  input_len: 96
  output_len: 24
  normalization_strategy: global  # global, instance, feature_wise

model:
  input_dim: 7
  embed_dim: 256
  output_dim: 1
  output_len: 24
  num_layers: 4
  kernel_type: rbf  # rbf, film_learnable, etc.
  evolution_type: diffusion  # diffusion, wave, schrodinger, cnn
  interference_type: causal  # causal, standard
  grid_size: 100
  num_steps: 4
  dropout: 0.1

training:
  batch_size: 64
  lr: 1e-4
  epochs: 10
  warmup_epochs: 2
  grad_clip: 1.0
  weight_decay: 0.01
  log_interval: 50

wandb:
  use_wandb: false
  project_name: "tfn-experiments"
  experiment_name: "my-experiment"
```

### Available Models

- `enhanced_tfn_regressor`: Main regression model
- `enhanced_tfn_classifier`: Classification model
- `enhanced_tfn_language_model`: Language modeling
- `transformer_classifier`: Baseline transformer
- `performer_classifier`: Baseline performer
- `lstm_regressor`: Baseline LSTM
- `cnn_regressor`: Baseline CNN

### Available Datasets

- **Time Series**: `ett`, `jena`, `electricity`
- **NLP**: `arxiv`, `pg19`, `imdb`, `wikitext`
- **PDE**: `burgers`, `darcy`
- **Classification**: `glue_sst2`, `glue_mrpc`, etc.

## Training

### Command Line Training

```bash
# Basic training
python train.py --config configs/ett.yaml

# Override parameters
python train.py --config configs/ett.yaml --set model.embed_dim=512 training.epochs=100

# Specify device
python train.py --config configs/ett.yaml --device cuda

# Use specific checkpoint directory
python train.py --config configs/ett.yaml --set checkpoint_dir="./my_checkpoints"
```

### Training with RevIN (Reversible Instance Normalization)

For time series forecasting with instance normalization:

```yaml
data:
  normalization_strategy: instance  # This triggers RevIN wrapper

model:
  input_dim: 7  # Required for RevIN
  # ... other model config
```

The system automatically wraps the model with RevIN when `normalization_strategy: instance` is used.

## Hyperparameter Search

### Command Line Search

```bash
# Run hyperparameter search
python hyperparameter_search.py --config configs/searches/ett_regression_search.yaml

# Override search parameters
python hyperparameter_search.py --config configs/searches/ett_regression_search.yaml --set search_space.params.model.embed_dim.values="[128,256,512]"
```

### Notebook Search

```python
from hyperparameter_search import run_search
import yaml

# Load search configuration
with open('configs/searches/ett_regression_search.yaml', 'r') as f:
    search_config = yaml.safe_load(f)

# Run search
results = run_search(search_config, output_dir="./search_results")
```

## Data Loading

### Time Series Data

```python
from data.timeseries_loader import ETTDataset

# Load ETT dataset
train_ds, val_ds, test_ds = ETTDataset.get_splits(
    csv_path='data/ETTh1.csv',
    input_len=96,
    output_len=24,
    normalization_strategy='global'  # global, instance, feature_wise
)
```

### PDE Data

```python
from data.pde_loader import BurgersDataset, DarcyFlowDataset

# Load Burgers' equation
train_ds, val_ds, test_ds = BurgersDataset.get_splits(
    file_path='burgers_data.mat',
    target_timestep=10
)

# Load Darcy flow
train_ds, val_ds, test_ds = DarcyFlowDataset.get_splits(
    file_path='darcy_data.mat',
    target_timestep=0  # Steady-state
)
```

### NLP Data

```python
from data.pg19_loader import PG19Dataset

# Load PG19 dataset
train_ds, val_ds, test_ds = PG19Dataset.get_splits(
    file_path='data/pg19.csv',
    tokenizer_name='gpt2',
    max_length=512
)
```

## Model Architecture

### Enhanced TFN Components

1. **Field Projection**: Projects tokens to continuous fields using kernels
2. **Field Evolution**: Evolves fields using PDE-inspired dynamics
3. **Field Interference**: Handles token interactions
4. **Field Sampling**: Samples evolved fields back to tokens

### Kernel Types

- `rbf`: Radial basis function kernel
- `film_learnable`: Feature-wise linear modulation kernel
- `data_dependent`: Data-dependent kernel

### Evolution Types

- `diffusion`: Diffusion equation evolution
- `wave`: Wave equation evolution
- `schrodinger`: Schrödinger equation evolution
- `cnn`: Convolutional evolution
- `spatially_varying_pde`: Spatially varying PDE evolution
- `modernized_cnn`: Modernized CNN evolution

### Interference Types

- `causal`: Causal attention-like interference
- `standard`: Standard interference

## Normalization Strategies

### Global Normalization

```yaml
data:
  normalization_strategy: global
```

- Fits scaler on training data
- Applies same transformation to all splits
- Metrics reported on original scale

### Instance Normalization

```yaml
data:
  normalization_strategy: instance
```

- Normalizes each sequence independently
- Automatically triggers RevIN wrapper
- Metrics reported on original scale via RevIN

### Feature-wise Normalization

```yaml
data:
  normalization_strategy: feature_wise
```

- Normalizes each feature independently
- Metrics reported on original scale

## Checkpoint Management

### Automatic Checkpoint Directory

The system automatically handles checkpoint directories:

```yaml
# Default behavior
checkpoint_dir: "checkpoints"  # Uses default

# Custom directory
checkpoint_dir: "./my_checkpoints/"

# Kaggle environment (automatic fallback)
checkpoint_dir: "checkpoints"  # Redirects to /kaggle/working/tfn_checkpoints
```

### Checkpoint Files

- `experiment_YYYYMMDD_HHMMSS_latest.pt`: Latest model state
- `experiment_YYYYMMDD_HHMMSS_best.pt`: Best model state
- `experiment_YYYYMMDD_HHMMSS_epoch_N.pt`: Epoch N model state
- `experiment_YYYYMMDD_HHMMSS_history.json`: Training history

## Weights & Biases Integration

### Installation

```bash
pip install wandb
wandb login
```

### Configuration

```yaml
wandb:
  use_wandb: true
  project_name: "tfn-experiments"
  experiment_name: "my-experiment"
  tags: ["time-series", "forecasting"]
  notes: "Experiment description"
```

### Usage

```bash
# Enable wandb
python train.py --config configs/ett_wandb.yaml

# Disable wandb
python train.py --config configs/ett_wandb.yaml --set wandb.use_wandb=false
```

## PDE Benchmarking

### Supported PDEs

- **Burgers' Equation**: 1D shock wave formation
- **Darcy Flow**: 2D porous media fluid flow
- **Generic PDE**: Any PDE following FNO conventions

### Usage

```bash
# Train on Burgers' equation
python train.py --config configs/burgers.yaml

# Train on Darcy flow
python train.py --config configs/darcy.yaml
```

### Dataset Format

PDE datasets are stored as `.mat` files:

```matlab
a = [n_samples, n_spatial_points];     % Initial conditions
u = [n_samples, n_spatial_points, n_timesteps];  % Solutions
x = [n_spatial_points] or [n_spatial_points, 2]; % Spatial grid
```

## Notebook Usage

### Direct Function Calls

```python
from train import run_training
from hyperparameter_search import run_search

# Training
history = run_training(config, device="auto")

# Hyperparameter search
results = run_search(search_config, output_dir="./results")
```

### Configuration Modification

```python
# Load and modify
config = yaml.safe_load(open('configs/ett.yaml'))
config['model']['embed_dim'] = 512
config['training']['epochs'] = 100

# Run with modified config
history = run_training(config)
```

### Iterative Experimentation

```python
# Test different configurations
for embed_dim in [128, 256, 512]:
    for lr in [1e-4, 1e-3]:
        config['model']['embed_dim'] = embed_dim
        config['training']['lr'] = lr
        
        history = run_training(config, device="cpu")
        final_loss = history['val_loss'][-1]
        print(f"embed_dim={embed_dim}, lr={lr}: loss={final_loss}")
```

## Troubleshooting

### Common Issues

1. **Checkpoint Directory Permissions**
   ```
   ❌ Permission denied creating checkpoint directory
   ```
   Solution: Set `checkpoint_dir` to a writable location

2. **Scaler Compatibility Warnings**
   ```
   ⚠️ Calculating regression metrics on a normalized scale
   ```
   Solution: Use `normalization_strategy: instance` with RevIN

3. **Memory Issues**
   ```
   CUDA out of memory
   ```
   Solution: Reduce `batch_size` or `grid_size`

4. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'wandb'
   ```
   Solution: Install wandb or set `wandb.use_wandb: false`

### Debugging Tips

1. **Check Configuration**
   ```python
   print(f"Model: {config['model_name']}")
   print(f"Embed dim: {config['model']['embed_dim']}")
   ```

2. **Monitor Training**
   ```python
   history = run_training(config)
   print(f"Final loss: {history['val_loss'][-1]}")
   ```

3. **Test Components**
   ```python
   from model.utils import build_model
   model = build_model('enhanced_tfn_regressor', config['model'])
   ```

### Performance Optimization

1. **Use GPU**: Set `device="cuda"`
2. **Reduce Grid Size**: Lower `grid_size` for faster training
3. **Batch Size**: Increase `batch_size` for better GPU utilization
4. **Mixed Precision**: Consider using `torch.cuda.amp` for faster training

## File Structure

```
TokenFieldNetwork/
├── configs/           # Configuration files
│   ├── *.yaml        # Dataset configurations
│   ├── tests/        # Test configurations
│   └── searches/     # Hyperparameter search configs
├── core/             # Core TFN components
├── data/             # Data loaders
├── model/            # Model definitions
├── src/              # Training utilities
├── test/             # Unit tests
├── checkpoints/      # Model checkpoints
├── train.py          # Main training script
├── hyperparameter_search.py  # Hyperparameter search
└── README.md         # Main documentation
```

This guide covers all aspects of the Token Field Network system. For specific questions, refer to the individual component documentation or create an issue on the repository. 