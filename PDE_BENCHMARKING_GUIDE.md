# PDE Benchmarking Guide for Token Field Network

This guide explains how to use the Token Field Network (TFN) with PDE benchmark datasets commonly used in Fourier Neural Operator (FNO) research.

## Overview

The PDE loader provides a unified interface for loading physics-informed neural network datasets, including:
- **Burgers' Equation**: 1D shock wave formation
- **Darcy Flow**: 2D porous media fluid flow
- **Generic PDE**: Any PDE dataset following FNO conventions

## Dataset Format

FNO benchmark datasets are typically stored as `.mat` files with the following structure:

```matlab
% Example .mat file structure
a = [n_samples, n_spatial_points];     % Initial conditions
u = [n_samples, n_spatial_points, n_timesteps];  % Solutions
x = [n_spatial_points] or [n_spatial_points, 2]; % Spatial grid
```

### Key Components

- **Initial Conditions (`a`)**: Input to the model (e.g., initial velocity profile for Burgers')
- **Solutions (`u`)**: Target outputs (e.g., velocity profile at future timesteps)
- **Grid (`x`)**: Spatial coordinates where the PDE is evaluated

## Usage

### 1. Basic Usage

```python
from data.pde_loader import PDEDataset, BurgersDataset, DarcyFlowDataset

# Load a generic PDE dataset
train_ds, val_ds, test_ds = PDEDataset.get_splits(
    file_path='path/to/your/data.mat',
    target_timestep=10,
    normalize=True
)

# Load Burgers' equation specifically
train_ds, val_ds, test_ds = BurgersDataset.get_splits(
    file_path='path/to/burgers_data.mat',
    target_timestep=10
)

# Load Darcy flow (steady-state)
train_ds, val_ds, test_ds = DarcyFlowDataset.get_splits(
    file_path='path/to/darcy_data.mat',
    target_timestep=0  # Steady-state problem
)
```

### 2. Configuration Files

Use the provided configuration files for easy experimentation:

```bash
# Train on Burgers' equation
python train.py --config configs/burgers.yaml

# Train on Darcy flow
python train.py --config configs/darcy.yaml

# Use generic PDE configuration
python train.py --config configs/pde_generic.yaml
```

### 3. Custom Parameters

```python
# Advanced configuration
train_ds, val_ds, test_ds = PDEDataset.get_splits(
    file_path='data.mat',
    initial_condition_key='a',      # Key for initial conditions
    solution_key='u',               # Key for solutions
    grid_key='x',                   # Key for spatial grid
    input_timesteps=1,              # Number of input timesteps
    target_timestep=10,             # Target timestep to predict
    split_frac={'train': 0.8, 'val': 0.1, 'test': 0.1},
    normalize=True,
    normalization_strategy='global'  # 'global' or 'instance'
)
```

## Dataset Types

### Burgers' Equation

**Physics**: Models shock wave formation in 1D
**Input**: Initial velocity profile
**Output**: Velocity profile at future timestep
**Grid**: 1D spatial coordinates

```python
# Example usage
train_ds, val_ds, test_ds = BurgersDataset.get_splits(
    file_path='burgers_data.mat',
    target_timestep=10  # Predict solution at t=10
)
```

### Darcy Flow

**Physics**: Models fluid flow through porous media
**Input**: Permeability field
**Output**: Pressure field
**Grid**: 2D spatial coordinates

```python
# Example usage
train_ds, val_ds, test_ds = DarcyFlowDataset.get_splits(
    file_path='darcy_data.mat',
    target_timestep=0  # Steady-state problem
)
```

## Data Format Requirements

### Input Format

The loader expects `.mat` files with specific array orientations:

```python
# Correct format
a.shape = (n_samples, n_spatial_points)           # Initial conditions
u.shape = (n_samples, n_spatial_points, n_timesteps)  # Solutions
x.shape = (n_spatial_points,) or (n_spatial_points, 2)  # Grid
```

### Output Format

The loader returns dictionaries compatible with TFN:

```python
sample = dataset[0]
print(sample.keys())
# Output: ['inputs', 'targets', 'positions']

print(sample['inputs'].shape)    # [n_spatial_points, input_dim]
print(sample['targets'].shape)   # [n_spatial_points, output_dim]
print(sample['positions'].shape) # [n_spatial_points, spatial_dim]
```

## Normalization Strategies

### Global Normalization

Normalizes across all samples using global statistics:

```python
# Default behavior
train_ds, val_ds, test_ds = PDEDataset.get_splits(
    file_path='data.mat',
    normalize=True,
    normalization_strategy='global'
)
```

### Instance Normalization

Normalizes each sample individually:

```python
train_ds, val_ds, test_ds = PDEDataset.get_splits(
    file_path='data.mat',
    normalize=True,
    normalization_strategy='instance'
)
```

## Integration with Training Pipeline

### 1. Using the Registry

The PDE datasets are automatically registered in the data registry:

```python
from data.registry import get_dataset_config

# Check available datasets
config = get_dataset_config('burgers')
print(config['description'])
```

### 2. Training Script

```bash
# Train with default configuration
python train.py --config configs/burgers.yaml

# Train with custom parameters
python train.py \
    --config configs/burgers.yaml \
    --data.file_path /path/to/custom/data.mat \
    --data.target_timestep 15
```

### 3. Hyperparameter Search

```bash
# Run hyperparameter search
python hyperparameter_search.py \
    --config configs/searches/pde_search.yaml \
    --dataset burgers
```

## Testing

### Run the Test Suite

```bash
python test_pde_loader.py
```

This will:
- Create synthetic test datasets
- Test all dataset types
- Verify data loading speed
- Check tensor shapes and formats

### Manual Testing

```python
from data.pde_loader import PDEDataset
import torch

# Create a small test dataset
train_ds, val_ds, test_ds = PDEDataset.get_splits(
    file_path='test_data.mat',
    target_timestep=5
)

# Test a batch
batch = [train_ds[i] for i in range(4)]
inputs = torch.stack([item['inputs'] for item in batch])
targets = torch.stack([item['targets'] for item in batch])
positions = torch.stack([item['positions'] for item in batch])

print(f"Batch shapes:")
print(f"  Inputs: {inputs.shape}")
print(f"  Targets: {targets.shape}")
print(f"  Positions: {positions.shape}")
```

## Common Issues and Solutions

### 1. Shape Mismatch Errors

**Problem**: `ValueError: Mismatch in spatial points`

**Solution**: Ensure your `.mat` file has consistent dimensions:
- `a.shape[1] == u.shape[1] == x.shape[0]`
- For 2D problems: `x.shape[1] == 2`

### 2. Timestep Index Errors

**Problem**: `IndexError: index X is out of bounds for dimension 2`

**Solution**: Check your target timestep:
- For time-dependent problems: `target_timestep < u.shape[2]`
- For steady-state problems: Use `target_timestep=0`

### 3. Grid Orientation Issues

**Problem**: Incorrect spatial coordinates

**Solution**: The loader automatically handles common orientations:
- 1D: `x.shape = (n_points,)`
- 2D: `x.shape = (n_points, 2)`

## Performance Considerations

### Memory Usage

- Large datasets may require chunked loading
- Consider using `instance` normalization for memory efficiency
- Use smaller batch sizes for high-resolution grids

### Loading Speed

- `.mat` files are loaded once during initialization
- Subsequent access is from memory
- Consider converting to `.h5` format for very large datasets

## Example Workflows

### 1. Quick Start with Synthetic Data

```python
# Generate synthetic data for testing
from test_pde_loader import create_synthetic_burgers_data

create_synthetic_burgers_data('data/synthetic_burgers.mat')

# Train on synthetic data
python train.py --config configs/tests/burgers_test.yaml
```

### 2. Real Dataset Integration

```python
# 1. Download FNO benchmark dataset
# 2. Place .mat file in data/ directory
# 3. Update config file with correct path
# 4. Train model

python train.py --config configs/burgers.yaml
```

### 3. Custom PDE Dataset

```python
# 1. Prepare your data in the required format
# 2. Save as .mat file
# 3. Use generic PDE loader

train_ds, val_ds, test_ds = PDEDataset.get_splits(
    file_path='your_custom_data.mat',
    initial_condition_key='your_input_key',
    solution_key='your_output_key',
    grid_key='your_grid_key'
)
```

## Next Steps

1. **Download FNO Benchmark Datasets**: Get real PDE datasets from FNO repositories
2. **Experiment with Different PDEs**: Try Navier-Stokes, heat equation, etc.
3. **Compare with Baselines**: Benchmark against FNO, PITT, and other methods
4. **Extend to Higher Dimensions**: Adapt for 3D spatial problems
5. **Add More PDE Types**: Implement specialized loaders for other equations

## References

- [Fourier Neural Operator Paper](https://arxiv.org/abs/2010.08895)
- [FNO Benchmark Datasets](https://github.com/zongyi-li/fourier_neural_operator)
- [Physics-Informed Neural Networks](https://arxiv.org/abs/1711.10561) 