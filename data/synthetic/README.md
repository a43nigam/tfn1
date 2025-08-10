# Synthetic Dataset Generation

This directory contains the infrastructure for generating and loading synthetic datasets for TFN research. The datasets are designed to test specific aspects of the Token Field Network architecture.

## Overview

Three synthetic datasets are provided:

1. **Heat Equation** (`heat_equation.pt`) - Tests PDE learning and spatial-temporal evolution
2. **Delayed Copy** (`delayed_copy.pt`) - Tests long-range dependencies and memory retention  
3. **Irregular Sampling** (`irregular_sampling.pt`) - Tests continuous position handling and temporal interpolation

## Quick Start

### Generate All Datasets
```bash
cd data/synthetic
python generate_tasks.py
```

This will create three `.pt` files (~883MB total):
- `heat_equation.pt` (~80MB)
- `delayed_copy.pt` (~800MB) 
- `irregular_sampling.pt` (~2.4MB)

### Verify Data Integrity
```bash
python test_data.py
```

### Test Loader Integration
```bash
# From project root
python -c "
from data.registry import create_dataset
train_ds = create_dataset('heat_equation', {'data': {}}, split='train')
print(f'Dataset created: {len(train_ds)} samples')
"
```

## Dataset Details

### Heat Equation Dataset
- **Purpose**: Test spatial-temporal PDE learning
- **Size**: 1000 samples, 200 timesteps, 100 spatial points
- **Physics**: 1D heat diffusion with finite-difference solver
- **Initial Conditions**: Gaussian bumps, step functions, Fourier series
- **Format**: `initial_conditions` [1000, 100], `solutions` [1000, 200, 100]
- **TFN Format**: `inputs` [batch, 100, 1], `targets` [batch, 100, 1], `positions` [batch, 100, 1]

### Delayed Copy Dataset  
- **Purpose**: Test long-range memory and dependency tracking
- **Size**: 5000 samples, 10000 sequence length, delay=9900
- **Task**: Copy 10-token pattern after 9900 timestep delay
- **Vocabulary**: 11 tokens (0-9 + delimiter)
- **Format**: `inputs` [5000, 10000], `targets` [5000, 10000]

### Irregular Sampling Dataset
- **Purpose**: Test non-uniform temporal spacing handling
- **Size**: 1000 samples, 200 irregular time points
- **Signal**: Mixture of sinusoids with Poisson timestamps
- **Noise**: Variable Gaussian noise (5-20%)
- **Format**: `inputs` [1000, 200], `targets` [1000, 200], `positions` [1000, 200]

## Usage

### Via Registry (Recommended)
```python
from data.registry import create_dataset

# Create train/val/test splits
config = {'data': {}}
train_ds = create_dataset('heat_equation', config, split='train')
val_ds = create_dataset('heat_equation', config, split='val') 
test_ds = create_dataset('heat_equation', config, split='test')

# Use with DataLoader
from torch.utils.data import DataLoader
dataloader = DataLoader(train_ds, batch_size=32, shuffle=True)

for batch in dataloader:
    inputs = batch['inputs']    # Model inputs
    targets = batch['targets']  # Ground truth targets
    # Additional keys: 'positions', 'grid', 'vocab_size', etc.
```

### Direct Instantiation
```python
from data.synthetic_task_loader import SyntheticTaskDataset

# Factory method for splits
train_ds, val_ds, test_ds = SyntheticTaskDataset.get_splits(
    'data/synthetic/heat_equation.pt'
)

# Single dataset
dataset = SyntheticTaskDataset(
    'data/synthetic/delayed_copy.pt', 
    split='train'
)
```

### Custom Split Ratios
```python
# Custom 70/15/15 split
custom_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
train_ds, val_ds, test_ds = SyntheticTaskDataset.get_splits(
    'data/synthetic/irregular_sampling.pt',
    split_ratios=custom_ratios
)
```

## Data Format Standardization

The loader automatically standardizes different dataset formats:

| Dataset | Input Shape | Target Shape | Additional Keys |
|---------|-------------|--------------|-----------------|
| Heat Equation | `[batch, 100, 1]` | `[batch, 100, 1]` | `positions`, `grid` |
| Delayed Copy | `[batch, 10000]` | `[batch, 10000]` | `vocab_size`, `delay` |
| Irregular Sampling | `[batch, 200, 1]` | `[batch, 200, 1]` | `positions` |

## Registry Integration

All datasets are registered in `data/registry.py`:

```python
DATASET_REGISTRY = {
    'heat_equation': {
        'class': SyntheticTaskDataset,
        'task_type': 'regression',
        'defaults': {'file_path': 'data/synthetic/heat_equation.pt'}
    },
    'delayed_copy': {
        'class': SyntheticTaskDataset,
        'task_type': 'classification', 
        'defaults': {'file_path': 'data/synthetic/delayed_copy.pt'}
    },
    'irregular_sampling': {
        'class': SyntheticTaskDataset,
        'task_type': 'regression',
        'defaults': {'file_path': 'data/synthetic/irregular_sampling.pt'}
    }
}
```

## Configuration Examples

### YAML Config Files
```yaml
# config/heat_equation_example.yaml
data:
  dataset: heat_equation
  batch_size: 32
  
model:
  name: tfn_unified
  # ... model config

training:
  epochs: 100
  # ... training config
```

### Training Command
```bash
python train.py --config configs/heat_equation_example.yaml
```

## File Structure
```
data/synthetic/
├── README.md                    # This file
├── generate_tasks.py           # Data generation script
├── test_data.py               # Data integrity tests
├── test_loader.py             # Loader integration tests
├── heat_equation.pt           # Generated dataset (excluded from git)
├── delayed_copy.pt            # Generated dataset (excluded from git)
└── irregular_sampling.pt      # Generated dataset (excluded from git)
```

## Notes

- **Git Exclusion**: The large `.pt` files are excluded from git (see `.gitignore`)
- **Local Generation**: Run `generate_tasks.py` after cloning the repository
- **Reproducibility**: Uses fixed seeds for deterministic data generation and splits
- **Memory Usage**: Datasets are loaded into memory for fast training
- **Extensibility**: Easy to add new synthetic tasks by following the same pattern

## Contributing

To add a new synthetic dataset:

1. Add generation function to `generate_tasks.py`
2. Update `SyntheticTaskDataset._prepare_data()` for new format
3. Register in `data/registry.py`
4. Add tests to `test_data.py` and `test_loader.py`
5. Update this README with dataset details 