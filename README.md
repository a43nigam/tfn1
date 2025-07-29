# Token Field Network (TFN)

A novel deep learning architecture that replaces attention with field projection and evolution. Each token emits a continuous field, which is evolved over a spatial grid and sampled back to update tokens.

## 🏗️ Architecture Overview

TFN operates through three core phases:
1. **Field Projection**: Tokens emit continuous fields across spatial domain
2. **Field Evolution**: Fields evolve over time using learned dynamics
3. **Field Sampling**: Evolved fields are sampled back to update tokens

This approach provides:
- **Mathematical Rigor**: Fully differentiable field-based attention
- **Physics-Inspired**: Leverages continuous field dynamics
- **Modular Design**: Independent components for field emission, evolution, and sampling
- **Numerical Stability**: Careful handling of tensor operations and gradients

## 📋 Table of Contents

- [Models](#models)
- [Kernels](#kernels)
- [Field Evolution Types](#field-evolution-types)
- [Interference Types](#interference-types)
- [Datasets](#datasets)
- [Training](#training)
- [Hyperparameter Search](#hyperparameter-search)
- [Configuration](#configuration)
- [Installation](#installation)

## 🤖 Models

### Core TFN Models

#### `tfn_classifier`
- **Purpose**: Classification tasks (sentiment analysis, text classification)
- **Compatible Datasets**: IMDB, ArXiv, GLUE, synthetic classification
- **Required Parameters**:
  - `vocab_size`: Vocabulary size
  - `embed_dim`: Embedding dimension
  - `num_classes`: Number of output classes
  - `kernel_type`: Field emission kernel type
  - `evolution_type`: Field evolution strategy
- **Optional Parameters**:
  - `num_layers`: Number of TFN layers (default: 2)
  - `grid_size`: Spatial grid size (default: 100)
  - `time_steps`: Evolution time steps (default: 3)
  - `dropout`: Dropout rate (default: 0.1)
  - `use_enhanced`: Use enhanced TFN layers (default: False)
  - `interference_type`: Field interference type (default: "standard")
  - `positional_embedding_strategy`: Positional embedding strategy
  - `calendar_features`: Calendar features for time-based embeddings
  - `feature_cardinalities`: JSON string of feature cardinalities

#### `tfn_regressor`
- **Purpose**: Regression tasks (time series forecasting, continuous prediction)
- **Compatible Datasets**: ETT, Jena Climate, Stock Market, synthetic regression
- **Required Parameters**:
  - `input_dim`: Input feature dimension
  - `embed_dim`: Embedding dimension
  - `output_dim`: Output feature dimension
  - `output_len`: Output sequence length
  - `kernel_type`: Field emission kernel type
  - `evolution_type`: Field evolution strategy
- **Optional Parameters**: Same as classifier

#### `tfn_language_model`
- **Purpose**: Language modeling and text generation
- **Compatible Datasets**: WikiText, PG19, synthetic copy
- **Required Parameters**:
  - `vocab_size`: Vocabulary size
  - `embed_dim`: Embedding dimension
  - `kernel_type`: Field emission kernel type
  - `evolution_type`: Field evolution strategy
- **Optional Parameters**:
  - `seq_len`: Sequence length (default: 512)
  - `grid_size`: Spatial grid size (default: 256)
  - Other parameters same as classifier

#### `tfn_vision`
- **Purpose**: Image classification and vision tasks
- **Compatible Datasets**: Image datasets (2D spatial domain)
- **Required Parameters**:
  - `vocab_size`: Vocabulary size
  - `embed_dim`: Embedding dimension
  - `num_classes`: Number of output classes
- **Optional Parameters**:
  - `num_layers`: Number of TFN layers
  - `num_evolution_steps`: Evolution steps
  - `field_dim`: Field dimension
  - `grid_height`, `grid_width`: 2D grid dimensions
  - `use_dynamic_positions`: Use dynamic positioning
  - `learnable_sigma`: Learnable field spread
  - `learnable_out_sigma`: Learnable output spread
  - `out_sigma_scale`: Output spread scale

### Enhanced TFN Models

#### `enhanced_tfn_classifier`
- **Purpose**: Enhanced classification with advanced field dynamics
- **Features**: Improved field evolution, adaptive time stepping, physics constraints
- **Compatible Datasets**: Same as base classifier
- **Additional Parameters**:
  - `kernel_hidden_dim`: Hidden dimension for data-dependent kernels
  - `evolution_hidden_dim`: Hidden dimension for evolution predictors
  - `num_frequencies`: Number of frequencies for multi-frequency kernels
  - `min_dt`, `max_dt`: Adaptive time stepping bounds

#### `enhanced_tfn_regressor`
- **Purpose**: Enhanced regression with advanced field dynamics
- **Features**: Same as enhanced classifier
- **Compatible Datasets**: Same as base regressor

### Baseline Models

#### `transformer_baseline`
- **Purpose**: Standard Transformer for comparison
- **Compatible Datasets**: All datasets
- **Parameters**: Standard Transformer parameters

#### `performer_baseline`
- **Purpose**: Performer (linear attention) for comparison
- **Compatible Datasets**: All datasets
- **Parameters**: Performer-specific parameters

#### `lstm_baseline`
- **Purpose**: LSTM baseline for comparison
- **Compatible Datasets**: All datasets
- **Parameters**: LSTM-specific parameters

#### `cnn_baseline`
- **Purpose**: CNN baseline for comparison
- **Compatible Datasets**: All datasets
- **Parameters**: CNN-specific parameters

## 🔬 Kernels

Kernels determine how tokens emit fields across the spatial domain. Each kernel encodes different physical intuitions.

### Available Kernels

#### `rbf` (Radial Basis Function)
- **Mathematical Form**: `K(z, μ, σ) = exp(-||z - μ||²/(2σ²))`
- **Properties**: Smooth, infinitely differentiable, infinite support
- **Best For**: General-purpose field emission, smooth interactions
- **Models**: All TFN models
- **Parameters**:
  - `min_sigma`: Minimum spread parameter (default: 0.1)
  - `max_sigma`: Maximum spread parameter (default: 10.0)

#### `compact`
- **Mathematical Form**: `K(z, μ, r) = max(0, 1 - ||z - μ||/r)`
- **Properties**: Finite support, piecewise linear, computationally efficient
- **Best For**: Local interactions, computational efficiency
- **Models**: All TFN models
- **Parameters**:
  - `min_radius`: Minimum radius (default: 0.1)
  - `max_radius`: Maximum radius (default: 5.0)

#### `fourier`
- **Mathematical Form**: `K(z, μ, f) = cos(2πf||z - μ||)`
- **Properties**: Oscillatory, frequency-dependent, good for periodic patterns
- **Best For**: Periodic data, frequency analysis
- **Models**: All TFN models
- **Parameters**:
  - `min_freq`: Minimum frequency (default: 0.1)
  - `max_freq`: Maximum frequency (default: 10.0)

#### `learnable`
- **Mathematical Form**: Learned kernel function via neural network
- **Properties**: Data-adaptive, highly flexible
- **Best For**: Complex, data-specific interactions
- **Models**: All TFN models
- **Parameters**:
  - `hidden_dim`: Hidden dimension for kernel network (default: 64)

#### `data_dependent_rbf`
- **Mathematical Form**: RBF with data-dependent spread parameters
- **Properties**: Adaptive to token content, maintains smoothness
- **Best For**: Content-aware field emission
- **Models**: All TFN models
- **Parameters**:
  - `hidden_dim`: Hidden dimension for spread predictor (default: 32)
  - `min_sigma`, `max_sigma`: Spread bounds

#### `data_dependent_compact`
- **Mathematical Form**: Compact kernel with data-dependent radius
- **Properties**: Adaptive local interactions
- **Best For**: Content-aware local interactions
- **Models**: All TFN models
- **Parameters**:
  - `hidden_dim`: Hidden dimension for radius predictor (default: 32)
  - `min_radius`, `max_radius`: Radius bounds

#### `multi_frequency_fourier`
- **Mathematical Form**: Sum of multiple Fourier components
- **Properties**: Multi-scale frequency analysis
- **Best For**: Multi-scale periodic patterns
- **Models**: All TFN models
- **Parameters**:
  - `num_frequencies`: Number of frequency components (default: 8)
  - `min_freq`, `max_freq`: Frequency bounds

#### `film_learnable`
- **Mathematical Form**: Feature-wise linear modulation of learned kernel
- **Properties**: Highly adaptive, feature-specific modulation
- **Best For**: Complex, feature-dependent interactions
- **Models**: All TFN models
- **Parameters**:
  - `hidden_dim`: Hidden dimension for FiLM network (default: 64)

## 🌊 Field Evolution Types

Field evolution determines how continuous fields evolve over time in the spatial domain.

### Available Evolution Types

#### `cnn`
- **Method**: Convolutional neural network-based evolution
- **Properties**: Learned spatial dynamics, efficient computation
- **Best For**: General-purpose field evolution
- **Models**: All TFN models
- **Parameters**:
  - `hidden_dim`: Hidden dimension for CNN layers (default: 128)

#### `modernized_cnn`
- **Method**: Modern CNN with depthwise convolutions and GLU
- **Properties**: Improved efficiency, better gradient flow
- **Best For**: Large-scale models, efficiency-critical applications
- **Models**: All TFN models
- **Parameters**:
  - `hidden_dim`: Hidden dimension (default: 64)
  - `kernel_sizes`: List of kernel sizes (default: [3, 5, 7])
  - `use_glu`: Use Gated Linear Units (default: True)
  - `use_depthwise`: Use depthwise convolutions (default: True)

#### `pde` / `diffusion`
- **Method**: Partial differential equation-based evolution (diffusion)
- **Mathematical Form**: `∂F/∂t = α∇²F`
- **Properties**: Physics-inspired, smooth evolution
- **Best For**: Smooth field dynamics, physics-constrained problems
- **Models**: All TFN models
- **Parameters**:
  - `dt`: Time step size (default: 0.01)

#### `wave`
- **Method**: Wave equation-based evolution
- **Mathematical Form**: `∂²F/∂t² = c²∇²F`
- **Properties**: Oscillatory dynamics, wave-like propagation
- **Best For**: Oscillatory patterns, wave-like phenomena
- **Models**: All TFN models
- **Parameters**:
  - `dt`: Time step size (default: 0.01)

#### `schrodinger`
- **Method**: Schrödinger equation-based evolution
- **Mathematical Form**: `iℏ∂F/∂t = ĤF`
- **Properties**: Complex-valued evolution, quantum-inspired
- **Best For**: Complex-valued fields, quantum-inspired dynamics
- **Models**: All TFN models
- **Parameters**:
  - `dt`: Time step size (default: 0.01)

#### `spatially_varying_pde`
- **Method**: PDE with spatially varying coefficients
- **Properties**: Adaptive to spatial structure
- **Best For**: Spatially inhomogeneous problems
- **Models**: All TFN models
- **Parameters**:
  - `hidden_dim`: Hidden dimension for coefficient prediction (default: 64)
  - `kernel_size`: Kernel size for spatial averaging (default: 3)

#### `adaptive_time_stepping`
- **Method**: Adaptive time stepping based on field gradients
- **Properties**: Automatic time step adjustment
- **Best For**: Stability-critical applications
- **Models**: All TFN models
- **Parameters**:
  - `base_dt`: Base time step (default: 0.01)
  - `min_dt`: Minimum time step (default: 0.001)
  - `max_dt`: Maximum time step (default: 0.1)
  - `hidden_dim`: Hidden dimension for time step prediction (default: 32)

## ⚡ Interference Types

Field interference determines how different token fields interact with each other.

### Available Interference Types

#### `standard`
- **Method**: Standard token field interference
- **Properties**: Multi-head interference, learnable coupling
- **Best For**: General-purpose field interactions
- **Models**: All TFN models
- **Parameters**:
  - `num_heads`: Number of interference heads (default: 8)
  - `interference_types`: Types of interference ("constructive", "destructive", "phase")

#### `causal`
- **Method**: Causal field interference (respects temporal order)
- **Properties**: Temporal causality, autoregressive behavior
- **Best For**: Sequential data, autoregressive generation
- **Models**: All TFN models
- **Parameters**: Same as standard

#### `multi_scale`
- **Method**: Multi-scale field interference
- **Properties**: Multi-resolution interactions
- **Best For**: Multi-scale patterns, hierarchical structure
- **Models**: All TFN models
- **Parameters**:
  - `scales`: Number of scales (default: 4)
  - Other parameters same as standard

#### `physics_constrained`
- **Method**: Physics-constrained interference
- **Properties**: Energy conservation, symmetry constraints
- **Best For**: Physics-inspired problems, constrained dynamics
- **Models**: All TFN models
- **Parameters**:
  - `energy_weight`: Energy conservation weight (default: 0.1)
  - `symmetry_weight`: Symmetry constraint weight (default: 0.1)
  - Other parameters same as standard

## 📊 Datasets

### Time Series Datasets

#### ETT (Electricity Transformer Temperature)
- **Type**: Multivariate time series forecasting
- **Features**: 7 variables (temperature, load, etc.)
- **Task**: Regression (predict future values)
- **Compatible Models**: `tfn_regressor`, `enhanced_tfn_regressor`
- **Configuration**: `configs/ett.yaml`
- **Special Features**: Instance normalization, time-based embeddings

#### Jena Climate
- **Type**: Climate time series data
- **Features**: 14 weather variables
- **Task**: Regression (predict weather variables)
- **Compatible Models**: `tfn_regressor`, `enhanced_tfn_regressor`
- **Configuration**: `configs/jena.yaml`

#### Stock Market
- **Type**: Financial time series data
- **Features**: OHLCV (Open, High, Low, Close, Volume)
- **Task**: Regression (predict stock prices)
- **Compatible Models**: `tfn_regressor`, `enhanced_tfn_regressor`
- **Configuration**: `configs/stock.yaml`

### NLP Datasets

#### IMDB
- **Type**: Sentiment analysis
- **Features**: Movie reviews
- **Task**: Classification (positive/negative sentiment)
- **Compatible Models**: `tfn_classifier`, `enhanced_tfn_classifier`
- **Configuration**: `configs/imdb.yaml`

#### WikiText
- **Type**: Language modeling
- **Features**: Wikipedia articles
- **Task**: Language modeling (next token prediction)
- **Compatible Models**: `tfn_language_model`
- **Configuration**: `configs/wikitext.yaml`

#### PG19
- **Type**: Language modeling
- **Features**: Project Gutenberg books
- **Task**: Language modeling (next token prediction)
- **Compatible Models**: `tfn_language_model`
- **Configuration**: `configs/pg19.yaml`

#### ArXiv
- **Type**: Text classification
- **Features**: ArXiv paper abstracts
- **Task**: Classification (subject categories)
- **Compatible Models**: `tfn_classifier`, `enhanced_tfn_classifier`
- **Configuration**: `configs/arxiv.yaml`

#### GLUE (SST-2)
- **Type**: Sentiment analysis
- **Features**: Stanford Sentiment Treebank
- **Task**: Classification (positive/negative sentiment)
- **Compatible Models**: `tfn_classifier`, `enhanced_tfn_classifier`
- **Configuration**: `configs/glue_sst2.yaml`

### Synthetic Datasets

#### Synthetic Copy
- **Type**: Synthetic sequence copying
- **Features**: Random integer sequences
- **Task**: Copy task (reproduce input sequence)
- **Compatible Models**: All TFN models
- **Configuration**: `configs/synthetic_copy.yaml`

## 🚀 Training

### Basic Training

```bash
# Train with YAML config
python train.py --config configs/ett.yaml

# Override config parameters
python train.py --config configs/ett.yaml \
    --model.embed_dim 512 \
    --model.num_layers 6 \
    --training.lr 1e-3 \
    --training.batch_size 128

# Use specific model from registry
python train.py --config configs/ett.yaml \
    --model_name enhanced_tfn_regressor
```

### CLI Parameters

#### Model Parameters
- `--model.task`: Task type (classification/regression)
- `--model.vocab_size`: Vocabulary size for classification
- `--model.input_dim`: Input dimension for regression
- `--model.output_dim`: Output dimension for regression
- `--model.output_len`: Output sequence length for regression
- `--model.num_classes`: Number of classes for classification
- `--model.embed_dim`: Embedding dimension
- `--model.num_layers`: Number of TFN layers
- `--model.kernel_type`: Kernel type for field projection
- `--model.evolution_type`: Evolution type for field evolution
- `--model.interference_type`: Interference type for field interference
- `--model.grid_size`: Grid size for field discretization
- `--model.time_steps`: Number of evolution time steps
- `--model.dropout`: Dropout rate
- `--model.use_enhanced`: Use enhanced TFN layers
- `--model.pos_min`, `--model.pos_max`: Position bounds
- `--model.positional_embedding_strategy`: Positional embedding strategy
- `--model.calendar_features`: Calendar features for time-based embeddings
- `--model.feature_cardinalities`: JSON string of feature cardinalities

#### Training Parameters
- `--training.lr`: Learning rate
- `--training.batch_size`: Batch size
- `--training.epochs`: Number of epochs
- `--training.weight_decay`: Weight decay
- `--training.optimizer`: Optimizer type
- `--training.warmup_epochs`: Warmup epochs
- `--training.grad_clip`: Gradient clipping
- `--training.log_interval`: Logging interval

#### Data Parameters
- `--data.dataset`: Dataset name
- `--data.seq_len`: Sequence length
- `--data.vocab_size`: Vocabulary size
- `--data.pad_idx`: Padding index
- `--data.dataset_size`: Dataset size
- `--data.csv_path`: CSV path for time series data
- `--data.max_length`: Max sequence length for NLP tokenization
- `--data.tokenizer_name`: Tokenizer name for NLP datasets
- `--data.normalization_strategy`: Normalization strategy
- `--data.instance_normalize`: Apply instance normalization
- `--data.input_len`: Input window length for time series
- `--data.output_len`: Output window length for time series

#### Convenience Parameters
- `--model_name`: Model name from registry
- `--device`: Device to use (cuda/cpu/auto)

## 🔍 Hyperparameter Search

### Basic Hyperparameter Search

```bash
# Search across multiple models and parameters
python hyperparameter_search.py \
    --models tfn_classifier enhanced_tfn_classifier \
    --param_sweep embed_dim:128,256,512 num_layers:2,4,6 kernel_type:rbf,compact \
    --epochs 20 --patience 5 \
    --output_dir ./search_results

# Search with specific config
python hyperparameter_search.py \
    --config configs/ett.yaml \
    --models tfn_regressor \
    --param_sweep embed_dim:128,256,512 evolution_type:cnn,pde \
    --epochs 30 --patience 8 \
    --output_dir ./ett_search
```

### Search Parameters

#### Search Configuration
- `--models`: List of model names to search
- `--param_sweep`: Parameter sweep specification
- `--epochs`: Maximum epochs per trial
- `--patience`: Early stopping patience
- `--min_epochs`: Minimum epochs before early stopping
- `--output_dir`: Output directory for results
- `--seed`: Random seed for reproducibility

#### Parameter Sweep Format
```
param1:value1,value2,value3 param2:value1,value2 param3:value1
```

Examples:
- `embed_dim:128,256,512 num_layers:2,4,6`
- `kernel_type:rbf,compact,learnable evolution_type:cnn,pde`
- `dropout:0.1,0.2,0.3 lr:1e-4,1e-3`

### Search Results

The hyperparameter search generates:
- **Trial Results**: Individual trial results in JSON format
- **Summary**: Overall search summary with best configurations
- **Search Config**: Search configuration for reproducibility

Results are saved in the specified output directory with structure:
```
output_dir/
├── search_config.json
├── summary.json
└── trials/
    ├── trial_001.json
    ├── trial_002.json
    └── ...
```

## ⚙️ Configuration

### YAML Configuration Files

Configuration files are located in `configs/` and specify all training parameters:

```yaml
model_name: tfn_regressor
task: time_series

data:
  dataset_name: ett
  csv_path: data/ETTh1.csv
  input_len: 336
  output_len: 96
  normalization_strategy: instance
  instance_normalize: true

model:
  input_dim: 7
  embed_dim: 256
  output_dim: 1
  output_len: 96
  num_layers: 4
  dropout: 0.2
  kernel_type: rbf
  evolution_type: cnn
  interference_type: standard
  positional_embedding_strategy: time_based
  calendar_features: ["hour", "day_of_week", "day_of_month", "month"]

training:
  batch_size: 64
  lr: 1e-4
  epochs: 20
  warmup_epochs: 5
  grad_clip: 1.0
  log_interval: 50
```

### Available Configuration Files

- `configs/ett.yaml`: ETT time series forecasting
- `configs/ett_instance_normalization.yaml`: ETT with instance normalization
- `configs/ett_time_based_embeddings.yaml`: ETT with time-based embeddings
- `configs/ett_combined_improvements.yaml`: ETT with both improvements
- `configs/imdb.yaml`: IMDB sentiment analysis
- `configs/wikitext.yaml`: WikiText language modeling
- `configs/stock.yaml`: Stock market prediction
- `configs/jena.yaml`: Jena climate prediction
- `configs/arxiv.yaml`: ArXiv text classification
- `configs/pg19.yaml`: PG19 language modeling
- `configs/synthetic_copy.yaml`: Synthetic copy task

## 📦 Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy
- Pandas
- Transformers (for NLP datasets)
- Scikit-learn
- Matplotlib (for visualization)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/TokenFieldNetwork.git
cd TokenFieldNetwork

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Quick Start

```bash
# Train on ETT dataset
python train.py --config configs/ett.yaml

# Run hyperparameter search
python hyperparameter_search.py \
    --models tfn_regressor \
    --param_sweep embed_dim:128,256,512 \
    --epochs 20 \
    --output_dir ./results
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest test/

# Run specific test
python -m pytest test/test_modular_implementation.py

# Run with coverage
python -m pytest --cov=core --cov=model test/
```

## 📚 Examples

### Time Series Forecasting

```python
from model import registry
from data_pipeline import get_dataloader

# Load data
train_loader = get_dataloader(config, split='train')

# Build model
model = registry.build_model('tfn_regressor', config['model'])

# Train
trainer = Trainer(model, train_loader, config['training'])
trainer.train()
```

### Text Classification

```python
# Load IMDB data
config = {'data': {'dataset_name': 'imdb'}}
train_loader = get_dataloader(config, split='train')

# Build classifier
model = registry.build_model('tfn_classifier', config['model'])

# Train
trainer = Trainer(model, train_loader, config['training'])
trainer.train()
```

### Language Modeling

```python
# Load WikiText data
config = {'data': {'dataset_name': 'wikitext'}}
train_loader = get_dataloader(config, split='train')

# Build language model
model = registry.build_model('tfn_language_model', config['model'])

# Train
trainer = Trainer(model, train_loader, config['training'])
trainer.train()
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by continuous field dynamics in physics
- Built on PyTorch ecosystem
- Uses HuggingFace Transformers for NLP datasets
- Implements novel field-based attention mechanism

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@article{tokenfieldnetwork,
  title={Token Field Networks: Replacing Attention with Field Projection and Evolution},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
``` 