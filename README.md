# Token Field Network (TFN)

A novel deep learning architecture that replaces attention with field projection and evolution. Each token emits a continuous field, which is evolved over a spatial grid and sampled back to update tokens.

## 🏗️ Architecture Overview

TFN operates through three core phases:
1. **Field Projection**: Tokens emit continuous fields across spatial domain using learnable kernels
2. **Field Evolution**: Fields evolve over time using physics-inspired or learned dynamics
3. **Field Sampling**: Evolved fields are sampled back to update token representations

This approach provides:
- **Mathematical Rigor**: Fully differentiable field-based attention mechanism
- **Physics-Inspired**: Leverages continuous field dynamics (diffusion, wave, Schrödinger equations)
- **Modular Design**: Independent, testable components for field emission, evolution, and sampling
- **Numerical Stability**: Careful handling of tensor operations and gradient flow

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Models](#models)
- [Core Components](#core-components)
- [Datasets](#datasets)
- [Training](#training)
- [Configuration](#configuration)
- [Advanced Features](#advanced-features)
- [Installation](#installation)
- [Examples](#examples)

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/your-repo/TokenFieldNetwork.git
cd TokenFieldNetwork
pip install -r requirements.txt

# Train on time series data with instance normalization
python train.py --config configs/ett_combined_improvements.yaml

# Train on text classification
python train.py --config configs/imdb.yaml

# Run hyperparameter search
python hyperparameter_search.py \
    --models tfn_regressor enhanced_tfn_regressor \
    --param_sweep embed_dim:128,256,512 kernel_type:rbf,compact \
    --epochs 20 --output_dir ./search_results
```

## 🤖 Models

### Unified TFN Architecture

The codebase provides a **unified TFN model** (`TFN`) that handles both classification and regression tasks through configuration:

```python
from model.tfn_unified import TFN

# Classification model
classifier = TFN(
    task="classification",
    vocab_size=30522,
    num_classes=2,
    embed_dim=128,
    kernel_type="rbf",
    evolution_type="cnn"
)

# Regression model  
regressor = TFN(
    task="regression",
    input_dim=7,
    output_dim=1,
    output_len=24,
    embed_dim=128,
    kernel_type="rbf",
    evolution_type="cnn"
)
```

### Model Registry

The model registry provides pre-configured model templates:

#### **Core TFN Models**
- **`tfn_classifier`**: Text/sequence classification
- **`tfn_regressor`**: Time series forecasting and regression
- **`tfn_language_model`**: Language modeling and text generation
- **`tfn_vision`**: 2D image classification (uses `ImageTFN`)

#### **Enhanced TFN Models**
- **`enhanced_tfn_classifier`**: Advanced classification with field interference
- **`enhanced_tfn_regressor`**: Advanced regression with physics constraints
- **`enhanced_tfn_language_model`**: Advanced language modeling with unified dynamics

#### **Baseline Models**
- **`transformer_classifier/regressor`**: Standard Transformer baselines
- **`performer_classifier/regressor`**: Linear attention baselines
- **`lstm_classifier/regressor`**: LSTM baselines
- **`cnn_classifier/regressor`**: CNN baselines

### Model Comparison

| Feature | Base TFN | Enhanced TFN |
|---------|----------|--------------|
| **Field Evolution** | CNN, PDE | CNN, PDE, Diffusion, Wave, Schrödinger, Spatially-varying PDE |
| **Field Interference** | Optional | Integrated (Standard, Causal, Multi-scale) |
| **Physics Constraints** | ❌ | ✅ (Energy conservation, symmetry) |
| **Kernels** | RBF, Compact, Fourier | + Learnable, FiLM, Data-dependent |
| **Complexity** | Low | High |
| **Use Case** | General tasks | Physics-inspired, complex sequences |

## 🔧 Core Components

### Field Kernels (`core/kernels.py`)

Kernels determine how tokens emit fields across the spatial domain:

#### **Basic Kernels**
- **`rbf`**: Radial Basis Function - `K(z,μ,σ) = exp(-||z-μ||²/(2σ²))`
- **`compact`**: Finite support - `K(z,μ,r) = max(0, 1-||z-μ||/r)`
- **`fourier`**: Oscillatory - `K(z,μ,f) = cos(2πf||z-μ||)`

#### **Advanced Kernels**
- **`learnable`**: Neural network-based adaptive kernel
- **`film_learnable`**: Feature-wise Linear Modulation kernel
- **`data_dependent_rbf`**: Content-aware RBF with learned parameters
- **`multi_frequency_fourier`**: Multi-scale frequency analysis

### Field Evolution (`core/field_evolution.py`)

Evolution types control how fields change over time:

#### **Base Evolution Types**
- **`cnn`**: Learned convolutional dynamics
- **`pde`**: Diffusion equation - `∂F/∂t = α∇²F`

#### **Enhanced Evolution Types** 
- **`diffusion`**: Standard diffusion dynamics
- **`wave`**: Wave equation - `∂²F/∂t² = c²∇²F`
- **`schrodinger`**: Quantum-inspired - `iℏ∂F/∂t = ĤF`
- **`spatially_varying_pde`**: Adaptive PDE coefficients
- **`modernized_cnn`**: Efficient CNN with depthwise convolutions

### Field Interference (`core/field_interference.py`)

Controls how token fields interact:

- **`standard`**: Multi-head interference with learnable coupling
- **`causal`**: Respects temporal causality for autoregressive tasks
- **`multi_scale`**: Multi-resolution field interactions
- **`physics_constrained`**: Energy conservation and symmetry constraints

### Positional Embeddings (`model/shared_layers.py`)

Modular positional encoding strategies:

- **`learned`**: Standard learnable absolute positions
- **`sinusoidal`**: Transformer-style sinusoidal embeddings
- **`time_based`**: Calendar-aware embeddings for time series

#### Time-Based Embeddings Example
```python
# For time series with temporal patterns
positional_embedding_strategy: "time_based"
calendar_features: ["hour", "day_of_week", "day_of_month", "month", "is_weekend"]
feature_cardinalities:
  hour: 24
  day_of_week: 7
  day_of_month: 31
  month: 12
  is_weekend: 2
```

## 📊 Datasets

### Time Series Datasets

#### **ETT (Electricity Transformer Temperature)**
- **Features**: 7 variables (temperature, load, oil temperature, etc.)
- **Task**: Multivariate forecasting
- **Special Features**: Instance normalization, time-based embeddings
- **Configs**: 
  - `configs/ett.yaml` - Basic configuration
  - `configs/ett_instance_normalization.yaml` - With instance normalization
  - `configs/ett_time_based_embeddings.yaml` - With temporal embeddings
  - `configs/ett_combined_improvements.yaml` - Both features combined

#### **Other Time Series**
- **Jena Climate**: 14 weather variables (`configs/jena.yaml`)
- **Stock Market**: OHLCV financial data (`configs/stock.yaml`)

### NLP Datasets

#### **Text Classification**
- **IMDB**: Movie review sentiment analysis (`configs/imdb.yaml`)
- **ArXiv**: Paper abstract classification (`configs/arxiv.yaml`)

#### **Language Modeling**
- **WikiText**: Wikipedia articles (`configs/wikitext.yaml`)
- **PG19**: Project Gutenberg books (`configs/pg19.yaml`)

#### **Synthetic Tasks**
- **Synthetic Copy**: Sequence copying task (`configs/synthetic_copy.yaml`)

### Data Normalization Strategies

The codebase supports multiple normalization approaches:

#### **For Time Series**
- **`global`**: Dataset-wide standardization (default)
- **`instance`**: Per-sample normalization (removes scale, preserves patterns)
- **`feature_wise`**: Per-feature standardization

```yaml
data:
  normalization_strategy: "instance"  # or "global", "feature_wise"
  instance_normalize: true  # Apply during data loading
```

## 🚀 Training

### Basic Training

```bash
# Train with config file
python train.py --config configs/ett_combined_improvements.yaml

# Override parameters
python train.py --config configs/imdb.yaml \
    --model.embed_dim 256 \
    --model.num_layers 4 \
    --training.lr 1e-4 \
    --training.batch_size 64
```

### Model Selection

```bash
# Use specific model from registry
python train.py --config configs/ett.yaml \
    --model_name enhanced_tfn_regressor

# Available model names:
# - tfn_classifier, tfn_regressor, tfn_language_model, tfn_vision
# - enhanced_tfn_classifier, enhanced_tfn_regressor, enhanced_tfn_language_model  
# - transformer_classifier, performer_classifier, lstm_classifier, cnn_classifier
# - transformer_regressor, performer_regressor, lstm_regressor, cnn_regressor
```

### Hyperparameter Search

```bash
# Search across models and parameters
python hyperparameter_search.py \
    --models tfn_regressor enhanced_tfn_regressor \
    --param_sweep embed_dim:128,256,512 kernel_type:rbf,compact evolution_type:cnn,pde \
    --epochs 30 --patience 8 \
    --output_dir ./search_results

# Search with specific config base
python hyperparameter_search.py \
    --config configs/ett.yaml \
    --models tfn_regressor \
    --param_sweep embed_dim:128,256 evolution_type:cnn,pde,diffusion \
    --epochs 20 --output_dir ./ett_search
```

## ⚙️ Configuration

### YAML Configuration Structure

```yaml
# Model selection (optional - can use model_name instead)
model_name: enhanced_tfn_regressor

# Data configuration
data:
  dataset_name: ett
  csv_path: data/ETTh1.csv
  input_len: 96      # Input sequence length
  output_len: 24     # Output sequence length
  normalization_strategy: instance
  instance_normalize: true

# Model architecture
model:
  task: regression
  input_dim: 7       # Number of input features
  embed_dim: 128     # Embedding dimension
  output_dim: 1      # Number of output features
  output_len: 24     # Output sequence length
  num_layers: 2      # Number of TFN layers
  kernel_type: rbf   # Field emission kernel
  evolution_type: cnn # Field evolution method
  interference_type: standard # Field interference (enhanced models only)
  grid_size: 100     # Spatial discretization
  time_steps: 3      # Evolution steps
  dropout: 0.1
  
  # Advanced features
  use_enhanced: false
  positional_embedding_strategy: time_based
  calendar_features: ["hour", "day_of_week", "month"]
  feature_cardinalities:
    hour: 24
    day_of_week: 7
    month: 12

# Training configuration  
training:
  batch_size: 32
  lr: 1e-3
  epochs: 50
  weight_decay: 1e-4
  optimizer: adamw
  warmup_epochs: 5
  grad_clip: 1.0
  log_interval: 100
```

### Parameter Override

```bash
# Override nested parameters
python train.py --config configs/ett.yaml \
    --model.embed_dim 512 \
    --model.kernel_type compact \
    --data.normalization_strategy global \
    --training.lr 5e-4
```

## 🔬 Advanced Features

### Instance Normalization for Time Series

Instance normalization normalizes each time series sample individually, helping the model focus on patterns rather than absolute values:

```yaml
data:
  normalization_strategy: "instance"
  instance_normalize: true
```

**Benefits:**
- Scale-invariant pattern recognition
- Better generalization across different value ranges
- Improved training stability for time series

### Time-Based Embeddings

Calendar-aware positional embeddings capture temporal patterns:

```yaml
model:
  positional_embedding_strategy: "time_based"
  calendar_features: ["hour", "day_of_week", "day_of_month", "month", "is_weekend"]
  feature_cardinalities:
    hour: 24
    day_of_week: 7
    day_of_month: 31
    month: 12
    is_weekend: 2
```

**Benefits:**
- Captures daily, weekly, monthly patterns
- Better than pure positional encoding for time series
- Handles irregular temporal dependencies

### Physics-Inspired Evolution

Enhanced models support physics-based field evolution:

```yaml
model:
  evolution_type: diffusion  # or wave, schrodinger
  use_physics_constraints: true
  constraint_weight: 0.1
```

**Available Physics Types:**
- **Diffusion**: Smooth spreading dynamics
- **Wave**: Oscillatory propagation
- **Schrödinger**: Quantum-inspired complex evolution

## 📦 Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy, Pandas
- Transformers (for NLP datasets)
- Scikit-learn
- Matplotlib (optional, for visualization)

### Setup

```bash
# Clone repository
git clone https://github.com/your-repo/TokenFieldNetwork.git
cd TokenFieldNetwork

# Install dependencies
pip install torch numpy pandas transformers scikit-learn matplotlib

# Verify installation
python test_modular_implementation.py
```

## 📚 Examples

### Time Series Forecasting with Advanced Features

```python
from model.tfn_unified import TFN
from data_pipeline import get_dataloader
import yaml

# Load config with advanced features
with open('configs/ett_combined_improvements.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Get data loader
train_loader = get_dataloader(config, split='train')

# Build model with time-based embeddings and instance normalization
model = TFN(
    task="regression",
    input_dim=7,
    output_dim=1,
    output_len=24,
    embed_dim=128,
    num_layers=2,
    kernel_type="rbf",
    evolution_type="cnn",
    positional_embedding_strategy="time_based",
    calendar_features=["hour", "day_of_week", "month"],
    feature_cardinalities={"hour": 24, "day_of_week": 7, "month": 12}
)

# Train
from src.trainer import Trainer
trainer = Trainer(model, train_loader, config['training'])
trainer.train()
```

### Enhanced TFN with Physics Constraints

```python
from model.tfn_enhanced import EnhancedTFNRegressor

# Build enhanced model with physics-inspired evolution
model = EnhancedTFNRegressor(
    input_dim=7,
    embed_dim=128,
    output_dim=1,
    output_len=24,
    num_layers=1,
    kernel_type="film_learnable",
    evolution_type="diffusion",
    interference_type="causal",
    use_physics_constraints=True,
    constraint_weight=0.1
)
```

### Text Classification

```python
# Load IMDB sentiment analysis
config = {
    'data': {'dataset_name': 'imdb', 'max_length': 512},
    'model': {
        'task': 'classification',
        'vocab_size': 30522,
        'num_classes': 2,
        'embed_dim': 128,
        'kernel_type': 'rbf',
        'evolution_type': 'cnn'
    }
}

train_loader = get_dataloader(config, split='train')
model = TFN(**config['model'])
```

### Model Registry Usage

```python
from model import registry

# Get model configuration
config = registry.get_model_config('enhanced_tfn_regressor')
print(f"Required params: {config['required_params']}")
print(f"Evolution types: {config['evolution_types']}")

# Build model from registry
model_params = {
    'input_dim': 7,
    'embed_dim': 256,
    'output_dim': 1,
    'output_len': 24,
    'kernel_type': 'rbf',
    'interference_type': 'causal'
}
model = registry.MODEL_REGISTRY['enhanced_tfn_regressor']['class'](**model_params)
```

## 🧪 Testing

```bash
# Run all tests
python test_modular_implementation.py

# Test model compatibility
python test/test_model_compatibility.py

# Run with pytest (if available)
python -m pytest test/ -v
```

## 🛠️ Development

### Code Structure

```
TokenFieldNetwork/
├── core/                    # Core TFN components
│   ├── kernels.py          # Field emission kernels
│   ├── field_evolution.py  # Field evolution dynamics
│   ├── field_interference.py # Field interference mechanisms
│   ├── field_projection.py # Field projection logic
│   ├── field_sampling.py   # Field sampling back to tokens
│   ├── unified_field_dynamics.py # Integrated dynamics
│   └── utils.py            # Utility functions
├── model/                   # Model architectures
│   ├── tfn_unified.py      # Main TFN model
│   ├── tfn_enhanced.py     # Enhanced TFN with advanced features
│   ├── tfn_pytorch.py      # 2D image TFN
│   ├── baselines.py        # Baseline models
│   ├── shared_layers.py    # Shared components (embeddings, etc.)
│   └── registry.py         # Model registry and configuration
├── data/                    # Data loading and processing
├── configs/                 # Configuration files
├── src/                     # Training infrastructure
│   ├── trainer.py          # Training loop
│   ├── metrics.py          # Evaluation metrics
│   └── task_strategies.py  # Task-specific logic
└── train.py                # Main training script
```

### Adding New Components

1. **New Kernel**: Add to `core/kernels.py`, inherit from `KernelBasis`
2. **New Evolution**: Add to `core/field_evolution.py`, inherit from `FieldEvolution`
3. **New Model**: Add to `model/registry.py` with configuration
4. **New Dataset**: Add loader to `data/` and config to `configs/`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-kernel`)
3. Implement your changes with proper typing and docstrings
4. Add tests for new functionality
5. Ensure `python test_modular_implementation.py` passes
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by continuous field dynamics in physics
- Built on the PyTorch ecosystem
- Uses HuggingFace Transformers for NLP datasets
- Implements novel field-based attention mechanisms

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@article{tokenfieldnetwork2024,
  title={Token Field Networks: Replacing Attention with Field Projection and Evolution},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
``` 