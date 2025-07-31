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
- **Simplified Architecture**: Streamlined components focused on core TFN principles

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
- **`enhanced_tfn_regressor`**: Advanced regression with unified dynamics
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
| **Kernels** | RBF, Compact, Fourier | + Data-dependent, FiLM, Multi-frequency |
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
- **`film_learnable`**: Feature-wise Linear Modulation kernel
- **`data_dependent_rbf`**: Content-aware RBF with learned parameters
- **`data_dependent_compact`**: Content-aware compact kernel
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
- **`modernized_cnn`**: Efficient CNN with multi-scale convolutions

### Field Interference (`core/field_interference.py`)

Controls how token fields interact:

- **`standard`**: Multi-head interference with learnable coupling
- **`causal`**: Respects temporal causality for autoregressive tasks
- **`multi_scale`**: Multi-resolution field interactions

### Unified Field Dynamics (`core/unified_field_dynamics.py`)

Combines evolution and interference in a mathematically sound framework:

- **Linear Evolution**: `L(F)` - Physics-inspired field dynamics
- **Nonlinear Interference**: `I(F)` - Token field interactions
- **Combined Dynamics**: `∂F/∂t = L(F) + I(F)`

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
- **PG-19**: Long-form text generation (`configs/pg19.yaml`)
- **WikiText**: Wikipedia articles (`configs/wikitext.yaml`)

## 🎯 Training

### Basic Training

```bash
# Single model training
python train.py --config configs/ett.yaml

# Multi-GPU training
python train.py --config configs/ett.yaml --gpus 0,1,2,3
```

### Hyperparameter Search

```bash
# Grid search over model parameters
python hyperparameter_search.py \
    --models enhanced_tfn_regressor \
    --param_sweep "embed_dim:128,256 kernel_type:rbf,compact evolution_type:cnn,diffusion" \
    --epochs 50 \
    --output_dir ./search_results

# Bayesian optimization
python hyperparameter_search.py \
    --models enhanced_tfn_regressor \
    --optimizer bayesian \
    --n_trials 100 \
    --epochs 30 \
    --output_dir ./bayesian_search
```

### Advanced Training Features

- **Instance Normalization**: Normalize each sequence independently
- **Time-Based Embeddings**: Calendar-aware positional encoding
- **Multi-Scale Processing**: Handle varying sequence lengths
- **Gradient Clipping**: Prevent gradient explosion
- **Learning Rate Scheduling**: Adaptive learning rates

## ⚙️ Configuration

### Model Configuration

```yaml
model:
  # Core parameters
  embed_dim: 128
  num_layers: 2
  kernel_type: rbf
  evolution_type: cnn
  
  # Field interference
  interference_type: standard  # standard, causal, multiscale
  
  # Grid settings
  grid_size: 100
  
  # Positional embeddings
  positional_embedding_strategy: learned  # learned, sinusoidal, time_based
  max_seq_len: 512
```

### Training Configuration

```yaml
training:
  # Optimization
  optimizer: adam
  learning_rate: 0.001
  weight_decay: 0.01
  
  # Training loop
  epochs: 100
  batch_size: 32
  gradient_clip: 1.0
  
  # Regularization
  dropout: 0.1
  
  # Monitoring
  log_interval: 100
  eval_interval: 1000
```

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
  evolution_type: diffusion  # or wave, schrodinger, spatially_varying_pde
```

**Available Physics Types:**
- **Diffusion**: Smooth spreading dynamics
- **Wave**: Oscillatory propagation  
- **Schrödinger**: Quantum-inspired complex evolution
- **Spatially-varying PDE**: Adaptive coefficients based on local field properties

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

### Enhanced TFN with Physics-Inspired Evolution

```python
from model.tfn_enhanced import EnhancedTFNRegressor

# Model with diffusion evolution and causal interference
model = EnhancedTFNRegressor(
    input_dim=7,
    embed_dim=128,
    output_dim=1,
    output_len=24,
    num_layers=2,
    kernel_type="rbf",
    evolution_type="diffusion",
    interference_type="causal",
    grid_size=100
)

# The model automatically learns proper scaling for evolution and interference
# No need to manually tune dt or interference_weight parameters
```

## 🧪 Testing

### Unit Tests

Comprehensive unit tests verify mathematical correctness:

```bash
# Test kernel mathematical properties
python -m pytest test/test_kernels.py -v

# Test evolution mathematical properties  
python -m pytest test/test_evolution.py -v

# Test full model integration
python test_modular_implementation.py
```

### Test Coverage

- **Kernel Tests**: Verify RBF, Compact, Fourier kernel properties
- **Evolution Tests**: Test diffusion, wave, CNN evolution methods
- **Numerical Stability**: Ensure finite outputs with extreme inputs
- **Mathematical Properties**: Verify conservation laws and physical constraints

## 🔬 Research Features

### Recent Improvements

- **Simplified Architecture**: Removed redundant components for cleaner implementation
- **Enhanced Numerical Stability**: Fixed unsafe parameter modifications
- **Improved Causal Interference**: Safe, differentiable causal masking
- **Streamlined Hyperparameters**: Let model learn scaling internally
- **Comprehensive Testing**: Unit tests for all core mathematical functions

### Key Design Principles

1. **Mathematical Rigor**: All components are fully differentiable
2. **Numerical Stability**: Careful handling of tensor operations
3. **Modular Design**: Independent, testable components
4. **Physics-Inspired**: Leverage continuous field dynamics
5. **Simplified Interface**: Focus on core TFN principles

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

We welcome contributions! Please see CONTRIBUTING.md for guidelines.

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