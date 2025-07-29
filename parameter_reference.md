# Token Field Network Parameter Reference

This document provides a comprehensive reference for all parameters available in the training scripts.

## 📋 Universal CLI Parameters

These parameters work with all models and training scripts.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--config` | str | `configs/tests/synthetic_copy_test.yaml` | Path to YAML config file |
| `--model_name` | str | `tfn_classifier` | Model to use (see model registry) |
| `--output_dir` | str | `./outputs` | Directory to save logs and checkpoints |
| `--seed` | int | `42` | Random seed for reproducibility |
| `--disable_logging` | flag | `False` | Disable logging to external services |
| `--learning_rate` | float | `0.001` | Learning rate for optimizer |
| `--batch_size` | int | `32` | Batch size for training |
| `--epochs` | int | `50` | Number of training epochs |
| `--weight_decay` | float | `0.01` | L2 regularization strength |
| `--optimizer` | str | `adamw` | Optimizer to use (adamw, sgd, etc.) |
| `--warmup_epochs` | int | `5` | Number of warmup epochs for learning rate scheduling |

## 🤖 Model-Specific Parameters

### Universal Model Parameters

| Parameter | Type | Default | Description | Models |
|-----------|------|---------|-------------|--------|
| `--model.embed_dim` | int | `128` | Embedding dimension | All TFN models |
| `--model.num_layers` | int | `2` | Number of model layers | All TFN models |
| `--model.dropout` | float | `0.1` | Dropout rate for regularization | All models |
| `--model.d_model` | int | `256` | Main hidden dimension | Transformer models |
| `--model.n_layers` | int | `4` | Number of layers | Transformer models |
| `--model.n_heads` | int | `8` | Number of attention heads | Transformer models |
| `--model.hidden_dim` | int | `128` | Hidden dimension | LSTM models |
| `--model.proj_dim` | int | `64` | Projection dimension | Performer models |
| `--model.num_filters` | int | `64` | Number of filters | CNN models |
| `--model.filter_sizes` | str | `"3,4,5"` | Filter sizes (comma-separated) | CNN models |
| `--model.bidirectional` | flag | `False` | Use bidirectional LSTM | LSTM models |

### TFN-Specific Parameters

| Parameter | Type | Default | Description | Enhanced Only |
|-----------|------|---------|-------------|---------------|
| `--model.kernel_type` | str | `rbf` | Kernel type for field projection | ❌ |
| `--model.evolution_type` | str | `cnn` | Evolution type for field dynamics | ❌ |
| `--model.interference_type` | str | `standard` | Interference type for field interactions | ❌ |
| `--model.grid_size` | int | `100` | Spatial grid resolution | ❌ |
| `--model.time_steps` | int | `3` | Number of evolution time steps | ❌ |
| `--model.use_enhanced` | flag | `False` | Enable enhanced TFN features | ✅ |
| `--model.kernel_hidden_dim` | int | `64` | Hidden dimension for kernel predictors | ✅ |
| `--model.evolution_hidden_dim` | int | `32` | Hidden dimension for evolution predictors | ✅ |
| `--model.num_frequencies` | int | `8` | Number of frequencies for Fourier kernels | ✅ |
| `--model.min_dt` | float | `0.001` | Minimum time step for adaptive evolution | ✅ |
| `--model.max_dt` | float | `0.1` | Maximum time step for adaptive evolution | ✅ |
| `--model.use_modernized_cnn` | flag | `False` | Use modernized CNN evolution | ✅ |

### Task-Specific Parameters

#### Classification Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model.num_classes` | int | `2` | Number of output classes |
| `--data.vocab_size` | int | `10000` | Vocabulary size for text data |
| `--data.max_length` | int | `512` | Maximum sequence length |
| `--data.tokenizer_name` | str | `bert-base-uncased` | Tokenizer name for NLP datasets |

#### Regression Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model.input_dim` | int | `1` | Input feature dimension |
| `--model.output_dim` | int | `1` | Output feature dimension |
| `--model.output_len` | int | `1` | Output sequence length |
| `--data.input_len` | int | `100` | Input window length |
| `--data.output_len` | int | `10` | Output window length |
| `--data.context_length` | int | `512` | Input context length |
| `--data.prediction_length` | int | `10` | Prediction length |

## 📊 Kernel Types Reference

### Standard Kernels (Available in All Models)

| Kernel Type | Description | Mathematical Form | Use Case |
|-------------|-------------|-------------------|----------|
| `rbf` | Radial Basis Function | `exp(-||x-y||²/2σ²)` | Standard field projection |
| `compact` | Compact support kernel | `max(0, 1-||x-y||/r)` | Local field interactions |
| `fourier` | Fourier basis kernel | `cos(ω(x-y))` | Frequency domain processing |
| `learnable` | Learnable kernel parameters | `f_θ(x, y)` | Adaptive field projection |

### Enhanced Kernels (Require `use_enhanced=True`)

| Kernel Type | Description | Mathematical Form | Use Case |
|-------------|-------------|-------------------|----------|
| `data_dependent_rbf` | Data-dependent RBF | `exp(-||x-y||²/2σ(x,y)²)` | Enhanced with learned parameters |
| `data_dependent_compact` | Data-dependent compact | `max(0, 1-||x-y||/r(x,y))` | Enhanced local interactions |
| `multi_frequency_fourier` | Multi-frequency Fourier | `Σᵢ cos(ωᵢ(x-y))` | Enhanced frequency processing |
| `film_learnable` | FiLM-conditioned learnable | `γ(x)f_θ(x,y) + β(x)` | Advanced adaptive kernels |

## 🔄 Evolution Types Reference

### Standard Evolution (Available in All Models)

| Evolution Type | Description | Mathematical Form | Use Case |
|----------------|-------------|-------------------|----------|
| `cnn` | Convolutional Neural Network | `∂F/∂t = Conv(F)` | Standard spatial evolution |
| `pde` | Partial Differential Equation | `∂F/∂t = L(F)` | Physics-inspired evolution |
| `diffusion` | Diffusion equation | `∂F/∂t = D∇²F` | Smooth field evolution |
| `wave` | Wave equation | `∂²F/∂t² = c²∇²F` | Oscillatory dynamics |
| `schrodinger` | Schrödinger equation | `iℏ∂F/∂t = ĤF` | Quantum-inspired evolution |

### Enhanced Evolution (Require `use_enhanced=True`)

| Evolution Type | Description | Mathematical Form | Use Case |
|----------------|-------------|-------------------|----------|
| `spatially_varying_pde` | Spatially-varying PDE | `∂F/∂t = L(x)F` | Enhanced physics modeling |
| `modernized_cnn` | Modernized CNN | `∂F/∂t = DepthwiseConv(F)` | Enhanced spatial processing |
| `adaptive_time_stepping` | Adaptive time stepping | `∂F/∂t = L(F, dt(x))` | Adaptive evolution control |

## 🌊 Interference Types Reference

### Standard Interference (Available in All Models)

| Interference Type | Description | Mathematical Form | Use Case |
|-------------------|-------------|-------------------|----------|
| `standard` | Standard field interference | `F₁ + F₂` | Basic field interactions |
| `causal` | Causal interference | `F₁ + F₂ * mask(t)` | Time-series applications |

### Enhanced Interference (Require `use_enhanced=True`)

| Interference Type | Description | Mathematical Form | Use Case |
|-------------------|-------------|-------------------|----------|
| `multiscale` | Multi-scale interference | `Σᵢ Fᵢ * scale_factor(i)` | Multi-resolution processing |
| `physics` | Physics-constrained interference | `F₁ + F₂ + constraints` | Physics-informed interactions |

## 🔍 Hyperparameter Search Parameters

### Search-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--models` | list | `["tfn_classifier"]` | Models to search |
| `--param_sweep` | str | `"embed_dim:128,256"` | Parameter sweep specification |
| `--patience` | int | `5` | Early stopping patience |
| `--min_epochs` | int | `3` | Minimum epochs before early stopping |
| `--track_flops` | flag | `False` | Enable FLOPS tracking |

### Parameter Sweep Format

The `--param_sweep` argument uses the following format:
```
"param1:value1,value2,value3 param2:value1,value2 param3:value1"
```

Examples:
```bash
# Basic sweep
--param_sweep "embed_dim:128,256,512 num_layers:2,4"

# Comprehensive sweep
--param_sweep "embed_dim:128,256 kernel_type:rbf,compact,data_dependent_rbf evolution_type:cnn,diffusion learning_rate:1e-3,1e-4"
```

## 📈 Data Parameters

### Universal Data Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--data.seq_len` | int | `512` | Sequence length for synthetic data |
| `--data.dataset_size` | int | `10000` | Dataset size for synthetic data |
| `--data.pad_idx` | int | `0` | Padding index for synthetic data |
| `--data.csv_path` | str | `None` | CSV path for ETT data |

### Task-Specific Data Parameters

#### Classification Data
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--data.vocab_size` | int | `10000` | Vocabulary size |
| `--data.max_length` | int | `512` | Maximum sequence length |
| `--data.tokenizer_name` | str | `bert-base-uncased` | Tokenizer name |

#### Regression Data
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--data.input_len` | int | `100` | Input window length |
| `--data.output_len` | int | `10` | Output window length |
| `--data.context_length` | int | `512` | Input context length |
| `--data.prediction_length` | int | `10` | Prediction length |

## ⚠️ Parameter Compatibility Matrix

### Model Compatibility

| Parameter | TFN | Enhanced TFN | Transformer | Performer | LSTM | CNN |
|-----------|-----|--------------|-------------|-----------|------|-----|
| `kernel_type` | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| `evolution_type` | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| `interference_type` | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| `n_heads` | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ |
| `hidden_dim` | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| `num_filters` | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

### Enhanced Feature Requirements

| Feature | Requires `use_enhanced=True` | Additional Requirements |
|---------|------------------------------|------------------------|
| `data_dependent_rbf` | ✅ | `kernel_hidden_dim` |
| `data_dependent_compact` | ✅ | `kernel_hidden_dim` |
| `multi_frequency_fourier` | ✅ | `num_frequencies` |
| `film_learnable` | ✅ | `kernel_hidden_dim` |
| `spatially_varying_pde` | ✅ | `evolution_hidden_dim` |
| `modernized_cnn` | ✅ | None |
| `adaptive_time_stepping` | ✅ | `min_dt`, `max_dt` |
| `multiscale` interference | ✅ | None |
| `physics` interference | ✅ | None |

## 🎯 Performance Guidelines

### Memory Usage
- **Grid Size**: Memory scales quadratically with `grid_size`
- **Time Steps**: Memory scales linearly with `time_steps`
- **Enhanced Features**: Additional memory for predictors

### Recommended Settings by Use Case

#### Quick Experiments
```bash
--model.embed_dim 128 \
--model.num_layers 2 \
--model.grid_size 50 \
--model.time_steps 3 \
--batch_size 32
```

#### Standard Training
```bash
--model.embed_dim 256 \
--model.num_layers 4 \
--model.grid_size 100 \
--model.time_steps 5 \
--batch_size 16
```

#### Advanced Research
```bash
--model.use_enhanced \
--model.embed_dim 512 \
--model.num_layers 6 \
--model.grid_size 200 \
--model.time_steps 7 \
--model.kernel_type data_dependent_rbf \
--model.evolution_type spatially_varying_pde \
--model.interference_type multiscale \
--batch_size 8
```

## 🔧 Debugging Parameters

### Verbose Logging
```bash
--batch_size 8 \
--epochs 5 \
--model.grid_size 25 \
--model.time_steps 2
```

### Memory Monitoring
```bash
--track_flops \
--model.grid_size 50 \
--batch_size 4
```

### Parameter Validation
The training scripts automatically warn about unsupported parameters:
```bash
python train.py --model_name tfn_classifier --model.n_heads 8
# Output: ⚠️ Warning: n_heads parameter not supported by tfn_classifier
``` 