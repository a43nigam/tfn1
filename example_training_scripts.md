# Example Training Scripts for Token Field Network

This document provides comprehensive examples of how to use the training scripts with all available parameters and flags.

## 🚀 Basic Training Scripts

### 1. Basic TFN Classification Training

```bash
python train.py \
    --config configs/tests/synthetic_copy_test.yaml \
    --model_name tfn_classifier \
    --output_dir ./outputs/basic_tfn \
    --seed 42 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --epochs 50 \
    --weight_decay 0.01 \
    --optimizer adamw \
    --warmup_epochs 5 \
    --model.embed_dim 256 \
    --model.num_layers 4 \
    --model.dropout 0.1 \
    --model.kernel_type rbf \
    --model.evolution_type cnn \
    --model.grid_size 100 \
    --model.time_steps 3 \
    --data.seq_len 512 \
    --data.vocab_size 10000 \
    --data.dataset_size 10000
```

### 2. Enhanced TFN with Advanced Features

```bash
python train.py \
    --config configs/tests/synthetic_copy_test.yaml \
    --model_name tfn_classifier \
    --output_dir ./outputs/enhanced_tfn \
    --seed 42 \
    --learning_rate 0.0005 \
    --batch_size 16 \
    --epochs 100 \
    --weight_decay 0.001 \
    --optimizer adamw \
    --warmup_epochs 10 \
    --model.embed_dim 512 \
    --model.num_layers 6 \
    --model.dropout 0.2 \
    --model.use_enhanced \
    --model.kernel_type data_dependent_rbf \
    --model.evolution_type spatially_varying_pde \
    --model.interference_type multiscale \
    --model.grid_size 200 \
    --model.time_steps 5 \
    --model.kernel_hidden_dim 128 \
    --model.evolution_hidden_dim 64 \
    --model.min_dt 0.001 \
    --model.max_dt 0.1 \
    --data.seq_len 1024 \
    --data.vocab_size 20000 \
    --data.dataset_size 50000
```

### 3. TFN Regression Training

```bash
python train.py \
    --config configs/tests/synthetic_copy_test.yaml \
    --model_name tfn_regressor \
    --output_dir ./outputs/tfn_regression \
    --seed 42 \
    --learning_rate 0.001 \
    --batch_size 64 \
    --epochs 30 \
    --weight_decay 0.01 \
    --optimizer adamw \
    --warmup_epochs 3 \
    --model.embed_dim 128 \
    --model.num_layers 3 \
    --model.dropout 0.1 \
    --model.kernel_type compact \
    --model.evolution_type diffusion \
    --model.interference_type standard \
    --model.grid_size 50 \
    --model.time_steps 4 \
    --model.input_dim 10 \
    --model.output_dim 1 \
    --model.output_len 5 \
    --data.input_len 100 \
    --data.output_len 5
```

### 4. Transformer Baseline Comparison

```bash
python train.py \
    --config configs/tests/synthetic_copy_test.yaml \
    --model_name transformer_regressor \
    --output_dir ./outputs/transformer_baseline \
    --seed 42 \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --epochs 50 \
    --weight_decay 0.01 \
    --optimizer adamw \
    --warmup_epochs 5 \
    --model.d_model 256 \
    --model.n_layers 4 \
    --model.n_heads 8 \
    --model.dropout 0.1 \
    --data.context_length 512 \
    --data.prediction_length 10
```

## 🔍 Hyperparameter Search Examples

### 1. Basic Hyperparameter Search

```bash
python hyperparameter_search.py \
    --config configs/tests/synthetic_copy_test.yaml \
    --models tfn_classifier enhanced_tfn_classifier \
    --param_sweep "embed_dim:128,256,512 num_layers:2,4,6 kernel_type:rbf,compact,data_dependent_rbf evolution_type:cnn,diffusion,spatially_varying_pde learning_rate:1e-3,1e-4" \
    --output_dir ./search_results/basic_search \
    --epochs 20 \
    --patience 5 \
    --min_epochs 3 \
    --seed 42
```

### 2. Comprehensive Enhanced TFN Search

```bash
python hyperparameter_search.py \
    --config configs/tests/synthetic_copy_test.yaml \
    --models enhanced_tfn_classifier \
    --param_sweep "embed_dim:256,512 num_layers:4,6,8 kernel_type:data_dependent_rbf,data_dependent_compact,multi_frequency_fourier evolution_type:spatially_varying_pde,modernized_cnn,adaptive_time_stepping interference_type:standard,causal,multiscale,physics grid_size:100,200 time_steps:3,5,7 learning_rate:5e-4,1e-4,5e-5" \
    --output_dir ./search_results/enhanced_search \
    --epochs 50 \
    --patience 10 \
    --min_epochs 5 \
    --seed 42 \
    --track_flops
```

### 3. Multi-Model Comparison Search

```bash
python hyperparameter_search.py \
    --config configs/tests/synthetic_copy_test.yaml \
    --models tfn_classifier enhanced_tfn_classifier transformer_regressor performer_regressor \
    --param_sweep "embed_dim:128,256 num_layers:2,4 kernel_type:rbf,compact evolution_type:cnn,diffusion learning_rate:1e-3,1e-4" \
    --output_dir ./search_results/model_comparison \
    --epochs 30 \
    --patience 7 \
    --min_epochs 5 \
    --seed 42
```

### 4. Regression Task Search

```bash
python hyperparameter_search.py \
    --config configs/tests/synthetic_copy_test.yaml \
    --models tfn_regressor enhanced_tfn_regressor \
    --param_sweep "embed_dim:64,128,256 num_layers:2,3,4 kernel_type:rbf,compact,fourier evolution_type:cnn,diffusion,wave interference_type:standard,causal grid_size:50,100 time_steps:3,5 learning_rate:1e-3,5e-4,1e-4" \
    --output_dir ./search_results/regression_search \
    --epochs 40 \
    --patience 8 \
    --min_epochs 5 \
    --seed 42
```

## 📊 Parameter Tables

### Kernel Types

| Kernel Type | Description | Enhanced Only | Use Case |
|-------------|-------------|---------------|----------|
| `rbf` | Radial Basis Function | ❌ | Standard field projection |
| `compact` | Compact support kernel | ❌ | Local field interactions |
| `fourier` | Fourier basis kernel | ❌ | Frequency domain processing |
| `learnable` | Learnable kernel parameters | ❌ | Adaptive field projection |
| `data_dependent_rbf` | Data-dependent RBF | ✅ | Enhanced with learned parameters |
| `data_dependent_compact` | Data-dependent compact | ✅ | Enhanced local interactions |
| `multi_frequency_fourier` | Multi-frequency Fourier | ✅ | Enhanced frequency processing |
| `film_learnable` | FiLM-conditioned learnable | ✅ | Advanced adaptive kernels |

### Evolution Types

| Evolution Type | Description | Enhanced Only | Use Case |
|----------------|-------------|---------------|----------|
| `cnn` | Convolutional Neural Network | ❌ | Standard spatial evolution |
| `pde` | Partial Differential Equation | ❌ | Physics-inspired evolution |
| `diffusion` | Diffusion equation | ❌ | Smooth field evolution |
| `wave` | Wave equation | ❌ | Oscillatory dynamics |
| `schrodinger` | Schrödinger equation | ❌ | Quantum-inspired evolution |
| `spatially_varying_pde` | Spatially-varying PDE | ✅ | Enhanced physics modeling |
| `modernized_cnn` | Modernized CNN | ✅ | Enhanced spatial processing |
| `adaptive_time_stepping` | Adaptive time stepping | ✅ | Adaptive evolution control |

### Interference Types

| Interference Type | Description | Enhanced Only | Use Case |
|-------------------|-------------|---------------|----------|
| `standard` | Standard field interference | ❌ | Basic field interactions |
| `causal` | Causal interference | ❌ | Time-series applications |
| `multiscale` | Multi-scale interference | ✅ | Multi-resolution processing |
| `physics` | Physics-constrained interference | ✅ | Physics-informed interactions |

### Additional Enhanced Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_enhanced` | bool | False | Enable enhanced TFN features |
| `kernel_hidden_dim` | int | 64 | Hidden dimension for kernel predictors |
| `evolution_hidden_dim` | int | 32 | Hidden dimension for evolution predictors |
| `num_frequencies` | int | 8 | Number of frequencies for Fourier kernels |
| `min_dt` | float | 0.001 | Minimum time step for adaptive evolution |
| `max_dt` | float | 0.1 | Maximum time step for adaptive evolution |

## 🎯 Advanced Usage Examples

### 1. Physics-Informed TFN

```bash
python train.py \
    --model_name enhanced_tfn_classifier \
    --model.use_enhanced \
    --model.kernel_type data_dependent_rbf \
    --model.evolution_type spatially_varying_pde \
    --model.interference_type physics \
    --model.kernel_hidden_dim 128 \
    --model.evolution_hidden_dim 64 \
    --model.grid_size 200 \
    --model.time_steps 7 \
    --model.min_dt 0.0005 \
    --model.max_dt 0.05
```

### 2. Multi-Scale Processing

```bash
python train.py \
    --model_name enhanced_tfn_classifier \
    --model.use_enhanced \
    --model.kernel_type multi_frequency_fourier \
    --model.evolution_type modernized_cnn \
    --model.interference_type multiscale \
    --model.num_frequencies 16 \
    --model.grid_size 256 \
    --model.time_steps 5
```

### 3. Adaptive Evolution

```bash
python train.py \
    --model_name enhanced_tfn_classifier \
    --model.use_enhanced \
    --model.kernel_type film_learnable \
    --model.evolution_type adaptive_time_stepping \
    --model.interference_type standard \
    --model.min_dt 0.001 \
    --model.max_dt 0.1 \
    --model.time_steps 10
```

## ⚠️ Parameter Compatibility Notes

### Enhanced Features Requirements
- `data_dependent_rbf`, `data_dependent_compact`, `multi_frequency_fourier`, `film_learnable` require `use_enhanced=True`
- `spatially_varying_pde`, `modernized_cnn`, `adaptive_time_stepping` require `use_enhanced=True`
- `multiscale`, `physics` interference types require `use_enhanced=True`

### Memory Considerations
- Larger `grid_size` increases memory usage quadratically
- More `time_steps` increases computation linearly
- Enhanced features require additional memory for predictors

### Performance Tips
- Start with smaller models (`embed_dim=128`, `num_layers=2`) for initial testing
- Use `grid_size=50-100` for quick experiments
- Enable `track_flops` in hyperparameter search to monitor computational cost
- Use `patience=5-10` for early stopping to save computation time

## 🔧 Debugging and Monitoring

### Enable Detailed Logging
```bash
python train.py \
    --model_name tfn_classifier \
    --model.kernel_type rbf \
    --model.evolution_type cnn \
    --model.grid_size 50 \
    --model.time_steps 3 \
    --epochs 5 \
    --batch_size 8  # Small batch for debugging
```

### Monitor Parameter Validation
The training scripts now warn about unsupported parameters:
```bash
python train.py --model_name tfn_classifier --model.n_heads 8
# Output: ⚠️ Warning: n_heads parameter not supported by tfn_classifier
```

### Track Computational Cost
```bash
python hyperparameter_search.py \
    --models tfn_classifier \
    --param_sweep "embed_dim:128,256" \
    --track_flops \
    --epochs 10
``` 