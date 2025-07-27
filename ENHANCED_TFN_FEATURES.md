# Enhanced TFN Features Implementation

This document outlines the enhanced features implemented for the Token Field Network (TFN) architecture, addressing the suggestions for improved kernel parameters, Fourier features, and evolution mechanisms.

## 1. Data-Dependent Kernel Parameters

### ✅ **IMPLEMENTED**

**What was suggested**: Instead of a single learnable sigma for all tokens, use a small neural network to predict unique sigma for each token based on its embedding.

**What we implemented**:

#### `DataDependentRBFKernel`
- Uses a small neural network to predict unique sigma values for each token from embeddings
- Allows some tokens to have wide, global influence while others have sharp, localized influence
- Mathematical formulation: `K(z, μ, σ(E)) = exp(-||z - μ||²/(2σ(E)²))`

#### `DataDependentCompactKernel`
- Predicts unique radius values for each token from embeddings
- Important tokens can have larger influence radius while less important ones have smaller radius
- Mathematical formulation: `K(z, μ, r(E)) = max(0, 1 - ||z - μ||/r(E))`

**Usage**:
```bash
python train.py --config configs/enhanced_tfn_demo.yaml --model.kernel_type data_dependent_rbf
```

## 2. Richer Fourier Features

### ✅ **IMPLEMENTED**

**What was suggested**: Map distance to a higher-dimensional vector of sine and cosine functions with multiple frequencies, inspired by Fourier Feature Networks.

**What we implemented**:

#### `MultiFrequencyFourierKernel`
- Maps distance to multiple sine and cosine functions with different frequencies
- Provides more capacity to represent complex, multi-frequency patterns
- Mathematical formulation: `K(z, μ, ω) = [cos(ω₁||z - μ||), sin(ω₁||z - μ||), ..., cos(ωₙ||z - μ||), sin(ωₙ||z - μ||)]`

**Usage**:
```bash
python train.py --config configs/enhanced_tfn_demo.yaml --model.kernel_type multi_frequency_fourier --model.num_frequencies 8
```

## 3. Improved Learnable Kernel

### ✅ **IMPLEMENTED**

**What was suggested**: Use FiLM (Feature-wise Linear Modulation) layers where the token's embedding generates scale and shift applied to activations inside the MLP.

**What we implemented**:

#### `FiLMLearnableKernel`
- Uses FiLM conditioning where token embeddings generate scale and shift parameters
- Applies these parameters to activations inside the kernel network
- Makes the kernel's shape truly data-dependent
- Mathematical formulation: `K(z, μ, θ(E)) = φ(||z - μ||, θ(E))` where θ(E) are FiLM parameters

**Usage**:
```bash
python train.py --config configs/enhanced_tfn_demo.yaml --model.kernel_type film_learnable
```

## 4. Spatially-Varying PDE Coefficients

### ✅ **IMPLEMENTED**

**What was suggested**: Make PDE coefficients spatially varying using a small CNN that takes the current field as input and outputs a map of diffusion coefficients.

**What we implemented**:

#### `SpatiallyVaryingPDEFieldEvolver`
- Uses a CNN to predict spatially-varying diffusion coefficients and wave speeds
- Takes current field state as input and outputs coefficient maps
- Enables non-linear PDE evolution
- Supports both diffusion and wave equations

**Usage**:
```bash
python train.py --config configs/enhanced_tfn_demo.yaml --model.evolution_type spatially_varying_pde
```

## 5. Modernized CNN Evolver

### ✅ **IMPLEMENTED**

**What was suggested**: Upgrade CNN evolution with modern components like depthwise separable convolutions, larger kernel sizes, and gated linear units (GLUs).

**What we implemented**:

#### `ModernizedCNNFieldEvolver`
- Incorporates depthwise separable convolutions for efficiency
- Uses multiple kernel sizes (3, 5, 7) for multi-scale processing
- Implements gated linear units (GLUs) for better information flow
- Includes residual connections and layer normalization

**Usage**:
```bash
python train.py --config configs/enhanced_tfn_demo.yaml --model.evolution_type modernized_cnn
```

## 6. Adaptive Time-Stepping

### ✅ **IMPLEMENTED**

**What was suggested**: Use a small network to predict optimal dt at each evolution step based on the field's current rate of change.

**What we implemented**:

#### `AdaptiveTimeSteppingEvolver`
- Uses a neural network to predict optimal time steps based on field characteristics
- Uses smaller steps during periods of high activity and larger ones when stable
- Monitors field rate of change and field statistics to determine optimal dt
- Supports both diffusion and wave evolution types

**Usage**:
```bash
python train.py --config configs/enhanced_tfn_demo.yaml --model.evolution_type adaptive_time_stepping
```

## CLI Flags and Configuration

### New CLI Flags Added:

```bash
# Kernel options
--model.kernel_type                    # Kernel type selection
--model.use_data_dependent_kernels     # Enable data-dependent features
--model.kernel_hidden_dim              # Hidden dim for kernel predictors
--model.num_frequencies                # Number of frequencies for Fourier

# Evolution options  
--model.evolution_type                 # Evolution type selection
--model.use_spatially_varying_pde      # Enable spatially-varying PDE
--model.use_adaptive_time_stepping     # Enable adaptive time stepping
--model.use_modernized_cnn             # Use modernized CNN
--model.evolution_hidden_dim           # Hidden dim for evolution predictors
--model.min_dt                         # Minimum time step
--model.max_dt                         # Maximum time step
```

### Available Options:

**Kernel Types**:
- `rbf` - Standard RBF kernel
- `compact` - Compact support kernel
- `fourier` - Simple Fourier kernel
- `learnable` - Basic learnable kernel
- `data_dependent_rbf` - Data-dependent RBF kernel
- `data_dependent_compact` - Data-dependent compact kernel
- `multi_frequency_fourier` - Multi-frequency Fourier kernel
- `film_learnable` - FiLM-conditioned learnable kernel

**Evolution Types**:
- `cnn` - Standard CNN evolution
- `pde` - Basic PDE evolution
- `diffusion` - Diffusion equation
- `wave` - Wave equation
- `schrodinger` - Schrödinger equation
- `spatially_varying_pde` - Spatially-varying PDE coefficients
- `modernized_cnn` - Modernized CNN with depthwise separable convolutions
- `adaptive_time_stepping` - Adaptive time stepping

## Example Configurations

### Basic Enhanced TFN:
```yaml
model:
  kernel_type: "data_dependent_rbf"
  evolution_type: "spatially_varying_pde"
  use_data_dependent_kernels: true
  use_spatially_varying_pde: true
```

### Multi-Frequency Fourier:
```yaml
model:
  kernel_type: "multi_frequency_fourier"
  num_frequencies: 8
  evolution_type: "modernized_cnn"
```

### Adaptive Time Stepping:
```yaml
model:
  kernel_type: "film_learnable"
  evolution_type: "adaptive_time_stepping"
  min_dt: 0.001
  max_dt: 0.1
```

## Implementation Details

### Modular Design
All new features are implemented as separate classes that extend the base `KernelBasis` and `FieldEvolver` classes, ensuring:
- Clean separation of concerns
- Easy toggling through CLI flags
- Backward compatibility with existing code
- Unit testability in isolation

### Numerical Stability
- All learnable parameters are properly clamped to valid ranges
- Sigmoid activations ensure positive coefficients where needed
- Proper handling of edge cases in spatial derivatives
- Stability monitoring in adaptive time stepping

### Performance Considerations
- Efficient tensor operations using broadcasting
- Minimal computational overhead for new features
- Optional features that can be disabled if not needed
- Memory-efficient implementations

## Testing and Validation

To test the enhanced features:

```bash
# Test data-dependent kernels
python train.py --config configs/enhanced_tfn_demo.yaml --model.kernel_type data_dependent_rbf

# Test spatially-varying PDE
python train.py --config configs/enhanced_tfn_demo.yaml --model.evolution_type spatially_varying_pde

# Test multi-frequency Fourier
python train.py --config configs/enhanced_tfn_demo.yaml --model.kernel_type multi_frequency_fourier

# Test adaptive time stepping
python train.py --config configs/enhanced_tfn_demo.yaml --model.evolution_type adaptive_time_stepping
```

## Future Extensions

The modular design allows for easy extension with:
- Additional kernel types
- New evolution mechanisms
- Hybrid approaches combining multiple features
- Task-specific optimizations
- Advanced physics constraints

All features are designed to be composable and can be combined in various ways to suit different applications and datasets. 