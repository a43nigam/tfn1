# PARN: Physics-Aware Reversible Normalization

## Overview

PARN (Physics-Aware Reversible Normalization) is a novel normalization technique designed specifically for physics-inspired deep learning models like Token Field Networks (TFN). Unlike standard normalization approaches, PARN decouples location and scale normalization and can feed preserved statistics back to the model as conditioning features.

## Key Innovations

### 1. Decoupled Normalization
PARN explicitly decouples the normalization of location (mean) and scale (standard deviation), allowing fine-grained control over which physical properties are removed from the signal:

- **Location Mode**: Removes only the mean, preserving signal volatility
- **Scale Mode**: Removes only the standard deviation, preserving signal magnitude  
- **Full Mode**: Removes both location and scale (equivalent to standard normalization)

### 2. Statistic Re-injection as Conditioning Features
The core innovation of PARN is its ability to feed preserved statistics back to the model as additional features. This tells the model: "Here is a clean, stable signal to process, and by the way, here is the physical context I removed for you."

### 3. Reversible Framework for Inductive Models
PARN creates a fully reversible pipeline specifically designed for architectures that rely on inductive biases (like physics). It's not just another normalization technique; it's a strategy to make normalization synergistic with the model's design.

## Implementation Details

### Core Components

#### PARN Module (`model/wrappers.py`)
```python
class PARN(nn.Module):
    def __init__(self, num_features: int, mode: str = 'location', eps: float = 1e-5):
        # mode: 'location', 'scale', or 'full'
```

**Key Methods:**
- `forward(x, operation)`: Handles both normalization ('norm') and denormalization ('denorm')
- `_normalize(x)`: Applies mode-specific normalization and returns preserved statistics
- `_denormalize(x)`: Reverses the normalization process
- `_get_statistics(x)`: Computes mean and standard deviation

#### PARNModel Wrapper
```python
class PARNModel(nn.Module):
    def __init__(self, base_model, num_features, mode='location'):
```

**Key Features:**
- Automatically handles normalization/denormalization
- Always injects preserved statistics as additional features
- Slices output to extract core predictions before denormalization
- Maintains compatibility with existing model interfaces

### Statistics Injection

When `inject_stats=True`, PARN concatenates preserved statistics with the normalized input:

```python
# For 'location' mode: preserves scale statistics
x_normalized = x - mean
stats = {'scale': stdev}

# For 'scale' mode: preserves location statistics  
x_normalized = x / stdev
stats = {'location': mean}

# For 'full' mode: preserves both
x_normalized = (x - mean) / stdev
stats = {'location': mean, 'scale': stdev}
```

The statistics are then concatenated as additional features:
```python
x_with_stats = torch.cat([x_normalized, stats_tensor], dim=-1)
```

### Output Processing

The PARNModel wrapper automatically handles output processing:

1. **Input Augmentation**: Statistics are concatenated with normalized input
2. **Model Processing**: Base model processes the augmented input
3. **Output Slicing**: Only the core features (excluding statistics) are extracted
4. **Denormalization**: The core features are denormalized back to original scale

```python
# The wrapper automatically slices the output to get core predictions
output_normalized = output_augmented[:, :, :self.parn.num_features]
output_denormalized = self.parn(output_normalized, operation='denorm')
```

## Usage Examples

### Basic Usage

```python
from model.wrappers import PARN, create_parn_wrapper

# Create PARN module
parn = PARN(num_features=7, mode='location')

# Normalize input
x_normalized, stats = parn(x, 'norm')

# Denormalize output  
x_denormalized, _ = parn(x_normalized, 'denorm')
```

### Model Wrapping

```python
from model.wrappers import create_parn_wrapper

# Wrap a base model with PARN
base_model = TFNRegressor(...)
parn_model = create_parn_wrapper(
    base_model, 
    num_features=7, 
    mode='location'
)

# Use normally - PARN handles normalization automatically
output = parn_model(input_data)
```

### Configuration File

```yaml
model:
  input_dim: 7
  embed_dim: 64
  output_dim: 7
  kernel_type: "rbf"
  evolution_type: "pde"
  
  # PARN configuration
  normalization:
    type: "parn"
    mode: "location"  # or "scale", "full"
    num_features: 7
```

## Integration with Training Pipeline

PARN is seamlessly integrated into the existing training pipeline through the `run_training` function in `train.py`. The system automatically:

1. **Detects PARN Strategy**: Checks for `normalization_strategy: "parn"` in data configuration
2. **Dynamic Input Dimension Adjustment**: Automatically calculates new input dimensions based on PARN mode:
   - `location` or `scale` mode: `input_dim * 2`
   - `full` mode: `input_dim * 3`
3. **Model Rebuilding**: Re-builds the base model with adjusted input dimensions
4. **PARN Wrapping**: Wraps the rebuilt model with PARNModel
5. **Informative Logging**: Provides clear feedback about the integration process

### Configuration Example

```yaml
data:
  normalization_strategy: "parn"  # Enable PARN
  parn_mode: "location"          # Choose normalization mode
```

### Integration Logic

The integration follows this flow in `train.py`:

```python
normalization_strategy = config.get('data', {}).get('normalization_strategy')

if normalization_strategy == 'parn':
    parn_mode = config.get('data', {}).get('parn_mode', 'location')
    original_input_dim = model_cfg['input_dim']
    
    # Adjust input dimension based on mode
    if parn_mode == 'location' or parn_mode == 'scale':
        model_cfg['input_dim'] = original_input_dim * 2
    elif parn_mode == 'full':
        model_cfg['input_dim'] = original_input_dim * 3
    
    # Re-build model with adjusted dimensions
    model = build_model(model_name, model_cfg, data_cfg).to(device)
    
    # Wrap with PARN
    model_to_train = PARNModel(base_model=model, num_features=original_input_dim, mode=parn_mode)
```

## Comparison with RevIN

| Feature | RevIN | PARN |
|---------|-------|------|
| Normalization | Full (location + scale) | Decoupled (location/scale/full) |
| Statistics | Not preserved | Preserved and injectable |
| Physics Awareness | No | Yes (statistics as context) |
| Reversibility | Yes | Yes |
| Inductive Bias Support | Limited | Enhanced |

## Mathematical Foundation

### Normalization Modes

**Location Mode:**
```
x_norm = (x - μ) * γ + β
stats = {scale: σ}
```

**Scale Mode:**
```
x_norm = (x / σ) * γ + β  
stats = {location: μ}
```

**Full Mode:**
```
x_norm = ((x - μ) / σ) * γ + β
stats = {location: μ, scale: σ}
```

Where:
- `μ`: Mean across time dimension
- `σ`: Standard deviation across time dimension  
- `γ, β`: Learnable affine parameters

### Reversibility

PARN maintains perfect reversibility (within numerical precision):
```
x_denorm = ((x_norm - β) / γ) * σ + μ  # Location mode
x_denorm = ((x_norm - β) / γ) * σ       # Scale mode
x_denorm = ((x_norm - β) / γ) * σ + μ   # Full mode
```

## Testing

Comprehensive tests are provided in `test/test_parn.py`:

- PARN initialization and parameter validation
- Normalization modes (location, scale, full)
- Reversibility verification
- Statistics injection functionality
- Model wrapper integration
- Shape handling for different tensor dimensions

Run tests with:
```bash
PYTHONPATH=/path/to/TokenFieldNetwork python test/test_parn.py
```

## Benefits for TFN

1. **Physics-Informed Conditioning**: Preserved statistics provide physical context to the field evolution process
2. **Stable Training**: Decoupled normalization allows fine-tuning of signal properties
3. **Interpretability**: Statistics injection makes the model's use of physical context explicit
4. **Flexibility**: Different modes can be tested to find optimal normalization strategy

## Future Directions

1. **Adaptive Mode Selection**: Automatically choose normalization mode based on data characteristics
2. **Multi-Scale Statistics**: Preserve statistics at multiple temporal scales
3. **Physics-Guided Injection**: Use domain knowledge to determine which statistics to inject
4. **Cross-Domain Validation**: Test PARN on other physics-inspired architectures

## References

This implementation is based on the novel PARN approach described in the project documentation. The key innovation is the combination of decoupled normalization with statistics re-injection as conditioning features, specifically designed for physics-aware models like TFN. 