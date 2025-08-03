# Development Documentation

This document contains key development information, architectural decisions, and implementation details for the Token Field Network project.

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Data Pipeline](#data-pipeline)
- [Model Registry](#model-registry)
- [Training System](#training-system)
- [Hyperparameter Search](#hyperparameter-search)
- [Wandb Integration](#wandb-integration)
- [Notebook Usage](#notebook-usage)

## 🏗️ Architecture Overview

### Core Components

The TFN architecture consists of three main phases:

1. **Field Projection**: Tokens emit continuous fields across spatial domain
2. **Field Evolution**: Fields evolve over time using physics-inspired dynamics
3. **Field Sampling**: Evolved fields are sampled back to update tokens

### Key Design Principles

- **Mathematical Rigor**: Fully differentiable field-based attention mechanism
- **Physics-Inspired**: Leverages continuous field dynamics
- **Modular Design**: Independent, testable components
- **Numerical Stability**: Careful handling of tensor operations

## 📊 Data Pipeline

### Registry Pattern Implementation

The data pipeline uses a centralized registry system that follows the same pattern as the model registry:

```python
# Data Loader Registry
DATASET_REGISTRY = {
    'synthetic': {
        'class': SyntheticCopyDataset,
        'task_type': 'copy',
        'required_params': ['dataset_size', 'seq_len', 'vocab_size'],
        'optional_params': ['pad_idx', 'task', 'num_classes'],
        'defaults': {
            'dataset_size': 1000,
            'seq_len': 50,
            'vocab_size': 20,
            'pad_idx': 0,
            'task': 'copy',
            'num_classes': 2
        },
        'factory_method': None,
        'split_method': None,
        'description': 'Synthetic dataset for testing copy and classification tasks'
    }
}
```

### Benefits

- **Extensibility**: Adding new datasets requires only registration
- **Centralized Configuration**: All dataset parameters in one place
- **Type Safety**: Required parameters are validated
- **Default Values**: Sensible defaults for optional parameters

## 🤖 Model Registry

### Unified TFN Architecture

The codebase provides a unified TFN model that handles both classification and regression tasks:

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

### Available Models

#### Core TFN Models
- `tfn_classifier`: Text/sequence classification
- `tfn_regressor`: Time series forecasting and regression
- `tfn_language_model`: Language modeling and text generation
- `tfn_vision`: 2D image classification

#### Enhanced TFN Models
- `enhanced_tfn_classifier`: Advanced classification with field interference
- `enhanced_tfn_regressor`: Advanced regression with unified dynamics
- `enhanced_tfn_language_model`: Advanced language modeling

#### Baseline Models
- `transformer_classifier/regressor`: Standard Transformer baselines
- `performer_classifier/regressor`: Linear attention baselines

## 🎯 Training System

### Strategy Pattern

The training system uses a strategy pattern for task-specific logic:

```python
class TaskStrategy(ABC):
    @abstractmethod
    def get_criterion(self) -> nn.Module:
        pass
    
    @abstractmethod
    def process_forward_pass(self, model, x, y) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
    
    @abstractmethod
    def calculate_metrics(self, logits, targets, scaler=None) -> Dict[str, float]:
        pass
```

### Available Strategies

- **ClassificationStrategy**: For classification and NER tasks
- **RegressionStrategy**: For regression and time-series tasks
- **LanguageModelingStrategy**: For language modeling tasks

### Callback System

The Trainer supports epoch-end callbacks for custom behavior:

```python
def on_epoch_end(epoch: int, metrics: Dict[str, Any], trainer_instance) -> bool:
    # Custom epoch-end logic
    return True  # Continue training, False to stop early

trainer.fit(on_epoch_end=on_epoch_end)
```

## 🔍 Hyperparameter Search

### DRY Principle Implementation

The hyperparameter search system eliminates code duplication by using the Trainer's `fit()` method with callbacks:

```python
# Define epoch-end callback for hyperparameter search
def on_epoch_end(epoch: int, metrics: Dict[str, Any], trainer_instance) -> bool:
    # Task-specific logging
    # Trial logging
    # Early stopping check
    return True  # Continue training

# Use the Trainer's fit method with our callback
trainer.fit(on_epoch_end=on_epoch_end)
```

### Key Features

- **No Code Duplication**: Uses Trainer's fit() method
- **Task-Aware Logging**: Different metrics for regression vs classification
- **Early Stopping**: Configurable patience and minimum epochs
- **Scaler Integration**: Proper metric denormalization for regression
- **FLOPS Tracking**: Performance monitoring

### Metric Handling

For regression tasks, the system correctly handles:
- **Scaler Extraction**: Gets scaler from dataset for denormalization
- **None Value Handling**: Properly formats None accuracy values
- **Task-Specific Logging**: Shows MSE/MAE for regression, loss/accuracy for classification

## 📈 Wandb Integration

### Configuration

Wandb integration is configured through the config file:

```yaml
wandb:
  use_wandb: true
  project_name: "tfn-time-series"
  experiment_name: "ett-forecasting-v1"
```

### Features

- **Automatic Tracking**: Training/validation metrics, model info, system info
- **Configuration Options**: Flexible project and experiment naming
- **Integration Points**: Works with CLI, notebooks, and hyperparameter search

## 📓 Notebook Usage

### Notebook-Friendly Functions

The project provides notebook-friendly functions for easy experimentation:

```python
from train import run_training
from hyperparameter_search import run_search

# Training
history = run_training(config)

# Hyperparameter search
results = run_search(search_config)
```

### Example Usage

```python
# Load and modify config
with open('configs/ett.yaml', 'r') as f:
    config = yaml.safe_load(f)

config['model']['embed_dim'] = 512
config['training']['epochs'] = 100

# Run training directly
history = run_training(config)
```

## 🧪 Testing

### Test Files

The project includes comprehensive test files:

- `test_ett_training.py`: ETT training example
- `test_wandb_simple.py`: Wandb integration testing
- `test_wandb_integration.py`: Comprehensive wandb testing
- `test_data_registry.py`: Data loading testing
- `test_notebook_functions.py`: Notebook functionality testing
- `test_standardized_data_format.py`: Data format testing
- `test_hyperparameter_search.py`: Hyperparameter search testing
- `test_modular_implementation.py`: Modular implementation testing

### Running Tests

```bash
# Run specific test
python test_ett_training.py

# Run all tests (if pytest is available)
pytest test_*.py
```

## 🔧 Development Guidelines

### Code Organization

- **Core Components**: `core/` - Field projection, evolution, sampling
- **Models**: `model/` - TFN implementations and registry
- **Data**: `data/` - Dataset loaders and registry
- **Training**: `src/` - Trainer, strategies, metrics
- **Configuration**: `configs/` - YAML configuration files

### Best Practices

1. **DRY Principle**: Avoid code duplication
2. **Strategy Pattern**: Use for task-specific logic
3. **Registry Pattern**: Centralized configuration
4. **Type Hints**: Include for all functions
5. **Documentation**: Docstrings for all classes and methods
6. **Testing**: Unit tests for all components

### Adding New Features

1. **Models**: Register in `model/registry.py`
2. **Datasets**: Register in `data/registry.py`
3. **Strategies**: Implement `TaskStrategy` interface
4. **Configurations**: Add YAML files to `configs/`
5. **Tests**: Create corresponding test files 