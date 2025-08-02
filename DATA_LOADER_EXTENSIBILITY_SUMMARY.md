# Data Loader Extensibility Summary

## Problem Solved

The `dataloader_factory` function in `data_pipeline.py` used a long `if/elif` chain to select dataset loaders based on `dataset_name`. Adding new datasets required modifying this central function, which violated the open/closed principle and made the system non-extensible.

## Solution Implemented

### 1. **Registry Pattern Implementation**

Created a centralized registry system in `data/registry.py` that follows the same pattern as the model registry:

```python
# Data Loader Registry with all parameters and task types
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
    },
    # ... other datasets
}
```

### 2. **Unified Dataset Creation**

The registry provides a `create_dataset()` function that handles both:
- **Datasets with `get_splits()` method** (like ETT, Jena, Stock)
- **Datasets with direct instantiation** (like GLUE, ArXiv, PG19)

```python
def create_dataset(dataset_name: str, config: Dict[str, Any], split: str = 'train') -> Dataset:
    """Create a dataset instance using the registry."""
    dataset_config = get_dataset_config(dataset_name)
    dataset_class = dataset_config['class']
    
    # Handle different creation patterns
    if dataset_config['split_method'] == 'get_splits':
        # For datasets with get_splits method
        train_ds, val_ds, test_ds = dataset_class.get_splits(**call_params)
        # Return appropriate split
    else:
        # For datasets with direct instantiation
        return dataset_class(**valid_params)
```

### 3. **Simplified Factory Function**

The `dataloader_factory` function was dramatically simplified:

```python
def dataloader_factory(config: Dict[str, Any], split: str = 'train') -> Dataset:
    """Factory to select the correct dataset based on config['data']['dataset_name']."""
    from data.registry import create_dataset
    
    data_cfg = config["data"]
    dataset_name = data_cfg.get("dataset_name", "synthetic")
    
    try:
        return create_dataset(dataset_name, config, split)
    except Exception as e:
        raise ValueError(f"Failed to create dataset '{dataset_name}': {e}")
```

## Benefits Achieved

### ✅ **Extensibility**
Adding new datasets now requires only:
1. Creating the dataset class
2. Registering it in the registry
3. No modification of central factory functions

```python
# Example: Adding a new dataset
from data.registry import register_dataset

register_dataset(
    name='my_custom_dataset',
    dataset_class=MyCustomDataset,
    task_type='regression',
    required_params=['param1', 'param2'],
    optional_params=['param3'],
    defaults={'param1': 'default_value'},
    description='My custom dataset description'
)
```

### ✅ **Centralized Configuration**
All dataset metadata is centralized in one place:
- Required/optional parameters
- Default values
- Task compatibility
- Descriptions

### ✅ **Type Safety and Validation**
The registry provides validation functions:
```python
# Validate dataset-task compatibility
validate_dataset_task_compatibility('ett', 'regression')  # True
validate_dataset_task_compatibility('ett', 'classification')  # False

# Get parameter information
get_required_params('ett')  # ['csv_path', 'input_len', 'output_len']
get_optional_params('ett')  # ['normalization_strategy', 'instance_normalize']
```

### ✅ **Discovery and Documentation**
Easy to discover available datasets and their capabilities:
```python
# List all available datasets
list_available_datasets()  # ['synthetic', 'ett', 'jena', 'stock', ...]

# List datasets by task
list_datasets_by_task('regression')  # ['ett', 'jena', 'stock']

# Get dataset description
get_dataset_description('ett')  # 'Electricity Transformer Temperature dataset...'
```

### ✅ **Backward Compatibility**
All existing configurations continue to work without modification:
```yaml
# configs/ett.yaml - unchanged
data:
  dataset_name: ett
  csv_path: data/ETTh1.csv
  input_len: 96
  output_len: 24
```

## Files Modified

### Core Registry
- **`data/registry.py`** - New registry system with all dataset configurations

### Updated Factory
- **`data_pipeline.py`** - Simplified `dataloader_factory` to use registry

### Testing
- **`test_data_registry.py`** - Comprehensive test suite for registry functionality

## Registry Functions

### Core Functions
- `get_dataset_config(dataset_name)` - Get full dataset configuration
- `create_dataset(dataset_name, config, split)` - Create dataset instance
- `register_dataset(...)` - Register new dataset

### Discovery Functions
- `list_available_datasets()` - List all registered datasets
- `list_datasets_by_task(task)` - List datasets for specific task
- `get_task_compatibility(task)` - Get task-dataset compatibility

### Validation Functions
- `validate_dataset_task_compatibility(dataset, task)` - Validate compatibility
- `get_required_params(dataset_name)` - Get required parameters
- `get_optional_params(dataset_name)` - Get optional parameters
- `get_dataset_defaults(dataset_name)` - Get default values

### Documentation Functions
- `get_dataset_description(dataset_name)` - Get dataset description

## Adding New Datasets

### Step 1: Create Dataset Class
```python
from torch.utils.data import Dataset

class MyCustomDataset(Dataset):
    def __init__(self, param1, param2, **kwargs):
        self.param1 = param1
        self.param2 = param2
        # ... implementation
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {'inputs': ..., 'targets': ...}
```

### Step 2: Register in Registry
```python
from data.registry import register_dataset

register_dataset(
    name='my_custom_dataset',
    dataset_class=MyCustomDataset,
    task_type='regression',
    required_params=['param1', 'param2'],
    optional_params=['param3'],
    defaults={'param1': 'default_value'},
    description='My custom dataset for regression tasks'
)
```

### Step 3: Use in Configuration
```yaml
# configs/my_custom.yaml
data:
  dataset_name: my_custom_dataset
  param1: custom_value
  param2: another_value
  param3: optional_value
```

## Testing Results

✅ **Registry Functions** - All basic registry operations work correctly
✅ **Dataset Creation** - Both synthetic and real datasets create successfully
✅ **Custom Registration** - New datasets can be registered dynamically
✅ **Factory Integration** - `dataloader_factory` works with registry
✅ **Error Handling** - Proper error messages for unknown datasets

## Comparison: Before vs After

### Before (Non-Extensible)
```python
def dataloader_factory(config, split):
    dataset_name = config["data"]["dataset_name"]
    
    if dataset_name == "ett":
        # 20+ lines of ETT-specific logic
    elif dataset_name == "jena":
        # 20+ lines of Jena-specific logic
    elif dataset_name == "stock":
        # 20+ lines of Stock-specific logic
    # ... more elif blocks for each dataset
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
```

### After (Extensible)
```python
def dataloader_factory(config, split):
    from data.registry import create_dataset
    
    dataset_name = config["data"]["dataset_name"]
    return create_dataset(dataset_name, config, split)
```

## Benefits Summary

1. **🔧 Extensibility** - Add new datasets without modifying core code
2. **📋 Centralized Configuration** - All dataset metadata in one place
3. **✅ Validation** - Built-in parameter and compatibility validation
4. **🔍 Discovery** - Easy to find available datasets and capabilities
5. **📚 Documentation** - Built-in descriptions and parameter documentation
6. **🔄 Backward Compatibility** - Existing configurations work unchanged
7. **🧪 Testability** - Registry pattern enables comprehensive testing

## Future Extensions

The registry pattern enables several future improvements:

1. **Dynamic Dataset Loading** - Load datasets from external sources
2. **Dataset Versioning** - Support multiple versions of the same dataset
3. **Parameter Validation** - Validate parameter types and ranges
4. **Auto-Documentation** - Generate documentation from registry metadata
5. **Dataset Metrics** - Track dataset usage and performance metrics

This refactoring makes the data loading system much more maintainable and extensible, following software engineering best practices and enabling rapid development of new datasets. 