# Data Loading Refactoring Summary

## Problem Solved

The data loading system was fragile due to inconsistent batch formats across different data loaders. The `_unpack_batch` method in `src/trainer.py` used a long if/elif chain to guess the format of data batches, making it brittle and prone to breaking when new data loaders were added.

## Solution Implemented

### 1. Standardized Batch Format

All data loaders now return a **single, consistent format**:

**For Regression/Copy Tasks:**
```python
{
    'inputs': torch.Tensor,      # Input data
    'targets': torch.Tensor      # Target data
}
```

**For Classification Tasks:**
```python
{
    'inputs': torch.Tensor,      # Input data
    'labels': torch.Tensor       # Class labels
}
```

**For NLP Tasks (with attention masks):**
```python
{
    'inputs': torch.Tensor,      # Input tokens
    'attention_mask': torch.Tensor,  # Attention mask
    'labels': torch.Tensor       # Labels/targets
}
```

### 2. Updated Data Loaders

The following data loaders were updated to use the standardized format:

- **Timeseries datasets** (`data/timeseries_loader.py`): `{'input': ..., 'target': ...}` → `{'inputs': ..., 'targets': ...}`
- **NLP datasets** (`data/nlp_loader.py`): `{'input_ids': ..., 'labels': ...}` → `{'inputs': ..., 'labels': ...}`
- **Language modeling datasets** (`data/wikitext_loader.py`, `data/pg19_loader.py`): `{'input_ids': ..., 'labels': ...}` → `{'inputs': ..., 'labels': ...}`
- **GLUE datasets** (`data/glue_loader.py`): `{'input_ids': ..., 'labels': ...}` → `{'inputs': ..., 'labels': ...}`
- **ArXiv datasets** (`data/arxiv_loader.py`): `{'input_ids': ..., 'labels': ...}` → `{'inputs': ..., 'labels': ...}`
- **Stock market datasets** (`data/stock_loader.py`): `{'input': ..., 'target': ...}` → `{'inputs': ..., 'targets': ...}`
- **Jena climate datasets** (`data/jena_loader.py`): `{'input': ..., 'target': ...}` → `{'inputs': ..., 'targets': ...}`
- **Synthetic datasets** (`data_pipeline.py`): `{'source': ..., 'target': ...}` → `{'inputs': ..., 'targets': ...}`

### 3. Simplified Trainer

The `_unpack_batch` method in `src/trainer.py` was simplified from a complex if/elif chain to a clean, robust implementation:

**Before (fragile):**
```python
def _unpack_batch(self, batch: Dict[str, Any]):
    # Complex if/elif chain with multiple legacy formats
    if "inputs" in batch and "targets" in batch:
        # Standard format
    elif "inputs" in batch and "labels" in batch:
        # Classification format
    elif "input_ids" in batch and "labels" in batch:
        # Language modeling format
    elif "input" in batch and "target" in batch:
        # Legacy format
    elif "source" in batch and "target" in batch:
        # Legacy format
    else:
        # Error handling
```

**After (robust):**
```python
def _unpack_batch(self, batch: Dict[str, Any]):
    # Standardized format only
    if "inputs" in batch and "targets" in batch:
        # Regression/copy tasks
        return model_input, targets
    elif "inputs" in batch and "labels" in batch:
        # Classification tasks
        return model_input, labels
    else:
        # Clear error message with debugging info
        raise ValueError(f"Invalid batch format: keys {list(batch.keys())}")
```

### 4. Updated Collate Functions

The collate functions were updated to work with the standardized format:

- **`pad_collate`**: Now expects and returns standardized format
- **`language_modeling_collate`**: Updated to use `'inputs'` instead of `'input_ids'`

## Benefits

### 1. **Robustness**
- No more fragile if/elif chains
- Clear error messages when batch format is incorrect
- Consistent behavior across all data loaders

### 2. **Maintainability**
- Single source of truth for batch format
- Easy to add new data loaders
- Clear documentation of expected format

### 3. **Debugging**
- Clear error messages with available keys
- Consistent format makes debugging easier
- No more guessing about batch structure

### 4. **Extensibility**
- New data loaders can easily adopt the standard format
- No need to update trainer for new formats
- Backward compatibility maintained through clear migration path

## Testing

Comprehensive tests were created (`test_standardized_data_format.py`) that verify:

✅ **All data loaders return standardized format**
- Synthetic datasets (regression/classification)
- Timeseries datasets
- NLP datasets
- Language modeling datasets

✅ **Trainer correctly unpacks standardized batches**
- Regression batches with `{'inputs': ..., 'targets': ...}`
- Classification batches with `{'inputs': ..., 'labels': ...}`
- NLP batches with attention masks

✅ **Invalid formats are properly rejected**
- Legacy formats are caught with clear error messages
- Debugging information is provided

## Migration Guide

For developers adding new data loaders:

1. **Use standardized keys**: `'inputs'`, `'targets'`, `'labels'`
2. **Include attention masks** for NLP tasks: `'attention_mask'`
3. **Test with the standardized format test suite**
4. **No need to update trainer** - it already handles the standard format

## Files Modified

### Core Files
- `src/trainer.py` - Simplified `_unpack_batch` method
- `data_pipeline.py` - Updated `pad_collate` and synthetic datasets

### Data Loaders
- `data/timeseries_loader.py` - Updated ETTDataset
- `data/nlp_loader.py` - Updated NLPDataset
- `data/wikitext_loader.py` - Updated WikiTextDataset
- `data/pg19_loader.py` - Updated PG19Dataset
- `data/glue_loader.py` - Updated GLUEDataset
- `data/arxiv_loader.py` - Updated ArxivDataset
- `data/stock_loader.py` - Updated StockMarketDataset
- `data/jena_loader.py` - Updated JenaClimateDataset

### Test Files
- `test_standardized_data_format.py` - Comprehensive test suite

## Conclusion

The data loading system is now **robust, maintainable, and extensible**. The standardized format eliminates the fragility of the previous implementation while providing clear error messages and debugging information. All existing functionality is preserved, and new data loaders can easily adopt the standard format. 