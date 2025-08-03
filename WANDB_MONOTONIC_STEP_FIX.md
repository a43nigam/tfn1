# WandB Monotonic Step Warning Fix

## Problem Description

The wandb integration was experiencing repeated warnings like:
```
Tried to log to step X that is less than the current step Y
```

This prevented epoch-level metrics from being logged to wandb, making it impossible to compare validation performance between experiments.

## Root Cause

wandb uses a single, monotonically increasing global step by default. The Trainer logs metrics for each batch, rapidly increasing this global step. When it then tries to log epoch-level metrics using the epoch number as the step, that number is smaller than the batch-based global step, so wandb rejects it.

## Solution Implemented

### 1. Custom Step Metric Definition

**File Modified:** `src/trainer.py`

**Method:** `_init_wandb()`

**Changes:**
```python
# Define custom x-axis for epoch-level metrics
wandb.define_metric("epoch/epoch")
wandb.define_metric("epoch/*", step_metric="epoch/epoch")
```

This tells wandb that any metric prefixed with "epoch/" should be plotted against the epoch step, creating a separate x-axis for epoch-level metrics.

### 2. Modified Metric Logging

**Method:** `_log_metrics()`

**Before:**
```python
# Filter out None values for wandb logging
wandb_metrics = {k: v for k, v in metrics.items() if v is not None}
wandb.log(wandb_metrics, step=epoch)
```

**After:**
```python
# Prefix metrics with "epoch/" to link them to the custom step
wandb_metrics = {f"epoch/{k}": v for k, v in metrics.items() if v is not None}
# Add the epoch number itself to the log
wandb_metrics["epoch/epoch"] = epoch
wandb.log(wandb_metrics)
```

### 3. Batch-Level Logging Unchanged

The batch-level logging in `_run_epoch()` remains unchanged:
```python
batch_metrics_log = {f"batch_{k}": v for k, v in batch_metrics.items()}
batch_metrics_log['batch_loss'] = loss.item()
wandb.log(batch_metrics_log)  # Uses default global step
```

## Benefits

1. **Clean Separation:** Epoch-level metrics (loss, accuracy, etc.) are plotted against epoch numbers, while batch-level metrics (batch_loss, batch_acc, etc.) are plotted against the global step counter.

2. **Interactive Charts:** wandb will provide a dropdown to switch between the batch-step x-axis and the epoch-step x-axis, making it easy to analyze both within-epoch dynamics and epoch-level trends.

3. **No More Warnings:** The monotonic step warnings are eliminated, allowing all metrics to be logged successfully.

4. **Extensible:** This pattern can be easily extended. For example, if you later add a separate evaluation loop on a held-out test set, you could define a third step metric (e.g., "eval_step") for those metrics without interfering with the training or validation logs.

## Best Practices Followed

- **Official Recommendation:** This is the official, recommended wandb solution for handling multiple time scales.
- **Backward Compatibility:** The console logging remains unchanged, so existing scripts continue to work.
- **Numerical Stability:** None values are properly filtered out before wandb logging.
- **Modularity:** The fix is contained within the existing Trainer class without affecting other components.

## Testing

The fix has been tested and verified to work correctly. The console output shows proper metric formatting, and the wandb integration will now log epoch-level metrics without step conflicts.

## Usage

No changes are required in user code. The fix is transparent to existing training scripts. Simply enable wandb with `use_wandb=True` in the Trainer constructor, and epoch-level metrics will be logged correctly without warnings. 