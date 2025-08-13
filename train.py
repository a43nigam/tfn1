from __future__ import annotations

import argparse
import yaml
import torch
import os
from data_pipeline import get_dataloader
from model import registry
from src.trainer import Trainer
from src import metrics
from model.utils import build_model as shared_build_model  # centralised builder
from src.task_strategies import TaskStrategy, ClassificationStrategy, RegressionStrategy, LanguageModelingStrategy, PDEStrategy
from typing import Any, Dict, Optional

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TFN models.")
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to YAML config file.")
    
    # Keep essential, high-level overrides
    parser.add_argument("--model_name", type=str, default=None,
                       help="Model name to use (overrides YAML)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use: cuda | cpu | auto (default: auto-detect)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for logs and results")
    
    # Generic key-value override system
    parser.add_argument("--set", nargs='+', default=[],
                       help="Override config parameters with key=value pairs. "
                            "Example: --set model.embed_dim=512 training.lr=1e-5 "
                            "data.batch_size=32 model.kernel_type=rbf")
    
    # Keep for convenience (commonly overridden parameters)
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of training epochs (convenience flag)")
    parser.add_argument("--lr", type=float, default=None,
                       help="Learning rate (convenience flag)")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size (convenience flag)")
    
    return parser.parse_args()

# -----------------------------------------------------------------------------
# Config update logic
# -----------------------------------------------------------------------------

def update_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Update config with command line arguments using generic override system."""
    
    # Handle direct convenience arguments
    convenience_mappings = {
        'model_name': 'model_name',
        'epochs': 'training.epochs', 
        'lr': 'training.lr',
        'batch_size': 'training.batch_size',
        'device': 'training.device',
        'output_dir': 'output_dir'
    }
    
    for arg_name, config_path in convenience_mappings.items():
        arg_value = getattr(args, arg_name, None)
        if arg_value is not None:
            _set_nested_config_value(config, config_path, arg_value)
    
    # Handle generic --set overrides
    for override in args.set:
        try:
            key, value = override.split('=', 1)
        except ValueError:
            raise ValueError(f"Invalid override format: '{override}'. Expected 'key=value'")
        
        # Smart type casting
        typed_value = _cast_config_value(value)
        _set_nested_config_value(config, key, typed_value)
    
    return config


def _set_nested_config_value(config: Dict[str, Any], key_path: str, value: Any) -> None:
    """Set a nested configuration value using dot notation."""
    keys = key_path.split('.')
    d = config
    
    # Navigate to nested dictionary, creating as needed
    for k in keys[:-1]:
        if k not in d or not isinstance(d.get(k), dict):
            d[k] = {}
        d = d[k]
    
    # Set the final value
    d[keys[-1]] = value


def _cast_config_value(value: str) -> Any:
    """Intelligently cast string values to appropriate types."""
    # Strip whitespace
    value = value.strip()
    
    # Handle boolean strings
    if value.lower() in ('true', 'yes', 'on', '1'):
        return True
    elif value.lower() in ('false', 'no', 'off', '0'):
        return False
    
    # Handle None/null
    if value.lower() in ('none', 'null', ''):
        return None
    
    # Handle numbers
    try:
        # Scientific notation or decimal
        if 'e' in value.lower() or '.' in value:
            return float(value)
        else:
            return int(value)
    except ValueError:
        pass
    
    # Handle lists (comma-separated)
    if ',' in value:
        items = [_cast_config_value(item.strip()) for item in value.split(',')]
        return items
    
    # Keep as string
    return value

# -----------------------------------------------------------------------------
# Model instantiation logic
# -----------------------------------------------------------------------------

def build_model(model_name: str, model_cfg: dict, data_cfg: dict = None) -> torch.nn.Module:
    """Thin wrapper around the centralised shared_build_model utility."""
    return shared_build_model(model_name, model_cfg, data_cfg)

def create_task_strategy(task_type: str, config: Dict[str, Any], scaler: Optional[Any] = None, target_col_idx: int = 0) -> TaskStrategy:
    """Factory function to create the appropriate task strategy."""
    
    # Check if a specific strategy is requested in the config
    strategy_name = config.get("strategy", None)
    
    if strategy_name == "pde":
        # Use PDE strategy for physics-informed neural networks
        return PDEStrategy(scaler=scaler, target_col_idx=target_col_idx)
    elif task_type == "classification" or task_type == "ner":
        return ClassificationStrategy()
    elif task_type in ("regression", "time_series"):
        # Pass the scaler to the strategy
        return RegressionStrategy(scaler=scaler, target_col_idx=target_col_idx)
    elif task_type == "language_modeling":
        return LanguageModelingStrategy()
    else:
        raise ValueError(f"No strategy available for task type: {task_type}")

def print_training_info(cfg: Dict[str, Any], model_name: str, model_info: Dict[str, Any], 
                       model: torch.nn.Module, device: torch.device, train_cfg: Dict[str, Any]) -> None:
    """Print basic training information that appears before training starts."""
    
    # Dataset info
    data_cfg = cfg["data"]
    print(f"\n📊 DATASET:")
    print(f"   Name: {data_cfg.get('dataset_name', 'unknown')}")
    print(f"   File: {data_cfg.get('file_path', data_cfg.get('csv_path', 'N/A'))}")
    if 'input_len' in data_cfg:
        print(f"   Input Length: {data_cfg['input_len']}")
    if 'output_len' in data_cfg:
        print(f"   Output Length: {data_cfg['output_len']}")
    if 'max_length' in data_cfg:
        print(f"   Max Length: {data_cfg['max_length']}")
    
    print("\n" + "="*80)
    print("🎬 STARTING TRAINING...")
    print("="*80 + "\n")

def validate_config_parameters(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """
    Validate that all provided command-line arguments are valid for the specified model.
    
    Args:
        config: The configuration dictionary
        args: Command line arguments
    
    Raises:
        ValueError: If invalid parameters are found
    """
    model_name = config.get("model_name", "tfn_classifier")
    
    try:
        model_info = registry.get_model_config(model_name)
    except ValueError as e:
        raise ValueError(f"Invalid model name '{model_name}': {e}")
    
    # Get all allowed parameters for this model
    allowed_params = set(
        model_info.get("required_params", []) + 
        model_info.get("optional_params", [])
    )
    
    # Check all --set arguments
    invalid_params = []
    for override in args.set:
        try:
            key, value = override.split('=', 1)
        except ValueError:
            raise ValueError(f"Invalid override format: '{override}'. Expected 'key=value'")
        
        # Check if this is a model parameter
        if key.startswith('model.'):
            param_name = key.split('.', 1)[1]  # Remove 'model.' prefix
            if param_name not in allowed_params:
                invalid_params.append((key, param_name))
    
    # Report all invalid parameters at once
    if invalid_params:
        error_msg = f"Invalid parameters for model '{model_name}':\n"
        for full_key, param_name in invalid_params:
            error_msg += f"  - '{full_key}' (unknown parameter '{param_name}')\n"
        error_msg += f"\nValid parameters for {model_name}:\n"
        for param in sorted(allowed_params):
            error_msg += f"  - model.{param}\n"
        raise ValueError(error_msg)

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def run_training(config: Dict[str, Any], device: str = "auto") -> Dict[str, Any]:
    """
    Run training with the given configuration.
    
    Args:
        config: Configuration dictionary
        device: Device to use ("auto", "cuda", "cpu")
    
    Returns:
        Dictionary containing training history and results
    """
    # Device detection
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print(f"device: {device}")
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            print(f"   gpu: {torch.cuda.get_device_name()}")
            print(f"   memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        except Exception as e:
            print(f"   GPU info unavailable: {e}")
    elif device.type == "cuda" and not torch.cuda.is_available():
        print(" cuda requested, unavailable - using cpu")
        device = torch.device("cpu")

    model_name = config.get("model_name", "tfn_classifier")
    
    # Universal: Print split sizes for all datasets
    from data_pipeline import dataloader_factory
    data_cfg = config["data"]
    train_ds = dataloader_factory(config, split="train")
    val_ds = dataloader_factory(config, split="val")
    test_ds = dataloader_factory(config, split="test")
    dataset_name = data_cfg.get("dataset_name", "unknown")
    total_size = len(train_ds) + len(val_ds) + len(test_ds)
    print(f"{dataset_name} splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}, total={total_size}")
    
    train_loader = get_dataloader(config, split='train')
    val_loader = get_dataloader(config, split='val')
    dataset_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    # ADD THIS LINE to get your W&B config
    wandb_cfg = config.get("wandb", {})

    # ------------------------------------------------------------------
    # Auto-fill model config with dataset-dependent attributes
    # ------------------------------------------------------------------
    for attr in ("vocab_size", "input_dim", "num_classes"):
        if hasattr(train_ds, attr) and attr not in model_cfg:
            model_cfg[attr] = getattr(train_ds, attr)

    if model_name.startswith("tfn") and config["data"].get("dataset_name", "").startswith("pg19"):
        from data.pg19_loader import PG19Dataset
        file_path = config["data"]["file_path"]
        tokenizer_name = config["data"].get("tokenizer_name", "gpt2")
        max_length = config["data"].get("max_length", 512)
        split_frac = config["data"].get("split_frac", {"train": 0.8, "val": 0.1, "test": 0.1})
        text_col = config["data"].get("text_col", "text")
        train_ds, val_ds, test_ds = PG19Dataset.get_splits(file_path, tokenizer_name, max_length, split_frac, text_col)
        print(f"PG19 splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # Build the model using the registry (now handles normalization strategies)
    model_to_train = build_model(model_name, model_cfg, data_cfg).to(device)
    model_info = registry.get_model_config(model_name)
    task_type = model_info['task_type']

    # --- Get scaler from dataset for regression tasks ---
    scaler = getattr(train_ds, 'scaler', None)
    target_col_idx = getattr(train_ds, 'target_col', 0)

    # Create the strategy object using the new factory with scaler
    strategy = create_task_strategy(task_type, config, scaler=scaler, target_col_idx=target_col_idx)

    # Print comprehensive training information
    print_training_info(config, model_name, model_info, model_to_train, device, train_cfg)

    lr_value = train_cfg.get("lr", 1e-3)
    if isinstance(lr_value, str):
        try:
            lr_value = float(lr_value)
        except ValueError:
            raise ValueError(f"invalid learning rate: {lr_value!r}")

    # Determine checkpoint directory with Kaggle environment detection
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    
    # If we can't write to the specified directory, try Kaggle working directory
    if os.path.exists('/kaggle/working') and not os.access(checkpoint_dir, os.W_OK):
        checkpoint_dir = '/kaggle/working/tfn_checkpoints'
        print(f"using kaggle working directory for checkpoints: {checkpoint_dir}")
    
    # Create trainer
    trainer = Trainer(
        model=model_to_train, # Use the model_to_train here
        train_loader=train_loader,
        val_loader=val_loader,
        strategy=strategy,
        device=device,
        lr=lr_value,
        weight_decay=train_cfg.get("weight_decay", 0.0),
        epochs=train_cfg.get("epochs", 10),
        grad_clip=float(train_cfg.get("grad_clip", 1.0)),
        log_interval=train_cfg.get("log_interval", 100),
        warmup_epochs=train_cfg.get("warmup_epochs", 1),
        track_flops=True,  # Enable FLOPs tracking
        # ADD THESE LINES to pass the W&B settings to the Trainer
        use_wandb=wandb_cfg.get('use_wandb', False),
        project_name=wandb_cfg.get('project_name', 'tfn-experiments'),
        experiment_name=wandb_cfg.get('experiment_name'),
        # ADD THIS LINE to pass the checkpoint directory
        checkpoint_dir=checkpoint_dir
    )

    # Run training and return history
    # Pass full config for detailed logging after W&B starts
    full_config = {
        "model_name": model_name,
        "model_info": model_info,
        "model": model_cfg,
        "training": train_cfg,
        "data": data_cfg
    }
    history = trainer.fit(full_config=full_config)
    return history


def main() -> None:
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = update_config_with_args(cfg, args)
    
    # Step 1: Pre-flight validation of command-line arguments
    validate_config_parameters(cfg, args)
    
    # Device detection with --device flag support
    device = args.device or "auto"
    
    # Run training
    run_training(cfg, device=device)

if __name__ == "__main__":
    main() 