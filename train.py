from __future__ import annotations

import argparse
import yaml
import torch
from data_pipeline import get_dataloader
from model import registry
from src.trainer import Trainer
from src import metrics
from model.utils import build_model as shared_build_model  # centralised builder
from src.task_strategies import TaskStrategy, ClassificationStrategy, RegressionStrategy, LanguageModelingStrategy
from typing import Any, Dict

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

def create_task_strategy(task_type: str, config: Dict[str, Any]) -> TaskStrategy:
    """Factory function to create the appropriate task strategy."""
    if task_type == "classification" or task_type == "ner":
        return ClassificationStrategy()
    elif task_type in ("regression", "time_series"):
        return RegressionStrategy()
    elif task_type == "language_modeling":
        return LanguageModelingStrategy()
    else:
        raise ValueError(f"No strategy available for task type: {task_type}")

def print_training_info(cfg: Dict[str, Any], model_name: str, model_info: Dict[str, Any], 
                       model: torch.nn.Module, device: torch.device, train_cfg: Dict[str, Any]) -> None:
    """Print comprehensive training information including hyperparameters and model details."""
    
    print("\n" + "="*80)
    print("🚀 TRAINING CONFIGURATION")
    print("="*80)
    
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
    
    # Model info
    print(f"\n🤖 MODEL:")
    print(f"   Type: {model_name}")
    print(f"   Task: {model_info['task_type']}")
    print(f"   Components: {', '.join(model_info.get('components', []))}")
    print(f"   Evolution Types: {', '.join(model_info.get('evolution_types', []))}")
    
    # Model hyperparameters
    model_cfg = cfg["model"]
    print(f"\n⚙️  MODEL HYPERPARAMETERS:")
    for key, value in model_cfg.items():
        print(f"   {key}: {value}")
    
    # Training hyperparameters
    print(f"\n🎯 TRAINING HYPERPARAMETERS:")
    for key, value in train_cfg.items():
        print(f"   {key}: {value}")
    
    # Model architecture details
    print(f"\n🏗️  MODEL ARCHITECTURE:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"   Device: {device}")
    
    # Print model structure (first few layers)
    print(f"\n📋 MODEL STRUCTURE (first 3 layers):")
    for i, (name, module) in enumerate(model.named_modules()):
        if i >= 3:  # Only show first 3 layers
            break
        if len(list(module.children())) == 0:  # Leaf modules only
            print(f"   {name}: {module}")
    
    print("\n" + "="*80)
    print("🎬 STARTING TRAINING...")
    print("="*80 + "\n")

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = update_config_with_args(cfg, args)
    
    # Device detection with --device flag support
    if args.device is not None:
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🔧 Using device: {device}")
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            print(f"   GPU: {torch.cuda.get_device_name()}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        except Exception as e:
            print(f"   GPU info unavailable: {e}")
    elif device.type == "cuda" and not torch.cuda.is_available():
        print("   ⚠️  CUDA requested but not available - falling back to CPU")
        device = torch.device("cpu")
        print(f"   Using CPU instead")

    model_name = cfg.get("model_name") or args.model_name or "tfn_classifier"
    # Universal: Print split sizes for all datasets
    from data_pipeline import dataloader_factory
    data_cfg = cfg["data"]
    train_ds = dataloader_factory(cfg, split="train")
    val_ds = dataloader_factory(cfg, split="val")
    test_ds = dataloader_factory(cfg, split="test")
    dataset_name = data_cfg.get("dataset_name", "unknown")
    total_size = len(train_ds) + len(val_ds) + len(test_ds)
    print(f"{dataset_name} splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}, total={total_size}")
    train_loader = get_dataloader(cfg, split='train')
    val_loader = get_dataloader(cfg, split='val')
    dataset_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    # ------------------------------------------------------------------
    # Auto-fill model config with dataset-dependent attributes
    # ------------------------------------------------------------------
    for attr in ("vocab_size", "input_dim", "num_classes"):
        if hasattr(train_ds, attr) and attr not in model_cfg:
            model_cfg[attr] = getattr(train_ds, attr)

    if model_name.startswith("tfn") and "pg19" in cfg["data"]["dataset_name"]:
        from data.pg19_loader import PG19Dataset
        file_path = cfg["data"]["file_path"]
        tokenizer_name = cfg["data"].get("tokenizer_name", "gpt2")
        max_length = cfg["data"].get("max_length", 512)
        split_frac = cfg["data"].get("split_frac", {"train": 0.8, "val": 0.1, "test": 0.1})
        text_col = cfg["data"].get("text_col", "text")
        train_ds, val_ds, test_ds = PG19Dataset.get_splits(file_path, tokenizer_name, max_length, split_frac, text_col)
        print(f"PG19 splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    model = build_model(model_name, model_cfg, data_cfg).to(device)
    model_info = registry.get_model_config(model_name)
    task_type = model_info['task_type']

    # --- KEY CHANGES START HERE ---
    
    # 1. Create the strategy object using the new factory
    strategy = create_task_strategy(task_type, cfg)

    # Print comprehensive training information
    print_training_info(cfg, model_name, model_info, model, device, train_cfg)

    lr_value = train_cfg.get("lr", 1e-3)
    if isinstance(lr_value, str):
        try:
            lr_value = float(lr_value)
        except ValueError:
            raise ValueError(f"Invalid learning rate value: {lr_value!r}")

    # 2. Inject the strategy object into the Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        strategy=strategy,  # CHANGED from task=task_type
        device=device,
        lr=lr_value,
        weight_decay=train_cfg.get("weight_decay", 0.0),
        epochs=train_cfg.get("epochs", args.epochs or 10),
        grad_clip=float(train_cfg.get("grad_clip", 1.0)),
        log_interval=train_cfg.get("log_interval", 100),
        warmup_epochs=train_cfg.get("warmup_epochs", 1),
        track_flops=True  # Enable FLOPs tracking
    )

    trainer.fit()

if __name__ == "__main__":
    main() 