from __future__ import annotations

import argparse
import yaml
import torch
from data_pipeline import get_dataloader
from model import registry
from src.trainer import Trainer
from src import metrics
from typing import Any, Dict

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TFN on the synthetic copy or regression task")
    parser.add_argument("--config", type=str, default="configs/tests/synthetic_copy_test.yaml", help="Path to YAML config file.")
    parser.add_argument("--model_name", type=str, default=None, help="Model to use (e.g., tfn_classifier, tfn_regressor, transformer_regressor, etc.)")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save logs and checkpoints.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--disable_logging", action="store_true", help="Disable logging to external services.")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate for optimizer.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--weight_decay", type=float, default=None, help="L2 regularization strength.")
    parser.add_argument("--optimizer", type=str, default=None, help="Optimizer to use (adamw, sgd, etc.)")
    parser.add_argument("--model.d_model", type=int, default=None, help="Main hidden dimension of the model.")
    parser.add_argument("--model.n_layers", type=int, default=None, help="Number of model layers.")
    parser.add_argument("--model.n_heads", type=int, default=None, help="Number of attention heads.")
    parser.add_argument("--model.dropout", type=float, default=None, help="Dropout rate for regularization.")
    parser.add_argument("--data.context_length", type=int, default=None, help="Input sequence length.")
    parser.add_argument("--data.prediction_length", type=int, default=None, help="Prediction length (for forecasting tasks).")
    # Additional model flags
    parser.add_argument("--model.embed_dim", type=int, default=None, help="Embedding dimension for the model.")
    parser.add_argument("--model.output_dim", type=int, default=None, help="Output dimension for regression models.")
    parser.add_argument("--model.input_dim", type=int, default=None, help="Input feature dimension for regression models.")
    parser.add_argument("--model.hidden_dim", type=int, default=None, help="Hidden dimension for LSTM models.")
    parser.add_argument("--model.proj_dim", type=int, default=None, help="Projection dimension for Performer models.")
    parser.add_argument("--model.num_filters", type=int, default=None, help="Number of filters for CNN models.")
    parser.add_argument("--model.filter_sizes", type=str, default=None, help="Filter sizes for CNN models (comma-separated, e.g. '3,4,5').")
    parser.add_argument("--model.bidirectional", action="store_true", help="Use bidirectional LSTM (flag).")
    # Additional data flags
    parser.add_argument("--data.input_len", type=int, default=None, help="Input window length for time series.")
    parser.add_argument("--data.output_len", type=int, default=None, help="Output window length for time series.")
    parser.add_argument("--data.seq_len", type=int, default=None, help="Sequence length for synthetic data.")
    parser.add_argument("--data.vocab_size", type=int, default=None, help="Vocabulary size for synthetic data.")
    parser.add_argument("--data.pad_idx", type=int, default=None, help="Padding index for synthetic data.")
    parser.add_argument("--data.dataset_size", type=int, default=None, help="Dataset size for synthetic data.")
    parser.add_argument("--data.csv_path", type=str, default=None, help="CSV path for ETT data.")
    parser.add_argument("--data.max_length", type=int, default=None, help="Max sequence length for NLP tokenization.")
    parser.add_argument("--data.tokenizer_name", type=str, default=None, help="Tokenizer name for NLP datasets (e.g., 'bert-base-uncased', 'roberta-base').")
    return parser.parse_args()

# -----------------------------------------------------------------------------
# Config update logic
# -----------------------------------------------------------------------------

def update_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    # Special mappings for CLI arguments that don't follow dot notation
    special_mappings = {
        "learning_rate": "training.lr",
        "batch_size": "training.batch_size", 
        "epochs": "training.epochs",
        "weight_decay": "training.weight_decay",
        "optimizer": "training.optimizer",
    }
    
    for arg_key, arg_val in vars(args).items():
        if arg_key == "config" or arg_val is None:
            continue
            
        # Handle special mappings
        if arg_key in special_mappings:
            config_key = special_mappings[arg_key]
        else:
            config_key = arg_key
            
        keys = config_key.split(".")
        d = config
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = arg_val
    return config

# -----------------------------------------------------------------------------
# Model instantiation logic
# -----------------------------------------------------------------------------

def build_model(model_name: str, model_cfg: dict) -> torch.nn.Module:
    model_info = registry.get_model_config(model_name)
    model_cls = model_info['class']
    task_type = model_info['task_type']
    model_args = dict(model_info.get('defaults', {}))
    model_args.update(model_cfg)
    if 'task' in model_cls.__init__.__code__.co_varnames:
        model_args['task'] = task_type
    allowed = set(model_info.get('required_params', []) + model_info.get('optional_params', []))
    filtered_args = {k: v for k, v in model_args.items() if k in allowed or k == 'task'}
    return model_cls(**filtered_args)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    if model_name.startswith("tfn") and "pg19" in cfg["data"]["dataset_name"]:
        from data.pg19_loader import PG19Dataset
        file_path = cfg["data"]["file_path"]
        tokenizer_name = cfg["data"].get("tokenizer_name", "gpt2")
        max_length = cfg["data"].get("max_length", 512)
        split_frac = cfg["data"].get("split_frac", {"train": 0.8, "val": 0.1, "test": 0.1})
        text_col = cfg["data"].get("text_col", "text")
        train_ds, val_ds, test_ds = PG19Dataset.get_splits(file_path, tokenizer_name, max_length, split_frac, text_col)
        print(f"PG19 splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    model = build_model(model_name, model_cfg).to(device)
    model_info = registry.get_model_config(model_name)
    task_type = model_info['task_type']

    # Print comprehensive training information
    print_training_info(cfg, model_name, model_info, model, device, train_cfg)

    lr_value = train_cfg.get("lr", 1e-3)
    if isinstance(lr_value, str):
        try:
            lr_value = float(lr_value)
        except ValueError:
            raise ValueError(f"Invalid learning rate value: {lr_value!r}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task=task_type,
        device=device,
        lr=lr_value,
        weight_decay=train_cfg.get("weight_decay", 0.0),
        epochs=train_cfg.get("epochs", args.epochs or 10),
        grad_clip=float(train_cfg.get("grad_clip", 1.0)),
        log_interval=train_cfg.get("log_interval", 100),
    )

    trainer.fit()

if __name__ == "__main__":
    main() 