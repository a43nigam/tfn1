#!/usr/bin/env python3
"""
Test script to verify learning rate and FLOPs tracking functionality.
"""

import torch
import yaml
from data_pipeline import get_dataloader
from model import registry
from src.trainer import Trainer

def test_lr_flops_tracking():
    """Test learning rate and FLOPs tracking functionality."""
    
    # Simple config for testing
    config = {
        "data": {
            "dataset_name": "synthetic",
            "seq_len": 50,
            "dataset_size": 1000,
            "pad_idx": 0,
            "task": "classification"
        },
        "model": {
            "vocab_size": 20,
            "embed_dim": 64,
            "grid_size": 128,
            "kernel_type": "rbf",
            "evolution_type": "cnn",
            "time_steps": 3,
            "dropout": 0.1,
            "num_classes": 2
        },
        "training": {
            "batch_size": 32,
            "lr": 1e-3,
            "epochs": 3,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "log_interval": 10,
            "warmup_epochs": 1
        },
        "task": "classification"
    }
    
    print("🧪 Testing Learning Rate and FLOPs Tracking")
    print("=" * 50)
    
    # Get data loaders
    train_loader = get_dataloader(config, split='train')
    val_loader = get_dataloader(config, split='val')
    
    # Build model
    model_name = "tfn_classifier"
    model_info = registry.get_model_config(model_name)
    model_cls = model_info['class']
    model = model_cls.for_classification(**config["model"])
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer with FLOPs tracking enabled
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task="classification",
        device="cpu",
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
        epochs=config["training"]["epochs"],
        grad_clip=config["training"]["grad_clip"],
        log_interval=config["training"]["log_interval"],
        warmup_epochs=config["training"]["warmup_epochs"],
        track_flops=True
    )
    
    print("\n🚀 Starting training with LR and FLOPs tracking...")
    print("-" * 50)
    
    # Run training
    history = trainer.fit()
    
    print("\n📊 Training Results:")
    print("-" * 50)
    print(f"Learning rates tracked: {len(history['learning_rates'])} epochs")
    print(f"Learning rate progression: {[f'{lr:.6f}' for lr in history['learning_rates']]}")
    
    if 'flops_stats' in history and history['flops_stats']:
        flops_stats = history['flops_stats']
        print(f"\n🔢 FLOPs Statistics:")
        print(f"  Total FLOPS: {flops_stats['total_flops']:,}")
        print(f"  Avg FLOPS per pass: {flops_stats['avg_flops_per_pass']:,.0f}")
        print(f"  FLOPS per second: {flops_stats['flops_per_second']:,.0f}")
    else:
        print("\n⚠️  No FLOPs statistics available")
    
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    test_lr_flops_tracking() 