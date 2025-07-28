#!/usr/bin/env python3
"""
Test script to verify the validation metrics fix.
"""

import torch
import yaml
from data_pipeline import get_dataloader
from model import registry
from src.trainer import Trainer

def test_validation_consistency():
    """Test that validation loss and metrics are consistent."""
    
    print("🧪 Testing Validation Metrics Fix")
    print("=" * 50)
    
    # Load a regression configuration
    config = {
        "data": {
            "dataset_name": "synthetic",
            "task": "regression",
            "dataset_size": 1000,
            "seq_len": 20,
            "pad_idx": 0
        },
        "model": {
            "task": "regression",
            "vocab_size": 20,  # Required for synthetic dataset
            "input_dim": 8,
            "output_dim": 8,
            "output_len": 20,
            "embed_dim": 64,
            "num_layers": 2,
            "grid_size": 64,
            "kernel_type": "rbf",
            "evolution_type": "cnn",
            "time_steps": 2,
            "dropout": 0.1
        },
        "training": {
            "batch_size": 16,
            "lr": 1e-3,
            "epochs": 2,
            "weight_decay": 0.01,
            "optimizer": "adamw",
            "warmup_epochs": 1,
            "grad_clip": 1.0,
            "log_interval": 10
        }
    }
    
    print("📊 Configuration:")
    print(f"   Dataset: {config['data']['dataset_name']}")
    print(f"   Input dim: {config['model']['input_dim']}")
    print(f"   Output dim: {config['model']['output_dim']}")
    print(f"   Output len: {config['model']['output_len']}")
    
    # Get data loaders
    train_loader = get_dataloader(config, split='train')
    val_loader = get_dataloader(config, split='val')
    
    print(f"✅ Train samples: {len(train_loader.dataset)}")
    print(f"✅ Val samples: {len(val_loader.dataset)}")
    
    # Test batch shapes
    batch = next(iter(train_loader))
    print(f"📦 Batch shapes:")
    print(f"   Source: {batch['source'].shape}")
    print(f"   Target: {batch['target'].shape}")
    
    # Create model
    model_name = "tfn_regressor"
    model_info = registry.get_model_config(model_name)
    model_cls = model_info['class']
    
    # Get model parameters
    model_args = dict(model_info.get('defaults', {}))
    model_args.update(config['model'])
    
    # Create model instance
    model = model_cls(**model_args)
    print(f"✅ Model created: {type(model).__name__}")
    print(f"✅ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = batch['source']
    y = batch['target']
    with torch.no_grad():
        preds = model(x)
        print(f"🔍 Model output shape: {preds.shape}")
        print(f"🔍 Target shape: {y.shape}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task="regression",
        device="cpu",
        lr=float(config['training']['lr']),
        epochs=int(config['training']['epochs'])
    )
    
    print("✅ Trainer created successfully!")
    
    # Run a few training steps to test consistency
    print("\n🚀 Running test training...")
    history = trainer.fit()
    
    # Check for consistency
    print("\n📊 Results Analysis:")
    for epoch in range(len(history['train_loss'])):
        train_loss = history['train_loss'][epoch]
        train_mse = history['train_mse'][epoch]
        val_loss = history['val_loss'][epoch]
        val_mse = history['val_mse'][epoch]
        
        print(f"Epoch {epoch+1}:")
        print(f"  Train - Loss: {train_loss:.6f}, MSE: {train_mse:.6f}")
        print(f"  Val   - Loss: {val_loss:.6f}, MSE: {val_mse:.6f}")
        
        # Check if loss and MSE are consistent (should be very close)
        if train_loss is not None and train_mse is not None:
            loss_mse_diff = abs(train_loss - train_mse)
            if loss_mse_diff > 1e-6:
                print(f"  ⚠️  Train Loss/MSE difference: {loss_mse_diff:.6f}")
            else:
                print(f"  ✅ Train Loss/MSE consistent")
        
        if val_loss is not None and val_mse is not None:
            loss_mse_diff = abs(val_loss - val_mse)
            if loss_mse_diff > 1e-6:
                print(f"  ⚠️  Val Loss/MSE difference: {loss_mse_diff:.6f}")
            else:
                print(f"  ✅ Val Loss/MSE consistent")
    
    print("\n🎉 Validation metrics fix test completed!")

if __name__ == "__main__":
    test_validation_consistency() 