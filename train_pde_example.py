#!/usr/bin/env python3
"""
Training script for PDE datasets using Token Field Network.

This script demonstrates how to train the TFN model on PDE datasets like
Burgers' Equation or Darcy Flow. You'll need to provide your own .mat files.

REQUIRED DATA FORMAT:
Your .mat file should contain:
- 'a': Initial conditions [n_samples, n_spatial_points]
- 'u': Solutions [n_samples, n_spatial_points, n_timesteps] 
- 'x': Spatial grid coordinates [n_spatial_points] or [n_spatial_points, 2] for 2D

Example data sources:
- FNO repository: https://github.com/neuraloperator/neuraloperator
- Burgers' Equation: 1D shock wave modeling
- Darcy Flow: 2D porous media flow
"""

import os
import torch
import yaml
from pathlib import Path
from train import build_model, create_task_strategy
from data_pipeline import get_dataloader


def create_pde_config(dataset_name: str, file_path: str, target_timestep: int = 10):
    """
    Create a configuration for PDE training.
    
    Args:
        dataset_name: 'burgers' or 'darcy'
        file_path: Path to your .mat file
        target_timestep: Which timestep to predict (0 for steady-state like Darcy)
    """
    
    # Base configuration
    config = {
        "model_name": "tfn_regressor",
        "task": "regression",
        "strategy": "pde",  # Use PDEStrategy
        "data": {
            "dataset_name": dataset_name,
            "file_path": file_path,
            "target_timestep": target_timestep,
            "normalize": True,
            "normalization_strategy": "global"
        },
        "model": {
            "input_dim": 1,
            "embed_dim": 256,
            "output_dim": 1,
            "num_layers": 4,
            "dropout": 0.1,
            "kernel_type": "rbf",
            "evolution_type": "cnn",
            "positional_embedding_strategy": "continuous"  # Better for PDE data
        },
        "training": {
            "batch_size": 32,
            "lr": 1e-4,
            "epochs": 100,
            "warmup_epochs": 10,
            "grad_clip": 1.0,
            "log_interval": 50,
            "save_interval": 10
        }
    }
    
    return config


def create_synthetic_pde_data():
    """
    Create synthetic PDE data for testing if you don't have real data.
    This creates a simple 1D Burgers' equation simulation.
    """
    import numpy as np
    from scipy.io import savemat
    
    print("Creating synthetic Burgers' equation data...")
    
    # Parameters
    n_samples = 1000
    n_spatial_points = 64
    n_timesteps = 20
    x_min, x_max = 0, 1
    
    # Spatial grid
    x = np.linspace(x_min, x_max, n_spatial_points)
    
    # Create synthetic initial conditions (random sine waves)
    a = np.zeros((n_samples, n_spatial_points))
    for i in range(n_samples):
        # Random frequency and phase
        freq = np.random.uniform(1, 5)
        phase = np.random.uniform(0, 2*np.pi)
        amplitude = np.random.uniform(0.5, 1.5)
        a[i] = amplitude * np.sin(freq * 2 * np.pi * x + phase)
    
    # Simple time evolution (simplified Burgers' equation)
    u = np.zeros((n_samples, n_spatial_points, n_timesteps))
    u[:, :, 0] = a  # Initial condition
    
    # Simple advection: shift and damp
    for t in range(1, n_timesteps):
        # Shift right and damp
        u[:, :, t] = 0.9 * np.roll(u[:, :, t-1], shift=1, axis=1)
    
    # Save as .mat file
    data_dict = {
        'a': a,  # Initial conditions
        'u': u,  # Solutions over time
        'x': x   # Spatial grid
    }
    
    output_path = "data/synthetic_burgers.mat"
    os.makedirs("data", exist_ok=True)
    savemat(output_path, data_dict)
    
    print(f"Synthetic data saved to: {output_path}")
    print(f"Shape: {n_samples} samples, {n_spatial_points} spatial points, {n_timesteps} timesteps")
    
    return output_path


def train_pde_model(config, output_dir="checkpoints"):
    """
    Train the TFN model on PDE data.
    
    Args:
        config: Training configuration
        output_dir: Directory to save checkpoints
    """
    print(f"Training TFN on {config['data']['dataset_name']} dataset...")
    print(f"Data file: {config['data']['file_path']}")
    print(f"Target timestep: {config['data']['target_timestep']}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create data loaders
    train_loader = get_dataloader(config, 'train')
    val_loader = get_dataloader(config, 'val')
    test_loader = get_dataloader(config, 'test')
    
    if train_loader is None:
        print("❌ Failed to create data loaders. Check your data file path.")
        return
    
    print(f"✓ Data loaders created successfully")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    
    # Create model
    model = build_model(config['model_name'], config['model'], config['data'])
    print(f"✓ Model created: {config['model_name']}")
    
    # Create task strategy
    strategy = create_task_strategy(config["task"], config)
    print(f"✓ Task strategy created: {type(strategy).__name__}")
    
    # Import trainer
    from src.trainer import Trainer
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        strategy=strategy,
        device="cpu",
        lr=config["training"]["lr"],
        weight_decay=0.0,
        epochs=config["training"]["epochs"],
        grad_clip=config["training"]["grad_clip"],
        log_interval=config["training"]["log_interval"],
        warmup_epochs=config["training"]["warmup_epochs"],
        checkpoint_dir=output_dir
    )
    
    print("✓ Trainer created successfully")
    
    # Train the model
    print("\n🚀 Starting training...")
    trainer.fit()
    
    print(f"\n✅ Training completed!")
    print(f"Best model saved to: {trainer.checkpoint_dir}/experiment_*_best.pt")
    print(f"Training history saved to: {trainer.checkpoint_dir}/experiment_*_history.json")


def main():
    """Main function to run PDE training."""
    print("Token Field Network - PDE Training")
    print("=" * 50)
    
    # Check if you have real PDE data
    print("\n📋 PDE Data Requirements:")
    print("Your .mat file should contain:")
    print("  - 'a': Initial conditions [n_samples, n_spatial_points]")
    print("  - 'u': Solutions [n_samples, n_spatial_points, n_timesteps]")
    print("  - 'x': Spatial grid [n_spatial_points] or [n_spatial_points, 2]")
    print("\nExample data sources:")
    print("  - FNO repository: https://github.com/neuraloperator/neuraloperator")
    print("  - Burgers' Equation: 1D shock wave modeling")
    print("  - Darcy Flow: 2D porous media flow")
    
    # Ask user for data choice
    print("\n" + "="*50)
    print("Choose your data source:")
    print("1. Use synthetic data (for testing)")
    print("2. Use your own .mat file")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        # Create synthetic data
        file_path = create_synthetic_pde_data()
        dataset_name = "burgers"
        target_timestep = 10
        print(f"\n✅ Using synthetic data: {file_path}")
        
    elif choice == "2":
        # Use user's data
        file_path = input("\nEnter path to your .mat file: ").strip()
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return
        
        dataset_name = input("Enter dataset name (burgers/darcy): ").strip().lower()
        if dataset_name not in ["burgers", "darcy"]:
            print("❌ Dataset name must be 'burgers' or 'darcy'")
            return
        
        target_timestep = int(input("Enter target timestep (0 for steady-state like Darcy): ").strip())
        print(f"\n✅ Using your data: {file_path}")
        
    else:
        print("❌ Invalid choice")
        return
    
    # Create configuration
    config = create_pde_config(dataset_name, file_path, target_timestep)
    
    # Train the model
    train_pde_model(config)
    
    print("\n🎉 Training script completed!")
    print("\nNext steps:")
    print("1. Check the checkpoints directory for saved models")
    print("2. Use the best model for inference")
    print("3. Compare results with FNO baselines")


if __name__ == "__main__":
    main() 