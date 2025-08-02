#!/usr/bin/env python3
"""
Test script to run ETT training with the original configuration.
"""

# This is the configuration dictionary you pass to the training function.
training_config = {
    'model_name': 'enhanced_tfn_regressor',
    'data': {
        'dataset_name': 'ett',
        'csv_path': 'data/ETTh1.csv',
        'input_len': 96,
        'output_len': 24
    },
    'model': {
        'input_dim': 7,
        'embed_dim': 128,
        'output_dim': 1,
        'output_len': 24,
        'num_layers': 2,
    },
    'training': {
        'epochs': 15,
        'batch_size': 32,
        'lr': 1e-4,
    },

    # Add this new section to your configuration
    'wandb': {
        'use_wandb': True,  # This is the master switch to enable W&B
        'project_name': 'TFN Paper Experiments',  # The project where your runs will be saved
        'experiment_name': 'ETT_Baseline_Run_1' # A specific name for this training run
    }
}

# Make sure this import is in your notebook
from train import run_training

# Execute the training run
# Because 'use_wandb' is True, this will automatically start a W&B run
print("🚀 Starting ETT training with fixed metrics handling...")
history = run_training(training_config, device="cuda")

print("✅ Training completed successfully!")
print(f"Final training loss: {history['train_loss'][-1] if history['train_loss'] else 'N/A'}")
print(f"Final validation loss: {history['val_loss'][-1] if history['val_loss'] else 'N/A'}") 