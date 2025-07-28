#!/usr/bin/env python3
"""
Memory-safe WikiText training script for Colab/Kaggle environments.
"""

import torch
import yaml
from train import build_model, print_training_info
from data_pipeline import get_dataloader
from model import registry
from src.trainer import Trainer
from src.memory_monitor import MemoryMonitor, print_memory_usage, suggest_batch_size, estimate_memory_usage

def main():
    print("🚀 Memory-Safe WikiText Training")
    print("=" * 50)
    
    # Check initial memory
    print_memory_usage("Initial")
    
    # Load configuration
    config_path = "configs/wikitext_memory_efficient.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    print(f"📋 Using config: {config_path}")
    
    # Memory-safe data loading
    with MemoryMonitor("Dataset Loading"):
        train_loader = get_dataloader(config, split='train')
        val_loader = get_dataloader(config, split='val')
        
        print(f"✅ Train samples: {len(train_loader.dataset)}")
        print(f"✅ Val samples: {len(val_loader.dataset)}")
    
    # Memory-safe model creation
    with MemoryMonitor("Model Creation"):
        model_name = "enhanced_tfn_language_model"
        model_info = registry.get_model_config(model_name)
        model_cls = model_info['class']
        
        # Get model parameters
        model_args = dict(model_info.get('defaults', {}))
        model_args.update(config['model'])
        
        # Create model instance
        model = model_cls(**model_args)
        print(f"✅ Model created: {type(model).__name__}")
        print(f"✅ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Estimate memory usage
    batch_size = config['training']['batch_size']
    seq_len = config['data']['max_length']
    vocab_size = config['model']['vocab_size']
    embed_dim = config['model']['embed_dim']
    
    estimated_memory = estimate_memory_usage(batch_size, seq_len, vocab_size, embed_dim)
    print(f"📊 Estimated memory usage: {estimated_memory:.1f} MB")
    
    # Suggest safe batch size if needed
    if estimated_memory > 1000:  # If estimated > 1GB
        suggested_batch_size = suggest_batch_size(seq_len, vocab_size, embed_dim, target_memory_mb=800)
        print(f"⚠️  Suggested batch size: {suggested_batch_size} (current: {batch_size})")
        
        if suggested_batch_size < batch_size:
            print("🔄 Adjusting batch size for memory safety...")
            config['training']['batch_size'] = suggested_batch_size
    
    # Print training info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_training_info(config, model_name, model_info, model, device, config['training'])
    
    # Memory-safe trainer creation
    with MemoryMonitor("Trainer Creation"):
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            task="language_modeling",
            device=device,
            lr=float(config['training']['lr']),
            epochs=int(config['training']['epochs']),
            weight_decay=float(config['training']['weight_decay']),
            grad_clip=float(config['training']['grad_clip']),
            log_interval=int(config['training']['log_interval'])
        )
        
        print("✅ Trainer created successfully!")
    
    print("\n🎉 Memory-safe WikiText training setup is ready!")
    print("\n📝 To start training, run:")
    print(f"   python train.py --config {config_path} --model_name {model_name}")
    print("\n📝 For even more memory efficiency:")
    print("   python train.py --config configs/wikitext_memory_efficient.yaml --model_name enhanced_tfn_language_model")

if __name__ == "__main__":
    main() 