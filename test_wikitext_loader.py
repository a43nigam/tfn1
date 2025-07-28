#!/usr/bin/env python3
"""
Test script for WikiText dataset loader and training pipeline.
"""

import torch
import yaml
from data_pipeline import get_dataloader
from model import registry
from src.trainer import Trainer

def test_wikitext_loader():
    """Test WikiText dataset loading and basic training setup."""
    
    print("🧪 Testing WikiText Dataset Loader")
    print("=" * 50)
    
    # Load test configuration
    with open("configs/wikitext_test.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    print(f"✅ Loaded config: {config['data']['dataset_name']}")
    
    # Test dataset loading
    try:
        train_loader = get_dataloader(config, split='train')
        val_loader = get_dataloader(config, split='val')
        test_loader = get_dataloader(config, split='test')
        
        print(f"✅ Train dataset: {len(train_loader.dataset)} samples")
        print(f"✅ Val dataset: {len(val_loader.dataset)} samples") 
        print(f"✅ Test dataset: {len(test_loader.dataset)} samples")
        
        # Test batch loading
        batch = next(iter(train_loader))
        print(f"✅ Batch keys: {list(batch.keys())}")
        print(f"✅ Input shape: {batch['input_ids'].shape}")
        print(f"✅ Labels shape: {batch['labels'].shape}")
        print(f"✅ Attention mask shape: {batch['attention_mask'].shape}")
        
        # Test model creation
        model_name = "enhanced_tfn_language_model"
        model_info = registry.get_model_config(model_name)
        print(f"✅ Model info: {model_info['task_type']}")
        
        # Test task compatibility
        task = "language_modeling"
        task_config = registry.get_task_compatibility(task)
        print(f"✅ Compatible models: {task_config['models']}")
        print(f"✅ Compatible datasets: {task_config['datasets']}")
        
        print("\n🎉 WikiText loader test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ WikiText loader test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_setup():
    """Test complete training setup with WikiText."""
    
    print("\n🧪 Testing Complete Training Setup")
    print("=" * 50)
    
    try:
        # Load configuration
        with open("configs/wikitext_test.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        # Get data loaders
        train_loader = get_dataloader(config, split='train')
        val_loader = get_dataloader(config, split='val')
        
        # Create model
        model_name = "enhanced_tfn_language_model"
        model_info = registry.get_model_config(model_name)
        model_cls = model_info['class']
        
        # Get model parameters
        model_args = dict(model_info.get('defaults', {}))
        model_args.update(config['model'])
        
        # Create model instance
        model = model_cls(**model_args)
        print(f"✅ Created model: {type(model).__name__}")
        print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        batch = next(iter(train_loader))
        x = batch['input_ids']
        with torch.no_grad():
            output = model(x)
        print(f"✅ Forward pass successful: {output.shape}")
        
        # Test trainer creation
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            task="language_modeling",
            device="cpu",
            lr=float(config['training']['lr']),
            epochs=int(config['training']['epochs'])
        )
        print(f"✅ Trainer created successfully")
        
        print("\n🎉 Complete training setup test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Training setup test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 WikiText Support Test Suite")
    print("=" * 60)
    
    # Run tests
    test1_passed = test_wikitext_loader()
    test2_passed = test_training_setup()
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED! WikiText support is ready.")
        print("\n📝 Usage:")
        print("  python train.py --config configs/wikitext_test.yaml --model_name enhanced_tfn_language_model")
        print("  python train.py --config configs/wikitext.yaml --model_name enhanced_tfn_language_model")
    else:
        print("\n❌ Some tests failed. Please check the implementation.") 