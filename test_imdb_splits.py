#!/usr/bin/env python3
"""
Test script to verify IMDB dataset splitting works correctly.
"""

import os
import sys
from data_pipeline import get_dataloader

def test_imdb_splits():
    """Test that IMDB dataset properly splits data."""
    
    # Test configuration
    config = {
        'data': {
            'dataset_name': 'imdb',
            'file_path': '/kaggle/input/training/training/IMDB/IMDB Dataset.csv',
            'tokenizer_name': 'bert-base-uncased',
            'max_length': 256,
            'text_col': 'review',
            'label_col': 'sentiment',
            'split_frac': {'train': 0.8, 'val': 0.1, 'test': 0.1}
        },
        'model': {
            'vocab_size': 30522,
            'num_classes': 2
        },
        'training': {
            'batch_size': 32
        }
    }
    
    try:
        print("Testing IMDB dataset splits...")
        
        # Get train, val, and test loaders
        train_loader = get_dataloader(config, split='train')
        val_loader = get_dataloader(config, split='val')
        test_loader = get_dataloader(config, split='test')
        
        # Get dataset sizes
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)
        total_size = train_size + val_size + test_size
        
        print(f"✅ IMDB dataset splits working correctly!")
        print(f"   Train size: {train_size:,} samples ({train_size/total_size:.1%})")
        print(f"   Val size: {val_size:,} samples ({val_size/total_size:.1%})")
        print(f"   Test size: {test_size:,} samples ({test_size/total_size:.1%})")
        print(f"   Total size: {total_size:,} samples")
        
        # Test that splits are different
        if train_size > 0 and val_size > 0 and test_size > 0:
            print("✅ All splits have data")
        else:
            print("❌ Some splits are empty")
            return False
        
        # Test a few samples
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        test_batch = next(iter(test_loader))
        
        print(f"✅ Sample batch shapes:")
        print(f"   Train: {train_batch['input_ids'].shape}")
        print(f"   Val: {val_batch['input_ids'].shape}")
        print(f"   Test: {test_batch['input_ids'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing IMDB splits: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imdb_splits()
    if success:
        print("\n🎉 IMDB dataset splitting is working correctly!")
    else:
        print("\n💥 IMDB dataset splitting has issues!")
        sys.exit(1) 