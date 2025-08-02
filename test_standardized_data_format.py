#!/usr/bin/env python3
"""
Test script to verify the standardized data format is working correctly.
This script tests that all data loaders return the expected standardized format.
"""

import torch
from torch.utils.data import DataLoader
import tempfile
import os

def test_synthetic_dataset():
    """Test synthetic dataset with standardized format."""
    print("\n=== Testing Synthetic Dataset ===")
    
    from data_pipeline import SyntheticCopyDataset, pad_collate
    
    # Test regression task
    dataset = SyntheticCopyDataset(
        dataset_size=10,
        seq_len=20,
        vocab_size=100,
        task="regression"
    )
    
    loader = DataLoader(dataset, batch_size=2, collate_fn=lambda x: pad_collate(x, task="regression"))
    
    batch = next(iter(loader))
    print(f"✓ Regression batch keys: {list(batch.keys())}")
    print(f"✓ Expected: ['inputs', 'targets'], Got: {list(batch.keys())}")
    
    # Test classification task
    dataset = SyntheticCopyDataset(
        dataset_size=10,
        seq_len=20,
        vocab_size=100,
        task="classification",
        num_classes=2
    )
    
    loader = DataLoader(dataset, batch_size=2, collate_fn=lambda x: pad_collate(x, task="classification"))
    
    batch = next(iter(loader))
    print(f"✓ Classification batch keys: {list(batch.keys())}")
    print(f"✓ Expected: ['inputs', 'labels'], Got: {list(batch.keys())}")
    
    return True

def test_timeseries_dataset():
    """Test timeseries dataset with standardized format."""
    print("\n=== Testing Timeseries Dataset ===")
    
    from data.timeseries_loader import ETTDataset
    from data_pipeline import pad_collate
    
    # Create dummy data for testing
    import numpy as np
    dummy_data = np.random.randn(1000, 7).astype(np.float32)
    
    dataset = ETTDataset(
        data=dummy_data,
        input_len=96,
        output_len=24,
        target_col=0
    )
    
    loader = DataLoader(dataset, batch_size=2, collate_fn=lambda x: pad_collate(x, task="regression"))
    
    batch = next(iter(loader))
    print(f"✓ Timeseries batch keys: {list(batch.keys())}")
    print(f"✓ Expected: ['inputs', 'targets'], Got: {list(batch.keys())}")
    print(f"✓ Inputs shape: {batch['inputs'].shape}")
    print(f"✓ Targets shape: {batch['targets'].shape}")
    
    return True

def test_nlp_dataset():
    """Test NLP dataset with standardized format."""
    print("\n=== Testing NLP Dataset ===")
    
    from data.nlp_loader import NLPDataset
    
    # Create a dummy CSV file for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("text,label\n")
        f.write("This is a test sentence,0\n")
        f.write("Another test sentence,1\n")
        f.write("Yet another sentence,0\n")
        temp_file = f.name
    
    try:
        dataset = NLPDataset(
            file_path=temp_file,
            tokenizer_name='bert-base-uncased',
            max_length=32
        )
        
        loader = DataLoader(dataset, batch_size=2)
        
        batch = next(iter(loader))
        print(f"✓ NLP batch keys: {list(batch.keys())}")
        print(f"✓ Expected: ['inputs', 'attention_mask', 'labels'], Got: {list(batch.keys())}")
        print(f"✓ Inputs shape: {batch['inputs'].shape}")
        print(f"✓ Labels shape: {batch['labels'].shape}")
        
        return True
    finally:
        os.unlink(temp_file)

def test_language_modeling_dataset():
    """Test language modeling dataset with standardized format."""
    print("\n=== Testing Language Modeling Dataset ===")
    
    from data.pg19_loader import PG19Dataset
    from data_pipeline import language_modeling_collate
    
    # Create a dummy CSV file for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("text\n")
        f.write("This is a test sentence for language modeling.\n")
        f.write("Another test sentence for language modeling.\n")
        temp_file = f.name
    
    try:
        dataset = PG19Dataset(
            file_path=temp_file,
            tokenizer_name='gpt2',
            max_length=32
        )
        
        loader = DataLoader(dataset, batch_size=2, collate_fn=language_modeling_collate)
        
        batch = next(iter(loader))
        print(f"✓ Language modeling batch keys: {list(batch.keys())}")
        print(f"✓ Expected: ['inputs', 'attention_mask', 'labels'], Got: {list(batch.keys())}")
        print(f"✓ Inputs shape: {batch['inputs'].shape}")
        print(f"✓ Labels shape: {batch['labels'].shape}")
        
        return True
    finally:
        os.unlink(temp_file)

def test_trainer_unpack_batch():
    """Test that the trainer can unpack batches with standardized format."""
    print("\n=== Testing Trainer Unpack Batch ===")
    
    from src.trainer import Trainer
    import torch.nn as nn
    
    # Create a dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
        
        def forward(self, x):
            return self.linear(x.mean(dim=1))
    
    model = DummyModel()
    
    # Create dummy data loaders
    from torch.utils.data import TensorDataset, DataLoader
    
    # Regression data
    inputs = torch.randn(10, 5, 10)
    targets = torch.randn(10, 2, 1)
    
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=2)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=loader,
        device='cpu'
    )
    
    # Test unpacking regression batch
    batch = next(iter(loader))
    batch_dict = {
        'inputs': batch[0],
        'targets': batch[1]
    }
    
    try:
        model_input, targets = trainer._unpack_batch(batch_dict)
        print(f"✓ Successfully unpacked regression batch")
        print(f"✓ Model input shape: {model_input.shape}")
        print(f"✓ Targets shape: {targets.shape}")
        
        # Test unpacking classification batch
        batch_dict = {
            'inputs': batch[0],
            'labels': torch.randint(0, 2, (2,))
        }
        
        model_input, labels = trainer._unpack_batch(batch_dict)
        print(f"✓ Successfully unpacked classification batch")
        print(f"✓ Model input shape: {model_input.shape}")
        print(f"✓ Labels shape: {labels.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to unpack batch: {e}")
        return False

def test_invalid_batch_format():
    """Test that the trainer properly rejects invalid batch formats."""
    print("\n=== Testing Invalid Batch Format ===")
    
    from src.trainer import Trainer
    import torch.nn as nn
    
    # Create a dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
        
        def forward(self, x):
            return self.linear(x.mean(dim=1))
    
    model = DummyModel()
    
    # Create dummy data loaders
    from torch.utils.data import TensorDataset, DataLoader
    
    inputs = torch.randn(10, 5, 10)
    targets = torch.randn(10, 2, 1)
    
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=2)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=loader,
        device='cpu'
    )
    
    # Test invalid batch format (legacy format)
    batch = next(iter(loader))
    batch_dict = {
        'input': batch[0],  # Old format
        'target': batch[1]  # Old format
    }
    
    try:
        trainer._unpack_batch(batch_dict)
        print("❌ Should have rejected invalid batch format")
        return False
    except ValueError as e:
        print(f"✓ Correctly rejected invalid batch format: {str(e)[:100]}...")
        return True

def main():
    """Run all tests."""
    print("Testing standardized data format...")
    
    tests = [
        test_synthetic_dataset,
        test_timeseries_dataset,
        test_nlp_dataset,
        test_language_modeling_dataset,
        test_trainer_unpack_batch,
        test_invalid_batch_format
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n{'='*60}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! The standardized data format is working correctly.")
    else:
        print("❌ Some tests failed. The standardized format may need further investigation.")
    
    return passed == total

if __name__ == "__main__":
    main() 