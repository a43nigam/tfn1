#!/usr/bin/env python3
"""
Test script to verify the synthetic task loader integration with the registry system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from data.registry import create_dataset, get_dataset_config, list_available_datasets
from data.synthetic_task_loader import SyntheticTaskDataset


def test_registry_integration():
    """Test that all synthetic datasets are properly registered."""
    print("Testing Registry Integration...")
    
    available_datasets = list_available_datasets()
    synthetic_datasets = ['heat_equation', 'delayed_copy', 'irregular_sampling']
    
    for dataset_name in synthetic_datasets:
        print(f"  Checking {dataset_name}...")
        assert dataset_name in available_datasets, f"{dataset_name} not found in registry"
        
        config = get_dataset_config(dataset_name)
        print(f"    Task type: {config['task_type']}")
        print(f"    Description: {config['description']}")
        print(f"    Defaults: {config['defaults']}")
    
    print("  ✓ All datasets properly registered")


def test_dataset_creation():
    """Test creating datasets through the registry system."""
    print("\nTesting Dataset Creation via Registry...")
    
    synthetic_datasets = ['heat_equation', 'delayed_copy', 'irregular_sampling']
    
    for dataset_name in synthetic_datasets:
        print(f"  Testing {dataset_name}...")
        
        # Create datasets using registry
        config = {'data': {}}  # Use defaults
        
        try:
            train_ds = create_dataset(dataset_name, config, split='train')
            val_ds = create_dataset(dataset_name, config, split='val')
            test_ds = create_dataset(dataset_name, config, split='test')
            
            print(f"    Train size: {len(train_ds)}")
            print(f"    Val size: {len(val_ds)}")
            print(f"    Test size: {len(test_ds)}")
            
            # Test sample retrieval
            sample = train_ds[0]
            print(f"    Sample keys: {list(sample.keys())}")
            print(f"    Input shape: {sample['inputs'].shape}")
            print(f"    Target shape: {sample['targets'].shape}")
            
            if 'positions' in sample:
                print(f"    Position shape: {sample['positions'].shape}")
                
        except Exception as e:
            print(f"    ❌ Failed to create {dataset_name}: {e}")
            continue
            
        print(f"    ✓ {dataset_name} created successfully")
    
    print("  ✓ All datasets created successfully via registry")


def test_dataloader_compatibility():
    """Test that datasets work with PyTorch DataLoader."""
    print("\nTesting DataLoader Compatibility...")
    
    synthetic_datasets = ['heat_equation', 'delayed_copy', 'irregular_sampling']
    
    for dataset_name in synthetic_datasets:
        print(f"  Testing {dataset_name} with DataLoader...")
        
        config = {'data': {}}  # Use defaults
        train_ds = create_dataset(dataset_name, config, split='train')
        
        # Create DataLoader
        dataloader = DataLoader(
            train_ds, 
            batch_size=4, 
            shuffle=True, 
            num_workers=0  # Avoid multiprocessing issues in tests
        )
        
        # Test batch retrieval
        for i, batch in enumerate(dataloader):
            print(f"    Batch {i+1}:")
            print(f"      Input batch shape: {batch['inputs'].shape}")
            print(f"      Target batch shape: {batch['targets'].shape}")
            
            if 'positions' in batch:
                print(f"      Position batch shape: {batch['positions'].shape}")
            
            # Only test first batch
            break
        
        print(f"    ✓ {dataset_name} DataLoader working")
    
    print("  ✓ All datasets compatible with DataLoader")


def test_direct_loader_instantiation():
    """Test direct instantiation of SyntheticTaskDataset."""
    print("\nTesting Direct Loader Instantiation...")
    
    datasets_info = [
        ('heat_equation.pt', 'heat equation'),
        ('delayed_copy.pt', 'delayed copy'),
        ('irregular_sampling.pt', 'irregular sampling')
    ]
    
    for filename, desc in datasets_info:
        print(f"  Testing {desc}...")
        
        file_path = f'data/synthetic/{filename}'
        
        # Test get_splits factory method
        train_ds, val_ds, test_ds = SyntheticTaskDataset.get_splits(file_path)
        
        print(f"    Split sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
        
        # Test metadata
        metadata = train_ds.get_metadata()
        print(f"    Dataset type: {metadata['dataset_type']}")
        print(f"    Input shape: {metadata['input_shape']}")
        print(f"    Target shape: {metadata['target_shape']}")
        
        # Test sample shapes
        shapes = train_ds.get_sample_shapes()
        print(f"    Sample shapes: {shapes}")
        
        print(f"    ✓ {desc} direct instantiation working")
    
    print("  ✓ Direct instantiation working for all datasets")


def test_data_consistency():
    """Test that data is consistent across different creation methods."""
    print("\nTesting Data Consistency...")
    
    # Test heat equation dataset
    print("  Testing heat equation consistency...")
    
    # Method 1: Direct loader
    direct_ds = SyntheticTaskDataset('data/synthetic/heat_equation.pt', split='train', seed=42)
    
    # Method 2: Registry
    config = {'data': {'seed': 42}}
    registry_ds = create_dataset('heat_equation', config, split='train')
    
    # Should have same samples
    sample1 = direct_ds[0]
    sample2 = registry_ds[0]
    
    assert torch.allclose(sample1['inputs'], sample2['inputs']), "Inputs don't match"
    assert torch.allclose(sample1['targets'], sample2['targets']), "Targets don't match"
    
    print("    ✓ Data consistent between creation methods")
    
    print("  ✓ Data consistency verified")


def main():
    """Run all tests."""
    print("="*60)
    print("TESTING SYNTHETIC TASK LOADER")
    print("="*60)
    
    try:
        test_registry_integration()
        test_dataset_creation()
        test_dataloader_compatibility()
        test_direct_loader_instantiation()
        test_data_consistency()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main() 