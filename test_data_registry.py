#!/usr/bin/env python3
"""
Test the data registry pattern implementation.
"""

import yaml
from data.registry import (
    get_dataset_config, get_task_compatibility, get_required_params,
    get_optional_params, get_dataset_defaults, validate_dataset_task_compatibility,
    list_available_datasets, list_datasets_by_task, get_dataset_description,
    register_dataset, create_dataset
)

def test_registry_functions():
    """Test basic registry functions."""
    print("Testing registry functions...")
    
    # Test listing datasets
    datasets = list_available_datasets()
    print(f"  Available datasets: {datasets}")
    assert len(datasets) > 0, "No datasets found in registry"
    
    # Test getting dataset config
    config = get_dataset_config('ett')
    print(f"  ETT config: {config['task_type']}")
    assert config['task_type'] == 'regression'
    
    # Test task compatibility
    regression_datasets = list_datasets_by_task('regression')
    print(f"  Regression datasets: {regression_datasets}")
    assert 'ett' in regression_datasets
    
    # Test parameter functions
    required = get_required_params('ett')
    print(f"  ETT required params: {required}")
    assert 'csv_path' in required
    
    optional = get_optional_params('ett')
    print(f"  ETT optional params: {optional}")
    
    defaults = get_dataset_defaults('ett')
    print(f"  ETT defaults: {defaults}")
    assert 'csv_path' in defaults
    
    # Test validation
    assert validate_dataset_task_compatibility('ett', 'regression')
    assert not validate_dataset_task_compatibility('ett', 'classification')
    
    # Test description
    desc = get_dataset_description('ett')
    print(f"  ETT description: {desc}")
    assert len(desc) > 0
    
    print("  ✅ Registry functions work correctly!")

def test_dataset_creation():
    """Test dataset creation through registry."""
    print("\nTesting dataset creation...")
    
    # Test synthetic dataset
    config = {
        'data': {
            'dataset_name': 'synthetic',
            'dataset_size': 100,
            'seq_len': 20,
            'vocab_size': 10,
            'task': 'copy'
        }
    }
    
    try:
        dataset = create_dataset('synthetic', config, 'train')
        print(f"  ✅ Synthetic dataset created: {len(dataset)} samples")
        assert len(dataset) == 100
    except Exception as e:
        print(f"  ❌ Synthetic dataset creation failed: {e}")
        return False
    
    # Test ETT dataset
    config = {
        'data': {
            'dataset_name': 'ett',
            'csv_path': 'data/ETTh1.csv',
            'input_len': 96,
            'output_len': 24
        }
    }
    
    try:
        dataset = create_dataset('ett', config, 'train')
        print(f"  ✅ ETT dataset created: {len(dataset)} samples")
        assert len(dataset) > 0
    except Exception as e:
        print(f"  ❌ ETT dataset creation failed: {e}")
        return False
    
    print("  ✅ Dataset creation works correctly!")
    return True

def test_custom_dataset_registration():
    """Test registering a custom dataset."""
    print("\nTesting custom dataset registration...")
    
    from torch.utils.data import Dataset
    
    class CustomDataset(Dataset):
        def __init__(self, size=100, **kwargs):
            self.size = size
            self.data = list(range(size))
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return {'inputs': torch.tensor([self.data[idx]]), 'targets': torch.tensor([self.data[idx]])}
    
    # Register custom dataset
    register_dataset(
        name='custom_test',
        dataset_class=CustomDataset,
        task_type='regression',
        required_params=['size'],
        optional_params=[],
        defaults={'size': 100},
        description='Custom test dataset'
    )
    
    # Test it's registered
    assert 'custom_test' in list_available_datasets()
    
    # Test creation
    config = {'data': {'dataset_name': 'custom_test', 'size': 50}}
    dataset = create_dataset('custom_test', config, 'train')
    assert len(dataset) == 50
    
    print("  ✅ Custom dataset registration works!")

def test_dataloader_factory_integration():
    """Test that dataloader_factory works with the registry."""
    print("\nTesting dataloader_factory integration...")
    
    from data_pipeline import dataloader_factory
    
    # Test synthetic dataset
    config = {
        'data': {
            'dataset_name': 'synthetic',
            'dataset_size': 100,
            'seq_len': 20,
            'vocab_size': 10,
            'task': 'copy'
        }
    }
    
    try:
        dataset = dataloader_factory(config, 'train')
        print(f"  ✅ Factory created synthetic dataset: {len(dataset)} samples")
        assert len(dataset) == 100
    except Exception as e:
        print(f"  ❌ Factory failed for synthetic: {e}")
        return False
    
    # Test ETT dataset
    config = {
        'data': {
            'dataset_name': 'ett',
            'csv_path': 'data/ETTh1.csv',
            'input_len': 96,
            'output_len': 24
        }
    }
    
    try:
        dataset = dataloader_factory(config, 'train')
        print(f"  ✅ Factory created ETT dataset: {len(dataset)} samples")
        assert len(dataset) > 0
    except Exception as e:
        print(f"  ❌ Factory failed for ETT: {e}")
        return False
    
    print("  ✅ Dataloader factory integration works!")
    return True

def test_error_handling():
    """Test error handling for unknown datasets."""
    print("\nTesting error handling...")
    
    # Test unknown dataset
    try:
        get_dataset_config('unknown_dataset')
        print("  ❌ Should have raised ValueError for unknown dataset")
        return False
    except ValueError:
        print("  ✅ Correctly raised ValueError for unknown dataset")
    
    # Test invalid dataset creation
    config = {'data': {'dataset_name': 'unknown_dataset'}}
    try:
        create_dataset('unknown_dataset', config, 'train')
        print("  ❌ Should have raised ValueError for unknown dataset creation")
        return False
    except ValueError:
        print("  ✅ Correctly raised ValueError for unknown dataset creation")
    
    print("  ✅ Error handling works correctly!")
    return True

def main():
    """Run all tests."""
    print("🧪 Testing Data Registry Pattern")
    print("=" * 50)
    
    tests = [
        test_registry_functions,
        test_dataset_creation,
        test_custom_dataset_registration,
        test_dataloader_factory_integration,
        test_error_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Data registry pattern is working correctly.")
    else:
        print("❌ Some tests failed. The registry pattern may need further investigation.")
    
    return passed == total

if __name__ == "__main__":
    main() 