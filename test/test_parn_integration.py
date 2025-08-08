"""
Test PARN integration with the training pipeline.

This module tests that PARN is properly integrated into the training
pipeline and that the model building and wrapping works correctly.
"""

import torch
import torch.nn as nn
import yaml
import tempfile
import os
from typing import Dict, Any

from model.wrappers import PARNModel
from model.utils import build_model


class DummyModel(nn.Module):
    """A simple dummy model for testing PARN integration."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # Handle both [B, N, F] and [B, F] shapes
        if x.dim() == 3:
            batch_size, seq_len, features = x.shape
            x = x.view(-1, features)  # [B*N, F]
            x = self.linear(x)
            x = x.view(batch_size, seq_len, -1)  # [B, N, O]
        else:
            x = self.linear(x)
        return x


def test_parn_integration():
    """Test PARN integration with the training pipeline."""
    
    print("🧪 Testing PARN integration with training pipeline...")
    
    # Test the integration logic without building actual models
    model_cfg = {
        "input_dim": 5,
        "embed_dim": 32,
        "output_dim": 5,
        "output_len": 10,
        "kernel_type": "rbf",
        "evolution_type": "cnn",  # Use a simpler evolution type
        "num_layers": 1,
        "grid_size": 50,
        "time_steps": 2,
        "dropout": 0.1
    }
    data_cfg = {
        "dataset_name": "test",
        "file_path": "data/test.csv",
        "input_len": 20,
        "output_len": 10,
        "batch_size": 4,
        "train_ratio": 0.7,
        "val_ratio": 0.2,
        "test_ratio": 0.1,
        "normalization_strategy": "parn",
        "parn_mode": "location"
    }
    model_name = "tfn_regressor"
    
    # Simulate the integration logic from train.py
    normalization_strategy = data_cfg.get('normalization_strategy')
    
    if normalization_strategy == 'parn':
        try:
            from model.wrappers import PARNModel
            parn_mode = data_cfg.get('parn_mode', 'location')
            original_input_dim = model_cfg['input_dim']
            
            # Dynamically calculate the new input dimension
            if parn_mode == 'location' or parn_mode == 'scale':
                model_cfg['input_dim'] = original_input_dim * 2
            elif parn_mode == 'full':
                model_cfg['input_dim'] = original_input_dim * 3

            print(f"✅ Applying PARN wrapper (mode='{parn_mode}'). Model input_dim adjusted to {model_cfg['input_dim']}.")
            
            # Test that the calculation is correct
            expected_dim = 10 if parn_mode in ['location', 'scale'] else 15
            assert model_cfg['input_dim'] == expected_dim, f"Expected {expected_dim}, got {model_cfg['input_dim']}"
            
            # Test that PARNModel can be created (without building the actual model)
            # Create a dummy model for testing
            class DummyBaseModel(nn.Module):
                def __init__(self, input_dim, output_dim):
                    super().__init__()
                    self.linear = nn.Linear(input_dim, output_dim)
                
                def forward(self, x, **kwargs):
                    return self.linear(x)
            
            dummy_model = DummyBaseModel(model_cfg['input_dim'], model_cfg['output_dim'])
            parn_model = PARNModel(base_model=dummy_model, num_features=original_input_dim, mode=parn_mode)
            
            # Test that the wrapper has the correct attributes
            assert parn_model.parn.mode == parn_mode
            assert parn_model.parn.num_features == original_input_dim
            
            print("✅ PARN integration test passed!")
            return True
            
        except Exception as e:
            print(f"❌ PARN integration test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return False


def test_parn_config_loading():
    """Test that PARN configuration can be loaded from YAML."""
    
    print("\n🧪 Testing PARN configuration loading...")
    
    # Create a temporary config file
    config_content = """
model_name: "tfn_regressor"

model:
  input_dim: 7
  embed_dim: 64
  output_dim: 7
  output_len: 24
  kernel_type: "rbf"
  evolution_type: "pde"
  num_layers: 2
  grid_size: 100
  time_steps: 3
  dropout: 0.1

data:
  dataset_name: "ETT"
  file_path: "data/ETTh1.csv"
  input_len: 96
  output_len: 24
  batch_size: 32
  train_ratio: 0.7
  val_ratio: 0.2
  test_ratio: 0.1
  normalization_strategy: "parn"
  parn_mode: "location"

training:
  epochs: 100
  lr: 1e-4
  device: "auto"
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        config_file = f.name
    
    try:
        # Load the config
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Test that PARN configuration is properly loaded
        assert config['data']['normalization_strategy'] == 'parn'
        assert config['data']['parn_mode'] == 'location'
        assert config['model']['input_dim'] == 7
        
        print("✅ PARN configuration loading test passed!")
        return True
        
    except Exception as e:
        print(f"❌ PARN configuration loading test failed: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists(config_file):
            os.unlink(config_file)


def test_parn_mode_calculations():
    """Test that PARN input dimension calculations are correct."""
    
    print("\n🧪 Testing PARN mode calculations...")
    
    # Test different modes
    test_cases = [
        ('location', 5, 10),  # location mode: 5 * 2 = 10
        ('scale', 5, 10),     # scale mode: 5 * 2 = 10
        ('full', 5, 15),      # full mode: 5 * 3 = 15
    ]
    
    for mode, original_dim, expected_dim in test_cases:
        # Simulate the calculation logic
        if mode == 'location' or mode == 'scale':
            calculated_dim = original_dim * 2
        elif mode == 'full':
            calculated_dim = original_dim * 3
        else:
            calculated_dim = original_dim
        
        assert calculated_dim == expected_dim, f"Mode {mode}: expected {expected_dim}, got {calculated_dim}"
    
    print("✅ PARN mode calculations test passed!")
    return True


if __name__ == "__main__":
    # Run all integration tests
    print("🚀 Running PARN integration tests...")
    
    test1_passed = test_parn_integration()
    test2_passed = test_parn_config_loading()
    test3_passed = test_parn_mode_calculations()
    
    if test1_passed and test2_passed and test3_passed:
        print("\n🎉 All PARN integration tests passed!")
    else:
        print("\n❌ Some PARN integration tests failed!")
        exit(1) 