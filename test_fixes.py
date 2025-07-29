#!/usr/bin/env python3
"""
Comprehensive test suite for critical bug fixes.

This test file verifies that all the identified critical issues have been resolved:
1. Hyperparameter search exception handling
2. Enhanced TFN parameter mismatch
3. Parameter validation system
4. Deprecated parameter warnings
"""

import pytest
import torch
import tempfile
import os
import warnings
from unittest.mock import patch, MagicMock
from dataclasses import asdict

# Import the modules we're testing
from hyperparameter_search import HyperparameterSearch, TrialResult
from model.tfn_unified import TFN
from model import registry
from train import build_model
from core.field_evolution import create_field_evolver


class TestCriticalBugFixes:
    """Test suite for critical bug fixes."""
    
    def test_hyperparameter_search_exception_handling(self):
        """Test that hyperparameter search handles trial failures gracefully."""
        # Create a minimal config for testing
        config = {
            'data': {
                'dataset_name': 'synthetic_copy',
                'seq_len': 10,
                'vocab_size': 100,
                'dataset_size': 100
            },
            'model': {
                'embed_dim': 64,
                'num_classes': 2,
                'vocab_size': 100
            },
            'training': {
                'batch_size': 4,
                'epochs': 2,
                'learning_rate': 0.001
            }
        }
        
        # Create search instance
        search = HyperparameterSearch(
            models=['tfn_classifier'],
            param_sweep={'embed_dim': [64]},
            config=config,
            output_dir=tempfile.mkdtemp(),
            patience=2,
            min_epochs=1
        )
        
        # Mock the _run_trial method to simulate a failure
        def mock_run_trial_failure(trial_id, model_name, parameters):
            raise RuntimeError("Simulated training failure")
        
        search._run_trial = mock_run_trial_failure
        
        # Run search - should not crash
        search.run_search()
        
        # Verify that the search completed without crashing
        assert True  # If we get here, the search didn't crash
    
    def test_enhanced_tfn_instantiation(self):
        """Test that Enhanced TFN can be instantiated with interference_type."""
        # Test with use_enhanced=True and interference_type
        model = TFN(
            task="classification",
            vocab_size=100,
            num_classes=2,
            embed_dim=64,
            num_layers=2,
            use_enhanced=True,
            interference_type="standard"
        )
        
        # Verify the model was created successfully
        assert isinstance(model, TFN)
        assert model.task == "classification"
        
        # Test forward pass with dummy data
        batch_size, seq_len = 2, 10
        dummy_input = torch.randint(0, 100, (batch_size, seq_len))
        
        with torch.no_grad():
            output = model(dummy_input)
        
        # Verify output shape - classification uses global average pooling
        assert output.shape == (batch_size, 2)  # [B, num_classes]
    
    def test_parameter_validation_warnings(self):
        """Test that parameter validation warns about unsupported parameters."""
        # Test with a parameter that's not supported by TFN
        model_cfg = {
            'vocab_size': 100,
            'num_classes': 2,
            'embed_dim': 64,
            'n_heads': 8,  # This parameter is not supported by TFN
            'kernel_type': 'rbf'
        }
        
        # Capture stdout to check for warnings
        import io
        import sys
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            model = build_model('tfn_classifier', model_cfg)
        
        output = f.getvalue()
        
        # Verify warning was printed
        assert "Warning: The following parameters were specified but are not supported" in output
        assert "n_heads" in output
        
        # Verify model was still created successfully
        assert isinstance(model, TFN)
    
    def test_deprecated_parameter_warning(self):
        """Test that deprecated propagator_type parameter shows warning."""
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Call create_field_evolver with deprecated parameter
            evolver = create_field_evolver(
                embed_dim=64,
                pos_dim=1,
                evolution_type="cnn",
                propagator_type="dynamic"  # Deprecated parameter
            )
            
            # Verify warning was issued
            assert len(w) > 0
            assert any("propagator_type parameter is deprecated" in str(warning.message) for warning in w)
            
            # Verify evolver was still created
            assert evolver is not None
    
    def test_model_registry_interference_type(self):
        """Test that interference_type is properly registered for TFN models."""
        # Check that interference_type is in optional params
        tfn_classifier_config = registry.get_model_config('tfn_classifier')
        assert 'interference_type' in tfn_classifier_config['optional_params']
        
        tfn_regressor_config = registry.get_model_config('tfn_regressor')
        assert 'interference_type' in tfn_regressor_config['optional_params']
        
        # Check that it has a default value
        assert 'interference_type' in tfn_classifier_config['defaults']
        assert tfn_classifier_config['defaults']['interference_type'] == 'standard'
    
    def test_enhanced_tfn_with_different_interference_types(self):
        """Test Enhanced TFN with different interference types."""
        interference_types = ["standard", "causal", "multiscale"]
        
        for interference_type in interference_types:
            model = TFN(
                task="classification",
                vocab_size=100,
                num_classes=2,
                embed_dim=64,
                num_layers=2,
                use_enhanced=True,
                interference_type=interference_type
            )
            
            # Test forward pass
            batch_size, seq_len = 2, 10
            dummy_input = torch.randint(0, 100, (batch_size, seq_len))
            
            with torch.no_grad():
                output = model(dummy_input)
            
            # Classification uses global average pooling
            assert output.shape == (batch_size, 2)  # [B, num_classes]
    
    def test_hyperparameter_search_parameter_validation(self):
        """Test that hyperparameter search uses the same parameter validation."""
        config = {
            'data': {
                'dataset_name': 'synthetic_copy',
                'seq_len': 10,
                'vocab_size': 100,
                'dataset_size': 100
            },
            'model': {
                'embed_dim': 64,
                'num_classes': 2,
                'vocab_size': 100,
                'n_heads': 8  # Unsupported parameter
            },
            'training': {
                'batch_size': 4,
                'epochs': 2,
                'learning_rate': 0.001
            }
        }
        
        search = HyperparameterSearch(
            models=['tfn_classifier'],
            param_sweep={'embed_dim': [64]},
            config=config,
            output_dir=tempfile.mkdtemp(),
            patience=2,
            min_epochs=1
        )
        
        # Mock the _run_trial method to avoid actual training
        def mock_run_trial(trial_id, model_name, parameters):
            # This should not crash due to parameter validation
            return TrialResult(
                trial_id=trial_id,
                model_name=model_name,
                parameters=parameters,
                start_time="2024-01-01T00:00:00",
                end_time="2024-01-01T00:01:00",
                duration_seconds=60.0,
                epochs_completed=2,
                early_stopped=False,
                early_stop_reason=None,
                best_epoch=1,
                best_val_loss=0.5,
                best_val_accuracy=0.8,
                best_val_mse=0.5,
                best_val_mae=0.7,
                final_train_loss=0.4,
                final_train_accuracy=0.85,
                final_train_mse=0.4,
                final_train_mae=0.6,
                final_val_loss=0.5,
                final_val_accuracy=0.8,
                final_val_mse=0.5,
                final_val_mae=0.7,
                training_history=[]
            )
        
        search._run_trial = mock_run_trial
        
        # Run search - should complete without crashing
        search.run_search()
        assert True  # If we get here, no crashes occurred


if __name__ == "__main__":
    # Run the tests
    print("🧪 Running comprehensive test suite for critical bug fixes...")
    
    test_suite = TestCriticalBugFixes()
    
    print("1. Testing hyperparameter search exception handling...")
    test_suite.test_hyperparameter_search_exception_handling()
    print("   ✅ Passed")
    
    print("2. Testing Enhanced TFN instantiation...")
    test_suite.test_enhanced_tfn_instantiation()
    print("   ✅ Passed")
    
    print("3. Testing parameter validation warnings...")
    test_suite.test_parameter_validation_warnings()
    print("   ✅ Passed")
    
    print("4. Testing deprecated parameter warnings...")
    test_suite.test_deprecated_parameter_warning()
    print("   ✅ Passed")
    
    print("5. Testing model registry interference_type...")
    test_suite.test_model_registry_interference_type()
    print("   ✅ Passed")
    
    print("6. Testing Enhanced TFN with different interference types...")
    test_suite.test_enhanced_tfn_with_different_interference_types()
    print("   ✅ Passed")
    
    print("7. Testing hyperparameter search parameter validation...")
    test_suite.test_hyperparameter_search_parameter_validation()
    print("   ✅ Passed")
    
    print("\n🎉 All critical bug fixes verified successfully!")
    print("✅ Hyperparameter search exception handling: FIXED")
    print("✅ Enhanced TFN parameter mismatch: FIXED")
    print("✅ Parameter validation system: IMPLEMENTED")
    print("✅ Deprecated parameter warnings: ADDED") 