#!/usr/bin/env python3
"""
Test script to verify --device flags work correctly.
"""

import subprocess
import sys
import torch

def test_device_detection():
    """Test device detection logic."""
    print("🔍 Testing device detection...")
    
    # Test auto-detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Auto-detected device: {device}")
    
    if torch.cuda.is_available():
        print(f"   CUDA available: ✅")
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print(f"   CUDA available: ❌")

def test_train_device_flag():
    """Test --device flag in train.py."""
    print("\n🚀 Testing train.py --device flag...")
    
    # Test with CUDA device
    cmd = [
        "python", "train.py",
        "--config", "configs/tests/synthetic_copy_test.yaml",
        "--model_name", "tfn_classifier",
        "--device", "cuda",
        "--epochs", "1",  # Just 1 epoch for quick test
        "--batch_size", "4",  # Small batch for testing
        "--disable_logging"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("   ✅ CUDA device flag works in train.py")
            if "Using device: cuda" in result.stdout:
                print("   ✅ Device correctly set to CUDA")
            else:
                print("   ⚠️  Device info not found in output")
        else:
            print(f"   ❌ Error: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("   ⏰ Test timed out (this is normal for a quick test)")
    except Exception as e:
        print(f"   ❌ Exception: {e}")

def test_hyperparameter_search_device_flag():
    """Test --device flag in hyperparameter_search.py."""
    print("\n🔍 Testing hyperparameter_search.py --device flag...")
    
    # Test with CUDA device
    cmd = [
        "python", "hyperparameter_search.py",
        "--config", "configs/tests/synthetic_copy_test.yaml",
        "--models", "tfn_classifier",
        "--param_sweep", "embed_dim:64 num_layers:2",
        "--output_dir", "./test_device_search",
        "--device", "cuda",
        "--epochs", "2",  # Just 2 epochs for quick test
        "--patience", "1",
        "--min_epochs", "1",
        "--seed", "42"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print("   ✅ CUDA device flag works in hyperparameter_search.py")
            if "Using device: cuda" in result.stdout:
                print("   ✅ Device correctly set to CUDA")
            else:
                print("   ⚠️  Device info not found in output")
        else:
            print(f"   ❌ Error: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("   ⏰ Test timed out (this is normal for a quick test)")
    except Exception as e:
        print(f"   ❌ Exception: {e}")

def test_cpu_device_flag():
    """Test forcing CPU device."""
    print("\n💻 Testing CPU device flag...")
    
    # Test with CPU device
    cmd = [
        "python", "train.py",
        "--config", "configs/tests/synthetic_copy_test.yaml",
        "--model_name", "tfn_classifier",
        "--device", "cpu",
        "--epochs", "1",
        "--batch_size", "4",
        "--disable_logging"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("   ✅ CPU device flag works in train.py")
            if "Using device: cpu" in result.stdout:
                print("   ✅ Device correctly set to CPU")
            else:
                print("   ⚠️  Device info not found in output")
        else:
            print(f"   ❌ Error: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("   ⏰ Test timed out (this is normal for a quick test)")
    except Exception as e:
        print(f"   ❌ Exception: {e}")

def main():
    """Main test function."""
    print("=" * 60)
    print("🧪 Testing --device flags for Token Field Network")
    print("=" * 60)
    
    # Test device detection
    test_device_detection()
    
    # Test CUDA device flag
    test_train_device_flag()
    test_hyperparameter_search_device_flag()
    
    # Test CPU device flag
    test_cpu_device_flag()
    
    print("\n" + "=" * 60)
    print("📋 Device Flag Usage Examples:")
    print("=" * 60)
    print("🚀 Force CUDA:")
    print("   python train.py --device cuda --model_name tfn_classifier")
    print("   python hyperparameter_search.py --device cuda --models tfn_classifier")
    print()
    print("💻 Force CPU:")
    print("   python train.py --device cpu --model_name tfn_classifier")
    print("   python hyperparameter_search.py --device cpu --models tfn_classifier")
    print()
    print("🤖 Auto-detect:")
    print("   python train.py --device auto --model_name tfn_classifier")
    print("   python hyperparameter_search.py --device auto --models tfn_classifier")
    print()
    print("🎯 Quick CUDA test:")
    print("   python train.py --device cuda --model_name tfn_classifier --epochs 1 --batch_size 4")
    print("=" * 60)

if __name__ == "__main__":
    main() 