import os
import sys
import subprocess
import yaml

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import registry

CONFIG_PATH = "configs/tests/synthetic_copy_test.yaml"
TRAIN_SCRIPT = "train.py"

# Universal/semi-universal flags to test
UNIVERSAL_FLAGS = {
    "--output_dir": "test/results/",  # dummy output dir
    "--seed": 42,
    "--disable_logging": None,  # boolean flag
    "--learning_rate": 1e-3,
    "--batch_size": 8,
    "--epochs": 1,
    "--weight_decay": 0.0,
    "--optimizer": "adam",
    # Model architecture
    "--model.d_model": 16,
    "--model.n_layers": 1,
    "--model.n_heads": 2,
    "--model.dropout": 0.1,
    # Data & sequencing
    "--data.context_length": 8,
    "--data.prediction_length": 1,
}

# Models to test (comprehensive list)
MODEL_NAMES = [
    "tfn_classifier",
    "enhanced_tfn_classifier",
    "transformer_classifier",
    "performer_classifier",
    "lstm_classifier",
    "cnn_classifier",
    "tfn_regressor",
    "transformer_regressor",
    "performer_regressor",
    "lstm_regressor",
    "cnn_regressor",
]

def build_flag_list():
    flags = []
    for k, v in UNIVERSAL_FLAGS.items():
        if v is None:
            flags.append(k)  # boolean flag
        else:
            flags.extend([k, str(v)])
    return flags

def test_lr_flops_tracking():
    """Test learning rate and FLOPs tracking functionality."""
    print("\n=== Testing LR and FLOPs Tracking ===")
    try:
        cmd = [sys.executable, "test_lr_flops_tracking.py"]
        print(f"[RUNNING] {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("[SUCCESS] LR and FLOPs tracking test passed")
            return True
        else:
            print(f"[FAIL] LR and FLOPs tracking test failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("[TIMEOUT] LR and FLOPs tracking test timed out")
        return False
    except Exception as e:
        print(f"[ERROR] LR and FLOPs tracking test error: {e}")
        return False

def test_hyperparameter_search():
    """Test hyperparameter search functionality."""
    print("\n=== Testing Hyperparameter Search ===")
    try:
        cmd = [sys.executable, "test_hyperparameter_search.py"]
        print(f"[RUNNING] {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("[SUCCESS] Hyperparameter search test passed")
            return True
        else:
            print(f"[FAIL] Hyperparameter search test failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("[TIMEOUT] Hyperparameter search test timed out")
        return False
    except Exception as e:
        print(f"[ERROR] Hyperparameter search test error: {e}")
        return False

def main():
    # Ensure output dir exists
    os.makedirs("test/results", exist_ok=True)
    
    # Load base config
    with open(CONFIG_PATH, "r") as f:
        base_cfg = yaml.safe_load(f)
    
    results = {}
    
    # Test model compatibility
    print("=== Testing Model Compatibility ===")
    for model_name in MODEL_NAMES:
        print(f"\n--- Testing model: {model_name} ---")
        cmd = [sys.executable, TRAIN_SCRIPT, "--config", CONFIG_PATH, "--model_name", model_name]
        cmd += build_flag_list()
        print(f"[RUNNING] {' '.join(cmd)}")
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                print(line, end='')  # stream output live
            proc.wait(timeout=300)
            results[model_name] = proc.returncode
            if proc.returncode == 0:
                print(f"[SUCCESS] {model_name}")
            else:
                print(f"[FAIL] {model_name} (return code {proc.returncode})")
        except subprocess.TimeoutExpired:
            proc.kill()
            print(f"[TIMEOUT] {model_name} (killed after 300s)")
            results[model_name] = -1
        except Exception as e:
            print(f"[ERROR] {model_name}: {e}")
            results[model_name] = -2
    
    # Test additional functionality
    lr_flops_success = test_lr_flops_tracking()
    hp_search_success = test_hyperparameter_search()
    
    # Print summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    print("\nModel Compatibility Results:")
    for model_name, code in results.items():
        status = "PASS" if code == 0 else "FAIL"
        print(f"  {model_name}: {status}")
    
    print(f"\nLR and FLOPs Tracking: {'PASS' if lr_flops_success else 'FAIL'}")
    print(f"Hyperparameter Search: {'PASS' if hp_search_success else 'FAIL'}")
    
    # Overall success
    model_successes = sum(1 for code in results.values() if code == 0)
    total_tests = len(results) + 2  # models + 2 additional tests
    overall_success = model_successes + (1 if lr_flops_success else 0) + (1 if hp_search_success else 0)
    
    print(f"\nOverall: {overall_success}/{total_tests} tests passed")
    
    if overall_success == total_tests:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check output above for details.")
    
    print("\nDone.")

if __name__ == "__main__":
    main() 