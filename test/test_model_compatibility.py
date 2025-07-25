import os
import sys
import subprocess
import yaml
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

# Models to test (sequence-compatible only)
MODEL_NAMES = [
    "tfn_classifier",
    "enhanced_tfn_classifier",
    "transformer_classifier",
    "performer_classifier",
    "lstm_classifier",
    "cnn_classifier",
]

def build_flag_list():
    flags = []
    for k, v in UNIVERSAL_FLAGS.items():
        if v is None:
            flags.append(k)  # boolean flag
        else:
            flags.extend([k, str(v)])
    return flags

def main():
    # Ensure output dir exists
    os.makedirs("test/results", exist_ok=True)
    # Load base config
    with open(CONFIG_PATH, "r") as f:
        base_cfg = yaml.safe_load(f)
    results = {}
    for model_name in MODEL_NAMES:
        print(f"\n=== Testing model: {model_name} ===")
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
    print("\n=== Summary ===")
    for model_name, code in results.items():
        status = "PASS" if code == 0 else "FAIL"
        print(f"{model_name}: {status}")
    print("\nDone.")

if __name__ == "__main__":
    main() 