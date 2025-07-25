import os
import sys
import subprocess
import yaml

CONFIG_PATH = "configs/tests/ett_test.yaml"
TRAIN_SCRIPT = "train.py"

# Regression models to test
REGRESSION_MODELS = [
    "tfn_regressor",
    "transformer_regressor",
    "performer_regressor",
    "lstm_regressor",
    "cnn_regressor",
]

# Minimal flags for fast CPU test (standardized names)
MINI_FLAGS = {
    "--output_dir": "test/results/",  # dummy output dir
    "--seed": 42,
    "--disable_logging": None,  # boolean flag
    "--learning_rate": 1e-3,
    "--batch_size": 2,
    "--epochs": 1,
    "--weight_decay": 0.0,
    "--optimizer": "adam",
    # Model architecture (standardized)
    "--model.embed_dim": 4,
    "--model.n_layers": 1,
    "--model.dropout": 0.0,
    # Data & sequencing (standardized)
    "--data.input_len": 8,
    "--data.output_len": 2,
}

def build_flag_list():
    flags = []
    for k, v in MINI_FLAGS.items():
        if v is None:
            flags.append(k)  # boolean flag
        else:
            flags.extend([k, str(v)])
    return flags

def main():
    os.makedirs("test/results", exist_ok=True)
    results = {}
    for model_name in REGRESSION_MODELS:
        print(f"\n=== Testing regression model: {model_name} ===")
        cmd = [sys.executable, TRAIN_SCRIPT, "--config", CONFIG_PATH, "--model_name", model_name]
        cmd += build_flag_list()
        print(f"[RUNNING] {' '.join(cmd)}")
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                print(line, end='')
            proc.wait(timeout=120)
            results[model_name] = proc.returncode
            if proc.returncode == 0:
                print(f"[SUCCESS] {model_name}")
            else:
                print(f"[FAIL] {model_name} (return code {proc.returncode})")
        except subprocess.TimeoutExpired:
            proc.kill()
            print(f"[TIMEOUT] {model_name} (killed after 120s)")
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