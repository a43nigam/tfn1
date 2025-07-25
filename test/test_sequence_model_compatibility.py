import os
import sys
import subprocess
import yaml

CONFIG_PATH = "configs/tests/synthetic_copy_test.yaml"
TRAIN_SCRIPT = "train.py"

# Sequence/classification models to test
SEQUENCE_MODELS = [
    "tfn_classifier",
    "enhanced_tfn_classifier",
    "transformer_classifier",
    "performer_classifier",
    "lstm_classifier",
    "cnn_classifier",
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
    # Data & synthetic sequence (standardized)
    "--data.seq_len": 8,
    "--data.vocab_size": 10,
    "--data.dataset_size": 10,
    # num_classes is set in the config, not as a flag
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
    for model_name in SEQUENCE_MODELS:
        print(f"\n=== Testing sequence model: {model_name} ===")
        cmd = [sys.executable, TRAIN_SCRIPT, "--config", CONFIG_PATH, "--model_name", model_name]
        cmd += build_flag_list()
        # Enhanced TFN requires diffusion evolution and valid head_dim
        if model_name == "enhanced_tfn_classifier":
            cmd += ["--model.evolution_type", "diffusion", "--model.embed_dim", "8", "--model.num_heads", "2"]
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