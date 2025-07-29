# Device Flags Guide for Token Field Network

## 🚀 Overview

Both `train.py` and `hyperparameter_search.py` now support a `--device` flag that allows you to explicitly specify which device to use for training.

## 📋 Available Device Options

### **1. `--device cuda`**
- Forces the use of CUDA/GPU
- Falls back to CPU if CUDA is not available
- Provides GPU information if available

### **2. `--device cpu`**
- Forces the use of CPU
- Useful for debugging or when you want to ensure CPU usage

### **3. `--device auto`**
- Auto-detects the best available device
- Uses CUDA if available, otherwise CPU
- Same as not specifying the flag

## 🎯 Usage Examples

### **Training Script (`train.py`)**

```bash
# Force CUDA
python train.py --device cuda --model_name tfn_classifier --epochs 10

# Force CPU
python train.py --device cpu --model_name tfn_classifier --epochs 10

# Auto-detect (default)
python train.py --device auto --model_name tfn_classifier --epochs 10
# or simply:
python train.py --model_name tfn_classifier --epochs 10
```

### **Hyperparameter Search (`hyperparameter_search.py`)**

```bash
# Force CUDA for hyperparameter search
python hyperparameter_search.py --device cuda --models tfn_classifier --param_sweep "embed_dim:64,128"

# Force CPU for hyperparameter search
python hyperparameter_search.py --device cpu --models tfn_classifier --param_sweep "embed_dim:64,128"

# Auto-detect for hyperparameter search
python hyperparameter_search.py --device auto --models tfn_classifier --param_sweep "embed_dim:64,128"
```

## 🔧 Advanced Usage

### **Combined with Other Flags**

```bash
# CUDA training with specific parameters
python train.py \
  --device cuda \
  --model_name tfn_classifier \
  --epochs 20 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --model.embed_dim 256 \
  --model.kernel_type rbf \
  --model.evolution_type cnn

# CUDA hyperparameter search with comprehensive sweep
python hyperparameter_search.py \
  --device cuda \
  --models tfn_classifier enhanced_tfn_classifier \
  --param_sweep "embed_dim:128,256 num_layers:2,4 kernel_type:rbf,compact evolution_type:cnn,diffusion learning_rate:1e-3,1e-4" \
  --output_dir ./cuda_search_results \
  --epochs 15 \
  --patience 5
```

### **Quick Testing Commands**

```bash
# Quick CUDA test (1 epoch, small batch)
python train.py --device cuda --model_name tfn_classifier --epochs 1 --batch_size 4

# Quick CPU test
python train.py --device cpu --model_name tfn_classifier --epochs 1 --batch_size 4

# Quick hyperparameter search test
python hyperparameter_search.py --device cuda --models tfn_classifier --param_sweep "embed_dim:64" --epochs 2 --patience 1
```

## 📊 Device Information Display

When you use the `--device` flag, the system will display:

```
🔧 Using device: cuda
   GPU: NVIDIA GeForce RTX 3080
   Memory: 10.0 GB
```

Or if CUDA is not available:

```
🔧 Using device: cuda
   ⚠️  CUDA requested but not available - falling back to CPU
   Using CPU instead
```

## ⚠️ Important Notes

### **CUDA Requirements**
- PyTorch must be compiled with CUDA support
- NVIDIA GPU drivers must be installed
- If CUDA is requested but not available, the system automatically falls back to CPU

### **Performance Considerations**
- **CUDA**: Best for large models and datasets, significantly faster training
- **CPU**: Good for debugging, small experiments, or when GPU memory is limited
- **Auto**: Automatically chooses the best available option

### **Memory Management**
- CUDA training uses more memory but is much faster
- CPU training uses less memory but is slower
- Monitor GPU memory usage with `nvidia-smi` when using CUDA

## 🧪 Testing Device Flags

Run the test script to verify device flags work correctly:

```bash
python test_device_flags.py
```

This will test:
- Device detection logic
- CUDA device flag in train.py
- CUDA device flag in hyperparameter_search.py
- CPU device flag in train.py
- Fallback behavior when CUDA is not available

## 🔍 Troubleshooting

### **CUDA Not Available**
If you see:
```
⚠️  CUDA requested but not available - falling back to CPU
```

This means:
1. PyTorch was not compiled with CUDA support
2. NVIDIA drivers are not installed
3. No NVIDIA GPU is present

**Solutions:**
- Install PyTorch with CUDA: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
- Install NVIDIA drivers
- Use CPU instead: `--device cpu`

### **Out of Memory Errors**
If you get CUDA out of memory errors:
- Reduce batch size: `--batch_size 16` or `--batch_size 8`
- Use CPU: `--device cpu`
- Reduce model size: `--model.embed_dim 128`

## 📈 Performance Comparison

| Device | Training Speed | Memory Usage | Best For |
|--------|----------------|--------------|----------|
| CUDA   | Very Fast      | High         | Production training, large models |
| CPU    | Slow           | Low          | Debugging, small experiments |
| Auto   | Optimal        | Adaptive     | General use |

## 🎯 Quick Reference

```bash
# Basic training with auto device detection
python train.py --model_name tfn_classifier

# Force CUDA for faster training
python train.py --device cuda --model_name tfn_classifier

# Force CPU for debugging
python train.py --device cpu --model_name tfn_classifier

# Hyperparameter search with CUDA
python hyperparameter_search.py --device cuda --models tfn_classifier --param_sweep "embed_dim:64,128"

# Quick test commands
python train.py --device cuda --model_name tfn_classifier --epochs 1 --batch_size 4
python hyperparameter_search.py --device cuda --models tfn_classifier --param_sweep "embed_dim:64" --epochs 2
``` 