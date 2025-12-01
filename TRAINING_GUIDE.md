# CNN Training Guide - Fixing Crashes

## What Was Fixed

The original CNN training was crashing due to:
1. **EfficientNet-B7** - Too large (66M parameters, 528x528 images)
2. **High memory usage** - Large batch sizes with huge images
3. **Windows multiprocessing issues** - num_workers causing crashes

## Changes Made

### 1. Model Size Reduction
- Changed from **EfficientNet-B7** → **EfficientNet-B3**
- Reduced image size from **528x528** → **300x300**
- Still powerful but uses ~80% less memory

### 2. Memory Optimizations
- Added `torch.cuda.empty_cache()` to clear GPU memory periodically
- Added gradient clipping to prevent exploding gradients
- Reduced default batch size from 16 → 8 (or 4 for safety)
- Set `num_workers=0` to avoid Windows multiprocessing crashes

### 3. Training Improvements
- Added learning rate scheduler
- Added weight decay for regularization
- Added data augmentation (ColorJitter)

## How to Train

### Option 1: Safe CNN Training (Recommended)
```bash
python train_cnn_safe.py --batch_size 4 --epochs 10
```

### Option 2: Lightweight CNN (Fastest)
```bash
python train_lightweight_cnn.py --batch_size 8 --epochs 10
```

### Option 3: Original Training Script
```bash
python train_models.py --model cnn --batch_size 4 --epochs 10
```

## Troubleshooting

### Still Getting Out of Memory?
1. Reduce batch size to 2:
   ```bash
   python train_cnn_safe.py --batch_size 2
   ```

2. Use the lightweight model instead:
   ```bash
   python train_lightweight_cnn.py --batch_size 4
   ```

3. Use CPU only (slower but won't crash):
   ```bash
   set CUDA_VISIBLE_DEVICES=
   python train_cnn_safe.py --batch_size 8
   ```

### Training Too Slow?
- If you have more GPU memory, increase batch size:
  ```bash
  python train_cnn_safe.py --batch_size 8
  ```

### Want Better Accuracy?
- Train for more epochs:
  ```bash
  python train_cnn_safe.py --epochs 20
  ```

## Model Comparison

| Model | Parameters | Image Size | Memory | Speed | Accuracy |
|-------|-----------|------------|--------|-------|----------|
| EfficientNet-B0 (Lightweight) | 5M | 224x224 | Low | Fast | Good |
| EfficientNet-B3 (Traditional) | 12M | 300x300 | Medium | Medium | Better |
| EfficientNet-B7 (Original) | 66M | 528x528 | Very High | Slow | Best |

## Recommended Settings

### Low GPU Memory (< 4GB)
```bash
python train_lightweight_cnn.py --batch_size 4
```

### Medium GPU Memory (4-8GB)
```bash
python train_cnn_safe.py --batch_size 4
```

### High GPU Memory (> 8GB)
```bash
python train_cnn_safe.py --batch_size 8
```

### CPU Only
```bash
set CUDA_VISIBLE_DEVICES=
python train_lightweight_cnn.py --batch_size 8
```
