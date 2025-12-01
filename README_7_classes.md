# Bird Species Classification - 7 Classes Setup

This project has been configured to train and test on only the **first 7 bird species** from the CUB-200-2011 dataset instead of all 200 classes.

## Classes Used

The 7 bird species are:
1. **Black_footed_Albatross** (Class 0)
2. **Laysan_Albatross** (Class 1) 
3. **Sooty_Albatross** (Class 2)
4. **Groove_billed_Ani** (Class 3)
5. **Crested_Auklet** (Class 4)
6. **Least_Auklet** (Class 5)
7. **Parakeet_Auklet** (Class 6)

## Dataset Statistics

- **Training samples**: 210 (30 per class)
- **Test samples**: 166 (varies per class: 11-30 samples)
- **Total images**: 376

## Quick Start

### 1. Test Data Loading
```bash
python test_data_loading.py
```

### 2. Train Models

Train all models:
```bash
python train_models.py --model all --epochs 5
```

Train specific models:
```bash
# Vision Transformer (ViT) - Best accuracy
python train_models.py --model vit --epochs 5 --batch_size 4

# Random Forest - Fastest
python train_models.py --model rf --batch_size 32

# Traditional CNN (EfficientNet-B3) - Memory-safe
python train_cnn_safe.py --batch_size 4 --epochs 10

# Lightweight CNN (EfficientNet-B0) - Fastest CNN
python train_lightweight_cnn.py --batch_size 8 --epochs 10
```

### 3. Test Inference
```bash
python test_inference.py
```

## Model Performance

Based on initial testing:

| Model | Validation Accuracy | Sample Test Accuracy |
|-------|-------------------|---------------------|
| **Vision Transformer (ViT)** | 90.96% | 100% (7/7) |
| **Random Forest** | 34.34% | 57.14% (4/7) |
| **Traditional CNN** | Not trained yet | - |

## Files Structure

```
├── models/
│   ├── vit_model.py              # Vision Transformer model
│   ├── traditional_cnn_model.py  # EfficientNet-B7 model
│   └── strong_random_forest_model.py  # Random Forest with HOG features
├── utils/
│   ├── data_loader.py            # Data loading utilities for 7 classes
│   └── preprocessing.py          # Image preprocessing utilities
├── train_models.py               # Training script
├── test_data_loading.py          # Test data loading functionality
├── test_inference.py             # Test trained models
└── app.py                        # Streamlit web application
```

## Model Details

### Vision Transformer (ViT)
- **Architecture**: ViT-Large (google/vit-large-patch16-224)
- **Input size**: 224×224 pixels
- **Pre-trained**: ImageNet-21k
- **Best performance**: 90.96% validation accuracy

### Random Forest
- **Features**: HOG (Histogram of Oriented Gradients)
- **Input size**: 96×96 pixels
- **Estimators**: 200 trees
- **Max depth**: 30

### Traditional CNN
- **Architecture**: EfficientNet-B3 (memory-optimized)
- **Input size**: 300×300 pixels
- **Pre-trained**: ImageNet-1k
- **Parameters**: ~12M (vs B7's 66M)
- **Memory**: Medium (optimized for Windows)

### Lightweight CNN
- **Architecture**: EfficientNet-B0
- **Input size**: 224×224 pixels
- **Pre-trained**: ImageNet-1k
- **Parameters**: ~5M
- **Memory**: Low (fastest training)

## Usage Tips

1. **Batch Size**: Use smaller batch sizes (4-8) for better memory management
2. **Training Time**: 
   - ViT: ~15 minutes for 2 epochs
   - CNN: ~10-20 minutes for 10 epochs
   - Random Forest: ~2-5 minutes
3. **Memory**: Models automatically use GPU if available, fallback to CPU
4. **Saved Models**: 
   - ViT: `models/best_model_weights/`
   - Random Forest: `models/random_forest_model.pkl`
   - Traditional CNN: `models/best_traditional_cnn_model.pth`
   - Lightweight CNN: `models/best_lightweight_cnn_model.pth`

## Memory Optimization

The CNN models have been optimized for Windows systems:
- Changed from EfficientNet-B7 (66M params, 528×528) to B3 (12M params, 300×300)
- Added GPU cache clearing during training
- Set `num_workers=0` to prevent Windows multiprocessing issues
- Added gradient clipping for stability

If you still encounter memory issues, see `TRAINING_GUIDE.md`

## Next Steps

To extend to more classes, modify the `num_classes` parameter in:
- `utils/data_loader.py` - `get_data_loaders()` function
- `train_models.py` - `--num_classes` argument
- Model initialization in each model file