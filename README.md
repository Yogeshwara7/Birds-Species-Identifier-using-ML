# 🐦 Bird Species Identification System

An image-based bird species identification system using multiple machine learning models including Vision Transformer (ViT), EfficientNet CNN, and Random Forest for accurate species classification.

## Features

- 🎯 Upload bird images for instant species identification
- 🤖 Multiple model options: ViT, EfficientNet-B3 CNN, and Random Forest
- 📊 High accuracy predictions with confidence scores
- 🖥️ User-friendly Streamlit web interface
- 🐦 Support for 7 bird species (expandable to 200)
- 🔧 Model training and fine-tuning capability
- 💾 Memory-optimized for Windows systems

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/Bird-Species-Identifier.git
cd Bird-Species-Identifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the CUB-200-2011 dataset:
   - Visit [CUB-200-2011 Dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
   - Download and extract the dataset
   - Place the extracted contents in the `data/CUB_200_2011` directory

## Supported Bird Species (7 Classes)

The system currently identifies these 7 bird species:
1. **Black footed Albatross**
2. **Laysan Albatross**
3. **Sooty Albatross**
4. **Groove billed Ani**
5. **Crested Auklet**
6. **Least Auklet**
7. **Parakeet Auklet**

*Note: The system can be expanded to all 200 species in the CUB-200-2011 dataset by modifying the `num_classes` parameter.*

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app** (uses pre-trained models):
   ```bash
   streamlit run app_7_classes.py
   ```
   Or on Windows:
   ```bash
   run_app.bat
   ```

3. **Upload a bird image** and get instant predictions!

4. **Train your own models** (optional):
   ```bash
   python train_cnn_safe.py --batch_size 4 --epochs 10
   ```

## Usage

### Running the Application

**Option 1: Using the batch file (Windows)**
```bash
run_app.bat
```

**Option 2: Direct command**
```bash
streamlit run app_7_classes.py
```

The app will open in your browser at `http://localhost:8501`

### Training Models

**Train all models:**
```bash
python train_models.py --model all --epochs 10 --batch_size 4
```

**Train specific models:**

1. **Vision Transformer (ViT)** - Best accuracy (~91%)
```bash
python train_models.py --model vit --epochs 5 --batch_size 4
```

2. **Traditional CNN (EfficientNet-B3)** - Memory-optimized
```bash
python train_cnn_safe.py --batch_size 4 --epochs 10
```

3. **Lightweight CNN (EfficientNet-B0)** - Fastest training
```bash
python train_lightweight_cnn.py --batch_size 8 --epochs 10
```

4. **Random Forest** - Fast inference
```bash
python train_models.py --model rf --batch_size 32
```

### Testing

Test data loading:
```bash
python test_data_loading.py
```

Test model inference:
```bash
python test_inference.py
```

## Project Structure

```
Bird-Species-Identifier/
├── app_7_classes.py                    # Main Streamlit application (7 classes)
├── app.py                              # Alternative Streamlit app
├── train_models.py                     # Universal training script
├── train_cnn_safe.py                   # Memory-safe CNN training
├── train_lightweight_cnn.py            # Lightweight CNN training
├── test_data_loading.py                # Test data loading
├── test_inference.py                   # Test model predictions
├── run_app.bat                         # Windows batch file to run app
├── models/
│   ├── vit_model.py                    # Vision Transformer implementation
│   ├── traditional_cnn_model.py        # EfficientNet-B3 CNN
│   ├── lightweight_cnn_model.py        # EfficientNet-B0 CNN
│   ├── strong_random_forest_model.py   # Random Forest with HOG features
│   ├── best_model_weights/             # Trained ViT weights
│   ├── best_traditional_cnn_model.pth  # Trained CNN weights
│   └── random_forest_model.pkl         # Trained RF weights
├── utils/
│   ├── data_loader.py                  # Dataset loading utilities
│   └── preprocessing.py                # Image preprocessing
├── data/
│   └── CUB_200_2011/                   # CUB dataset directory
├── requirements.txt                     # Project dependencies
├── README.md                           # Main documentation
├── README_7_classes.md                 # 7-class setup guide
└── TRAINING_GUIDE.md                   # CNN training troubleshooting
```

## Model Details

### 1. Vision Transformer (ViT) - Best Accuracy
- **Architecture**: ViT-Large (google/vit-large-patch16-224)
- **Pre-trained on**: ImageNet-21k
- **Input Size**: 224×224 pixels
- **Parameters**: ~304M
- **Validation Accuracy**: ~91%
- **Best for**: Highest accuracy predictions

### 2. Traditional CNN (EfficientNet-B3) - Balanced
- **Architecture**: EfficientNet-B3
- **Pre-trained on**: ImageNet-1k
- **Input Size**: 300×300 pixels
- **Parameters**: ~12M
- **Memory**: Medium (optimized for Windows)
- **Best for**: Good accuracy with reasonable memory usage

### 3. Lightweight CNN (EfficientNet-B0) - Fast
- **Architecture**: EfficientNet-B0
- **Pre-trained on**: ImageNet-1k
- **Input Size**: 224×224 pixels
- **Parameters**: ~5M
- **Memory**: Low
- **Best for**: Fast training and inference on limited hardware

### 4. Random Forest - Traditional ML
- **Features**: HOG (Histogram of Oriented Gradients)
- **Input Size**: 96×96 pixels
- **Estimators**: 200 trees
- **Validation Accuracy**: ~34%
- **Best for**: Fast inference, interpretable results

## Model Performance Comparison

| Model | Accuracy | Parameters | Memory | Training Time | Inference Speed |
|-------|----------|------------|--------|---------------|-----------------|
| **ViT-Large** | 91% | 304M | High | Slow | Medium |
| **EfficientNet-B3** | TBD | 12M | Medium | Medium | Fast |
| **EfficientNet-B0** | TBD | 5M | Low | Fast | Very Fast |
| **Random Forest** | 34% | N/A | Very Low | Very Fast | Very Fast |

## Requirements

- Python 3.8 or higher
- PyTorch 2.0+
- CUDA-capable GPU (recommended for training, but CPU works)
- 8GB+ RAM (4GB minimum for lightweight models)
- 10GB+ free disk space

### Key Dependencies
- `torch>=2.0.0` - Deep learning framework
- `torchvision>=0.15.0` - Computer vision utilities
- `transformers>=4.30.0` - Hugging Face transformers
- `streamlit>=1.24.0` - Web interface
- `scikit-learn>=1.2.0` - Random Forest model
- `opencv-python>=4.7.0` - Image processing
- `Pillow>=9.5.0` - Image handling

## Troubleshooting

### Out of Memory Errors
If you encounter GPU memory errors during training:

1. **Reduce batch size**:
   ```bash
   python train_cnn_safe.py --batch_size 2
   ```

2. **Use lightweight model**:
   ```bash
   python train_lightweight_cnn.py --batch_size 4
   ```

3. **Use CPU only** (slower but stable):
   ```bash
   set CUDA_VISIBLE_DEVICES=
   python train_cnn_safe.py --batch_size 8
   ```

See `TRAINING_GUIDE.md` for detailed troubleshooting.

### Windows Multiprocessing Issues
The data loaders are configured with `num_workers=0` to prevent Windows multiprocessing crashes.

## Dataset Statistics

- **Total Classes**: 7 (expandable to 200)
- **Training Samples**: 210 (30 per class)
- **Test Samples**: 166 (varies per class)
- **Total Images**: 376

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 