# 🐦 Bird Species Identification System

An image-based bird species identification system that uses a state-of-the-art Vision Transformer (ViT) model for accurate species classification.

## Features

- Upload bird images for species identification
- State-of-the-art Vision Transformer model
- High accuracy predictions with confidence scores
- User-friendly web interface
- Support for 200 bird species
- Model fine-tuning capability

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
   - Place the extracted contents in the `data` directory

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided URL (usually http://localhost:8501)

3. Upload a bird image to get the species prediction

4. (Optional) Use the "Fine-tune Model" button in the sidebar to improve the model's accuracy

## Project Structure

```
Bird-Species-Identifier/
├── app.py                 # Main Streamlit application
├── models/
│   └── cnn_model.py      # Vision Transformer model implementation
├── utils/
│   └── dataset.py        # Dataset handling and preprocessing
├── data/                 # Dataset directory
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Model Details

### Vision Transformer (ViT)
- Architecture: ViT-Large (pre-trained on ImageNet-21k)
- Input: RGB images (224x224)
- Features:
  - State-of-the-art accuracy
  - Robust to image variations
  - Fine-tuning capability
  - Confidence score predictions

## Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM
- 10GB+ free disk space

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 