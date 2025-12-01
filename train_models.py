#!/usr/bin/env python3
"""
Training script for bird classification models using first 7 classes only
"""

import os
import torch
import argparse
from utils.data_loader import get_data_loaders, get_class_names
from models.vit_model import CNNModel as ViTModel
from models.traditional_cnn_model import TraditionalCNNModel, train_model as train_cnn
from models.strong_random_forest_model import StrongRandomForestModel

def train_vit_model(data_dir, num_classes=7, epochs=5, batch_size=16, learning_rate=2e-5):
    """Train Vision Transformer model"""
    print(f"\n=== Training ViT Model on {num_classes} classes ===")
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(
        data_dir, num_classes=num_classes, batch_size=batch_size, model_type='vit'
    )
    
    # Initialize model
    model = ViTModel(num_classes=num_classes)
    
    # Train model
    model.train(train_loader, val_loader, num_epochs=epochs, learning_rate=learning_rate)
    
    print("ViT training completed!")
    return model

def train_traditional_cnn(data_dir, num_classes=7, epochs=10, batch_size=8, learning_rate=1e-4):
    """Train Traditional CNN (EfficientNet) model"""
    print(f"\n=== Training Traditional CNN Model on {num_classes} classes ===")
    
    # Use smaller batch size to prevent memory crashes
    print(f"Using batch size: {batch_size} (reduced to prevent memory issues)")
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(
        data_dir, num_classes=num_classes, batch_size=batch_size, model_type='cnn'
    )
    
    # Initialize model
    model = TraditionalCNNModel(num_classes=num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Model: EfficientNet-B3 (memory-efficient)")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Train model
    train_cnn(model, train_loader, val_loader, device, num_epochs=epochs, learning_rate=learning_rate)
    
    print("Traditional CNN training completed!")
    return model

def train_random_forest(data_dir, num_classes=7, batch_size=32):
    """Train Random Forest model"""
    print(f"\n=== Training Random Forest Model on {num_classes} classes ===")
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(
        data_dir, num_classes=num_classes, batch_size=batch_size, model_type='rf'
    )
    
    # Initialize model
    model = StrongRandomForestModel()
    
    # Train model
    model.train(train_loader, val_loader)
    
    print("Random Forest training completed!")
    return model

def main():
    parser = argparse.ArgumentParser(description='Train bird classification models on first 7 classes')
    parser.add_argument('--data_dir', type=str, default='data/CUB_200_2011', 
                        help='Path to CUB dataset directory')
    parser.add_argument('--model', type=str, choices=['vit', 'cnn', 'rf', 'all'], default='all',
                        help='Model to train: vit, cnn, rf, or all')
    parser.add_argument('--num_classes', type=int, default=7,
                        help='Number of classes to use (default: 7)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs for deep learning models')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate for training')
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} not found!")
        return
    
    # Print class information
    class_names = get_class_names(args.num_classes)
    print(f"Training on {args.num_classes} classes:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Train selected models
    if args.model in ['vit', 'all']:
        try:
            train_vit_model(
                args.data_dir, 
                num_classes=args.num_classes,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate
            )
        except Exception as e:
            print(f"Error training ViT model: {e}")
    
    if args.model in ['cnn', 'all']:
        try:
            train_traditional_cnn(
                args.data_dir,
                num_classes=args.num_classes,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=1e-4  # Different LR for CNN
            )
        except Exception as e:
            print(f"Error training CNN model: {e}")
    
    if args.model in ['rf', 'all']:
        try:
            train_random_forest(
                args.data_dir,
                num_classes=args.num_classes,
                batch_size=32  # Larger batch size for RF
            )
        except Exception as e:
            print(f"Error training Random Forest model: {e}")
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main()