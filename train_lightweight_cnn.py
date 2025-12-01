#!/usr/bin/env python3
"""
Training script for lightweight CNN model to avoid memory crashes
"""

import os
import torch
import argparse
from utils.data_loader import get_data_loaders, get_class_names
from models.lightweight_cnn_model import LightweightCNNModel, train_lightweight_cnn

def main():
    parser = argparse.ArgumentParser(description='Train lightweight CNN on first 7 bird classes')
    parser.add_argument('--data_dir', type=str, default='data/CUB_200_2011', 
                        help='Path to CUB dataset directory')
    parser.add_argument('--num_classes', type=int, default=7,
                        help='Number of classes to use (default: 7)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8,  # Smaller batch size
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for training')
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} not found!")
        return
    
    # Print class information
    class_names = get_class_names(args.num_classes)
    print(f"Training Lightweight CNN on {args.num_classes} classes:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Check available memory
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("Using CPU (this will be slower)")
    
    print(f"\n=== Training Lightweight CNN (EfficientNet-B0) ===")
    print(f"Image size: 224x224 (much smaller than original 528x528)")
    print(f"Batch size: {args.batch_size}")
    
    try:
        # Get data loaders with smaller batch size
        train_loader, val_loader = get_data_loaders(
            args.data_dir,
            num_classes=args.num_classes,
            batch_size=args.batch_size,
            model_type='cnn'
        )
        
        # Initialize lightweight model
        model = LightweightCNNModel(num_classes=args.num_classes)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Model loaded on: {device}")
        
        # Train model
        train_lightweight_cnn(
            model, 
            train_loader, 
            val_loader, 
            device, 
            num_epochs=args.epochs,
            learning_rate=args.learning_rate
        )
        
        print("✅ Lightweight CNN training completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        print("\nTroubleshooting tips:")
        print("1. Try smaller batch size: --batch_size 4")
        print("2. Close other applications to free memory")
        print("3. Use CPU only by setting CUDA_VISIBLE_DEVICES=''")

if __name__ == "__main__":
    main()