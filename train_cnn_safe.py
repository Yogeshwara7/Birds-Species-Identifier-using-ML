#!/usr/bin/env python3
"""
Safe CNN training script with memory optimizations for Windows
"""

import os
import torch
import argparse
from utils.data_loader import get_data_loaders, get_class_names
from models.traditional_cnn_model import TraditionalCNNModel, train_model

def main():
    parser = argparse.ArgumentParser(description='Train CNN safely on Windows')
    parser.add_argument('--data_dir', type=str, default='data/CUB_200_2011', 
                        help='Path to CUB dataset directory')
    parser.add_argument('--num_classes', type=int, default=7,
                        help='Number of classes to use (default: 7)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (use 4 for safety, 8 if you have more RAM)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for training')
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} not found!")
        return
    
    # Print class information
    class_names = get_class_names(args.num_classes)
    print(f"Training CNN on {args.num_classes} classes:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Check device and memory
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {total_memory:.1f} GB")
        
        if total_memory < 4:
            print("⚠️  Warning: Low GPU memory. Using batch_size=2")
            args.batch_size = 2
    else:
        print("Using CPU (training will be slower)")
    
    print(f"\n=== Training Configuration ===")
    print(f"Model: EfficientNet-B3")
    print(f"Image size: 300x300")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    
    try:
        # Get data loaders
        print("\nLoading data...")
        train_loader, val_loader = get_data_loaders(
            args.data_dir,
            num_classes=args.num_classes,
            batch_size=args.batch_size,
            model_type='cnn'
        )
        
        # Initialize model
        print("\nInitializing model...")
        model = TraditionalCNNModel(num_classes=args.num_classes)
        
        # Train model
        print("\nStarting training...")
        train_model(
            model, 
            train_loader, 
            val_loader, 
            device, 
            num_epochs=args.epochs,
            learning_rate=args.learning_rate
        )
        
        print("\n✅ CNN training completed successfully!")
        print(f"Best model saved to: models/best_traditional_cnn_model.pth")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n❌ GPU Out of Memory Error!")
            print("\nTry these solutions:")
            print("1. Reduce batch size: --batch_size 2")
            print("2. Use lightweight model: python train_lightweight_cnn.py --batch_size 4")
            print("3. Close other applications")
            print("4. Use CPU only: set CUDA_VISIBLE_DEVICES= before running")
        else:
            print(f"\n❌ Runtime Error: {e}")
            raise
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise

if __name__ == "__main__":
    main()
