#!/usr/bin/env python3
"""
Test script to verify data loading works correctly for first 7 classes
"""

import os
from utils.data_loader import get_data_loaders, get_class_names

def test_data_loading():
    """Test data loading functionality"""
    data_dir = 'data/CUB_200_2011'
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} not found!")
        return False
    
    print("Testing data loading for first 7 classes...")
    
    # Get class names
    class_names = get_class_names(7)
    print(f"\nClasses to be used:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")
    
    # Test data loaders for each model type
    model_types = ['vit', 'cnn', 'rf']
    
    for model_type in model_types:
        print(f"\n=== Testing {model_type.upper()} data loader ===")
        try:
            train_loader, val_loader = get_data_loaders(
                data_dir, 
                num_classes=7, 
                batch_size=8, 
                model_type=model_type
            )
            
            # Test one batch from train loader
            train_batch = next(iter(train_loader))
            images, labels = train_batch
            
            print(f"Train batch shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Label range: {labels.min().item()} - {labels.max().item()}")
            print(f"Unique labels in batch: {labels.unique().tolist()}")
            
            # Test one batch from val loader
            val_batch = next(iter(val_loader))
            images, labels = val_batch
            
            print(f"Val batch shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Label range: {labels.min().item()} - {labels.max().item()}")
            print(f"Unique labels in batch: {labels.unique().tolist()}")
            
        except Exception as e:
            print(f"Error testing {model_type} data loader: {e}")
            return False
    
    print("\n✅ All data loading tests passed!")
    return True

if __name__ == "__main__":
    test_data_loading()