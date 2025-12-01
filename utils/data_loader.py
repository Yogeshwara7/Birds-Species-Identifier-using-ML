import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

class CUBDataset(Dataset):
    """CUB-200-2011 Dataset with filtering for specific classes"""
    
    def __init__(self, data_dir, num_classes=7, transform=None, train=True):
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.transform = transform
        self.train = train
        
        # Load dataset files
        self.images_df = pd.read_csv(
            os.path.join(data_dir, 'images.txt'), 
            sep=' ', 
            names=['img_id', 'filepath']
        )
        
        self.labels_df = pd.read_csv(
            os.path.join(data_dir, 'image_class_labels.txt'), 
            sep=' ', 
            names=['img_id', 'target']
        )
        
        self.train_test_df = pd.read_csv(
            os.path.join(data_dir, 'train_test_split.txt'), 
            sep=' ', 
            names=['img_id', 'is_training_img']
        )
        
        # Merge dataframes
        self.data = self.images_df.merge(self.labels_df, on='img_id')
        self.data = self.data.merge(self.train_test_df, on='img_id')
        
        # Filter for first num_classes only (classes 1-7)
        self.data = self.data[self.data['target'] <= num_classes]
        
        # Filter for train/test split
        if train:
            self.data = self.data[self.data['is_training_img'] == 1]
        else:
            self.data = self.data[self.data['is_training_img'] == 0]
        
        # Reset index
        self.data = self.data.reset_index(drop=True)
        
        # Adjust labels to be 0-indexed (0-6 instead of 1-7)
        self.data['target'] = self.data['target'] - 1
        
        print(f"Loaded {len(self.data)} {'training' if train else 'test'} samples for {num_classes} classes")
        print(f"Class distribution: {self.data['target'].value_counts().sort_index().to_dict()}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, 'images', self.data.iloc[idx]['filepath'])
        image = Image.open(img_path).convert('RGB')
        target = self.data.iloc[idx]['target']
        
        if self.transform:
            image = self.transform(image)
        
        return image, target

def get_data_loaders(data_dir, num_classes=7, batch_size=32, model_type='vit'):
    """
    Create data loaders for training and validation
    
    Args:
        data_dir (str): Path to CUB dataset directory
        num_classes (int): Number of classes to use (default: 7)
        batch_size (int): Batch size for data loaders
        model_type (str): Type of model ('vit', 'cnn', or 'rf')
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    
    if model_type == 'vit':
        # ViT preprocessing - will be handled by ViTImageProcessor
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    elif model_type == 'cnn':
        # CNN preprocessing with memory-efficient image sizes
        train_transform = transforms.Compose([
            transforms.Resize(320),
            transforms.RandomCrop(300),  # Reduced from 528 to save memory
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(300),  # Reduced from 528 to save memory
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:  # Random Forest
        # Minimal preprocessing for RF - will extract features later
        transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
        ])
    
    # Create datasets
    if model_type == 'cnn':
        train_dataset = CUBDataset(data_dir, num_classes, train_transform, train=True)
        val_dataset = CUBDataset(data_dir, num_classes, val_transform, train=False)
    else:
        train_dataset = CUBDataset(data_dir, num_classes, transform, train=True)
        val_dataset = CUBDataset(data_dir, num_classes, transform, train=False)
    
    # Create data loaders (num_workers=0 for Windows to avoid crashes)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0  # Set to 0 on Windows to prevent multiprocessing issues
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0  # Set to 0 on Windows to prevent multiprocessing issues
    )
    
    return train_loader, val_loader

def get_class_names(num_classes=7):
    """Get the class names for the first num_classes"""
    class_names = [
        "Black_footed_Albatross",
        "Laysan_Albatross", 
        "Sooty_Albatross",
        "Groove_billed_Ani",
        "Crested_Auklet",
        "Least_Auklet",
        "Parakeet_Auklet"
    ]
    return class_names[:num_classes]