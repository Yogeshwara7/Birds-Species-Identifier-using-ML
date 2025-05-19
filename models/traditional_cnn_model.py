import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os # Import os for save/load
from PIL import Image # Import PIL for image handling if predict takes PIL Image
import torchvision.transforms as transforms # Import transforms for prediction
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights

class TraditionalCNNModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        # Load pre-trained EfficientNet-B7
        self.model = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
        
        # Modify the classifier for our number of classes
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(num_features, num_classes)
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        return self.model(x)

    # Add a predict method that handles PIL Images
    def predict(self, image):
        self.eval() # Set model to evaluation mode
        
        # Define the same transforms as used for CNN training/evaluation for consistency
        transform = transforms.Compose([
            transforms.Resize(600),  # EfficientNet-B7 expects larger input
            transforms.CenterCrop(528),  # EfficientNet-B7's optimal input size
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        if isinstance(image, Image.Image):
            inputs = transform(image).unsqueeze(0).to(self.device)
        else:
            raise ValueError("Input must be a PIL Image")

        with torch.no_grad():
            outputs = self(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            return predicted_class, confidence

    # Add save and load weights methods that work with the model instance
    def save_weights(self, path="models/best_traditional_cnn_model.pth"):
        torch.save(self.state_dict(), path)
        print(f"EfficientNet-B7 model saved to {path}")

    def load_weights(self, path="models/best_traditional_cnn_model.pth"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved weights found at {path}")
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.eval()
        print(f"EfficientNet-B7 model loaded from {path}")

# Standalone training function as you provided
def train_model(model, train_loader, val_loader, device, epochs=5, lr=0.0001):
    model.to(device)
    
    # Use AdamW optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Use cosine learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Use label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Gradient accumulation steps to handle large model
    accumulation_steps = 4
    optimizer.zero_grad()

    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Scale loss by accumulation steps
            loss = loss / accumulation_steps
            loss.backward()
            
            # Update weights after accumulation steps
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            running_loss += loss.item() * accumulation_steps
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Clear cache periodically
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Clear cache periodically
                torch.cuda.empty_cache()
        
        # Update learning rate
        scheduler.step()
        
        # Calculate metrics
        train_acc = 100 * correct / total
        val_acc = 100 * val_correct / val_total
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Training Loss: {running_loss/len(train_loader):.4f}, Training Acc: {train_acc:.2f}%')
        print(f'Validation Loss: {val_loss/len(val_loader):.4f}, Validation Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_weights()
    
    print("EfficientNet-B7 training complete.") 