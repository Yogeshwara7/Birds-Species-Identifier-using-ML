import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import os

class CNNModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Use a model pre-trained on ImageNet-21k for better fine-grained classification
        self.model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224')
        self.processor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224')
        self.model.to(self.device)
        self.model.eval()

    def preprocess_image(self, image):
        if isinstance(image, Image.Image):
            # Ensure image is in RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt")
            return inputs.to(self.device)
        else:
            raise ValueError("Input must be a PIL Image")

    def predict(self, image):
        with torch.no_grad():
            inputs = self.preprocess_image(image)
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
            return predicted_class, confidence

    def train(self, train_loader, val_loader, num_epochs=10):
        self.model.train()
        # Use a lower learning rate for fine-tuning
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5, weight_decay=0.01)
        # Use label smoothing for better generalization
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        # Gradient accumulation steps
        accumulation_steps = 4  # Accumulate gradients for 4 steps
        optimizer.zero_grad()  # Zero out gradients at the start

        best_val_acc = 0.0
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(images).logits
                loss = criterion(outputs, labels)
                
                # Scale loss by accumulation steps
                loss = loss / accumulation_steps
                loss.backward()

                # Update weights after accumulation steps
                if (batch_idx + 1) % accumulation_steps == 0:
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images).logits
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

            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Training Loss: {running_loss/len(train_loader):.4f}, Training Acc: {train_acc:.2f}%')
            print(f'Validation Loss: {val_loss/len(val_loader):.4f}, Validation Acc: {val_acc:.2f}%')

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_weights("models/best_model_weights")

    def save_weights(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'processor': self.processor
        }, path)

    def load_weights(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved weights found at {path}")
        # Load checkpoint with map_location to handle CPU-only loading and weights_only=False
        checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        # Move model to the device specified during initialization (CPU or CUDA)
        self.model.to(self.device) 