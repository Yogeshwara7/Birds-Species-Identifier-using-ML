import torch
import numpy as np
from PIL import Image
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score

class RandomForestModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RandomForestClassifier(n_estimators=200, max_depth=None, class_weight='balanced', n_jobs=-1)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def extract_hog_features(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Use the image size you specified
        resized = cv2.resize(gray, (96, 96))
        # Use the HOG parameters you specified
        hog = cv2.HOGDescriptor((96, 96), (32, 32), (16, 16), (16, 16), 9)
        features = hog.compute(resized)
        return features.flatten()

    def preprocess_image(self, image):
        if isinstance(image, Image.Image):
            # Ensure image is in RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')
            features = self.extract_hog_features(image)
            return features
        else:
            raise ValueError("Input must be a PIL Image")

    def predict(self, image):
        if not self.is_fitted:
            raise ValueError("Model needs to be trained before making predictions")
        
        features = self.preprocess_image(image)
        features = self.scaler.transform([features])
        
        # Return predicted class and confidence (max probability)
        probabilities = self.model.predict_proba(features)[0]
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        
        return predicted_class, confidence

    def train(self, train_loader, val_loader=None):
        print("Extracting features from training images for Random Forest...")
        X_train = []
        y_train = []
        
        # Iterate through the DataLoader
        for images, labels in tqdm(train_loader, desc="Processing training images for RF"):
            # Convert torch tensors to numpy arrays and then to PIL Images
            # Assuming images are in CxHxW format from DataLoader
            for img_tensor, label_tensor in zip(images, labels):
                # Convert tensor to numpy, then transpose to HxWx3 if necessary, and scale if needed
                img_np = img_tensor.permute(1, 2, 0).numpy() # Change from CxHxW to HxWxC for cv2
                # Assuming the input tensors are in float [0, 1], convert to uint8 [0, 255] for cv2
                img_np = (img_np * 255).astype(np.uint8)
                
                img_pil = Image.fromarray(img_np)
                
                features = self.extract_hog_features(img_pil)
                X_train.append(features)
                y_train.append(label_tensor.item())
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        print("Scaling features and training Random Forest model...")
        X_train = self.scaler.fit_transform(X_train)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        print("Random Forest training complete.")

        # Evaluate if val_loader is provided
        if val_loader is not None:
            self.evaluate(val_loader)

    def evaluate(self, val_loader):
        if not self.is_fitted:
            print("Model not fitted, skipping evaluation.")
            return

        print("Evaluating Random Forest model...")
        X_test = []
        y_test = []
        
        for images, labels in tqdm(val_loader, desc="Processing validation images for RF"):
            for img_tensor, label_tensor in zip(images, labels):
                 img_np = img_tensor.permute(1, 2, 0).numpy()
                 img_np = (img_np * 255).astype(np.uint8)
                 img_pil = Image.fromarray(img_np)
                 
                 features = self.extract_hog_features(img_pil)
                 X_test.append(features)
                 y_test.append(label_tensor.item())

        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        X_test = self.scaler.transform(X_test)
        preds = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        print(f"Validation Accuracy: {accuracy:.2%}")
        return accuracy

    def save_weights(self, path="models/best_rf_model.pkl"):
        if not self.is_fitted:
            raise ValueError("Model needs to be trained before saving")
        # Save both model and scaler
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)
        print(f"Random Forest model saved to {path}")

    def load_weights(self, path="models/best_rf_model.pkl"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved weights found at {path}")
        
        # Load both model and scaler
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_fitted = True
        print(f"Random Forest model loaded from {path}") 