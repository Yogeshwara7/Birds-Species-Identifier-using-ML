import cv2
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
from tqdm import tqdm

class StrongRandomForestModel:
    def __init__(self, n_classes=200):
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            class_weight='balanced_subsample',
            n_jobs=-1,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.n_classes = n_classes

    def extract_features(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Resize image
        image = cv2.resize(image, (96, 96))

        # HOG features
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hog = cv2.HOGDescriptor((96, 96), (16, 16), (8, 8), (8, 8), 9)
        hog_features = hog.compute(gray).flatten()

        # Color histogram features
        chans = cv2.split(image)
        hist_features = []
        for chan in chans:
            hist = cv2.calcHist([chan], [0], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            hist_features.extend(hist)

        return np.concatenate((hog_features, hist_features))

    def train(self, train_loader, val_loader=None):
        X, y = [], []
        for images, labels in tqdm(train_loader, desc="Extracting train features"):
            for img, label in zip(images, labels):
                features = self.extract_features(img)
                X.append(features)
                y.append(label.item())  # Convert tensor to int

        X = self.scaler.fit_transform(X)
        self.model.fit(X, y)
        self.is_fitted = True

        # Evaluate on validation set if provided
        if val_loader:
            val_predictions = []
            val_labels = []
            for images, labels in tqdm(val_loader, desc="Evaluating validation set"):
                for img, label in zip(images, labels):
                    pred, _ = self.predict(img)  # Get only the class prediction, ignore confidence
                    val_predictions.append(pred)
                    val_labels.append(label.item())

            acc = accuracy_score(val_labels, val_predictions)
            print(f"Validation Accuracy: {acc:.2%}")

    def predict(self, image):
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")

        features = self.extract_features(image)
        features = self.scaler.transform([features])
        probabilities = self.model.predict_proba(features)[0]
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        return predicted_class, confidence

    def evaluate(self, test_loader, label_names=None):
        predictions = []
        true_labels = []
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            for img, label in zip(images, labels):
                pred, _ = self.predict(img)  # Get only the class prediction, ignore confidence
                predictions.append(pred)
                true_labels.append(label.item())

        acc = accuracy_score(true_labels, predictions)
        print(f"Accuracy: {acc:.2%}")

        if label_names:
            print(classification_report(true_labels, predictions, target_names=label_names))

    def save_weights(self, path="models/best_rf_model.pkl"):
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)

    def load_weights(self, path="models/best_rf_model.pkl"):
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_fitted = True 