#!/usr/bin/env python3
"""
Test inference script for trained models on first 7 classes
"""

import os
from PIL import Image
from models.vit_model import CNNModel as ViTModel
from models.strong_random_forest_model import StrongRandomForestModel
from models.traditional_cnn_model import TraditionalCNNModel
from utils.data_loader import get_class_names

def test_model_inference(model_type='vit'):
    """Test model inference on a sample image"""
    
    # Get class names
    class_names = get_class_names(7)
    print(f"Testing {model_type.upper()} model inference...")
    print(f"Classes: {class_names}")
    
    # Load model
    if model_type == 'vit':
        model = ViTModel(num_classes=7)
        model_path = 'models/best_model_weights'
        if os.path.exists(model_path):
            model.load_weights(model_path)
            print("✅ Loaded ViT model weights")
        else:
            print("⚠️ No saved ViT weights found, using pre-trained model")
    
    elif model_type == 'rf':
        model = StrongRandomForestModel()
        model_path = 'models/random_forest_model.pkl'
        if os.path.exists(model_path):
            model.load_weights(model_path)
            print("✅ Loaded Random Forest model weights")
        else:
            print("❌ No saved Random Forest weights found")
            return
    
    elif model_type == 'cnn':
        model = TraditionalCNNModel(num_classes=7)
        model_path = 'models/best_traditional_cnn_model.pth'
        if os.path.exists(model_path):
            model.load_weights(model_path)
            print("✅ Loaded CNN model weights")
        else:
            print("⚠️ No saved CNN weights found, using pre-trained model")
    
    # Find a test image from the dataset
    test_image_paths = []
    data_dir = 'data/CUB_200_2011/images'
    
    # Get first image from each of the 7 classes
    for i in range(1, 8):  # Classes 1-7
        class_folder = f"{i:03d}.{class_names[i-1]}"
        class_path = os.path.join(data_dir, class_folder)
        if os.path.exists(class_path):
            images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                test_image_paths.append((os.path.join(class_path, images[0]), i-1, class_names[i-1]))
    
    if not test_image_paths:
        print("❌ No test images found")
        return
    
    print(f"\nTesting on {len(test_image_paths)} sample images...")
    
    correct = 0
    total = 0
    
    for image_path, true_label, true_class_name in test_image_paths:
        try:
            # Load and predict
            image = Image.open(image_path).convert('RGB')
            predicted_class, confidence = model.predict(image)
            predicted_class_name = class_names[predicted_class]
            
            # Check if correct
            is_correct = predicted_class == true_label
            if is_correct:
                correct += 1
            total += 1
            
            # Print result
            status = "✅" if is_correct else "❌"
            print(f"{status} True: {true_class_name} | Predicted: {predicted_class_name} | Confidence: {confidence:.3f}")
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
    
    accuracy = correct / total if total > 0 else 0
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.2%}")

def main():
    """Test all available models"""
    models_to_test = []
    
    # Check which models are available
    if os.path.exists('models/best_model_weights'):
        models_to_test.append('vit')
    if os.path.exists('models/random_forest_model.pkl'):
        models_to_test.append('rf')
    if os.path.exists('models/best_traditional_cnn_model.pth'):
        models_to_test.append('cnn')
    
    if not models_to_test:
        print("No trained models found. Please train a model first using train_models.py")
        return
    
    print(f"Found trained models: {models_to_test}")
    
    for model_type in models_to_test:
        print("\n" + "="*50)
        test_model_inference(model_type)
        print("="*50)

if __name__ == "__main__":
    main()