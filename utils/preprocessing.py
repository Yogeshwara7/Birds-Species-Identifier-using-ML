import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def preprocess_image(image, model_type="cnn"):
    """
    Preprocess the input image based on the selected model type.
    
    Args:
        image (numpy.ndarray): Input image
        model_type (str): Either "cnn" or "rf"
    
    Returns:
        numpy.ndarray: Preprocessed image
    """
    if model_type == "cnn":
        # CNN preprocessing
        # Resize to 224x224 (standard input size for many CNN architectures)
        resized = cv2.resize(image, (224, 224))
        # Convert to RGB if grayscale
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        # Normalize pixel values
        normalized = resized.astype(np.float32) / 255.0
        return normalized
    
    else:  # Random Forest preprocessing
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        # Resize to 80x80
        resized = cv2.resize(gray, (80, 80))
        # Normalize pixel values
        normalized = resized.astype(np.float32) / 255.0
        return normalized

def extract_features(image):
    """
    Extract handcrafted features for Random Forest model.
    
    Args:
        image (numpy.ndarray): Preprocessed grayscale image
    
    Returns:
        numpy.ndarray: Extracted features
    """
    features = []
    
    # 1. Color Histogram (8 bins)
    hist = cv2.calcHist([image], [0], None, [8], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    features.extend(hist)
    
    # 2. Texture Features (GLCM)
    # Convert to uint8 for GLCM
    image_uint8 = (image * 255).astype(np.uint8)
    glcm = graycomatrix(image_uint8, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True)
    
    # Calculate GLCM properties
    contrast = graycoprops(glcm, 'contrast').flatten()
    dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    
    features.extend(contrast)
    features.extend(dissimilarity)
    features.extend(homogeneity)
    features.extend(energy)
    features.extend(correlation)
    
    # 3. Shape Features
    # Edge detection
    edges = cv2.Canny(image_uint8, 100, 200)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    features.append(edge_density)
    
    # 4. Basic Statistics
    mean = np.mean(image)
    std = np.std(image)
    features.extend([mean, std])
    
    return np.array(features) 