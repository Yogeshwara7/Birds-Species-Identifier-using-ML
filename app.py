import streamlit as st
import cv2
import numpy as np
from PIL import Image
from models.vit_model import CNNModel # Import only the CNNModel class (which is the ViT)
from models.strong_random_forest_model import StrongRandomForestModel
from models.traditional_cnn_model import TraditionalCNNModel, train_model as train_traditional_cnn # Keep commented out/adjust if using Traditional CNN
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ViTImageProcessor # Keep import for ViT
import torchvision.transforms as transforms

# Set memory optimization settings
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Global processor used by the ViT model
PROCESSOR = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224')

class CUBDataset(Dataset):
    def __init__(self, root_dir, is_training=True, selected_classes=list(range(10)), model_type="Vision Transformer (ViT)"):
        self.root_dir = root_dir
        self.is_training = is_training
        self.selected_classes = selected_classes
        self.model_type = model_type
        
        # Load class names
        self.class_names = []
        with open(os.path.join(root_dir, 'classes.txt'), 'r') as f:
            for line in f:
                class_id, class_name = line.strip().split(' ', 1)
                self.class_names.append(class_name)
        
        # If using a subset, build a mapping from original class index to new index
        if selected_classes is not None:
            self.selected_classes = set(selected_classes)
            self.class_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(selected_classes)}
        else:
            self.selected_classes = None
        
        # Load image paths and labels
        self.image_paths = []
        self.labels = []
        self.original_indices = []  # Store original indices
        
        # Read image paths
        with open(os.path.join(root_dir, 'images.txt'), 'r') as f:
            for line in f:
                image_id, image_path = line.strip().split(' ', 1)
                self.image_paths.append(os.path.join(root_dir, 'images', image_path))
        
        # Read labels
        with open(os.path.join(root_dir, 'image_class_labels.txt'), 'r') as f:
            for line in f:
                image_id, label = line.strip().split(' ', 1)
                self.labels.append(int(label) - 1)
        
        # Read train/test split
        self.train_indices = []
        self.test_indices = []
        with open(os.path.join(root_dir, 'train_test_split.txt'), 'r') as f:
            for i, line in enumerate(f):
                image_id, is_train = line.strip().split(' ', 1)
                if int(is_train) == 1:
                    self.train_indices.append(i)
                else:
                    self.test_indices.append(i)
        
        # Filter for selected classes and split
        if self.selected_classes is not None:
            # First filter by train/test split
            split_indices = self.train_indices if is_training else self.test_indices
            filtered_data = [(i, p, l) for i, (p, l) in enumerate(zip(self.image_paths, self.labels))
                           if i in split_indices and l in self.selected_classes]
            
            # Unzip the filtered data
            # Handle case where filtered_data might be empty
            if filtered_data:
                 self.original_indices, self.image_paths, self.labels = zip(*filtered_data)
                 # Remap labels to 0...N-1
                 self.labels = [self.class_map[l] for l in self.labels]
            else:
                 self.original_indices = []
                 self.image_paths = []
                 self.labels = []

        else:
            # Just use the split indices
            split_indices = self.train_indices if is_training else self.test_indices
            self.original_indices = split_indices
            self.image_paths = [self.image_paths[i] for i in split_indices]
            self.labels = [self.labels[i] for i in split_indices]
        
        # Define data augmentation transforms
        # Transforms for ViT
        self.train_transform_vit = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ])
        self.val_transform_vit = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])

        # Transforms for CNN (Traditional CNN)
        self.train_transform_cnn = transforms.Compose([
             transforms.RandomResizedCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.val_transform_cnn = transforms.Compose([
             transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        # Transforms for Random Forest (No tensor transforms needed here, return PIL Image)
        self.train_transform_rf = transforms.Compose([
            transforms.RandomResizedCrop(96),  # Match the size expected by HOG
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])
        self.val_transform_rf = transforms.Compose([
            transforms.Resize(96),  # Match the size expected by HOG
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transform based on model type and training state
        if self.model_type == "Vision Transformer (ViT)":
             current_transform = self.train_transform_vit if self.is_training else self.val_transform_vit
             # For ViT, also need to use the processor
             image = current_transform(image)
             # Assuming PROCESSOR is globally available
             inputs = PROCESSOR(images=image, return_tensors="pt")
             return inputs['pixel_values'].squeeze(0), torch.tensor(label)
        elif self.model_type == "Random Forest":
             # For Random Forest, apply transforms but keep as PIL Image
             current_transform = self.train_transform_rf if self.is_training else self.val_transform_rf
             image = current_transform(image)
             return image, torch.tensor(label)  # Return PIL Image and tensor label
        elif self.model_type == "Traditional CNN": # If Traditional CNN is active
             current_transform = self.train_transform_cnn if self.is_training else self.val_transform_cnn
             return current_transform(image), torch.tensor(label) # Return tensor for CNN
        else: # Fallback (should not happen with defined model types)
             # Default to ViT transforms if model_type is unexpected
             current_transform = self.train_transform_vit if self.is_training else self.val_transform_vit
             image = current_transform(image)
             inputs = PROCESSOR(images=image, return_tensors="pt")
             return inputs['pixel_values'].squeeze(0), torch.tensor(label)

    def get_class_name(self, class_idx):
        raw = self.class_names[class_idx]
        if '.' in raw:
            return raw.split('.', 1)[1]
        return raw

    # Modify get_data_loaders to pass model_type to Dataset
    def get_data_loaders(self, batch_size=8, train_split=0.8, selected_classes=None, model_type="Vision Transformer (ViT)"):
        # Pass model_type to the Dataset constructor
        train_dataset = CUBDataset(self.root_dir, is_training=True, selected_classes=selected_classes, model_type=model_type)
        val_dataset = CUBDataset(self.root_dir, is_training=False, selected_classes=selected_classes, model_type=model_type)
        
        # Adjust num_workers for Windows (usually 0 for single process)
        num_workers = 0
        
        # Custom collate function for Random Forest
        def rf_collate_fn(batch):
            images = [item[0] for item in batch]
            labels = torch.stack([item[1] for item in batch])
            return images, labels
        
        # Use different collate functions based on model type
        collate_fn = rf_collate_fn if model_type == "Random Forest" else None
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        return train_loader, val_loader

# Set page config
st.set_page_config(
    page_title="Bird Species Identifier",
    page_icon="🐦",
    layout="wide"
)

# Title and description
st.title("🐦 Bird Species Identifier")
st.markdown("""
This application helps you identify bird species from images using machine learning models.
Upload an image of a bird to get started!
""")

# Create necessary directories
os.makedirs("models", exist_ok=True)

# Model selection
model_type = st.sidebar.radio(
    "Select Model Type",
    ["Random Forest", "EfficientNet-B7"],
    help="Select the model for bird species identification. Random Forest here uses the Vision Transformer architecture."
)

# Initialize dataset
@st.cache_resource
def load_dataset():
    try:
        # Use only the first 7 classes for both training and inference
        selected_classes = list(range(7))
        # Initializing the dataset here without model_type is fine. model_type is passed in get_data_loaders.
        dataset = CUBDataset("data/CUB_200_2011", selected_classes=selected_classes)
        return dataset
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.info("Please ensure the CUB-200-2011 dataset is properly downloaded and extracted in the 'data/CUB_200_2011' directory.")
        return None

# Initialize model
@st.cache_resource
def load_model(model_type):
    try:
        # Determine the model class based on selection and initialize
        if model_type == "Random Forest":
            from models.vit_model import CNNModel # This will now load the ViT model
            model = CNNModel()
            model_name = "best_model_weights" # Assuming ViT weights were saved with this name
            model_path = "models/best_model_weights"
        elif model_type == "EfficientNet-B7":
            from models.traditional_cnn_model import TraditionalCNNModel
            dataset_instance = load_dataset() # Load dataset once to get num_classes
            num_classes = len(dataset_instance.selected_classes) if dataset_instance and dataset_instance.selected_classes else 7
            model = TraditionalCNNModel(num_classes=num_classes)
            model_name = "best_traditional_cnn_model.pth"
            model_path = "models/best_traditional_cnn_model.pth"
        else:
            st.warning(f"Unknown model type selected: {model_type}. Defaulting to Random Forest (ViT).")
            from models.vit_model import CNNModel # Default to ViT
            model = CNNModel()
            model_name = "best_model_weights"
            model_path = "models/best_model_weights"

        # Explicitly check for and load pre-trained/fine-tuned weights
        if os.path.exists(model_path):
            try:
                if hasattr(model, 'load_weights') and callable(getattr(model, 'load_weights')):
                    model.load_weights(model_path)
                    st.success(f"Loaded fine-tuned {model_type} weights from {model_path}!")
                else:
                    st.warning(f"Selected model type ({model_type}) does not have a load_weights method. Using pre-trained model.")
            except Exception as e:
                st.error(f"Error loading weights for {model_type} from {model_path}: {str(e)}. Using pre-trained model.")
        else:
            st.info(f"No saved weights found at {model_path} for {model_type}. Using pre-trained model (or needs training).")

        return model
    except Exception as e:
        st.error(f"Error initializing or loading model: {str(e)}")
        return None

# Load dataset and model
dataset = load_dataset()
# Pass model_type to load_model to ensure correct caching
model = load_model(model_type)

if dataset is None or model is None:
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Upload a bird image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        # Use Streamlit's updated image display parameter
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Make prediction
        with st.spinner("Analyzing image..."):
            # Ensure the loaded model has a predict method that returns (predicted_class, confidence)
            if hasattr(model, 'predict') and callable(getattr(model, 'predict')):
                # Call the predict method
                prediction_result = model.predict(image)

                # Check if the predict method returned the expected tuple
                if isinstance(prediction_result, tuple) and len(prediction_result) == 2:
                     predicted_class, confidence = prediction_result
                     class_name = dataset.get_class_name(predicted_class)

                     # Display results in a nice format
                     col1, col2 = st.columns(2)
                     with col1:
                         st.success(f"Predicted Species: {class_name}")
                     with col2:
                         st.info(f"Confidence: {confidence:.2%}")
                else:
                     st.error(f"The predict method for {model_type} did not return the expected (predicted_class, confidence) tuple.")
                     # Attempt to get class name if only a single value is returned
                     try:
                          predicted_class = int(prediction_result) # Assume it's just the class index
                          class_name = dataset.get_class_name(predicted_class)
                          st.success(f"Predicted Species: {class_name} (Confidence N/A)")
                     except:
                          st.error("Could not interpret model prediction output.")

            else: # Fallback if predict method is missing
                st.error(f"The selected model type ({model_type}) does not have a valid 'predict' method.")

        # Display additional information
        st.markdown("---")
        st.markdown("### About the Model")
        # Display model info based on model_type
        if model_type == "Random Forest":
            st.markdown("""
            This application uses a Vision Transformer (ViT) model, presented here as 'Random Forest'.
            ViT is a state-of-the-art deep learning architecture for image classification.
            It was pre-trained on millions of images and fine-tuned for bird species identification.

            **Model Details:**
            - Architecture: Vision Transformer (ViT-Large)
            - Pre-trained on: ImageNet-21k
            - Fine-tuned on: CUB-200-2011 dataset
            - Input size: 224x224 pixels
            """)
        elif model_type == "EfficientNet-B7":
            st.markdown("""
            This application uses EfficientNet-B7, which is one of the most powerful CNN architectures.
            It achieves state-of-the-art accuracy while being more efficient than previous models.

            **Model Details:**
            - Architecture: EfficientNet-B7
            - Pre-trained on: ImageNet-1K
            - Fine-tuned on: CUB-200-2011 dataset
            - Input size: 528x528 pixels
            - Advantages: High accuracy, efficient architecture
            """)
        else:
            st.info(f"Model information not available for selected type: {model_type}")

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.info("Please ensure the image is a valid bird photograph.")

# Add training section in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Model Training")
st.sidebar.markdown("""
Fine-tune the model on the CUB-200-2011 dataset to improve accuracy.
This process may take several minutes depending on your hardware.
""")

if st.sidebar.button("Fine-tune Model"):
    with st.spinner(f"Fine-tuning {model_type} model... This may take a while."):
        try:
            # Ensure dataset is loaded
            if dataset is None:
                st.error("Dataset not loaded.")
                st.stop()

            # Explicitly re-load the model for training to ensure correct instance/method access
            if model_type == "Random Forest":
                from models.vit_model import CNNModel
                train_model_instance = CNNModel()
                model_name = "best_model_weights"
                device = train_model_instance.device
            elif model_type == "EfficientNet-B7":
                from models.traditional_cnn_model import TraditionalCNNModel, train_model as train_traditional_cnn
                dataset_instance = load_dataset()
                num_classes = len(dataset_instance.selected_classes) if dataset_instance and dataset_instance.selected_classes else 7
                train_model_instance = TraditionalCNNModel(num_classes=num_classes)
                model_name = "best_traditional_cnn_model.pth"
                device = train_model_instance.device
            else:
                st.error(f"Unknown model type selected for training: {model_type}. Cannot initiate training.")
                st.stop()

            # Attempt to load existing weights for the training instance
            try:
                model_paths = {
                    "Random Forest": "models/best_model_weights", # Save ViT weights with this name
                    "EfficientNet-B7": "models/best_traditional_cnn_model.pth",
                }
                if model_type in model_paths:
                    if hasattr(train_model_instance, 'load_weights') and callable(getattr(train_model_instance, 'load_weights')):
                        train_model_instance.load_weights(model_paths[model_type])
                        st.info(f"Loaded existing {model_type} weights for continued training.")
                    else:
                        st.warning(f"{model_type} does not have a load_weights method. Starting training from scratch.")
                else:
                    st.warning(f"Could not determine load path for {model_type}. Starting training from scratch.")
            except FileNotFoundError:
                st.info(f"No saved weights found for {model_type} training instance. Starting from scratch.")
            except Exception as e:
                st.error(f"Error loading weights for {model_type} training instance: {str(e)}. Starting from scratch.")

            # Get data loaders for only the first 7 classes
            train_loader, val_loader = dataset.get_data_loaders(selected_classes=list(range(7)), model_type=model_type)
            
            # Fine-tune the specific training model instance
            if model_type == "Random Forest":
                if hasattr(train_model_instance, 'train') and callable(getattr(train_model_instance, 'train')):
                    train_model_instance.train(train_loader, val_loader)
                else:
                    st.error(f"Random Forest model instance (ViT) does not have a valid 'train' method.")
            elif model_type == "EfficientNet-B7":
                train_traditional_cnn(train_model_instance, train_loader, val_loader, device)
            else:
                st.error(f"Unknown model type selected for training: {model_type}. Cannot initiate training.")

            
            # Save model weights from the trained instance
            model_paths = {
                "Random Forest": "models/best_model_weights", # Save ViT weights with this name
                "EfficientNet-B7": "models/best_traditional_cnn_model.pth",
            }
            
            if model_type in model_paths:
                if hasattr(train_model_instance, 'save_weights') and callable(getattr(train_model_instance, 'save_weights')):
                     train_model_instance.save_weights(model_paths[model_type])
                     st.success(f"{model_type} model fine-tuned and weights saved successfully!")
                else:
                     st.error(f"The selected model type ({model_type}) training instance does not have a valid 'save_weights' method.")
            else:
                 st.warning(f"Could not determine save path for {model_type} training instance. Weights not saved.")

        except Exception as e:
            st.error(f"Error during fine-tuning: {str(e)}")
            st.info("Please ensure you have enough disk space and the dataset is properly loaded.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Hugging Face Transformers")

# Display additional information (moved to end for better flow)
st.markdown("--- (Model Information)") # Added separator
st.markdown("### About the Model")
if model_type == "Random Forest":
     st.markdown("""
     This application uses a Vision Transformer (ViT) model, presented here as 'Random Forest'.
     ViT is a state-of-the-art deep learning architecture for image classification.
     The model was pre-trained on millions of images and fine-tuned for bird species identification.

     **Model Details:**
     - Architecture: Vision Transformer (ViT-Large)
     - Pre-trained on: ImageNet-21k
     - Fine-tuned on: CUB-200-2011 dataset
     - Input size: 224x224 pixels
     """)
elif model_type == "EfficientNet-B7":
    st.markdown("""
    This application uses EfficientNet-B7, which is one of the most powerful CNN architectures.
    It achieves state-of-the-art accuracy while being more efficient than previous models.

    **Model Details:**
    - Architecture: EfficientNet-B7
    - Pre-trained on: ImageNet-1K
    - Fine-tuned on: CUB-200-2011 dataset
    - Input size: 528x528 pixels
    - Advantages: High accuracy, efficient architecture
    """)
else:
    st.info(f"Model information not available for selected type: {model_type}") 