import streamlit as st
import os
from PIL import Image
from models.vit_model import CNNModel as ViTModel
from models.strong_random_forest_model import StrongRandomForestModel
from models.traditional_cnn_model import TraditionalCNNModel
from utils.data_loader import get_class_names

# Set page config
st.set_page_config(
    page_title="Bird Species Identifier - 7 Classes",
    page_icon="🐦",
    layout="wide"
)

# Title and description
st.title("🐦 Bird Species Identifier (7 Classes)")
st.markdown("""
This application identifies bird species from the first 7 classes of the CUB-200-2011 dataset.
Upload an image of a bird to get started!
""")

# Show the 7 classes
class_names = get_class_names(7)
st.sidebar.markdown("### 🐦 Supported Bird Species")
for i, name in enumerate(class_names):
    st.sidebar.markdown(f"{i+1}. **{name.replace('_', ' ')}**")

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model Type",
    ["Vision Transformer (ViT)", "Random Forest", "Traditional CNN"],
    help="Choose the model for bird species identification"
)

# Check available models
available_models = {}
if os.path.exists('models/best_model_weights'):
    available_models["Vision Transformer (ViT)"] = 'models/best_model_weights'
if os.path.exists('models/random_forest_model.pkl'):
    available_models["Random Forest"] = 'models/random_forest_model.pkl'
if os.path.exists('models/best_traditional_cnn_model.pth'):
    available_models["Traditional CNN"] = 'models/best_traditional_cnn_model.pth'

# Initialize model
@st.cache_resource
def load_model(model_type):
    try:
        if model_type == "Vision Transformer (ViT)":
            model = ViTModel(num_classes=7)
            if model_type in available_models:
                model.load_weights(available_models[model_type])
                st.success("Loaded trained ViT model!")
            else:
                st.warning("Using pre-trained ViT model (not fine-tuned)")
            return model
            
        elif model_type == "Random Forest":
            model = StrongRandomForestModel()
            if model_type in available_models:
                model.load_weights(available_models[model_type])
                st.success("Loaded trained Random Forest model!")
            else:
                st.error("Random Forest model not trained. Please train first.")
                return None
            return model
            
        elif model_type == "Traditional CNN":
            model = TraditionalCNNModel(num_classes=7)
            if model_type in available_models:
                model.load_weights(available_models[model_type])
                st.success("Loaded trained CNN model!")
            else:
                st.warning("Using pre-trained CNN model (not fine-tuned)")
            return model
            
    except Exception as e:
        st.error(f"Error loading {model_type}: {str(e)}")
        return None

# Display model availability
st.sidebar.markdown("### Model Status")
for model_name in ["Vision Transformer (ViT)", "Random Forest", "Traditional CNN"]:
    if model_name in available_models:
        st.sidebar.markdown(f"{model_name} - Trained")
    else:
        st.sidebar.markdown(f"{model_name} - Not trained")

# Load selected model
model = load_model(model_type)

if model is None:
    st.error(f"Could not load {model_type} model. Please train the model first.")
    st.info("Run: `python train_models.py --model <model_name>` to train a model")
    st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "Upload a bird image", 
    type=["jpg", "jpeg", "png"],
    help="Upload an image of one of the 7 supported bird species"
)

if uploaded_file is not None:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            # Make prediction
            with st.spinner("Analyzing image..."):
                predicted_class, confidence = model.predict(image)
                predicted_name = class_names[predicted_class].replace('_', ' ')
                
                # Display results
                st.success(f"**Predicted Species:** {predicted_name}")
                st.info(f"**Confidence:** {confidence:.1%}")
                
                # Confidence bar
                st.progress(confidence)
                
                # Additional info
                if confidence > 0.8:
                    st.balloons()
                    st.success("High confidence prediction!")
                elif confidence > 0.5:
                    st.warning("Moderate confidence - consider better lighting or angle")
                else:
                    st.error("Low confidence - image may not be one of the supported species")

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.info("Please ensure the image is a valid bird photograph.")

# Model information
st.markdown("---")
st.markdown("###Model Information")

if model_type == "Vision Transformer (ViT)":
    st.markdown("""
    **Vision Transformer (ViT-Large)**
    - Architecture: Transformer-based image classification
    - Pre-trained on: ImageNet-21k
    - Input size: 224×224 pixels
    - Fine-tuned on: 7 bird species from CUB-200-2011
    - Performance: ~91% validation accuracy
    """)
elif model_type == "Random Forest":
    st.markdown("""
    **Random Forest with HOG Features**
    - Features: Histogram of Oriented Gradients (HOG)
    - Trees: 200 estimators
    - Input size: 96×96 pixels
    - Performance: ~34% validation accuracy
    - Fast inference, good for resource-constrained environments
    """)
elif model_type == "Traditional CNN":
    st.markdown("""
    **EfficientNet-B7**
    - Architecture: Convolutional Neural Network
    - Pre-trained on: ImageNet-1k
    - Input size: 528×528 pixels
    - Efficient and accurate CNN architecture
    """)

# Training section
st.sidebar.markdown("---")
st.sidebar.markdown("### Model Training")

if st.sidebar.button("Train Selected Model"):
    model_map = {
        "Vision Transformer (ViT)": "vit",
        "Random Forest": "rf", 
        "Traditional CNN": "cnn"
    }
    
    selected_model_key = model_map.get(model_type, "vit")
    
    with st.spinner(f"Training {model_type}... This may take several minutes."):
        try:
            # Show training command
            st.info(f"Running: `python train_models.py --model {selected_model_key} --epochs 5`")
            
            # You could run the training here, but it's better to show the command
            st.success("Training command ready! Run it in your terminal for better progress tracking.")
            st.code(f"python train_models.py --model {selected_model_key} --epochs 5 --batch_size 4")
            
        except Exception as e:
            st.error(f"Training error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
<p>🐦 Built with Streamlit •Powered by PyTorch & Transformers</p>
<p><em>Trained on 7 bird species from the CUB-200-2011 dataset</em></p>
</div>
""", unsafe_allow_html=True)