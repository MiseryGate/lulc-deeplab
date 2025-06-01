import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
import glob

# Set page config
st.set_page_config(
    page_title="DeepLab Borneo Land Cover Prediction",
    page_icon="🌿",
    layout="wide"
)

# Class names and color map
CLASS_NAMES = [
    'Urban land',
    'Agriculture land', 
    'Rangeland',
    'Forest land',
    'Water',
    'Barren land',
    'Unknown'
]

COLOR_MAP = [
    [0, 255, 255],      # Urban - Cyan
    [255, 255, 0],      # Agriculture - Yellow
    [255, 0, 255],      # Rangeland - Magenta
    [0, 255, 0],        # Forest - Green
    [0, 0, 255],        # Water - Blue
    [255, 255, 255],    # Barren - White
    [0, 0, 0]           # Unknown - Black
]

# Configuration - Modify these paths as needed
MODEL_FOLDER = "./model" 
DEFAULT_ENCODER = "mobilenet_v2"
DEFAULT_CLASSES = 7

def fix_encoder_compatibility(model, encoder_name):
    """Fix encoder compatibility issues"""
    try:
        if hasattr(model, 'encoder'):
            encoder = model.encoder
            
            if not hasattr(encoder, '_out_indexes'):
                encoder_configs = {
                    'mobilenet_v2': [1, 2, 4, 6],
                    'resnet18': [1, 2, 3, 4],
                    'resnet34': [1, 2, 3, 4],
                    'resnet50': [1, 2, 3, 4],
                    'efficientnet-b0': [1, 2, 3, 4, 5],
                }
                encoder._out_indexes = encoder_configs.get(encoder_name, [1, 2, 3, 4])
            
            if not hasattr(encoder, '_out_channels'):
                encoder.eval()
                with torch.no_grad():
                    dummy_input = torch.randn(1, 3, 224, 224)
                    try:
                        features = encoder(dummy_input)
                        if isinstance(features, (list, tuple)):
                            out_channels = [f.shape[1] for f in features]
                        else:
                            out_channels = [features.shape[1]]
                        encoder._out_channels = out_channels
                    except:
                        encoder._out_channels = [64, 128, 256, 512, 1024]
            return True
    except:
        return False
    return False

@st.cache_resource
def load_model_from_folder(model_name, encoder_name=DEFAULT_ENCODER, num_classes=DEFAULT_CLASSES):
    """Load model from ./model folder"""
    try:
        model_path = os.path.join("./model", model_name)
        if not os.path.exists(model_path):
            return None, None, f"Model file not found: {model_path}"
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        import segmentation_models_pytorch as smp
        
        # Load checkpoint
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        except:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Try loading as complete model first
        try:
            if hasattr(checkpoint, 'eval'):
                model = checkpoint
            elif isinstance(checkpoint, dict) and 'model' in checkpoint:
                model = checkpoint['model']
            else:
                raise ValueError("Need to create model architecture")
            
            model = model.to(device)
            model.eval()
            fix_encoder_compatibility(model, encoder_name)
            
            # Test model
            dummy_input = torch.randn(1, 3, 512, 512).to(device)
            with torch.no_grad():
                test_output = model(dummy_input)
            
            return model, device, None
        except:
            # Load as state dict
            model = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights="imagenet",
                classes=num_classes,
                activation='softmax2d',
            )
            
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                return None, None, "Invalid checkpoint format"
            
            try:
                model.load_state_dict(state_dict, strict=True)
            except:
                model.load_state_dict(state_dict, strict=False)
            
            model = model.to(device)
            model.eval()
            fix_encoder_compatibility(model, encoder_name)
            
            return model, device, None
            
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"


def get_available_models():
    """Get list of available model files from ./model folder"""
    model_folder = "./model"
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
        return []
    
    model_files = glob.glob(os.path.join(model_folder, "*.pth"))
    return [os.path.basename(f) for f in model_files]

def preprocess_image(image, target_size=(512, 512)):
    """Preprocess image for model input"""
    # Ensure target size is divisible by 16
    h, w = target_size
    h = ((h + 15) // 16) * 16
    w = ((w + 15) // 16) * 16
    target_size = (h, w)
    
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image.astype(np.uint8))
    
    original_size = image.size
    original_size = (original_size[1], original_size[0])  # (height, width)
    image_vis = np.array(image).astype('uint8')
    
    # Resize and normalize
    image_resized = image.resize((target_size[1], target_size[0]), Image.BILINEAR)
    image_array = np.array(image_resized).astype(np.float32)
    
    # ImageNet normalization
    image_array = image_array / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_array = (image_array - mean) / std
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image_array)
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
    
    return image_tensor, original_size, image_vis

def predict_image(model, image, device, target_size=(512, 512)):
    """Run prediction on image"""
    try:
        image_tensor, original_size, image_vis = preprocess_image(image, target_size)
        image_tensor = image_tensor.float().to(device)
        
        model.eval()
        with torch.no_grad():
            pred_mask = model(image_tensor)
            
            if isinstance(pred_mask, dict):
                pred_mask = pred_mask.get('out', list(pred_mask.values())[0])
            elif isinstance(pred_mask, (list, tuple)):
                pred_mask = pred_mask[0]
            
            pred_mask = pred_mask.detach().squeeze().cpu().numpy()
            
            if len(pred_mask.shape) == 3 and pred_mask.shape[0] == len(CLASS_NAMES):
                pred_mask = np.transpose(pred_mask, (1, 2, 0))  # CHW to HWC
            
            # Convert to categorical and apply colors
            pred_categorical = np.argmax(pred_mask, axis=-1)
            pred_colored = np.zeros((*pred_categorical.shape, 3), dtype=np.uint8)
            
            for i, color in enumerate(COLOR_MAP):
                mask = pred_categorical == i
                pred_colored[mask] = color
            
            # Resize back to original size
            if original_size != target_size:
                pred_colored = cv2.resize(
                    pred_colored,
                    (original_size[1], original_size[0]),
                    interpolation=cv2.INTER_NEAREST
                )
            
            return pred_colored, pred_categorical, image_vis, original_size
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None, None

def get_class_distribution(pred_categorical):
    """Calculate class distribution"""
    unique, counts = np.unique(pred_categorical, return_counts=True)
    total_pixels = pred_categorical.size
    
    distribution = {}
    for class_idx, count in zip(unique, counts):
        if class_idx < len(CLASS_NAMES):
            class_name = CLASS_NAMES[class_idx]
            distribution[class_name] = (count / total_pixels) * 100
    
    return distribution

# Main App
st.title("🌿 DeepLab Borneo Land Cover Prediction")

# Model selection
st.sidebar.header("Model Selection")
available_models = get_available_models()

if not available_models:
    st.sidebar.error(f"No models found in '{MODEL_FOLDER}' folder")
    st.error(f"Please place your .pth model files in the '{MODEL_FOLDER}' folder")
    st.stop()

selected_model = st.sidebar.selectbox("Select Model", available_models)

# Model parameters
encoder_name = st.sidebar.selectbox("Encoder", ["mobilenet_v2", "resnet34", "resnet50", "efficientnet-b0"])
num_classes = st.sidebar.number_input("Classes", min_value=1, max_value=20, value=7)

# Load model
if selected_model:
    model, device, error = load_model_from_folder(selected_model, encoder_name, num_classes)
    
    if error:
        st.sidebar.error(error)
        st.stop()
    else:
        st.sidebar.success("✅ Model loaded")

# Main content
tab1, tab2 = st.tabs(["📤 Single Image", "📊 Batch Analysis"])

with tab1:
    st.header("Upload and Analyze Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['png', 'jpg', 'jpeg', 'tiff', 'tif']
    )
    
    if uploaded_file is not None and model is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("Predicted Land Cover")
            with st.spinner("Processing..."):
                pred_colored, pred_categorical, image_vis, original_size = predict_image(
                    model, image, device
                )
                
                if pred_colored is not None:
                    st.image(pred_colored, use_container_width=True)
        
        if pred_categorical is not None:
            # Distribution analysis
            st.subheader("Land Cover Distribution")
            distribution = get_class_distribution(pred_categorical)
            
            # Bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            classes = list(distribution.keys())
            percentages = list(distribution.values())
            colors = [np.array(COLOR_MAP[CLASS_NAMES.index(cls)]) / 255.0 for cls in classes]
            
            bars = ax.bar(classes, percentages, color=colors)
            ax.set_ylabel('Percentage (%)')
            ax.set_title('Land Cover Distribution')
            ax.tick_params(axis='x', rotation=45)
            
            for bar, pct in zip(bars, percentages):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{pct:.1f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Legend
            st.subheader("Legend")
            legend_cols = st.columns(len(classes))
            for i, (col, class_name) in enumerate(zip(legend_cols, classes)):
                color_rgb = COLOR_MAP[CLASS_NAMES.index(class_name)]
                col.markdown(
                    f'<div style="background-color: rgb({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]}); '
                    f'padding: 10px; text-align: center; border-radius: 5px; margin: 2px; color: white; text-shadow: 1px 1px 1px black;">'
                    f'<strong>{class_name}</strong></div>',
                    unsafe_allow_html=True
                )

with tab2:
    st.header("Batch Analysis")
    
    uploaded_files = st.file_uploader(
        "Choose multiple image files", 
        type=['png', 'jpg', 'jpeg', 'tiff', 'tif'],
        accept_multiple_files=True
    )
    
    if uploaded_files and model is not None:
        st.write(f"Selected {len(uploaded_files)} files")
        
        if st.button("Process All Images"):
            progress_bar = st.progress(0)
            results = []
            
            for i, file in enumerate(uploaded_files):
                try:
                    image = Image.open(file).convert('RGB')
                    _, pred_categorical, _, _ = predict_image(model, image, device)
                    
                    if pred_categorical is not None:
                        distribution = get_class_distribution(pred_categorical)
                        result = {'filename': file.name, **distribution}
                        results.append(result)
                
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            if results:
                results_df = pd.DataFrame(results)
                st.subheader("Batch Analysis Results")
                st.dataframe(results_df, use_container_width=True)
                
                # Summary statistics
                st.subheader("Summary Statistics")
                numeric_cols = [col for col in results_df.columns if col != 'filename']
                if numeric_cols:
                    summary_stats = results_df[numeric_cols].describe()
                    st.dataframe(summary_stats, use_container_width=True)

st.markdown("---")
st.markdown("🌿 **DeepLab Borneo Land Cover Analysis**")