import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import ee
import geemap.foliumap as geemap
import folium
import requests
from io import BytesIO
import cv2
from datetime import datetime, timedelta
import os
import pandas as pd

# Set page config
st.set_page_config(
    page_title="DeepLab Borneo Land Cover Prediction",
    page_icon="🌿",
    layout="wide"
)

# Class names for DeepGlobe dataset
CLASS_NAMES = [
    'Urban land',
    'Agriculture land', 
    'Rangeland',
    'Forest land',
    'Water',
    'Barren land',
    'Unknown'
]

# Color map for visualization
COLOR_MAP = [
    [0, 255, 255],      # Urban - Cyan
    [255, 255, 0],      # Agriculture - Yellow
    [255, 0, 255],      # Rangeland - Magenta
    [0, 255, 0],        # Forest - Green
    [0, 0, 255],        # Water - Blue
    [255, 255, 255],    # Barren - White
    [0, 0, 0]           # Unknown - Black
]

def fix_encoder_compatibility(model, encoder_name):
    """Fix encoder compatibility issues by setting missing attributes"""
    try:
        if hasattr(model, 'encoder'):
            encoder = model.encoder
            
            # Fix _out_indexes attribute
            if not hasattr(encoder, '_out_indexes'):
                st.info("Setting missing _out_indexes attribute...")
                
                # Define _out_indexes for different encoders
                encoder_configs = {
                    'mobilenet_v2': [1, 2, 4, 6],
                    'resnet18': [1, 2, 3, 4],
                    'resnet34': [1, 2, 3, 4],
                    'resnet50': [1, 2, 3, 4],
                    'resnet101': [1, 2, 3, 4],
                    'resnet152': [1, 2, 3, 4],
                    'resnext50_32x4d': [1, 2, 3, 4],
                    'resnext101_32x8d': [1, 2, 3, 4],
                    'efficientnet-b0': [1, 2, 3, 4, 5],
                    'efficientnet-b1': [1, 2, 3, 4, 5],
                    'efficientnet-b2': [1, 2, 3, 4, 5],
                    'efficientnet-b3': [1, 2, 3, 4, 5],
                    'efficientnet-b4': [1, 2, 3, 4, 5],
                    'efficientnet-b5': [1, 2, 3, 4, 5],
                    'efficientnet-b6': [1, 2, 3, 4, 5],
                    'efficientnet-b7': [1, 2, 3, 4, 5],
                    'vgg11': [3, 8, 15, 22, 29],
                    'vgg13': [5, 10, 17, 24, 31],
                    'vgg16': [5, 10, 17, 24, 31],
                    'vgg19': [5, 10, 17, 24, 31],
                    'densenet121': [4, 5, 7, 9],
                    'densenet169': [4, 5, 7, 9],
                    'densenet201': [4, 5, 7, 9],
                    'inceptionv3': [5, 6, 7, 8],
                    'xception': [2, 3, 4, 5],
                    'mit_b0': [2, 3, 4, 5],
                    'mit_b1': [2, 3, 4, 5],
                    'mit_b2': [2, 3, 4, 5],
                    'mit_b3': [2, 3, 4, 5],
                    'mit_b4': [2, 3, 4, 5],
                    'mit_b5': [2, 3, 4, 5],
                }
                
                # Set appropriate _out_indexes
                if encoder_name in encoder_configs:
                    encoder._out_indexes = encoder_configs[encoder_name]
                else:
                    # Default fallback
                    encoder._out_indexes = [1, 2, 3, 4]
                
                st.success(f"Set _out_indexes to: {encoder._out_indexes}")
            
            # Fix _out_channels attribute if missing
            if not hasattr(encoder, '_out_channels'):
                st.info("Setting missing _out_channels attribute...")
                
                # Try to infer output channels by running a forward pass
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
                        st.success(f"Inferred _out_channels: {out_channels}")
                    except Exception as e:
                        st.warning(f"Could not infer _out_channels: {e}")
                        # Set default values based on encoder type
                        if 'mobilenet' in encoder_name.lower():
                            encoder._out_channels = [16, 24, 32, 1280]
                        elif 'resnet' in encoder_name.lower():
                            encoder._out_channels = [64, 256, 512, 1024, 2048]
                        else:
                            encoder._out_channels = [64, 128, 256, 512, 1024]
            
            return True
    except Exception as e:
        st.error(f"Error fixing encoder compatibility: {e}")
        return False
    
    return False

@st.cache_resource
def load_model(model_path, encoder_name="mobilenet_v2", num_classes=7, activation='softmax2d'):
    """Load the trained DeepLab model with improved error handling and compatibility fixes"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        st.info(f"Using device: {device}")
        
        # Import segmentation_models_pytorch
        try:
            import segmentation_models_pytorch as smp
            smp_version = smp.__version__ if hasattr(smp, '__version__') else "unknown"
            st.info(f"Using segmentation_models_pytorch version: {smp_version}")
        except ImportError:
            st.error("segmentation_models_pytorch not installed. Please install it: pip install segmentation-models-pytorch")
            return None, None
        
        # Load checkpoint first to understand its structure
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        except Exception:
            try:
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                st.warning("Loaded checkpoint with weights_only=False")
            except Exception as e:
                st.error(f"Failed to load checkpoint: {e}")
                return None, None
        
        # Method 1: Try loading as complete model first
        try:
            st.info("Attempting to load complete model...")
            
            if hasattr(checkpoint, 'eval'):
                # checkpoint is the model itself
                model = checkpoint
                st.success("Loaded complete model object")
            elif isinstance(checkpoint, dict) and 'model' in checkpoint:
                if hasattr(checkpoint['model'], 'eval'):
                    # checkpoint['model'] is the complete model
                    model = checkpoint['model']
                    st.success("Loaded model from 'model' key")
                else:
                    raise ValueError("'model' key doesn't contain a complete model")
            else:
                raise ValueError("Checkpoint is not a complete model")
            
            model = model.to(device)
            model.eval()
            
            # Fix compatibility issues
            fix_encoder_compatibility(model, encoder_name)
            
            # Test the model with a dummy input
            try:
                dummy_input = torch.randn(1, 3, 512, 512).to(device)
                with torch.no_grad():
                    test_output = model(dummy_input)
                st.success("Model compatibility test passed!")
                return model, device
            except Exception as test_error:
                st.warning(f"Model test failed with complete model: {test_error}")
                # Fall through to state dict method
                
        except Exception as e1:
            st.info(f"Complete model loading failed: {e1}")
            st.info("Attempting to load as state dict...")
        
        # Method 2: Load as state dict (create new model and load weights)
        try:
            # Create model architecture
            try:
                model = smp.DeepLabV3Plus(
                    encoder_name=encoder_name,
                    encoder_weights="imagenet",
                    classes=num_classes,
                    activation=activation,
                )
                st.info("Created model with ImageNet encoder weights")
            except Exception as model_creation_error:
                st.warning(f"Failed to create model with ImageNet weights: {model_creation_error}")
                try:
                    model = smp.DeepLabV3Plus(
                        encoder_name=encoder_name,
                        encoder_weights=None,
                        classes=num_classes,
                        activation=activation,
                    )
                    st.info("Created model without pretrained weights")
                except Exception as fallback_error:
                    st.error(f"Failed to create model architecture: {fallback_error}")
                    return None, None
            
            # Load state dict
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    st.info("Loading from 'state_dict' key")
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    st.info("Loading from 'model_state_dict' key")
                elif 'model' in checkpoint and isinstance(checkpoint['model'], dict):
                    state_dict = checkpoint['model']
                    st.info("Loading from 'model' key (state dict)")
                else:
                    # Assume the checkpoint dict is the state dict
                    state_dict = checkpoint
                    st.info("Using checkpoint as state dict")
            else:
                st.error("Checkpoint format not recognized")
                return None, None
            
            # Load state dict with error handling
            try:
                model.load_state_dict(state_dict, strict=True)
                st.success("Loaded state dict with strict=True")
            except Exception as strict_error:
                st.warning(f"Strict loading failed: {strict_error}")
                try:
                    model.load_state_dict(state_dict, strict=False)
                    st.success("Loaded state dict with strict=False")
                except Exception as non_strict_error:
                    st.error(f"State dict loading failed: {non_strict_error}")
                    return None, None
            
            model = model.to(device)
            model.eval()
            
            # Fix compatibility issues
            fix_encoder_compatibility(model, encoder_name)
            
            # Test the model
            try:
                dummy_input = torch.randn(1, 3, 512, 512).to(device)
                with torch.no_grad():
                    test_output = model(dummy_input)
                st.success("Model loaded and tested successfully!")
                return model, device
            except Exception as test_error:
                st.error(f"Model test failed: {test_error}")
                
                # Try one more compatibility fix
                if "_out_indexes" in str(test_error):
                    st.info("Attempting additional compatibility fixes...")
                    try:
                        # Force set attributes on all encoder modules
                        for name, module in model.encoder.named_modules():
                            if hasattr(module, 'forward'):
                                if not hasattr(module, '_out_indexes'):
                                    setattr(module, '_out_indexes', [1, 2, 3, 4])
                                if not hasattr(module, '_out_channels'):
                                    setattr(module, '_out_channels', [64, 128, 256, 512])
                        
                        # Test again
                        with torch.no_grad():
                            test_output = model(dummy_input)
                        st.success("Compatibility fix successful!")
                        return model, device
                    except Exception as final_error:
                        st.error(f"Final compatibility fix failed: {final_error}")
                
                return None, None
                
        except Exception as e2:
            st.error(f"State dict loading failed: {e2}")
            return None, None
        
    except Exception as e:
        st.error(f"General error loading model: {str(e)}")
        return None, None

def reverse_one_hot(one_hot_mask):
    """
    Convert one-hot encoded mask back to categorical mask
    Args:
        one_hot_mask: numpy array of shape (H, W, num_classes)
    Returns:
        categorical_mask: numpy array of shape (H, W) with class indices
    """
    return np.argmax(one_hot_mask, axis=-1)

def colour_code_segmentation(image, label_values):
    """
    Apply color coding to segmentation mask
    Args:
        image: numpy array of shape (H, W) with class indices
        label_values: list of RGB color values for each class
    Returns:
        colored_mask: numpy array of shape (H, W, 3) with RGB colors
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]
    return x

def preprocess_image(image, target_size=(512, 512)):
    """
    Preprocess image for DeepLabV3Plus model - refined version
    
    Args:
        image: PIL Image, numpy array, or file path
        target_size: tuple (height, width) - must be divisible by 16
    
    Returns:
        preprocessed_tensor: torch tensor ready for model
        original_size: tuple of original (height, width) for post-processing
        image_vis: original image as uint8 numpy array for visualization
    """
    
    # Ensure target size is divisible by 16
    h, w = target_size
    h = ((h + 15) // 16) * 16  # Round up to nearest multiple of 16
    w = ((w + 15) // 16) * 16
    target_size = (h, w)
    
    # Handle different input types
    if isinstance(image, str):
        # If it's a file path
        image = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        # If it's a numpy array
        if len(image.shape) == 3:
            # Convert BGR to RGB if needed (OpenCV uses BGR)
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image.astype(np.uint8))
    elif not isinstance(image, Image.Image):
        raise ValueError("Image must be PIL Image, numpy array, or file path")
    
    # Store original size and create visualization image
    original_size = image.size  # PIL returns (width, height)
    original_size = (original_size[1], original_size[0])  # Convert to (height, width)
    image_vis = np.array(image).astype('uint8')  # For visualization
    
    # Resize image to target size
    image_resized = image.resize((target_size[1], target_size[0]), Image.BILINEAR)  # PIL resize takes (width, height)
    
    # Convert to numpy array and normalize
    image_array = np.array(image_resized).astype(np.float32) / 255.0
    
    # Convert to tensor format matching your script
    # Your script expects format for model input
    image_tensor = torch.from_numpy(image_array)
    image_tensor = image_tensor.permute(2, 0, 1)  # HWC to CHW
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    
    return image_tensor, original_size, image_vis

def predict_with_model_refined(model, image, device, select_classes, select_class_rgb_values, target_size=(512, 512)):
    """
    Run prediction on preprocessed image - refined to match your script approach
    
    Args:
        model: trained DeepLabV3Plus model
        image: input image (PIL, numpy array, or path)
        device: torch device
        select_classes: list of class names (e.g., ['urban_land', 'agriculture_land', ...])
        select_class_rgb_values: list of RGB color values for each class
        target_size: target size for model input
    
    Returns:
        pred_mask_colored: colored prediction mask
        pred_class_heatmaps: dictionary of heatmaps for each class
        original_size: original image dimensions
        image_vis: original image for visualization
    """
    try:
        # Preprocess image
        image_tensor, original_size, image_vis = preprocess_image(image, target_size)
        image_tensor = image_tensor.to(device)
        
        st.info(f"Input tensor shape: {image_tensor.shape}")
        
        # Run prediction
        model.eval()
        with torch.no_grad():
            try:
                # Get model output
                pred_mask = model(image_tensor)
                st.success("Model prediction successful")
            except Exception as prediction_error:
                st.error(f"Prediction failed: {prediction_error}")
                
                # Try to fix common issues and retry
                if "out_indexes" in str(prediction_error).lower():
                    st.info("Attempting to fix _out_indexes issue...")
                    try:
                        # Force fix encoder attributes
                        if hasattr(model, 'encoder'):
                            if not hasattr(model.encoder, '_out_indexes'):
                                model.encoder._out_indexes = [1, 2, 3, 4]
                            if not hasattr(model.encoder, '_out_channels'):
                                model.encoder._out_channels = [64, 128, 256, 512]
                        
                        # Retry prediction
                        pred_mask = model(image_tensor)
                        st.success("Prediction successful after fix!")
                        
                    except Exception as retry_error:
                        st.error(f"Retry failed: {retry_error}")
                        return None, None, None, None
                else:
                    return None, None, None, None
            
            # Handle different output formats
            if isinstance(pred_mask, dict):
                if 'out' in pred_mask:
                    pred_mask = pred_mask['out']
                elif 'logits' in pred_mask:
                    pred_mask = pred_mask['logits']
                else:
                    pred_mask = list(pred_mask.values())[0]
            elif isinstance(pred_mask, (list, tuple)):
                pred_mask = pred_mask[0]
            
            # Move to CPU and remove batch dimension
            pred_mask = pred_mask.detach().squeeze().cpu().numpy()
            
            st.info(f"Model output shape after squeeze: {pred_mask.shape}")
            
            # Convert from CHW to HWC format (matching your script)
            if len(pred_mask.shape) == 3 and pred_mask.shape[0] == len(select_classes):
                pred_mask = np.transpose(pred_mask, (1, 2, 0))  # CHW to HWC
                st.info(f"Converted to HWC format: {pred_mask.shape}")
            else:
                st.warning(f"Unexpected prediction shape: {pred_mask.shape}")
                # Handle case where output might already be in HWC or different format
                if len(pred_mask.shape) == 2:
                    # Single channel output - convert to one-hot
                    num_classes = len(select_classes)
                    pred_one_hot = np.zeros((pred_mask.shape[0], pred_mask.shape[1], num_classes))
                    for i in range(num_classes):
                        pred_one_hot[:, :, i] = (pred_mask == i).astype(float)
                    pred_mask = pred_one_hot
                elif pred_mask.shape[-1] != len(select_classes):
                    st.error(f"Output channels ({pred_mask.shape[-1]}) don't match number of classes ({len(select_classes)})")
                    return None, None, None, None
            
            # Extract heatmaps for each class (matching your script)
            pred_class_heatmaps = {}
            for class_name in select_classes:
                if class_name in select_classes:
                    class_idx = select_classes.index(class_name)
                    if class_idx < pred_mask.shape[-1]:
                        pred_class_heatmaps[class_name] = pred_mask[:, :, class_idx]
            
            # Convert to categorical mask and apply color coding (matching your script)
            pred_mask_categorical = reverse_one_hot(pred_mask)
            pred_mask_colored = colour_code_segmentation(pred_mask_categorical, select_class_rgb_values)
            
            # Resize back to original size if needed
            if original_size != target_size:
                pred_mask_colored = cv2.resize(
                    pred_mask_colored.astype(np.uint8),
                    (original_size[1], original_size[0]),  # cv2.resize expects (width, height)
                    interpolation=cv2.INTER_NEAREST
                )
                
                # Resize heatmaps too
                for class_name in pred_class_heatmaps:
                    pred_class_heatmaps[class_name] = cv2.resize(
                        pred_class_heatmaps[class_name],
                        (original_size[1], original_size[0]),
                        interpolation=cv2.INTER_LINEAR
                    )
                
                # Resize visualization image to match
                if image_vis.shape[:2] != original_size:
                    image_vis = cv2.resize(
                        image_vis,
                        (original_size[1], original_size[0]),
                        interpolation=cv2.INTER_LINEAR
                    )
            
            st.info(f"Final prediction shape: {pred_mask_colored.shape}")
            st.info(f"Unique classes in prediction: {np.unique(pred_mask_categorical)}")
        
        return pred_mask_colored, pred_class_heatmaps, original_size, image_vis
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None, None

def complete_prediction_pipeline_refined(model, image, device, select_classes, select_class_rgb_values, target_size=(512, 512)):
    """
    Complete pipeline matching your script's approach
    
    Args:
        model: trained model
        image: input image
        device: torch device
        select_classes: list of class names
        select_class_rgb_values: list of RGB values for each class
        target_size: model input size
    
    Returns:
        dict containing all prediction results
    """
    # Run prediction
    pred_mask_colored, pred_class_heatmaps, original_size, image_vis = predict_with_model_refined(
        model, image, device, select_classes, select_class_rgb_values, target_size
    )
    
    if pred_mask_colored is None:
        return None
    
    return {
        'original_image': image_vis,
        'predicted_mask': pred_mask_colored,
        'class_heatmaps': pred_class_heatmaps,
        'original_size': original_size
    }

def visualize_predictions(original_image, predicted_mask, class_heatmaps, select_classes):
    """
    Visualize predictions similar to your script
    
    Args:
        original_image: original image as numpy array
        predicted_mask: colored prediction mask
        class_heatmaps: dictionary of class heatmaps
        select_classes: list of class names
    """
    # Display original and prediction side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(original_image, use_container_width=True)
    
    with col2:
        st.subheader("Predicted Land Cover")
        st.image(predicted_mask, use_container_width=True)
    
    # Display individual class heatmaps
    if class_heatmaps:
        st.subheader("Class Probability Heatmaps")
        
        # Create columns for heatmaps
        num_classes = len(select_classes)
        cols = st.columns(min(num_classes, 4))  # Max 4 columns
        
        for i, class_name in enumerate(select_classes):
            col_idx = i % len(cols)
            
            if class_name in class_heatmaps:
                with cols[col_idx]:
                    st.write(f"**{class_name.replace('_', ' ').title()}**")
                    
                    # Normalize heatmap for display
                    heatmap = class_heatmaps[class_name]
                    heatmap_normalized = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8) * 255).astype(np.uint8)
                    
                    # Apply colormap
                    heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
                    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                    
                    st.image(heatmap_colored, use_container_width=True)

def get_class_distribution_refined(pred_mask_colored, select_class_rgb_values, select_classes):
    """
    Calculate class distribution from colored prediction mask
    
    Args:
        pred_mask_colored: colored prediction mask
        select_class_rgb_values: RGB values for each class
        select_classes: class names
    
    Returns:
        distribution: dictionary with class percentages
    """
    # Convert colored mask back to categorical
    h, w = pred_mask_colored.shape[:2]
    pred_categorical = np.zeros((h, w), dtype=np.int32)
    
    for i, rgb_value in enumerate(select_class_rgb_values):
        # Find pixels matching this RGB value
        mask = np.all(pred_mask_colored == rgb_value, axis=-1)
        pred_categorical[mask] = i
    
    # Calculate distribution
    unique, counts = np.unique(pred_categorical, return_counts=True)
    total_pixels = pred_categorical.size
    
    distribution = {}
    for class_idx, count in zip(unique, counts):
        if class_idx < len(select_classes):
            class_name = select_classes[class_idx].replace('_', ' ').title()
            distribution[class_name] = (count / total_pixels) * 100
    
    return distribution

# Example usage in your Streamlit app:
"""
# Define your classes and colors (matching your training setup)
select_classes = ['urban_land', 'agriculture_land', 'rangeland', 'forest_land', 'water', 'barren_land', 'unknown']
select_class_rgb_values = [
    [0, 255, 255],      # Urban - Cyan
    [255, 255, 0],      # Agriculture - Yellow  
    [255, 0, 255],      # Rangeland - Magenta
    [0, 255, 0],        # Forest - Green
    [0, 0, 255],        # Water - Blue
    [255, 255, 255],    # Barren - White
    [0, 0, 0]           # Unknown - Black
]

# Use the refined prediction pipeline
results = complete_prediction_pipeline_refined(
    model, image, device, select_classes, select_class_rgb_values, target_size=(512, 512)
)

if results:
    # Visualize results
    visualize_predictions(
        results['original_image'],
        results['predicted_mask'], 
        results['class_heatmaps'],
        select_classes
    )
    
    # Get class distribution
    distribution = get_class_distribution_refined(
        results['predicted_mask'], 
        select_class_rgb_values, 
        select_classes
    )
"""

# Initialize Google Earth Engine
@st.cache_resource
def init_earth_engine():
    """Initialize Google Earth Engine"""
    try:
        ee.Initialize()
        return True
    except Exception as e:
        st.warning(f"Google Earth Engine initialization failed: {str(e)}")
        st.info("Please authenticate with Google Earth Engine to download satellite data.")
        return False

def download_sentinel2_image(bbox, start_date, end_date, cloud_cover=20):
    """Download Sentinel-2 image from Google Earth Engine"""
    try:
        # Define area of interest
        aoi = ee.Geometry.Rectangle(bbox)
        
        # Get Sentinel-2 collection
        collection = ee.ImageCollection('COPERNICUS/S2_SR') \
            .filterBounds(aoi) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover))
        
        # Get the least cloudy image
        image = collection.sort('CLOUDY_PIXEL_PERCENTAGE').first()
        
        if image.getInfo() is None:
            return None, "No images found for the specified criteria"
        
        # Select RGB bands and clip to AOI
        rgb_image = image.select(['B4', 'B3', 'B2']).clip(aoi)
        
        # Get download URL
        url = rgb_image.getDownloadURL({
            'scale': 10,
            'crs': 'EPSG:4326',
            'region': aoi,
            'format': 'GEO_TIFF'
        })
        
        return url, None
    
    except Exception as e:
        return None, f"Error downloading image: {str(e)}"

def download_from_copernicus_browser():
    """Provide instructions for manual download from Copernicus Browser"""
    st.markdown("""
    ### Manual Download from Copernicus Browser
    
    1. Visit [Copernicus Browser](https://browser.dataspace.copernicus.eu/)
    2. Navigate to Borneo, Indonesia (coordinates: -2.5°S to 7°N, 108°E to 119°E)
    3. Select Sentinel-2 L2A data
    4. Choose recent dates with low cloud coverage
    5. Download the image in JPEG or PNG format
    6. Upload the downloaded image using the file uploader below
    """)

# Streamlit App Layout
st.title("🌿 DeepLab Borneo Land Cover Prediction")
st.markdown("Analyze land cover patterns in Borneo, Indonesia using your trained DeepLab model")

# Sidebar for model loading and settings
st.sidebar.header("Model Configuration")

# Model architecture selection
st.sidebar.subheader("Model Architecture")
encoder_name = st.sidebar.selectbox(
    "Encoder", 
    ["mobilenet_v2", "resnet34", "resnet50", "resnet101", "efficientnet-b0", "efficientnet-b3"],
    index=0,  # Default to mobilenet_v2
    help="Select the encoder used in your model"
)

activation = st.sidebar.selectbox(
    "Activation",
    ["softmax2d", "sigmoid", None],
    index=0,  # Default to softmax2d
    help="Activation function used in your model"
)

num_classes = st.sidebar.number_input(
    "Number of Classes", 
    min_value=1, max_value=20, value=7,
    help="Number of output classes in your model"
)

# Model upload
uploaded_model = st.sidebar.file_uploader(
    "Upload your trained DeepLab model (.pth)", 
    type=['pth'],
    help="Upload your trained DeepLab model file"
)

# Enhanced troubleshooting section
st.sidebar.subheader("🔧 Troubleshooting")
with st.sidebar.expander("Common Issues & Solutions"):
    st.markdown("""
    **1. '_out_indexes' Error:**
    - The app will auto-fix this
    - Update: `pip install --upgrade segmentation-models-pytorch`
    
    **2. Model Loading Issues:**
    - Try different loading methods in the app
    - Ensure model architecture matches
    
    **3. Recommended Model Saving:**
    ```python
    # Best practice
    torch.save(model.state_dict(), 'model.pth')
    
    # Alternative (may work better sometimes)
    torch.save(model, 'model.pth')
    ```
    
    **4. Version Compatibility:**
    - Check versions below
    - Try older SMP version if needed:
    ```bash
    pip install segmentation-models-pytorch==0.3.0
    ```
    """)

# Version info
st.sidebar.subheader("📋 System Info")
if st.sidebar.button("Check Versions"):
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        try:
            import segmentation_models_pytorch as smp
            smp_version = getattr(smp, '__version__', "unknown")
            st.success(f"SMP: {smp_version}")
        except ImportError:
            st.error("SMP: Not installed")
    
    with col2:
        import torch
        st.info(f"PyTorch: {torch.__version__}")
    
    try:
        import cv2
        st.sidebar.info(f"OpenCV: {cv2.__version__}")
    except ImportError:
        st.sidebar.warning("OpenCV: Not available")

if uploaded_model is not None:
    # Save uploaded model temporarily
    model_path = f"temp_model_{uploaded_model.name}"
    with open(model_path, "wb") as f:
        f.write(uploaded_model.getbuffer())
    
    # Load model with architecture parameters
    with st.spinner("Loading model... This may take a moment."):
        model, device = load_model(model_path, encoder_name, num_classes, activation)
    
    if model is not None:
        st.sidebar.success("✅ Model loaded successfully!")
        st.sidebar.info(f"Device: {device}")
        st.sidebar.info(f"Architecture: DeepLabV3Plus + {encoder_name}")
        
        # Clean up temporary file
        try:
            os.remove(model_path)
        except:
            pass
        
        # Main content tabs
        tab1, tab2, tab3 = st.tabs(["📤 Upload Image", "🛰️ Download Satellite Data", "📊 Batch Analysis"])
        
        with tab1:
            st.header("Upload and Analyze Image")
            
            uploaded_file = st.file_uploader(
                "Choose an image file", 
                type=['png', 'jpg', 'jpeg', 'tiff', 'tif'],
                help="Upload a satellite image of Borneo for land cover analysis"
            )
            
            if uploaded_file is not None:
                # Load and display original image
                image = Image.open(uploaded_file).convert('RGB')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original Image")
                    st.image(image, use_container_width=True)  # Fixed deprecated parameter
                
                # Preprocess and predict
                # Define your classes and colors (matching your training setup)
                    select_classes = ['urban_land', 'agriculture_land', 'rangeland', 'forest_land', 'water', 'barren_land', 'unknown']
                    select_class_rgb_values = [
                        [0, 255, 255],      # Urban - Cyan
                        [255, 255, 0],      # Agriculture - Yellow  
                        [255, 0, 255],      # Rangeland - Magenta
                        [0, 255, 0],        # Forest - Green
                        [0, 0, 255],        # Water - Blue
                        [255, 255, 255],    # Barren - White
                        [0, 0, 0]           # Unknown - Black
                    ]

                    # Preprocess and predict
                    with st.spinner("Processing image..."):
                        try:
                            target_size = (512, 512)  # Divisible by 16
                            
                            results = complete_prediction_pipeline_refined(
                                model, image, device, select_classes, select_class_rgb_values, target_size
                            )
                            
                            if results:
                                # Visualize results
                                visualize_predictions(
                                    results['original_image'],
                                    results['predicted_mask'], 
                                    results['class_heatmaps'],
                                    select_classes
                                )
                            else:
                                st.error("Failed to make prediction. Please check your model and try again.")
                                st.stop()
                                
                        except Exception as e:
                            st.error(f"Error during prediction: {str(e)}")
                            st.stop()

                    # Class distribution
                    st.subheader("Land Cover Distribution")  
                    distribution = get_class_distribution_refined(
                        results['predicted_mask'], 
                        select_class_rgb_values, 
                        select_classes
                    )
                
                # Create bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                classes = list(distribution.keys())
                percentages = list(distribution.values())

                # Fix the color mapping issue
                def get_color_for_class(cls, default_color=[128, 128, 128]):
                    """Get color for class name, handling case mismatches and missing classes"""
                    # Convert class name to match CLASS_NAMES format
                    cls_formatted = cls.lower().replace(' ', '_')
                    
                    # Create a mapping from formatted names to indices
                    class_name_mapping = {
                        'urban_land': 'Urban land',
                        'agriculture_land': 'Agriculture land',
                        'rangeland': 'Rangeland', 
                        'forest_land': 'Forest land',
                        'water': 'Water',
                        'barren_land': 'Barren land',
                        'unknown': 'Unknown'
                    }
                    
                    # Try to find the class in CLASS_NAMES
                    if cls_formatted in class_name_mapping:
                        mapped_name = class_name_mapping[cls_formatted]
                        if mapped_name in CLASS_NAMES:
                            return np.array(COLOR_MAP[CLASS_NAMES.index(mapped_name)]) / 255.0
                    
                    # Direct lookup attempt
                    if cls in CLASS_NAMES:
                        return np.array(COLOR_MAP[CLASS_NAMES.index(cls)]) / 255.0
                    
                    # Default color if not found
                    print(f"Warning: Class '{cls}' not found in CLASS_NAMES, using default color")
                    return np.array(default_color) / 255.0

                colors = [get_color_for_class(cls) for cls in classes]
                
                bars = ax.bar(classes, percentages, color=colors)
                ax.set_ylabel('Percentage (%)')
                ax.set_title('Land Cover Distribution')
                ax.tick_params(axis='x', rotation=45)
                
                # Add percentage labels on bars
                for bar, pct in zip(bars, percentages):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{pct:.1f}%', ha='center', va='bottom')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Legend
                st.subheader("Legend")
                predicted_classes = list(distribution.keys())
                if predicted_classes:
                    legend_cols = st.columns(len(predicted_classes))
                    
                    for i, (col, class_name) in enumerate(zip(legend_cols, predicted_classes)):
                        # Get color using the safe function
                        color_normalized = get_color_for_class(class_name)
                        color_rgb = (color_normalized * 255).astype(int)
                        
                        col.markdown(
                            f'<div style="background-color: rgb({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]}); '
                            f'padding: 10px; text-align: center; border-radius: 5px; margin: 2px; color: white; text-shadow: 1px 1px 1px black;">'
                            f'<strong>{class_name}</strong></div>',
                            unsafe_allow_html=True
                        )
                else:
                    # Fallback: show all available classes
                    legend_cols = st.columns(len(CLASS_NAMES))
                    for i, (col, class_name) in enumerate(zip(legend_cols, CLASS_NAMES)):
                        color_rgb = COLOR_MAP[i]
                        col.markdown(
                            f'<div style="background-color: rgb({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]}); '
                            f'padding: 10px; text-align: center; border-radius: 5px; margin: 2px; color: white; text-shadow: 1px 1px 1px black;">'
                            f'<strong>{class_name}</strong></div>',
                            unsafe_allow_html=True
                        )
        
        with tab2:
            st.header("Download Satellite Data for Borneo")
            
            # Initialize Earth Engine
            ee_initialized = init_earth_engine()
            
            if ee_initialized:
                st.success("Google Earth Engine initialized successfully!")
                
                # Borneo bounding box options
                borneo_regions = {
                    "Entire Borneo": [95, -4.5, 119.5, 7.5],
                    "Indonesian Borneo (Kalimantan)": [108, -4.5, 119, 4.5],
                    "East Kalimantan": [113, -2.5, 119, 3.5],
                    "Central Kalimantan": [111, -3.5, 115.5, 1],
                    "West Kalimantan": [108, -2, 112.5, 2.5],
                    "South Kalimantan": [114, -4, 116, -1.5]
                }
                
                selected_region = st.selectbox("Select Borneo Region", list(borneo_regions.keys()))
                bbox = borneo_regions[selected_region]
                
                # Date range selection
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
                with col2:
                    end_date = st.date_input("End Date", datetime.now())
                
                cloud_cover = st.slider("Maximum Cloud Cover (%)", 0, 50, 20)
                
                if st.button("Download Sentinel-2 Image"):
                    with st.spinner("Downloading satellite image..."):
                        url, error = download_sentinel2_image(
                            bbox, 
                            start_date.strftime('%Y-%m-%d'), 
                            end_date.strftime('%Y-%m-%d'),
                            cloud_cover
                        )
                        
                        if url:
                            st.success("Image downloaded successfully!")
                            st.markdown(f"[Download Image]({url})")
                        else:
                            st.error(error)
            else:
                # Manual download instructions
                download_from_copernicus_browser()
            
            # Alternative data sources
            st.subheader("Alternative Data Sources")
            st.markdown("""
            - **[EOS LandViewer](https://eos.com/landviewer/)** - Free Landsat and Sentinel data
            - **[USGS EarthExplorer](https://earthexplorer.usgs.gov/)** - Landsat archive
            - **[Copernicus Open Access Hub](https://scihub.copernicus.eu/)** - Sentinel data
            """)
        
        with tab3:
            st.header("Batch Analysis")
            st.info("Upload multiple images for batch processing")
            
            uploaded_files = st.file_uploader(
                "Choose multiple image files", 
                type=['png', 'jpg', 'jpeg', 'tiff', 'tif'],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                st.write(f"Selected {len(uploaded_files)} files")
                
                if st.button("Process All Images"):
                    progress_bar = st.progress(0)
                    results = []
                    
                    for i, file in enumerate(uploaded_files):
                        try:
                            image = Image.open(file).convert('RGB')
                            results = complete_prediction_pipeline_refined(
                                model, image, device, select_classes, select_class_rgb_values, (512, 512)
                            )

                            if results:
                                distribution = get_class_distribution_refined(
                                    results['predicted_mask'], 
                                    select_class_rgb_values, 
                                    select_classes
                                )
                                
                                results.append({
                                    'filename': file.name,
                                    **distribution
                                })
                            else:
                                st.warning(f"Failed to process {file.name}")
                        
                        except Exception as e:
                            st.error(f"Error processing {file.name}: {str(e)}")
                        
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    if results:
                        # Display results table
                        results_df = pd.DataFrame(results)
                        st.subheader("Batch Analysis Results")
                        st.dataframe(results_df, use_container_width=True)  # Fixed deprecated parameter
                        
                        # Summary statistics
                        st.subheader("Summary Statistics")
                        numeric_cols = [col for col in results_df.columns if col != 'filename']
                        if numeric_cols:
                            summary_stats = results_df[numeric_cols].describe()
                            st.dataframe(summary_stats, use_container_width=True)  # Fixed deprecated parameter

else:
    st.sidebar.warning("Please upload your trained DeepLab model to begin analysis")
    
    # Information about the app
    st.markdown("""
    ## About This Application
    
    This tool helps you test your trained DeepLab model on satellite imagery from Borneo, Indonesia. 
    
    ### Features:
    - 🔄 Load your trained PyTorch DeepLab model
    - 📤 Upload and analyze satellite images
    - 🛰️ Download satellite data from multiple sources
    - 📊 Batch processing capabilities
    - 📈 Land cover distribution analysis
    
    ### Data Sources Available:
    - Google Earth Engine (Sentinel-2, Landsat)
    - Copernicus Data Space Ecosystem
    - EOS LandViewer
    - USGS EarthExplorer
    
    ### Requirements:
    - Trained DeepLab model (.pth file, preferably saved as state_dict)
    - Satellite imagery of Borneo region
    - Google Earth Engine account (optional, for automatic downloads)
    
    ### Model Compatibility:
    For best results, save your model using:
    ```python
    # Recommended method
    torch.save(model.state_dict(), 'model.pth')
    
    # Instead of
    torch.save(model, 'model.pth')  # May cause loading issues
    ```
    """)

# Footer
st.markdown("---")
st.markdown("🌿 **DeepLab Borneo Land Cover Analysis** - Monitoring tropical forest changes")