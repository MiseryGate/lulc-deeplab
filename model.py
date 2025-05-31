from segmentation_models_pytorch.decoders.deeplabv3.model import DeepLabV3Plus
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

@st.cache_resource
def load_model(model_path, encoder_name="resnet34", num_classes=7):
    """Load the trained DeepLab model"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Try different loading methods
        try:
            # Method 1: Load complete model
            model = torch.load(model_path, map_location=device)
            if hasattr(model, 'eval'):
                model.eval()
                return model, device
            else:
                # If it's just state dict, continue to method 2
                checkpoint = model
        except Exception as e1:
            # Method 2: Load as state dict
            checkpoint = torch.load(model_path, map_location=device)
        
        # Create model architecture
        try:
            import segmentation_models_pytorch as smp
            
            model = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=None,  # Don't load pretrained weights
                in_channels=3,
                classes=num_classes,
            )
            
            # Load state dict
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                elif 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'])
                else:
                    # Try loading the checkpoint directly
                    model.load_state_dict(checkpoint)
            else:
                st.error("Unexpected checkpoint format")
                return None, None
                
            model = model.to(device)
            model.eval()
            return model, device
            
        except ImportError:
            st.error("segmentation_models_pytorch not installed. Please install it: pip install segmentation-models-pytorch")
            return None, None
        except Exception as e2:
            st.error(f"Error creating model architecture: {str(e2)}")
            st.info("Please ensure your model architecture settings match your trained model.")
            return None, None
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def preprocess_image(image, target_size=(512, 512)):
    """Preprocess image for model prediction"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Resize image
    image = image.resize(target_size, Image.BILINEAR)
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)

def predict_image(model, image_tensor, device):
    """Make prediction using the loaded model"""
    try:
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            
            # Handle different model output formats
            outputs = model(image_tensor)
            
            # Handle different output formats
            if isinstance(outputs, dict):
                if 'out' in outputs:
                    outputs = outputs['out']
                elif 'logits' in outputs:
                    outputs = outputs['logits']
                else:
                    # Take the first value if it's a dict
                    outputs = list(outputs.values())[0]
            elif isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            
            # Ensure outputs have the right shape
            if len(outputs.shape) == 3:
                outputs = outputs.unsqueeze(0)
            
            # Apply softmax and get predictions
            probs = F.softmax(outputs, dim=1)
            predictions = torch.argmax(probs, dim=1)
            
            return predictions.cpu().numpy(), probs.cpu().numpy()
            
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.info("Please check if your model architecture matches the expected input format.")
        return None, None

def colorize_prediction(prediction, color_map):
    """Convert prediction mask to colored image"""
    h, w = prediction.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for i, color in enumerate(color_map):
        colored[prediction == i] = color
    
    return colored

def get_class_distribution(prediction):
    """Calculate class distribution in the prediction"""
    unique, counts = np.unique(prediction, return_counts=True)
    total_pixels = prediction.size
    
    distribution = {}
    for class_idx, count in zip(unique, counts):
        if class_idx < len(CLASS_NAMES):
            distribution[CLASS_NAMES[class_idx]] = (count / total_pixels) * 100
    
    return distribution

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
    ["resnet34", "resnet50", "resnet101", "mobilenet_v2", "efficientnet-b0", "efficientnet-b3"],
    help="Select the encoder used in your model"
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

if uploaded_model is not None:
    # Save uploaded model temporarily
    model_path = f"temp_model_{uploaded_model.name}"
    with open(model_path, "wb") as f:
        f.write(uploaded_model.getbuffer())
    
    # Load model with architecture parameters
    with st.spinner("Loading model..."):
        # Store architecture info in session state for model loading
        st.session_state.encoder_name = encoder_name
        st.session_state.num_classes = num_classes
        
        model, device = load_model(model_path, encoder_name, num_classes)
    
    if model is not None:
        st.sidebar.success("✅ Model loaded successfully!")
        st.sidebar.info(f"Device: {device}")
        st.sidebar.info(f"Architecture: DeepLabV3Plus with {encoder_name}")
        
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
                    st.image(image, use_column_width=True)
                
                # Preprocess and predict
                with st.spinner("Processing image..."):
                    image_tensor = preprocess_image(image)
                    result = predict_image(model, image_tensor, device)
                    
                    if result[0] is not None:
                        predictions, probabilities = result
                        
                        # Get prediction mask
                        pred_mask = predictions[0]
                        
                        # Colorize prediction
                        colored_pred = colorize_prediction(pred_mask, COLOR_MAP)
                    else:
                        st.error("Failed to make prediction. Please check your model and try again.")
                        st.stop()
                
                with col2:
                    st.subheader("Land Cover Prediction")
                    st.image(colored_pred, use_column_width=True)
                
                # Class distribution
                st.subheader("Land Cover Distribution")
                distribution = get_class_distribution(pred_mask)
                
                # Create bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                classes = list(distribution.keys())
                percentages = list(distribution.values())
                colors = [np.array(COLOR_MAP[CLASS_NAMES.index(cls)])/255.0 for cls in classes]
                
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
                legend_cols = st.columns(len(CLASS_NAMES))
                for i, (col, class_name) in enumerate(zip(legend_cols, CLASS_NAMES)):
                    color_rgb = COLOR_MAP[i]
                    col.markdown(
                        f'<div style="background-color: rgb({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]}); '
                        f'padding: 10px; text-align: center; border-radius: 5px; margin: 2px;">'
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
                        image = Image.open(file).convert('RGB')
                        image_tensor = preprocess_image(image)
                        predictions, _ = predict_image(model, image_tensor, device)
                        
                        pred_mask = predictions[0]
                        distribution = get_class_distribution(pred_mask)
                        
                        results.append({
                            'filename': file.name,
                            **distribution
                        })
                        
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    # Display results table
                    import pandas as pd
                    results_df = pd.DataFrame(results)
                    st.subheader("Batch Analysis Results")
                    st.dataframe(results_df)
                    
                    # Summary statistics
                    st.subheader("Summary Statistics")
                    numeric_cols = [col for col in results_df.columns if col != 'filename']
                    summary_stats = results_df[numeric_cols].describe()
                    st.dataframe(summary_stats)

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
    - Trained DeepLab model (.pth file)
    - Satellite imagery of Borneo region
    - Google Earth Engine account (optional, for automatic downloads)
    """)

# Footer
st.markdown("---")
st.markdown("🌿 **DeepLab Borneo Land Cover Analysis** - Monitoring tropical forest changes")