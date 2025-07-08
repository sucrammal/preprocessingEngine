#!/usr/bin/env python
# coding: utf-8

# # Burner Detection Preprocessing Engine
# 
# **Pipeline Overview:** Download model → Process images → Evaluate burner classification
# 
# **Key Features:**
# 1. **Simple Evaluation**: Presence/absence of burners (binary classification)
# 2. **Advanced Evaluation**: Spatial IoU-based matching (object detection metrics)  
# 3. **Preprocessing Comparison**: Test different normalization techniques for lighting variations
# 4. **Visualization Demo**: Visual walkthrough of pipeline on sample image
# 
# **Preprocessing Methods Available:**
# - **Simple**: Standard 0-1 normalization (baseline)
# - **GCN**: Global Contrast Normalization (handles overall brightness differences)
# - **LCN**: Local Contrast Normalization (handles local lighting/shadow variations)

# ## Setup

# In your terminal, use the following commands to create a python3.11 kernel. Select the kernel "Python 3.11" at the top of the notebook to use it:
# 
# ```
# conda create -n my_env python=3.11 ipykernel
# conda activate my_env
# python -m ipykernel install --user --name=my_env --display-name="Python 3.11"   
# ```

# In[ ]:


import json
import os
import subprocess
import glob
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage
import cv2
from scipy import ndimage
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configuration
VIAM_CONFIG = {
    "model_name": os.getenv("VIAM_MODEL_NAME", "your-burner-detection-model"),
    "model_org_id": os.getenv("VIAM_MODEL_ORG_ID", "your-model-org-id"),
    "model_version": os.getenv("VIAM_MODEL_VERSION", "2024-XX-XXTXX-XX-XX"),
}

METADATA_DIR = os.getenv("METADATA_DIR", "metadata")
IMAGES_DIR = os.getenv("IMAGES_DIR", "data")
MODEL_DIR = os.getenv("MODEL_DIR", "models")

MODEL_INPUT_SIZE = None

INFERENCE_CONFIDENCE = float(os.getenv("INFERENCE_CONFIDENCE", "0.5"))
IOU_CONFIDENCE = float(os.getenv("IOU_CONFIDENCE", "0.5"))  # Fixed: was using INFERENCE_CONFIDENCE env var
PREPROCESSING_METHOD = os.getenv("PREPROCESSING_METHOD", "simple")

print("🔧 CONFIGURATION SUMMARY")
print("=" * 60)
print(f"Model: {VIAM_CONFIG['model_name']} v{VIAM_CONFIG['model_version']}")
print(f"Data: {len(glob.glob(os.path.join(METADATA_DIR, '*.json')))} metadata files")
print(f"Preprocessing method: {PREPROCESSING_METHOD}")
print(f"Inference confidence threshold: {INFERENCE_CONFIDENCE}")
print(f"IoU evaluation threshold: {IOU_CONFIDENCE}")
print("=" * 60)

# Quick preview of ground truth format
metadata_files_preview = glob.glob(os.path.join(METADATA_DIR, "*.json"))
if metadata_files_preview:
    print(f"\n📋 Sample ground truth format:")
    with open(metadata_files_preview[0], 'r') as f:
        sample_metadata = json.load(f)
    if 'annotations' in sample_metadata:
        for bbox in sample_metadata['annotations'].get('bboxes', [])[:3]:  # Show first 3
            print(f"  - Label: '{bbox.get('label', 'N/A')}'")
            print(f"    Normalized coordinates:")
            print(f"      x_min: {bbox.get('xMinNormalized', 'N/A'):.3f}")
            print(f"      y_min: {bbox.get('yMinNormalized', 'N/A'):.3f}") 
            print(f"      x_max: {bbox.get('xMaxNormalized', 'N/A'):.3f}")
            print(f"      y_max: {bbox.get('yMaxNormalized', 'N/A'):.3f}")
            width = bbox.get('xMaxNormalized', 0) - bbox.get('xMinNormalized', 0)
            height = bbox.get('yMaxNormalized', 0) - bbox.get('yMinNormalized', 0)
            print(f"    Box dimensions (normalized):")
            print(f"      width: {width:.3f}")
            print(f"      height: {height:.3f}")
    else:
        print("  No annotations found in sample metadata")


# ### Preprocessing and data checks: 
# - Select between simple normalization (simple), local contrast normalization (LCN), and global contrast normalization (GCN).
# - Check if all bounding box keys are present

# In[ ]:


def apply_global_contrast_normalization(image_array: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Apply Global Contrast Normalization (GCN)
    
    This enhances global contrast by standardizing the entire image to have
    zero mean and unit variance, then scaling to 0-1 range.
    """
    # Convert to float for calculations
    image_float = image_array.astype(np.float32)
    
    # Global mean and std across all pixels and channels
    global_mean = np.mean(image_float)
    global_std = np.std(image_float)
    
    # Standardize: zero mean, unit variance
    normalized = (image_float - global_mean) / (global_std + epsilon)
    
    # Scale to 0-1 range for model input
    # Use a more aggressive scaling to show the effect
    normalized = np.tanh(normalized * 0.5) * 0.5 + 0.5  # Tanh activation for contrast
    
    return np.clip(normalized, 0, 1)

def apply_local_contrast_normalization(image_array: np.ndarray, window_size: int = 9, epsilon: float = 1e-8) -> np.ndarray:
    """Apply Local Contrast Normalization (LCN)"""
    normalized = np.zeros_like(image_array, dtype=np.float32)
    
    for channel in range(image_array.shape[2]):
        channel_data = image_array[:, :, channel]
        local_mean = ndimage.uniform_filter(channel_data, size=window_size, mode='reflect')
        local_variance = ndimage.uniform_filter(channel_data**2, size=window_size, mode='reflect') - local_mean**2
        local_std = np.sqrt(np.maximum(local_variance, 0)) + epsilon
        channel_normalized = (channel_data - local_mean) / local_std
        channel_normalized = (channel_normalized - channel_normalized.min()) / (channel_normalized.max() - channel_normalized.min() + epsilon)
        normalized[:, :, channel] = channel_normalized
    
    return normalized


# In[ ]:


def preprocess_image(image_array: np.ndarray, target_size: Optional[Tuple[int, int]] = None,
                    normalization_method: str = "simple") -> np.ndarray:
    """
    Preprocess image array for model input
    
    Args:
        image_array: Input image as numpy array
        target_size: Target size for model input (None to keep original size)
        normalization_method: 'simple', 'gcn', or 'lcn'
    
    Returns:
        Preprocessed image array ready for model input
    """
    # Resize image if target_size is specified
    if target_size is not None:
        image_pil = Image.fromarray(image_array)
        image_resized = image_pil.resize(target_size)
        image_array = np.array(image_resized, dtype=np.float32)
    else:
        # Keep original size
        image_array = np.array(image_array, dtype=np.float32)
    
    # Apply normalization
    if normalization_method == "simple":
        # Simple 0-1 normalization
        normalized = image_array / 255.0
    elif normalization_method == "gcn":
        # Global Contrast Normalization
        normalized = apply_global_contrast_normalization(image_array)
    elif normalization_method == "lcn":
        # Local Contrast Normalization
        normalized = apply_local_contrast_normalization(image_array)
    else:
        raise ValueError(f"Unknown normalization method: {normalization_method}")
    
    return normalized

def visualize_preprocessing_effects(image_array: np.ndarray, methods: List[str] = ["simple", "gcn", "lcn"], 
                                   target_size: Optional[Tuple[int, int]] = None):
    """Visualize the effects of different preprocessing methods with detailed analysis"""
    
    # Create subplots: 2 rows (images + histograms), multiple columns
    fig, axes = plt.subplots(2, len(methods) + 1, figsize=(5 * (len(methods) + 1), 10))
    
    # Original image
    axes[0, 0].imshow(image_array.astype(np.uint8))
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # Original histogram
    axes[1, 0].hist(image_array.flatten(), bins=50, alpha=0.7, color='blue', range=(0, 255))
    axes[1, 0].set_title("Original Histogram")
    axes[1, 0].set_xlabel("Pixel Value")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_xlim(0, 255)
    
    # Store statistics for comparison
    stats = []
    stats.append({
        'method': 'Original',
        'mean': np.mean(image_array),
        'std': np.std(image_array),
        'min': np.min(image_array),
        'max': np.max(image_array)
    })
    
    # Preprocessed versions
    for i, method in enumerate(methods):
        preprocessed = preprocess_image(image_array, target_size, method)
        
        # For display, we need to handle the different ranges
        if method == "simple":
            # Simple is already 0-1, just scale to display
            display_image = (preprocessed * 255).astype(np.uint8)
            hist_data = preprocessed.flatten() * 255  # Scale for histogram
            hist_range = (0, 255)
        else:
            # For GCN and LCN, show the actual normalized values but scale for display
            # Clip to reasonable range for display
            display_preprocessed = np.clip(preprocessed, 0, 1)
            display_image = (display_preprocessed * 255).astype(np.uint8)
            hist_data = preprocessed.flatten()
            hist_range = (np.min(hist_data), np.max(hist_data))
        
        # Display preprocessed image
        axes[0, i + 1].imshow(display_image)
        axes[0, i + 1].set_title(f"{method.upper()} Preprocessing")
        axes[0, i + 1].axis('off')
        
        # Display histogram
        axes[1, i + 1].hist(hist_data, bins=50, alpha=0.7, 
                           color=['green', 'orange', 'red'][i], range=hist_range)
        axes[1, i + 1].set_title(f"{method.upper()} Histogram")
        axes[1, i + 1].set_xlabel("Normalized Value")
        axes[1, i + 1].set_ylabel("Frequency")
        
        # Store statistics
        stats.append({
            'method': method.upper(),
            'mean': np.mean(preprocessed),
            'std': np.std(preprocessed),
            'min': np.min(preprocessed),
            'max': np.max(preprocessed)
        })
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    print("\n📊 PREPROCESSING STATISTICS COMPARISON")
    print("=" * 80)
    print(f"{'Method':<10} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 50)
    for stat in stats:
        print(f"{stat['method']:<10} {stat['mean']:<10.3f} {stat['std']:<10.3f} "
              f"{stat['min']:<10.3f} {stat['max']:<10.3f}")
    
    print("\n💡 What to look for:")
    print("   - SIMPLE: Should have mean ≈ original/255, range [0,1]")
    print("   - GCN: Should have different mean/std, enhanced global contrast")  
    print("   - LCN: Should show enhanced local features, may have wider range")
    print("   - Histograms show the distribution of pixel values after normalization")

def visualize_preprocessing_detail_comparison(image_array: np.ndarray, 
                                            methods: List[str] = ["simple", "gcn", "lcn"],
                                            crop_size: int = 150):
    """
    Show detailed side-by-side comparison of preprocessing effects on a cropped region
    This makes the differences more visible by focusing on a smaller area
    """
    print(f"\n🔍 DETAILED PREPROCESSING COMPARISON")
    print("=" * 60)
    
    # Select a center crop to focus on details
    h, w = image_array.shape[:2]
    center_y, center_x = h // 2, w // 2
    y1 = max(0, center_y - crop_size // 2)
    y2 = min(h, center_y + crop_size // 2)
    x1 = max(0, center_x - crop_size // 2)
    x2 = min(w, center_x + crop_size // 2)
    
    cropped_original = image_array[y1:y2, x1:x2]
    
    # Create comparison figure
    fig, axes = plt.subplots(2, len(methods) + 1, figsize=(4 * (len(methods) + 1), 8))
    
    # Original cropped
    axes[0, 0].imshow(cropped_original.astype(np.uint8))
    axes[0, 0].set_title("Original (Cropped)")
    axes[0, 0].axis('off')
    
    # Show a line profile across the middle
    middle_line = cropped_original[cropped_original.shape[0]//2, :, 0]  # Red channel
    axes[1, 0].plot(middle_line, 'b-', linewidth=2, label='Original')
    axes[1, 0].set_title("Pixel Intensity Profile")
    axes[1, 0].set_xlabel("Pixel Position")
    axes[1, 0].set_ylabel("Intensity")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Process each method
    colors = ['green', 'orange', 'red']
    for i, method in enumerate(methods):
        # Apply preprocessing to full image then crop
        preprocessed_full = preprocess_image(image_array, None, method)
        preprocessed_crop = preprocessed_full[y1:y2, x1:x2]
        
        # Display cropped preprocessed image
        display_image = (preprocessed_crop * 255).astype(np.uint8)
        axes[0, i + 1].imshow(display_image)
        axes[0, i + 1].set_title(f"{method.upper()} (Cropped)")
        axes[0, i + 1].axis('off')
        
        # Show line profile for comparison
        processed_line = preprocessed_crop[preprocessed_crop.shape[0]//2, :, 0] * 255  # Scale for comparison
        axes[1, i + 1].plot(middle_line, 'b-', linewidth=2, alpha=0.7, label='Original')
        axes[1, i + 1].plot(processed_line, color=colors[i], linewidth=2, label=f'{method.upper()}')
        axes[1, i + 1].set_title(f"{method.upper()} vs Original")
        axes[1, i + 1].set_xlabel("Pixel Position")
        axes[1, i + 1].set_ylabel("Intensity")
        axes[1, i + 1].legend()
        axes[1, i + 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print observations
    print(f"\n📋 Observations from cropped region ({crop_size}x{crop_size} pixels):")
    print("   Line profiles show pixel intensity changes across the middle row")
    print("   - SIMPLE: Should closely follow original (just scaled)")
    print("   - GCN: May show enhanced contrast globally") 
    print("   - LCN: Should show enhanced local details and edge contrast")

print("✅ Preprocessing functions defined")


# ## Data validation and cleanup

# In[ ]:


def find_image_for_metadata(metadata_file: str) -> str:
    """Find corresponding image file for metadata"""
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    filename = metadata.get('fileName', '')
    fileExt = metadata.get('fileExt','')
    binary_id = metadata.get('id', '')
    
    # Try direct filename match
    image_path = os.path.join(IMAGES_DIR, filename) + fileExt
    if os.path.exists(image_path):
        return image_path
    
    # Try binary ID match
    for file in os.listdir(IMAGES_DIR):
        if binary_id in file and file.lower().endswith(('.jpg', '.jpeg', '.png')):
            return os.path.join(IMAGES_DIR, file)
    
    return None

def extract_burner_bounding_boxes(metadata: Dict) -> List[Tuple[float, float, float, float]]:
    """
    Extract burner bounding boxes from metadata
    
    Returns:
        List of tuples (xmin, ymin, xmax, ymax) in normalized coordinates
    """
    burner_boxes = []
    
    if 'annotations' in metadata:
        for bbox in metadata['annotations'].get('bboxes', []):
            label = bbox.get('label', '')
            if 'burner' in label.lower():
                # Check if all coordinates are present
                required_keys = ["xMinNormalized", "yMinNormalized", "xMaxNormalized", "yMaxNormalized"]
                if all(key in bbox for key in required_keys):
                    burner_box = (
                        bbox["xMinNormalized"],
                        bbox["yMinNormalized"],
                        bbox["xMaxNormalized"],
                        bbox["yMaxNormalized"]
                    )
                    burner_boxes.append(burner_box)
    
    return burner_boxes

def is_valid_bbox(bbox: Dict) -> bool:
    """Check if a bounding box has all required coordinate keys"""
    required_keys = ["xMinNormalized", "yMinNormalized", "xMaxNormalized", "yMaxNormalized"]
    return all(key in bbox for key in required_keys)


# In[ ]:


print(find_image_for_metadata("metadata/2024-10-23T17_50_47.707Z_BLOND-Design-Agency-London-Impulse-Stovetop-Banner.json"))


# # Dataset Creation

# In[ ]:


def create_dataset_dataframe(max_images: Optional[int] = None) -> pd.DataFrame:
    """
    Create dataset DataFrame with images and ground truth
    
    Returns:
        DataFrame with columns: image_name, image_path, image, burner_bounding_boxes
    """
    print("\n🗂️ CREATING DATASET DATAFRAME")
    print("=" * 60)
    
    # Get all metadata files
    metadata_files = glob.glob(os.path.join(METADATA_DIR, "*.json"))
    
    if max_images:
        metadata_files = metadata_files[:max_images]
        print(f"Processing first {max_images} images for testing")
    
    if not metadata_files:
        print("❌ No metadata files found!")
        return pd.DataFrame()
    
    # Initialize lists
    image_names = []
    image_paths = []
    images = []
    burner_bboxes = []
    
    print(f"📋 Processing {len(metadata_files)} metadata files...")
    
    # Use tqdm for progress bar
    for i, metadata_file in enumerate(tqdm(metadata_files, desc="Loading dataset", unit="file")):
        try:
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Find corresponding image
            image_path = find_image_for_metadata(metadata_file)
            if not image_path:
                print(f"⚠️  No image found for {os.path.basename(metadata_file)}")
                continue
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image)
            
            # Extract burner bounding boxes
            bboxes = extract_burner_bounding_boxes(metadata)
            
            # Add to lists
            image_names.append(os.path.basename(image_path))
            image_paths.append(image_path)
            images.append(image_array)
            burner_bboxes.append(bboxes)
                
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(metadata_file)}: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame({
        'image_name': image_names,
        'image_path': image_paths,
        'image': images,
        'burner_bounding_boxes': burner_bboxes
    })
    
    # Add derived columns
    df['num_burners'] = df['burner_bounding_boxes'].apply(len)
    df['has_burners'] = df['num_burners'] > 0
    
    print(f"\n✅ Dataset DataFrame created!")
    print(f"   Total images: {len(df)}")
    print(f"   Images with burners: {len(df[df['has_burners']])}")
    print(f"   Images without burners: {len(df[~df['has_burners']])}")
    print(f"   Total burner objects: {df['num_burners'].sum()}")
    print(f"   Average burners per image: {df['num_burners'].mean():.2f}")
    
    return df

def visualize_dataset_samples(df: pd.DataFrame, num_samples: int = 6):
    """Visualize sample images from the dataset"""
    print(f"\n📸 VISUALIZING {num_samples} DATASET SAMPLES")
    print("=" * 60)
    
    # Select diverse samples
    with_burners = df[df['has_burners']].head(num_samples // 2)
    without_burners = df[~df['has_burners']].head(num_samples // 2)
    samples = pd.concat([with_burners, without_burners])
    
    fig, axes = plt.subplots(2, num_samples // 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (_, row) in enumerate(samples.iterrows()):
        if i >= len(axes):
            break
            
        # Display image
        axes[i].imshow(row['image'])
        axes[i].set_title(f"{row['image_name']}\n{row['num_burners']} burners")
        axes[i].axis('off')
        
        # Draw bounding boxes
        for bbox in row['burner_bounding_boxes']:
            xmin, ymin, xmax, ymax = bbox
            h, w = row['image'].shape[:2]
            
            # Convert to pixel coordinates
            x1, y1 = int(xmin * w), int(ymin * h)
            x2, y2 = int(xmax * w), int(ymax * h)
            
            # Draw rectangle
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                   linewidth=2, edgecolor='red', facecolor='none')
            axes[i].add_patch(rect)
    
    plt.tight_layout()
    plt.show()


# In[ ]:


# Create the dataset
print("\n🚀 STEP 2: DATASET CREATION")
print("=" * 60)

# For testing, limit to first 100 images - remove this for full dataset
# dataset_df = create_dataset_dataframe(max_images=100)
dataset_df = create_dataset_dataframe()

if not dataset_df.empty:
    # Visualize samples
    visualize_dataset_samples(dataset_df)
    
    # Show preprocessing effects on a sample
    if len(dataset_df) > 0:
        sample_image = dataset_df.iloc[0]['image']
        print(f"\n🔍 PREPROCESSING EFFECTS ON SAMPLE IMAGE")
        print(f"Original image size: {sample_image.shape[:2]} (height x width)")
        print(f"Current preprocessing method: {PREPROCESSING_METHOD}")
        visualize_preprocessing_effects(sample_image)
        visualize_preprocessing_detail_comparison(sample_image)


# ## Setup Model

# In[ ]:


def download_model():
    """Download TFLite model from Viam"""
    import tarfile
    
    print("\n🤖 DOWNLOADING MODEL")
    print("=" * 60)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    cmd = [
        "viam", "packages", "export",
        "--org-id", VIAM_CONFIG["model_org_id"],
        "--name", VIAM_CONFIG["model_name"],
        "--version", VIAM_CONFIG["model_version"],
        "--type", "ml_model",
        "--destination", MODEL_DIR
    ]
    
    print("📥 Downloading model...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Model downloaded successfully")
        model_path_tflite = os.path.join(MODEL_DIR, VIAM_CONFIG["model_version"], VIAM_CONFIG["model_name"]) + ".tflite"
        model_path_targz = os.path.join(MODEL_DIR, VIAM_CONFIG["model_version"], VIAM_CONFIG["model_name"]) + ".tar.gz"
        
        # Look for existing .tflite file in directory
        tflite_files = glob.glob(model_path_tflite, recursive=True)
        if tflite_files:
            model_path = tflite_files[0]
            print(f"✅ Model found: {model_path}")
            return model_path
        
        # Extract .tar.gz files if needed
        tar_files = glob.glob(model_path_targz, recursive=True)
        if tar_files:
            for tar_file in tar_files:
                print(f"📦 Extracting {tar_file}...")
                try:
                    with tarfile.open(tar_file, 'r:gz') as tar:
                        tar.extractall(os.path.dirname(tar_file))
                    print(f"✅ Extracted {tar_file}")
                except Exception as e:
                    print(f"❌ Error extracting {tar_file}: {e}")
                    continue
            
            # Look for .tflite files after extraction
            tflite_files = glob.glob(model_path_targz, recursive=True)
            if tflite_files:
                model_path = tflite_files[0]
                print(f"✅ Model ready: {model_path}")
                return model_path
        
        print("❌ No .tflite file found")
        return None
    else:
        print(f"❌ Download failed: {result.stderr}")
        return None


# ### Load in model

# In[ ]:


def load_model(model_path: str):
    """Load TFLite model and return interpreter"""
    print(f"\n🔄 LOADING MODEL: {model_path}")
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("✅ Model loaded successfully")
    print(f"   Input shape: {input_details[0]['shape']}")
    print(f"   Input dtype: {input_details[0]['dtype']}")
    print(f"   Output tensors: {len(output_details)}")
    
    # Show what input size will be used
    input_shape = input_details[0]['shape']
    if len(input_shape) == 4:  # [batch, height, width, channels]
        if input_shape[1] > 1 and input_shape[2] > 1:
            print(f"   Model expects images resized to: {input_shape[2]}x{input_shape[1]} (width x height)")
        else:
            print(f"   Model accepts variable input sizes (no resizing needed)")
            print(f"   Input shape: {input_shape} (where -1 means dynamic)")
    else:
        print(f"   Model input shape is not standard image format: {input_shape}")
    
    return interpreter


# In[ ]:


def get_model_input_size(interpreter) -> Optional[Tuple[int, int]]:
    """
    Get the expected input size from the model interpreter
    
    Returns:
        Tuple of (width, height) if model has fixed input size, None if variable
    """
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    
    if len(input_shape) == 4:  # [batch, height, width, channels]
        if input_shape[1] > 1 and input_shape[2] > 1:
            return (input_shape[2], input_shape[1])  # (width, height)
        else:
            return None  # Variable input size
    else:
        return None  # Non-standard input shape


# In[ ]:


print("\n🚀 STEP 3: MODEL LOADING")
print("=" * 60)

model_path = download_model()
if model_path:
    model_interpreter = load_model(model_path)
    
    # Show what preprocessing will be used
    expected_size = get_model_input_size(model_interpreter)
    if expected_size:
        print(f"🔧 Images will be resized to {expected_size[0]}x{expected_size[1]} for inference")
    else:
        print(f"🔧 Images will keep their original size for inference")
        
else:
    print("❌ No model available - skipping inference steps")
    model_interpreter = None


# ### Infernece: Call the model!

# In[ ]:


def analyze_model_outputs(outputs: Dict[str, np.ndarray], confidence_threshold: float = 0.1) -> None:
    """
    Analyze and print detailed information about model outputs
    
    Args:
        outputs: Dictionary of model outputs
        confidence_threshold: Minimum confidence to show detections
    """
    print("\n🔍 MODEL OUTPUT ANALYSIS")
    print("=" * 60)
    
    for name, tensor in outputs.items():
        print(f"\nTensor: {name}")
        print(f"  Shape: {tensor.shape}")
        print(f"  Dtype: {tensor.dtype}")
        
        if name == 'StatefulPartitionedCall:0':  # Classes
            print(f"  Classes: {np.unique(tensor)}")
            print(f"  All classes: {tensor.flatten()[:10]}...")  # Show first 10
            
        elif name == 'StatefulPartitionedCall:1':  # Scores
            scores = tensor.squeeze() if tensor.ndim > 1 else tensor
            print(f"  Score range: {scores.min():.4f} - {scores.max():.4f}")
            print(f"  Scores > {confidence_threshold}: {np.sum(scores > confidence_threshold)}")
            high_scores = scores[scores > confidence_threshold]
            if len(high_scores) > 0:
                print(f"  High scores: {high_scores[:5]}...")  # Show first 5 high scores
                
        elif name == 'StatefulPartitionedCall:2':  # Number of detections
            print(f"  Number of detections: {tensor}")
            
        elif name == 'StatefulPartitionedCall:3':  # Bounding boxes
            boxes = tensor.squeeze(1) if tensor.ndim == 3 else tensor
            print(f"  Bounding box format: [ymin, xmin, ymax, xmax]")
            print(f"  Box coordinate range: [{boxes.min():.4f}, {boxes.max():.4f}]")
            
            # Show boxes for high-confidence detections
            scores = outputs.get('StatefulPartitionedCall:1', np.array([]))
            if scores.size > 0:
                scores_flat = scores.squeeze() if scores.ndim > 1 else scores
                high_conf_indices = np.where(scores_flat > confidence_threshold)[0]
                print(f"  High-confidence boxes ({len(high_conf_indices)} detections):")
                for i in high_conf_indices[:5]:  # Show first 5
                    ymin, xmin, ymax, xmax = boxes[i]
                    score = scores_flat[i]
                    print(f"    Box {i}: [{ymin:.3f}, {xmin:.3f}, {ymax:.3f}, {xmax:.3f}] (score: {score:.3f})")


# In[ ]:


def run_single_inference(image_array: np.ndarray, interpreter, 
                        normalization_method: str = "simple", 
                        confidence_threshold: float = 0.5,
                        debug: bool = False) -> List[Tuple[float, float, float, float, float]]:
    """
    Run inference on a single image
    
    The function automatically determines the required input size from the model:
    - If model has fixed input size (e.g., 640x640), image will be resized
    - If model accepts variable sizes (shape contains -1), original size is kept
    
    Args:
        image_array: Input image as numpy array
        interpreter: TFLite interpreter
        normalization_method: Preprocessing method ('simple', 'gcn', 'lcn')
        confidence_threshold: Minimum confidence score for detections
        debug: If True, print detailed analysis of model outputs
    
    Returns:
        List of tuples (xmin, ymin, xmax, ymax, confidence) for detected burners
    """
    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Get expected input size from model (or use None for no resizing)
    target_size = get_model_input_size(interpreter)
    
    # Preprocess image
    preprocessed = preprocess_image(image_array, target_size, normalization_method)
    
    # Convert to model's expected dtype
    input_dtype = input_details[0]['dtype']
    if input_dtype == np.uint8:
        preprocessed = np.clip(preprocessed * 255.0, 0, 255).astype(np.uint8)
    else:
        preprocessed = preprocessed.astype(np.float32)
    
    # Add batch dimension
    input_data = np.expand_dims(preprocessed, axis=0)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Get outputs
    outputs = {}
    for detail in output_details:
        outputs[detail['name']] = interpreter.get_tensor(detail['index'])
    
    # Debug output analysis if requested
    if debug:
        analyze_model_outputs(outputs, confidence_threshold=confidence_threshold)
    
    # Parse detections - handle specific tensor names from this model
    detections = []
    
    # Map specific tensor names to standard format
    classes = outputs.get('StatefulPartitionedCall:0', None)  # Class indices
    scores = outputs.get('StatefulPartitionedCall:1', None)   # Confidence scores
    boxes = outputs.get('StatefulPartitionedCall:3', None)    # Bounding boxes
    num_detections = outputs.get('StatefulPartitionedCall:2', None)  # Number of detections (optional)
    
    if boxes is not None and classes is not None and scores is not None:
        # Remove extra dimensions to get the right shapes
        if boxes.ndim == 3: boxes = boxes.squeeze(1)     # (32, 1, 4) -> (32, 4)
        if scores.ndim == 2: scores = scores.squeeze(1)  # (32, 1) -> (32,)
        if classes.ndim == 1: classes = classes          # (32,) -> (32,)
        
        if debug:
            print(f"Debug: Found {len(scores)} potential detections")
            print(f"Debug: boxes shape: {boxes.shape}, scores shape: {scores.shape}, classes shape: {classes.shape}")
        
        for i, score in enumerate(scores):
            # Apply confidence threshold and check for burner class (assuming class 0 is burner)
            if score > confidence_threshold and int(classes[i]) == 0:  
                # boxes format: [ymin, xmin, ymax, xmax] (typical TF format)
                ymin, xmin, ymax, xmax = boxes[i]
                detection = (float(xmin), float(ymin), float(xmax), float(ymax), float(score))
                detections.append(detection)
                if debug:
                    print(f"Debug: Added detection {i}: score={score:.3f}, box=({xmin:.3f}, {ymin:.3f}, {xmax:.3f}, {ymax:.3f})")
    else:
        print("⚠️  Missing required tensors:")
        print(f"  classes: {classes is not None}")
        print(f"  scores: {scores is not None}")
        print(f"  boxes: {boxes is not None}")
        print(f"  Available tensors: {list(outputs.keys())}")
    
    return detections


# In[ ]:


def test_inference_on_single_image(df: pd.DataFrame, interpreter, 
                                   image_index: int = 0, 
                                   confidence_threshold: float = 0.3,
                                   debug: bool = True):
    """
    Test inference on a single image with detailed output analysis
    
    Args:
        df: DataFrame with images
        interpreter: TFLite interpreter
        image_index: Index of image to test
        confidence_threshold: Confidence threshold for detections
        debug: Whether to show detailed analysis
    """
    if image_index >= len(df):
        print(f"❌ Image index {image_index} out of range (max: {len(df)-1})")
        return
    
    print(f"\n🧪 TESTING INFERENCE ON SINGLE IMAGE")
    print("=" * 60)
    
    row = df.iloc[image_index]
    print(f"Image: {row['image_name']}")
    print(f"Ground truth burners: {row['num_burners']}")
    print(f"Confidence threshold: {confidence_threshold}")
    
    # Run inference with debug enabled
    detections = run_single_inference(row['image'], interpreter, 
                                   normalization_method=PREPROCESSING_METHOD,
                                   confidence_threshold=confidence_threshold,
                                   debug=debug)
    
    print(f"\n📊 INFERENCE RESULTS")
    print("=" * 40)
    print(f"Found {len(detections)} detections:")
    
    for i, (xmin, ymin, xmax, ymax, confidence) in enumerate(detections):
        print(f"  Detection {i+1}: confidence={confidence:.3f}, "
              f"box=({xmin:.3f}, {ymin:.3f}, {xmax:.3f}, {ymax:.3f})")
        
        # Convert to pixel coordinates for visualization
        h, w = row['image'].shape[:2]
        x1, y1 = int(xmin * w), int(ymin * h)
        x2, y2 = int(xmax * w), int(ymax * h)
        print(f"    Pixel coords: ({x1}, {y1}) to ({x2}, {y2})")
    
    return detections


# In[ ]:


test_inference_on_single_image(
       dataset_df, 
       model_interpreter, 
       image_index=0,
       confidence_threshold=INFERENCE_CONFIDENCE,  # Use global variable
       debug=True
   )


# In[ ]:


def run_inference_on_dataframe(df: pd.DataFrame, interpreter, 
                              normalization_method: str = "simple",
                              confidence_threshold: float = 0.5,
                              debug: bool = False) -> pd.DataFrame:
    """Run inference on all images in the DataFrame"""
    
    print(f"\n🎯 RUNNING INFERENCE ON {len(df)} IMAGES")
    print("=" * 60)
    print(f"Using {normalization_method} normalization")
    
    inferred_bboxes = []
    
    # Use tqdm for progress bar
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing images", unit="img"):
        try:
            # Only enable debug for the first image to avoid spam
            debug_this_image = debug and i == 0
            detections = run_single_inference(row['image'], interpreter, normalization_method, 
                                           confidence_threshold, debug_this_image)
            inferred_bboxes.append(detections)
                
        except Exception as e:
            print(f"❌ Error processing {row['image_name']}: {e}")
            inferred_bboxes.append([])
    
    # Add results to DataFrame
    df_with_inference = df.copy()
    df_with_inference['inferred_burner_bboxes'] = inferred_bboxes
    df_with_inference['num_inferred_burners'] = df_with_inference['inferred_burner_bboxes'].apply(len)
    df_with_inference['has_inferred_burners'] = df_with_inference['num_inferred_burners'] > 0
    
    print(f"\n✅ Inference completed!")
    print(f"   Images with predicted burners: {len(df_with_inference[df_with_inference['has_inferred_burners']])}")
    print(f"   Total predicted burners: {df_with_inference['num_inferred_burners'].sum()}")
    print(f"   Average predicted burners per image: {df_with_inference['num_inferred_burners'].mean():.2f}")
    
    return df_with_inference

def visualize_inference_results(df: pd.DataFrame, num_samples: int = 6):
    """Visualize inference results"""
    print(f"\n📊 VISUALIZING INFERENCE RESULTS")
    print("=" * 60)
    
    # Select samples with ground truth burners
    samples = df[df['has_burners']].head(num_samples)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (_, row) in enumerate(samples.iterrows()):
        if i >= len(axes):
            break
            
        # Display image
        axes[i].imshow(row['image'])
        axes[i].set_title(f"{row['image_name']}\nGT: {row['num_burners']}, Pred: {row['num_inferred_burners']}")
        axes[i].axis('off')
        
        h, w = row['image'].shape[:2]
        
        # Draw ground truth bounding boxes (green)
        for bbox in row['burner_bounding_boxes']:
            xmin, ymin, xmax, ymax = bbox
            x1, y1 = int(xmin * w), int(ymin * h)
            x2, y2 = int(xmax * w), int(ymax * h)
            
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                   linewidth=2, edgecolor='green', facecolor='none')
            axes[i].add_patch(rect)
        
        # Draw inferred bounding boxes (red)
        for bbox in row['inferred_burner_bboxes']:
            xmin, ymin, xmax, ymax, confidence = bbox
            x1, y1 = int(xmin * w), int(ymin * h)
            x2, y2 = int(xmax * w), int(ymax * h)
            
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                   linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
            axes[i].add_patch(rect)
            
            # Add confidence score
            axes[i].text(x1, y1 - 5, f'{confidence:.2f}', 
                        color='red', fontsize=8, weight='bold')
    
    # Add legend
    import matplotlib.lines as mlines
    green_line = mlines.Line2D([], [], color='green', linewidth=2, label='Ground Truth')
    red_line = mlines.Line2D([], [], color='red', linewidth=2, linestyle='--', label='Predictions')
    plt.legend(handles=[green_line, red_line], loc='upper right', bbox_to_anchor=(1.1, 1.1))
    
    plt.tight_layout()
    plt.show()

# Run inference if model is available
if model_interpreter is not None:
    print("\n🚀 STEP 4: INFERENCE")
    print("=" * 60)

    dataset_df = run_inference_on_dataframe(dataset_df, model_interpreter, PREPROCESSING_METHOD, INFERENCE_CONFIDENCE)
    
    # Visualize results
    if not dataset_df.empty:
        visualize_inference_results(dataset_df)
else:
    print("\n⚠️  SKIPPING STEP 4: No model available")


# ## Evaluation: Simple presence/absence

# In[ ]:


def evaluate_presence_absence(df: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluate model performance using presence/absence method
    
    Returns:
        Dictionary with accuracy, precision, recall, f1 metrics
    """
    print("\n📊 PRESENCE/ABSENCE EVALUATION")
    print("=" * 60)
    
    # Calculate confusion matrix
    true_positives = len(df[(df['has_burners']) & (df['has_inferred_burners'])])
    false_positives = len(df[(~df['has_burners']) & (df['has_inferred_burners'])])
    false_negatives = len(df[(df['has_burners']) & (~df['has_inferred_burners'])])
    true_negatives = len(df[(~df['has_burners']) & (~df['has_inferred_burners'])])
    
    total = len(df)
    
    # Calculate metrics
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'true_negatives': true_negatives
    }
    
    print(f"Results:")
    print(f"   Accuracy:  {accuracy:.3f}")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall:    {recall:.3f}")
    print(f"   F1 Score:  {f1:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"   True Positives:  {true_positives}")
    print(f"   False Positives: {false_positives}")
    print(f"   False Negatives: {false_negatives}")
    print(f"   True Negatives:  {true_negatives}")
    
    return results


# ## Evaluation: Intersection over Union (IOU)

# In[ ]:


def calculate_iou(box1: Tuple[float, float, float, float], 
                 box2: Tuple[float, float, float, float]) -> float:
    """Calculate IoU between two bounding boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def evaluate_iou_matching(df: pd.DataFrame, iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    Evaluate model performance using IoU matching
    
    Returns:
        Dictionary with precision, recall, f1 metrics
    """
    print(f"\n🎯 IoU MATCHING EVALUATION (threshold={iou_threshold})")
    print("=" * 60)
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for _, row in df.iterrows():
        gt_boxes = row['burner_bounding_boxes']
        pred_boxes = [bbox[:4] for bbox in row['inferred_burner_bboxes']]  # Remove confidence
        
        # Match predictions to ground truth
        matched_gt = set()
        matched_pred = set()
        
        for pred_idx, pred_box in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                    
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                matched_gt.add(best_gt_idx)
                matched_pred.add(pred_idx)
                true_positives += 1
        
        # Count false positives and false negatives
        false_positives += len(pred_boxes) - len(matched_pred)
        false_negatives += len(gt_boxes) - len(matched_gt)
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    results = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'iou_threshold': iou_threshold
    }
    
    print(f"Results:")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall:    {recall:.3f}")
    print(f"   F1 Score:  {f1:.3f}")
    print(f"   True Positives:  {true_positives}")
    print(f"   False Positives: {false_positives}")
    print(f"   False Negatives: {false_negatives}")
    
    return results


# In[ ]:


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        Object with numpy types converted to Python native types
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def visualize_evaluation_results(presence_results: Dict, iou_results: Dict):
    """Visualize evaluation results"""
    print(f"\n📈 EVALUATION RESULTS VISUALIZATION")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Presence/Absence Results
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    values = [presence_results['accuracy'], presence_results['precision'], 
              presence_results['recall'], presence_results['f1']]
    
    axes[0].bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
    axes[0].set_title('Presence/Absence Evaluation')
    axes[0].set_ylabel('Score')
    axes[0].set_ylim(0, 1)
    
    # Add value labels on bars
    for i, v in enumerate(values):
        axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # IoU Results
    iou_metrics = ['Precision', 'Recall', 'F1']
    iou_values = [iou_results['precision'], iou_results['recall'], iou_results['f1']]
    
    axes[1].bar(iou_metrics, iou_values, color=['green', 'orange', 'red'])
    axes[1].set_title(f'IoU Matching Evaluation (threshold={iou_results["iou_threshold"]})')
    axes[1].set_ylabel('Score')
    axes[1].set_ylim(0, 1)
    
    # Add value labels on bars
    for i, v in enumerate(iou_values):
        axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

# Run evaluation if we have inference results
if model_interpreter is not None and 'inferred_burner_bboxes' in dataset_df.columns:
    print("\n🚀 STEP 5: EVALUATION")
    print("=" * 60)
    
    # Method 1: Presence/Absence
    presence_results = evaluate_presence_absence(dataset_df)
    
    # Method 2: IoU Matching
    iou_results = evaluate_iou_matching(dataset_df, iou_threshold=IOU_CONFIDENCE)
    
    # Visualize results
    visualize_evaluation_results(presence_results, iou_results)
    
    # Save results
    results_summary = {
        'presence_absence': presence_results,
        'iou_matching': iou_results,
        'dataset_info': {
            'total_images': len(dataset_df),
            'images_with_burners': len(dataset_df[dataset_df['has_burners']]),
            'total_gt_burners': dataset_df['num_burners'].sum(),
            'total_predicted_burners': dataset_df['num_inferred_burners'].sum()
        }
    }
    
    # Convert numpy types to Python native types for JSON serialization
    results_summary_serializable = convert_numpy_types(results_summary)
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(results_summary_serializable, f, indent=2)
    
    print(f"\n💾 Results saved to 'evaluation_results.json'")
    
else:
    print("\n⚠️  SKIPPING STEP 5: No inference results available")


# ## Final Summary

# In[ ]:


print("\n🏁 PIPELINE COMPLETE")
print("=" * 60)

if not dataset_df.empty:
    print(f"📊 Final Dataset Summary:")
    print(f"   Total images processed: {len(dataset_df)}")
    print(f"   Images with ground truth burners: {len(dataset_df[dataset_df['has_burners']])}")
    print(f"   Total ground truth burners: {dataset_df['num_burners'].sum()}")
    
    if 'inferred_burner_bboxes' in dataset_df.columns:
        print(f"   Images with predicted burners: {len(dataset_df[dataset_df['has_inferred_burners']])}")
        print(f"   Total predicted burners: {dataset_df['num_inferred_burners'].sum()}")
    
    # Save DataFrame
    dataset_df.to_pickle('burner_dataset_complete.pkl')
    print(f"\n💾 Complete dataset saved to 'burner_dataset_complete.pkl'")
    
    # Create summary CSV
    summary_columns = ['image_name', 'num_burners', 'has_burners']
    if 'inferred_burner_bboxes' in dataset_df.columns:
        summary_columns.extend(['num_inferred_burners', 'has_inferred_burners'])
    
    summary_df = dataset_df[summary_columns]
    summary_df.to_csv('dataset_summary.csv', index=False)
    print(f"📋 Summary saved to 'dataset_summary.csv'")
    
else:
    print("❌ No dataset created")

print(f"\n✅ All steps completed successfully!")
print(f"📁 Output files:")
print(f"   - burner_dataset_complete.pkl (complete dataset)")
print(f"   - dataset_summary.csv (summary statistics)")
if model_interpreter is not None:
    print(f"   - evaluation_results.json (evaluation metrics)")

