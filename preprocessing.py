# %% [markdown]
# # Burner Detection Pipeline - Step by Step
# 
# **Pipeline Overview:**
# 1. **Preprocessing**: Load and preprocess all images
# 2. **Dataset Creation**: Create DataFrame with images and ground truth
# 3. **Model Loading**: Download and load the TFLite model
# 4. **Inference**: Run inference once on all images
# 5. **Evaluation**: Analyze results using presence/absence and IoU methods
# 6. **Visualization**: Visual inspection at each step
# 
# **Note on Input Sizes:**
# - The model input size is determined dynamically from the model itself
# - If the model has fixed input dimensions (e.g., 640x640), images will be resized
# - If the model accepts variable input sizes (shape contains -1), original sizes are preserved

# %% [markdown]
# ## Setup and Configuration

# %%
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

# Model input size - will be determined dynamically from the model
MODEL_INPUT_SIZE = None

print("🔧 CONFIGURATION")
print("=" * 60)
print(f"Model: {VIAM_CONFIG['model_name']} v{VIAM_CONFIG['model_version']}")
print(f"Data Directory: {IMAGES_DIR}")
print(f"Metadata Directory: {METADATA_DIR}")
print(f"Model Directory: {MODEL_DIR}")
print(f"Model Input Size: Dynamic (determined from model)")
print(f"Available metadata files: {len(glob.glob(os.path.join(METADATA_DIR, '*.json')))}")

# %% [markdown]
# ## Step 1: Preprocessing Functions

# %%
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

# %% [markdown]
# ## Step 2: Dataset Creation

# %%
def find_image_for_metadata(metadata_file: str) -> Optional[str]:
    """Find corresponding image file for metadata"""
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    filename = metadata.get('fileName', '')
    binary_id = metadata.get('id', '')
    
    # Try direct filename match
    image_path = os.path.join(IMAGES_DIR, filename)
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
    
    for i, metadata_file in enumerate(metadata_files):
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
            
            # Progress update
            if (i + 1) % 500 == 0:
                print(f"   Progress: {i + 1}/{len(metadata_files)} images processed")
                
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

# Create the dataset
print("\n🚀 STEP 2: DATASET CREATION")
print("=" * 60)

# For testing, limit to first 100 images - remove this for full dataset
dataset_df = create_dataset_dataframe(max_images=100)

if not dataset_df.empty:
    # Visualize samples
    visualize_dataset_samples(dataset_df)
    
    # Show preprocessing effects on a sample
    if len(dataset_df) > 0:
        sample_image = dataset_df.iloc[0]['image']
        print(f"\n🔍 PREPROCESSING EFFECTS ON SAMPLE IMAGE")
        print(f"Original image size: {sample_image.shape[:2]} (height x width)")
        
        # Show full comparison with histograms and statistics
        visualize_preprocessing_effects(sample_image, target_size=None)
        
        # Show detailed comparison on cropped region
        visualize_preprocessing_detail_comparison(sample_image)

# %% [markdown]
# ## Step 3: Model Loading

# %%
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
        
        # Look for existing .tflite files
        tflite_files = glob.glob(os.path.join(MODEL_DIR, "**/*.tflite"), recursive=True)
        if tflite_files:
            model_path = tflite_files[0]
            print(f"✅ Model found: {model_path}")
            return model_path
        
        # Extract .tar.gz files if needed
        tar_files = glob.glob(os.path.join(MODEL_DIR, "**/*.tar.gz"), recursive=True)
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
            tflite_files = glob.glob(os.path.join(MODEL_DIR, "**/*.tflite"), recursive=True)
            if tflite_files:
                model_path = tflite_files[0]
                print(f"✅ Model ready: {model_path}")
                return model_path
        
        print("❌ No .tflite file found")
        return None
    else:
        print(f"❌ Download failed: {result.stderr}")
        return None

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

# Download and load model
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

# %% [markdown]
# ## Step 4: Inference

# %%
def run_single_inference(image_array: np.ndarray, interpreter, 
                        normalization_method: str = "simple") -> List[Tuple[float, float, float, float, float]]:
    """
    Run inference on a single image
    
    The function automatically determines the required input size from the model:
    - If model has fixed input size (e.g., 640x640), image will be resized
    - If model accepts variable sizes (shape contains -1), original size is kept
    
    Args:
        image_array: Input image as numpy array
        interpreter: TFLite interpreter
        normalization_method: Preprocessing method ('simple', 'gcn', 'lcn')
    
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
    
    # Parse detections (adjust tensor names based on your model)
    detections = []
    boxes = outputs.get('detection_boxes', outputs.get('boxes', None))
    classes = outputs.get('detection_classes', outputs.get('classes', None))
    scores = outputs.get('detection_scores', outputs.get('scores', None))
    
    if boxes is not None and classes is not None and scores is not None:
        # Remove batch dimension
        if boxes.ndim == 3: boxes = boxes[0]
        if classes.ndim == 2: classes = classes[0]
        if scores.ndim == 2: scores = scores[0]
        
        for i, score in enumerate(scores):
            if score > 0.5 and int(classes[i]) == 0:  # confidence threshold and burner class
                # boxes format: [ymin, xmin, ymax, xmax]
                ymin, xmin, ymax, xmax = boxes[i]
                detection = (float(xmin), float(ymin), float(xmax), float(ymax), float(score))
                detections.append(detection)
    
    return detections

def run_inference_on_dataframe(df: pd.DataFrame, interpreter, 
                              normalization_method: str = "simple") -> pd.DataFrame:
    """Run inference on all images in the DataFrame"""
    
    print(f"\n🎯 RUNNING INFERENCE ON {len(df)} IMAGES")
    print("=" * 60)
    print(f"Using {normalization_method} normalization")
    
    inferred_bboxes = []
    
    for i, row in df.iterrows():
        try:
            detections = run_single_inference(row['image'], interpreter, normalization_method)
            inferred_bboxes.append(detections)
            
            # Progress update
            if (i + 1) % 25 == 0:
                print(f"   Progress: {i + 1}/{len(df)} images processed")
                
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
    
    dataset_df = run_inference_on_dataframe(dataset_df, model_interpreter, "simple")
    
    # Visualize results
    if not dataset_df.empty:
        visualize_inference_results(dataset_df)
else:
    print("\n⚠️  SKIPPING STEP 4: No model available")

# %% [markdown]
# ## Step 5: Evaluation

# %%
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
    iou_results = evaluate_iou_matching(dataset_df, iou_threshold=0.5)
    
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
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n💾 Results saved to 'evaluation_results.json'")
    
else:
    print("\n⚠️  SKIPPING STEP 5: No inference results available")

# %% [markdown]
# ## Step 6: Final Summary and Export

# %%
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
