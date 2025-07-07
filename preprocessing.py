# %% [markdown]
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

# %% [markdown]
# ## Setup

# %%
import json
import os
import subprocess
import glob
from typing import Dict, List, Any, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from dotenv import load_dotenv
import matplotlib.pyplot as plt

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

print(f"Model: {VIAM_CONFIG['model_name']} v{VIAM_CONFIG['model_version']}")
print(f"Data: {len(glob.glob(os.path.join(METADATA_DIR, '*.json')))} metadata files")

# Quick preview of ground truth format
metadata_files_preview = glob.glob(os.path.join(METADATA_DIR, "*.json"))
if metadata_files_preview:
    print(f"\n📋 Sample ground truth format:")
    with open(metadata_files_preview[0], 'r') as f:
        sample_metadata = json.load(f)
    if 'annotations' in sample_metadata:
        for bbox in sample_metadata['annotations'].get('bboxes', [])[:3]:  # Show first 3
            print(f"  - Label: '{bbox.get('label', 'N/A')}'")
    else:
        print("  No annotations found in sample metadata")

# %% [markdown]
# ## Step 1: Download Model

# %%
def download_model():
    """Download TFLite model from Viam and extract if needed"""
    import tarfile
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    cmd = [
        "viam", "packages", "export",
        "--org-id", VIAM_CONFIG["model_org_id"],
        "--name", VIAM_CONFIG["model_name"],
        "--version", VIAM_CONFIG["model_version"],
        "--type", "ml_model",
        "--destination", MODEL_DIR
    ]
    
    print("Downloading model...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Model downloaded successfully")
        
        # First, check if .tflite file already exists (already extracted)
        tflite_files = glob.glob(os.path.join(MODEL_DIR, "**/*.tflite"), recursive=True)
        if tflite_files:
            model_path = tflite_files[0]
            print(f"✅ Model found: {model_path}")
            return model_path
        
        # If no .tflite file found, look for .tar.gz files to extract
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
            
            # Now look for .tflite files after extraction
            tflite_files = glob.glob(os.path.join(MODEL_DIR, "**/*.tflite"), recursive=True)
            if tflite_files:
                model_path = tflite_files[0]
                print(f"✅ Model ready: {model_path}")
                return model_path
            else:
                print("❌ No .tflite file found after extraction")
                return None
        else:
            print("❌ No .tar.gz files found to extract")
            return None
    else:
        print(f"❌ Download failed: {result.stderr}")
        return None

# Download model
model_path = download_model()

# %% [markdown]
# ## Core Processing Functions
# 
# These functions are used by both evaluation methods

# %%
def load_model(model_path: str):
    """Load TFLite model"""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image_path: str, target_size: tuple = (640, 640), input_dtype=np.float32, 
                    normalization_method: str = "simple") -> np.ndarray:
    """Preprocess image for model with different normalization techniques"""
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)
    image_array = np.array(image, dtype=np.float32)
    
    # Apply different normalization techniques
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
    
    # Convert to model's expected dtype
    if input_dtype == np.uint8:
        # For quantized models, convert back to 0-255 range
        normalized = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    else:
        # Keep as float32
        normalized = normalized.astype(np.float32)
    
    return np.expand_dims(normalized, axis=0)

def apply_global_contrast_normalization(image_array: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Apply Global Contrast Normalization (GCN)"""
    # Calculate global mean and std across all pixels and channels
    global_mean = np.mean(image_array)
    global_std = np.std(image_array)
    
    # Normalize: (X - μ) / (σ + ε)
    normalized = (image_array - global_mean) / (global_std + epsilon)
    
    # Scale back to reasonable range (0-1)
    normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min() + epsilon)
    
    return normalized

def apply_local_contrast_normalization(image_array: np.ndarray, window_size: int = 9, epsilon: float = 1e-8) -> np.ndarray:
    """Apply Local Contrast Normalization (LCN)"""
    from scipy import ndimage
    
    # Convert to grayscale for LCN calculation, then apply to all channels
    normalized = np.zeros_like(image_array, dtype=np.float32)
    
    for channel in range(image_array.shape[2]):
        channel_data = image_array[:, :, channel]
        
        # Calculate local mean using a uniform filter
        local_mean = ndimage.uniform_filter(channel_data, size=window_size, mode='reflect')
        
        # Calculate local standard deviation
        local_variance = ndimage.uniform_filter(channel_data**2, size=window_size, mode='reflect') - local_mean**2
        local_std = np.sqrt(np.maximum(local_variance, 0)) + epsilon
        
        # Apply LCN: (X - local_mean) / (local_std + ε)
        channel_normalized = (channel_data - local_mean) / local_std
        
        # Scale to 0-1 range
        channel_normalized = (channel_normalized - channel_normalized.min()) / (channel_normalized.max() - channel_normalized.min() + epsilon)
        
        normalized[:, :, channel] = channel_normalized
    
    return normalized

def run_inference(image_path: str, interpreter, normalization_method: str = "simple") -> List[Dict]:
    """Run inference and extract burner detections"""
    if not os.path.exists(image_path):
        return []
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Preprocess image with correct dtype and normalization method
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    target_size = (input_shape[1], input_shape[2])
    image_data = preprocess_image(image_path, target_size, input_dtype, normalization_method)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], image_data)
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
            if score > 0.5:  # Confidence threshold
                detections.append({
                    "class_id": int(classes[i]),
                    "confidence": float(score),
                    "bbox": [float(x) for x in boxes[i]]
                })
    
    return detections

def find_image_for_metadata(metadata_file: str) -> str:
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

def process_all_images(model_path: str):
    """Process all images and return results"""
    if not model_path or not os.path.exists(model_path):
        print("❌ Model not found")
        return []
    
    interpreter = load_model(model_path)
    metadata_files = glob.glob(os.path.join(METADATA_DIR, "*.json"))
    results = []
    
    print(f"Processing {len(metadata_files)} images...")
    print("\n=== Label Matching Logic ===")
    print("Ground Truth: Looking for labels containing 'burner' (case-insensitive)")
    print("Predictions: Looking for detections with class_id=0 (based on labels.txt)")
    print("=" * 50)
    
    for i, metadata_file in enumerate(metadata_files):
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Find corresponding image
        image_path = find_image_for_metadata(metadata_file)
        if not image_path:
            continue
        
        # Run inference
        detections = run_inference(image_path, interpreter, "simple")
        
        # Extract ground truth burner labels
        gt_burners = []
        all_gt_labels = []
        if 'annotations' in metadata:
            for bbox in metadata['annotations'].get('bboxes', []):
                label = bbox.get('label', '')
                all_gt_labels.append(label)
                if 'burner' in label.lower():
                    gt_burners.append(label)
        
        # Check if model detected burners (class_id 0 = burner based on labels.txt)
        pred_burners = [det for det in detections if det['class_id'] == 0]
        
        results.append({
            "file": os.path.basename(metadata_file),
            "image_path": image_path,
            "ground_truth_burners": len(gt_burners),
            "predicted_burners": len(pred_burners),
            "detections": detections,
            "has_burner_gt": len(gt_burners) > 0,
            "has_burner_pred": len(pred_burners) > 0,
            "all_gt_labels": all_gt_labels
        })
        
        # Print detailed results for first image only
        if i == 0:  # First image
            filename = os.path.basename(image_path)
            print(f"\n  📸 {filename}")
            print(f"     Ground truth: {all_gt_labels} → {len(gt_burners)} burners")
            print(f"     Predictions: {len(detections)} total detections")
            for j, det in enumerate(detections):
                print(f"       - Detection {j+1}: class_id={det['class_id']}, confidence={det['confidence']:.3f}")
            print(f"     Result: GT={len(gt_burners)}, Pred={len(pred_burners)}")
        elif (i + 1) % 500 == 0:
            # Progress update every 500 images
            print(f"  Progress: {i + 1}/{len(metadata_files)} images processed")
        
                 # No individual image output for images 2-8000+
    
    print(f"\n✅ Processing complete: {len(results)} images processed")
    return results

# %% [markdown]
# ## Advanced Evaluation with IoU Matching

# %%
def convert_bbox_format(bbox, format_type: str) -> List[float]:
    """Convert between different bounding box formats"""
    if format_type == "gt_to_pred":
        # Ground truth: {xMinNormalized, yMinNormalized, xMaxNormalized, yMaxNormalized}
        # to Model: [ymin, xmin, ymax, xmax]
        return [bbox["yMinNormalized"], bbox["xMinNormalized"], bbox["yMaxNormalized"], bbox["xMaxNormalized"]]
    elif format_type == "pred_to_gt":
        # Model: [ymin, xmin, ymax, xmax]
        # to Ground truth format: {xMinNormalized, yMinNormalized, xMaxNormalized, yMaxNormalized}
        return {
            "xMinNormalized": bbox[1],
            "yMinNormalized": bbox[0],
            "xMaxNormalized": bbox[3],
            "yMaxNormalized": bbox[2]
        }
    else:
        raise ValueError(f"Unknown format_type: {format_type}")

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate IoU between two bounding boxes in [ymin, xmin, ymax, xmax] format"""
    # box1 and box2 should be in format [ymin, xmin, ymax, xmax]
    y1_min, x1_min, y1_max, x1_max = box1
    y2_min, x2_min, y2_max, x2_max = box2
    
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

def evaluate_with_iou(results: List[Dict], iou_threshold: float = 0.5) -> Dict:
    """Evaluate predictions using IoU matching"""
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    detailed_results = []
    
    for result in results:
        if not result["inference_success"]:
            continue
            
        # Get ground truth burner boxes
        gt_boxes = []
        if 'annotations' in result.get('metadata', {}):
            for bbox in result['metadata']['annotations'].get('bboxes', []):
                if 'burner' in bbox.get('label', '').lower():
                    gt_boxes.append(convert_bbox_format(bbox, "gt_to_pred"))
        
        # Get predicted burner boxes
        pred_boxes = []
        for det in result.get('detections', []):
            if det['class_id'] == 0:  # burner class
                pred_boxes.append(det['bbox'])
        
        # Match predictions to ground truth
        matched_gt = set()
        matched_pred = set()
        matches = []
        
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
                matches.append({
                    'gt_idx': best_gt_idx,
                    'pred_idx': pred_idx,
                    'iou': best_iou,
                    'confidence': result['detections'][pred_idx]['confidence']
                })
                true_positives += 1
        
        # Count false positives and false negatives
        false_positives += len(pred_boxes) - len(matched_pred)
        false_negatives += len(gt_boxes) - len(matched_gt)
        
        detailed_results.append({
            'file': result['file'],
            'gt_boxes': len(gt_boxes),
            'pred_boxes': len(pred_boxes),
            'matches': matches,
            'tp': len(matches),
            'fp': len(pred_boxes) - len(matched_pred),
            'fn': len(gt_boxes) - len(matched_gt)
        })
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'detailed_results': detailed_results
    }

# %% [markdown]
# ## Visualization Pipeline

# %%
def visualize_single_image_pipeline(metadata_file: str, model_path: str):
    """Demonstrate the pipeline on a single sample image"""
    
    print("🔍 Loading sample image and metadata...")
    
    # Load metadata and find image
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    image_path = find_image_for_metadata(metadata_file)
    if not image_path:
        print(f"❌ No image found for {metadata_file}")
        return
    
    # Load model
    interpreter = load_model(model_path)
    original_image = Image.open(image_path).convert('RGB')
    
    # Step 1: Show ground truth
    print(f"\n📋 Step 1: Ground Truth Analysis")
    print(f"   Image: {os.path.basename(image_path)}")
    print(f"   Original size: {original_image.size}")
    
    all_gt_labels = []
    gt_boxes = []
    if 'annotations' in metadata:
        for bbox in metadata['annotations'].get('bboxes', []):
            label = bbox.get('label', '')
            all_gt_labels.append(label)
            if 'burner' in label.lower():
                gt_boxes.append(bbox)
    
    print(f"   All ground truth labels: {all_gt_labels}")
    print(f"   Burner labels found: {len(gt_boxes)} burners")
    
    # Step 2: Show preprocessing
    print(f"\n🔄 Step 2: Preprocessing")
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    target_size = (input_shape[1], input_shape[2])
    
    print(f"   Model expects: {target_size} pixels, {input_dtype} data type")
    preprocessed = preprocess_image(image_path, target_size, input_dtype)
    print(f"   Preprocessed shape: {preprocessed.shape}")
    
    # Step 3: Show inference results
    print(f"\n🎯 Step 3: Inference Results")
    detections = run_inference(image_path, interpreter, "simple")
    pred_boxes = [det for det in detections if det['class_id'] == 0]
    
    print(f"   Total detections: {len(detections)}")
    print(f"   Burner detections: {len(pred_boxes)}")
    
    for i, det in enumerate(pred_boxes):
        print(f"     - Burner {i+1}: confidence={det['confidence']:.3f}, bbox={det['bbox']}")
    
    # Step 4: Visual comparison
    print(f"\n📸 Step 4: Creating Visual Comparison")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original + Ground Truth
    axes[0].imshow(original_image)
    axes[0].set_title(f"Ground Truth\n({len(gt_boxes)} burners)")
    axes[0].axis('off')
    
    for bbox in gt_boxes:
        w, h = original_image.size
        x_min = int(bbox["xMinNormalized"] * w)
        y_min = int(bbox["yMinNormalized"] * h)
        x_max = int(bbox["xMaxNormalized"] * w)
        y_max = int(bbox["yMaxNormalized"] * h)
        
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                           fill=False, edgecolor='green', linewidth=3)
        axes[0].add_patch(rect)
        axes[0].text(x_min, y_min - 10, "GT", color='green', fontsize=12, weight='bold')
    
    # Preprocessed image
    if input_dtype == np.uint8:
        display_image = preprocessed[0].astype(np.uint8)
    else:
        display_image = (preprocessed[0] * 255).astype(np.uint8)
    
    axes[1].imshow(display_image)
    axes[1].set_title(f"Preprocessed\n{target_size}, {input_dtype}")
    axes[1].axis('off')
    
    # Original + Predictions
    axes[2].imshow(original_image)
    axes[2].set_title(f"Predictions\n({len(pred_boxes)} burners)")
    axes[2].axis('off')
    
    for i, det in enumerate(pred_boxes):
        w, h = original_image.size
        y_min = int(det['bbox'][0] * h)
        x_min = int(det['bbox'][1] * w)
        y_max = int(det['bbox'][2] * h)
        x_max = int(det['bbox'][3] * w)
        
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                           fill=False, edgecolor='red', linewidth=3)
        axes[2].add_patch(rect)
        axes[2].text(x_min, y_min - 10, f"{det['confidence']:.2f}", 
                    color='red', fontsize=12, weight='bold')
    
    plt.tight_layout()
    plt.savefig("pipeline_demo.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Final summary
    print(f"\n✅ Pipeline Demo Complete!")
    print(f"   📊 Ground Truth: {len(gt_boxes)} burners")
    print(f"   🎯 Predictions: {len(pred_boxes)} burners")
    print(f"   📸 Visualization saved: pipeline_demo.png")
    print(f"   🎪 Demo shows: GT (green) vs Predictions (red)")
    
    if len(gt_boxes) > 0 and len(pred_boxes) > 0:
        print(f"   ✅ Model found burners in image with burners!")
    elif len(gt_boxes) > 0 and len(pred_boxes) == 0:
        print(f"   ❌ Model missed burners that should be there")
    elif len(gt_boxes) == 0 and len(pred_boxes) > 0:
        print(f"   ❌ Model detected burners where there are none")
    else:
        print(f"   ✅ Model correctly found no burners in image with no burners")

def process_all_images_with_iou(model_path: str):
    """Process all images with full metadata for IoU evaluation"""
    if not model_path or not os.path.exists(model_path):
        print("❌ Model not found")
        return []
    
    interpreter = load_model(model_path)
    metadata_files = glob.glob(os.path.join(METADATA_DIR, "*.json"))
    results = []
    
    print(f"Processing {len(metadata_files)} images for IoU evaluation...")
    
    for i, metadata_file in enumerate(metadata_files):
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Find corresponding image
        image_path = find_image_for_metadata(metadata_file)
        if not image_path:
            continue
        
        # Run inference
        detections = run_inference(image_path, interpreter, "simple")
        
        results.append({
            "file": os.path.basename(metadata_file),
            "image_path": image_path,
            "metadata": metadata,
            "detections": detections,
            "inference_success": True
        })
        
        # Progress update every 500 images
        if (i + 1) % 500 == 0:
            print(f"  Progress: {i + 1}/{len(metadata_files)} images processed")
    
    print(f"\n✅ IoU evaluation complete: {len(results)} images processed")
    return results

# %% [markdown]
# ## Evaluation Methods
# 
# The notebook provides two independent evaluation approaches:
# 
# **Method 1: Simple Evaluation**
# - Binary classification: "Does the image contain burners?"
# - Counts burner instances, compares presence/absence
# - Fast, good for overall accuracy assessment
# 
# **Method 2: Advanced IoU Evaluation** 
# - Object detection metrics with spatial matching
# - Uses IoU thresholds to match predictions to ground truth
# - Provides precision, recall, F1 scores
# - More thorough, accounts for spatial accuracy

# %%
# Process all images
if model_path:
    # ===== EVALUATION METHOD 1: Simple Presence/Absence =====
    print("\n" + "="*60)
    print("📊 SIMPLE EVALUATION (Presence/Absence)")
    print("="*60)
    
    results = process_all_images(model_path)
    
    # ===== EVALUATION METHOD 2: Advanced IoU-Based =====
    print("\n" + "="*60)
    print("🎯 ADVANCED EVALUATION (IoU-Based)")
    print("="*60)
    
    iou_results = process_all_images_with_iou(model_path)
    if iou_results:
        evaluation = evaluate_with_iou(iou_results, iou_threshold=0.5)
        
        print(f"\n📊 IoU Evaluation Results (threshold=0.5):")
        print(f"   Precision: {evaluation['precision']:.3f}")
        print(f"   Recall: {evaluation['recall']:.3f}")
        print(f"   F1 Score: {evaluation['f1']:.3f}")
        print(f"   True Positives: {evaluation['true_positives']}")
        print(f"   False Positives: {evaluation['false_positives']}")
        print(f"   False Negatives: {evaluation['false_negatives']}")
        
        # Show detailed per-image results
        print(f"\n📋 Per-Image Results:")
        for detail in evaluation['detailed_results'][:5]:  # Show first 5
            print(f"   {detail['file']}: GT={detail['gt_boxes']}, Pred={detail['pred_boxes']}, "
                  f"TP={detail['tp']}, FP={detail['fp']}, FN={detail['fn']}")
    
else:
    print("⚠️  Skipping processing - no model available")
    results = []

# %% [markdown]
# ## Step 3: Check Burner Classification Accuracy

# %%
def analyze_burner_performance(results: List[Dict]):
    """Analyze burner detection performance"""
    if not results:
        print("No results to analyze")
        return
    
    total = len(results)
    true_positives = sum(1 for r in results if r['has_burner_gt'] and r['has_burner_pred'])
    false_positives = sum(1 for r in results if not r['has_burner_gt'] and r['has_burner_pred'])
    false_negatives = sum(1 for r in results if r['has_burner_gt'] and not r['has_burner_pred'])
    true_negatives = sum(1 for r in results if not r['has_burner_gt'] and not r['has_burner_pred'])
    
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    print("\n=== Simple Burner Classification Results ===")
    print(f"Total images: {total}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives: {true_positives}")
    print(f"  False Positives: {false_positives}")
    print(f"  False Negatives: {false_negatives}")
    print(f"  True Negatives: {true_negatives}")
    
    # Show misclassified examples
    print(f"\nMisclassifications:")
    for result in results:
        if result['has_burner_gt'] != result['has_burner_pred']:
            status = "Missing burner" if result['has_burner_gt'] else "False detection"
            print(f"  {result['file']}: {status}")

# Analyze results
analyze_burner_performance(results)

# %% [markdown]
# ## Export Results

# %%
if results:
    output_file = "burner_classification_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📄 Results saved to {output_file}")

# %%
# ===== PREPROCESSING METHOD COMPARISON =====
if model_path:
    print("\n" + "="*60)
    print("🧪 PREPROCESSING COMPARISON EXPERIMENT")
    print("="*60)
    
    # Test preprocessing methods on subset of images
    comparison_results = compare_preprocessing_methods(model_path, max_images=100)
    
    # Visualize preprocessing effects on a sample image
    metadata_files = glob.glob(os.path.join(METADATA_DIR, "*.json"))
    if metadata_files:
        sample_metadata = metadata_files[0]
        print(f"\nVisualizing preprocessing effects on: {os.path.basename(sample_metadata)}")
        visualize_preprocessing_effects(sample_metadata, model_path)

# %% [markdown]
# ## Preprocessing Method Comparison
# 
# Compare different normalization techniques to handle lighting variations

# %%
def process_images_with_preprocessing_method(model_path: str, normalization_method: str, 
                                           max_images: int = 100) -> List[Dict]:
    """Process subset of images with specific preprocessing method"""
    if not model_path or not os.path.exists(model_path):
        print(f"❌ Model not found")
        return []
    
    interpreter = load_model(model_path)
    metadata_files = glob.glob(os.path.join(METADATA_DIR, "*.json"))
    
    # Limit to subset for comparison (processing all 8000+ would take too long)
    test_files = metadata_files[:max_images]
    results = []
    
    print(f"Testing {normalization_method} normalization on {len(test_files)} images...")
    
    for i, metadata_file in enumerate(test_files):
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Find corresponding image
        image_path = find_image_for_metadata(metadata_file)
        if not image_path:
            continue
        
        # Run inference with specific normalization method
        detections = run_inference(image_path, interpreter, normalization_method)
        
        # Extract ground truth burner labels
        gt_burners = []
        if 'annotations' in metadata:
            for bbox in metadata['annotations'].get('bboxes', []):
                if 'burner' in bbox.get('label', '').lower():
                    gt_burners.append(bbox['label'])
        
        # Check if model detected burners
        pred_burners = [det for det in detections if det['class_id'] == 0]
        
        results.append({
            "file": os.path.basename(metadata_file),
            "image_path": image_path,
            "ground_truth_burners": len(gt_burners),
            "predicted_burners": len(pred_burners),
            "detections": detections,
            "has_burner_gt": len(gt_burners) > 0,
            "has_burner_pred": len(pred_burners) > 0,
            "normalization_method": normalization_method
        })
        
        # Progress update every 25 images for smaller batches
        if (i + 1) % 25 == 0:
            print(f"  Progress: {i + 1}/{len(test_files)} images processed")
    
    print(f"✅ {normalization_method} processing complete: {len(results)} images")
    return results

def compare_preprocessing_methods(model_path: str, max_images: int = 100):
    """Compare all three preprocessing methods"""
    methods = ["simple", "gcn", "lcn"]
    all_results = {}
    
    print("🧪 PREPROCESSING METHOD COMPARISON")
    print("=" * 60)
    print(f"Testing {len(methods)} normalization methods on {max_images} images each")
    print("Methods: Simple (0-1), GCN (Global Contrast), LCN (Local Contrast)")
    print("=" * 60)
    
    # Test each method
    for method in methods:
        print(f"\n🔬 Testing {method.upper()} normalization...")
        results = process_images_with_preprocessing_method(model_path, method, max_images)
        all_results[method] = results
    
    # Compare results
    print(f"\n📊 PREPROCESSING COMPARISON RESULTS")
    print("=" * 60)
    
    for method, results in all_results.items():
        if not results:
            continue
            
        total = len(results)
        true_positives = sum(1 for r in results if r['has_burner_gt'] and r['has_burner_pred'])
        false_positives = sum(1 for r in results if not r['has_burner_gt'] and r['has_burner_pred'])
        false_negatives = sum(1 for r in results if r['has_burner_gt'] and not r['has_burner_pred'])
        true_negatives = sum(1 for r in results if not r['has_burner_gt'] and not r['has_burner_pred'])
        
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Count total detections (regardless of correctness)
        total_detections = sum(r['predicted_burners'] for r in results)
        avg_detections = total_detections / total if total > 0 else 0
        
        print(f"\n🔸 {method.upper()} Normalization:")
        print(f"   Accuracy:  {accuracy:.3f}")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall:    {recall:.3f}")
        print(f"   F1 Score:  {f1:.3f}")
        print(f"   Avg Detections/Image: {avg_detections:.2f}")
        print(f"   TP: {true_positives}, FP: {false_positives}, FN: {false_negatives}, TN: {true_negatives}")
    
    # Find best method
    best_method = None
    best_f1 = 0
    
    for method, results in all_results.items():
        if not results:
            continue
        total = len(results)
        true_positives = sum(1 for r in results if r['has_burner_gt'] and r['has_burner_pred'])
        false_positives = sum(1 for r in results if not r['has_burner_gt'] and r['has_burner_pred'])
        false_negatives = sum(1 for r in results if r['has_burner_gt'] and not r['has_burner_pred'])
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_method = method
    
    if best_method:
        print(f"\n🏆 BEST PREPROCESSING METHOD: {best_method.upper()}")
        print(f"   Best F1 Score: {best_f1:.3f}")
        print(f"   💡 Recommended for production use!")
    
    # Save comparison results
    comparison_file = "preprocessing_comparison.json"
    with open(comparison_file, 'w') as f:
        # Convert results to JSON-serializable format
        serializable_results = {}
        for method, results in all_results.items():
            serializable_results[method] = []
            for result in results:
                # Remove non-serializable items
                clean_result = {k: v for k, v in result.items() if k != 'detections'}
                clean_result['detection_count'] = len(result.get('detections', []))
                serializable_results[method].append(clean_result)
        
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n📄 Comparison results saved to {comparison_file}")
    return all_results

def visualize_preprocessing_effects(metadata_file: str, model_path: str):
    """Visualize the effects of different preprocessing methods on a single image"""
    
    print("🔍 Visualizing preprocessing effects on sample image...")
    
    # Load metadata and find image
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    image_path = find_image_for_metadata(metadata_file)
    if not image_path:
        print(f"❌ No image found for {metadata_file}")
        return
    
    # Load original image
    original_image = Image.open(image_path).convert('RGB')
    
    # Test all three preprocessing methods
    methods = ["simple", "gcn", "lcn"]
    method_names = ["Simple (0-1)", "Global Contrast Norm", "Local Contrast Norm"]
    
    # Load model to get expected input format
    interpreter = load_model(model_path)
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    target_size = (input_shape[1], input_shape[2])
    
    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Apply each preprocessing method
    for i, (method, name) in enumerate(zip(methods, method_names)):
        preprocessed = preprocess_image(image_path, target_size, input_dtype, method)
        
        # Convert back to display format
        if input_dtype == np.uint8:
            display_image = preprocessed[0].astype(np.uint8)
        else:
            display_image = (preprocessed[0] * 255).astype(np.uint8)
        
        axes[i + 1].imshow(display_image)
        axes[i + 1].set_title(f"{name}")
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig("preprocessing_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"📸 Preprocessing comparison saved: preprocessing_comparison.png")

# %% [markdown]
# ## Pipeline Visualization Demo

# %%
# ===== PIPELINE DEMO: Visual Walkthrough =====
if model_path:
    print("\n" + "="*60)
    print("📸 PIPELINE VISUALIZATION DEMO")
    print("="*60)
    
    # Visualize one sample image to demonstrate the pipeline
    metadata_files = glob.glob(os.path.join(METADATA_DIR, "*.json"))
    if metadata_files:
        sample_metadata = metadata_files[0]  # Just take the first one
        print(f"Demonstrating pipeline on sample image: {os.path.basename(sample_metadata)}")
        visualize_single_image_pipeline(sample_metadata, model_path)
    else:
        print("⚠️  No metadata files found for visualization demo")
else:
    print("⚠️  Skipping visualization demo - no model available")
