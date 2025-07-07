# %% [markdown]
# # Burner Detection Preprocessing Engine
# 
# Simple workflow: Download model → Process images → Check burner classification accuracy

# %% [markdown]
# ## Setup

# %%
import json
import os
import subprocess
import glob
from typing import Dict, List, Any
import numpy as np
from PIL import Image
import tensorflow as tf
from dotenv import load_dotenv

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
# ## Step 2: Process Images

# %%
def load_model(model_path: str):
    """Load TFLite model"""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image_path: str, target_size: tuple = (640, 640), input_dtype=np.float32) -> np.ndarray:
    """Preprocess image for model"""
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)
    
    if input_dtype == np.uint8:
        # For quantized models (UINT8)
        image_array = np.array(image, dtype=np.uint8)
    else:
        # For float models (FLOAT32)
        image_array = np.array(image, dtype=np.float32) / 255.0
    
    return np.expand_dims(image_array, axis=0)

def run_inference(image_path: str, interpreter) -> List[Dict]:
    """Run inference and extract burner detections"""
    if not os.path.exists(image_path):
        return []
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Preprocess image with correct dtype
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    target_size = (input_shape[1], input_shape[2])
    image_data = preprocess_image(image_path, target_size, input_dtype)
    
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
    
    for metadata_file in metadata_files:
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Find corresponding image
        image_path = find_image_for_metadata(metadata_file)
        if not image_path:
            continue
        
        # Run inference
        detections = run_inference(image_path, interpreter)
        
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
        
    return results

# Process all images
if model_path:
    results = process_all_images(model_path)
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
    
    print("\n=== Burner Classification Results ===")
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
