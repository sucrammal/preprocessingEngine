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

# %% [markdown]
# ## Step 1: Download Model

# %%
def download_model():
    """Download TFLite model from Viam"""
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
        # Find .tflite file
        tflite_files = glob.glob(os.path.join(MODEL_DIR, "**/*.tflite"), recursive=True)
        if tflite_files:
            model_path = tflite_files[0]
            print(f"✅ Model downloaded: {model_path}")
            return model_path
        else:
            print("❌ No .tflite file found")
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

def preprocess_image(image_path: str, target_size: tuple = (640, 640)) -> np.ndarray:
    """Preprocess image for model"""
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)
    image_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)

def run_inference(image_path: str, interpreter) -> List[Dict]:
    """Run inference and extract burner detections"""
    if not os.path.exists(image_path):
        return []
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Preprocess image
    input_shape = input_details[0]['shape']
    target_size = (input_shape[1], input_shape[2])
    image_data = preprocess_image(image_path, target_size)
    
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
        if 'annotations' in metadata:
            for bbox in metadata['annotations'].get('bboxes', []):
                if 'burner' in bbox.get('label', '').lower():
                    gt_burners.append(bbox['label'])
        
        # Check if model detected burners (assuming class_id mapping)
        pred_burners = [det for det in detections if det['class_id'] == 1]  # Adjust class_id as needed
        
        results.append({
            "file": os.path.basename(metadata_file),
            "image_path": image_path,
            "ground_truth_burners": len(gt_burners),
            "predicted_burners": len(pred_burners),
            "detections": detections,
            "has_burner_gt": len(gt_burners) > 0,
            "has_burner_pred": len(pred_burners) > 0
        })
        
        print(f"  {os.path.basename(image_path)}: GT={len(gt_burners)}, Pred={len(pred_burners)}")
    
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
