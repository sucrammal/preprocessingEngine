# %% [markdown]
# # Preprocessing Engine for Viam Cloud Inference
# 
# This notebook processes images through Viam cloud inference to detect burners
# and compares results with ground truth metadata.

# %% [markdown]
# ## Setup and Imports

# %%
import json
import os
import subprocess
import glob
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# %% [markdown]
# ## Configuration
# 
# Configuration is loaded from environment variables (.env file)
# Create a .env file with your Viam organization details

# %%
# Viam Configuration - loaded from environment variables
VIAM_CONFIG = {
    "model_name": os.getenv("VIAM_MODEL_NAME", "your-burner-detection-model"),
    "model_org_id": os.getenv("VIAM_MODEL_ORG_ID", "your-model-org-id"),
    "model_version": os.getenv("VIAM_MODEL_VERSION", "2024-XX-XXTXX-XX-XX"),
    "org_id": os.getenv("VIAM_ORG_ID", "your-inference-org-id")
}

# Local paths - can be overridden via environment variables
METADATA_DIR = os.getenv("METADATA_DIR", "metadata")
IMAGES_DIR = os.getenv("IMAGES_DIR", "images")

# Print configuration (without showing sensitive values)
print("Current configuration:")
print(f"  Model name: {VIAM_CONFIG['model_name']}")
print(f"  Model org ID: {VIAM_CONFIG['model_org_id'][:8]}..." if len(VIAM_CONFIG['model_org_id']) > 8 else f"  Model org ID: {VIAM_CONFIG['model_org_id']}")
print(f"  Model version: {VIAM_CONFIG['model_version']}")
print(f"  Inference org ID: {VIAM_CONFIG['org_id'][:8]}..." if len(VIAM_CONFIG['org_id']) > 8 else f"  Inference org ID: {VIAM_CONFIG['org_id']}")
print(f"  Metadata dir: {METADATA_DIR}")
print(f"  Images dir: {IMAGES_DIR}")

# %% [markdown]
# ## Helper Functions

# %%
def load_metadata(metadata_path: str) -> Dict[str, Any]:
    """Load metadata JSON file"""
    with open(metadata_path, 'r') as f:
        return json.load(f)

def get_binary_data_id(metadata: Dict[str, Any]) -> str:
    """Extract binary data ID from metadata"""
    return metadata.get('id', '')

def run_viam_inference(binary_data_id: str, config: Dict[str, str]) -> Dict[str, Any]:
    """Run Viam cloud inference on a binary data ID"""
    cmd = [
        "viam", "infer",
        "--binary-data-id", binary_data_id,
        "--model-name", config["model_name"],
        "--model-org-id", config["model_org_id"],
        "--model-version", config["model_version"],
        "--org-id", config["org_id"]
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return {"success": True, "output": result.stdout, "error": None}
    except subprocess.CalledProcessError as e:
        return {"success": False, "output": None, "error": e.stderr}

def parse_inference_output(output: str) -> Dict[str, Any]:
    """Parse the output from viam infer command"""
    # This is a basic parser - you may need to adjust based on actual output format
    lines = output.strip().split('\n')
    
    inference_data = {
        "tensors": {},
        "annotations": []
    }
    
    # Parse tensor information
    in_tensors = False
    for line in lines:
        if "Output Tensors:" in line:
            in_tensors = True
            continue
        elif "Annotations:" in line:
            in_tensors = False
            continue
        elif in_tensors and "Tensor Name:" in line:
            # Parse tensor info - format: "Tensor Name: name Shape: [shape] Values: [values]"
            parts = line.split()
            if len(parts) >= 4:
                tensor_name = parts[2]
                # Extract shape and values - this is a simplified parser
                inference_data["tensors"][tensor_name] = line
    
    return inference_data

# %% [markdown]
# ## Load and Process Metadata Files

# %%
def get_metadata_files():
    """Get all metadata JSON files"""
    metadata_files = glob.glob(os.path.join(METADATA_DIR, "*.json"))
    return metadata_files

# Load metadata files
metadata_files = get_metadata_files()
print(f"Found {len(metadata_files)} metadata files")

# Display first metadata file structure
if metadata_files:
    sample_metadata = load_metadata(metadata_files[0])
    print("\nSample metadata structure:")
    print(json.dumps(sample_metadata, indent=2)[:500] + "...")

# %% [markdown]
# ## Debug and Check Data Availability

# %%
def debug_data_access():
    """Debug data access and binary data IDs"""
    if not metadata_files:
        print("No metadata files found!")
        return
    
    print("=== Debugging Data Access ===")
    print(f"Config org ID: {VIAM_CONFIG['org_id']}")
    print(f"Config model org ID: {VIAM_CONFIG['model_org_id']}")
    
    # Check first few metadata files
    for i, metadata_file in enumerate(metadata_files[:3]):
        print(f"\n--- Metadata file {i+1}: {os.path.basename(metadata_file)} ---")
        metadata = load_metadata(metadata_file)
        binary_data_id = get_binary_data_id(metadata)
        
        print(f"Binary data ID: {binary_data_id}")
        print(f"File org ID: {metadata['captureMetadata']['organizationId']}")
        print(f"Capture time: {metadata['timeRequested']}")
        print(f"URI: {metadata.get('uri', 'N/A')}")
        
        # Check if org IDs match
        if VIAM_CONFIG['org_id'] != metadata['captureMetadata']['organizationId']:
            print("⚠️  WARNING: Config org ID doesn't match file org ID")

# Run debug (uncomment to execute)
debug_data_access()

# %% [markdown]
# ## Run Cloud Inference on Single Image (Test)

# %%
def test_single_inference():
    """Test inference on a single image to see the output format"""
    if not metadata_files:
        print("No metadata files found!")
        return
    
    # Get the first metadata file
    test_metadata = load_metadata(metadata_files[0])
    binary_data_id = get_binary_data_id(test_metadata)
    
    print(f"Testing inference on binary data ID: {binary_data_id}")
    print(f"Ground truth from metadata:")
    
    # Display ground truth annotations
    if 'annotations' in test_metadata:
        for annotation in test_metadata['annotations'].get('bboxes', []):
            print(f"  - Label: {annotation['label']}")
            print(f"    Bbox: x_min={annotation['xMinNormalized']:.3f}, y_min={annotation['yMinNormalized']:.3f}")
            print(f"          x_max={annotation['xMaxNormalized']:.3f}, y_max={annotation['yMaxNormalized']:.3f}")
    
    # Run inference
    print("\nRunning cloud inference...")
    result = run_viam_inference(binary_data_id, VIAM_CONFIG)
    
    if result["success"]:
        print("✅ Inference successful!")
        print("\nRaw output:")
        print(result["output"])
        
        # Parse the output
        parsed = parse_inference_output(result["output"])
        print("\nParsed inference data:")
        print(json.dumps(parsed, indent=2))
    else:
        print("❌ Inference failed!")
        print(f"Error: {result['error']}")

# Run the test (uncomment to execute)
# test_single_inference()

# %% [markdown]
# ## Process All Images

# %%
def process_all_images():
    """Process all images through cloud inference"""
    results = []
    
    for metadata_file in metadata_files:
        print(f"\nProcessing {os.path.basename(metadata_file)}...")
        
        # Load metadata
        metadata = load_metadata(metadata_file)
        binary_data_id = get_binary_data_id(metadata)
        
        # Run inference
        inference_result = run_viam_inference(binary_data_id, VIAM_CONFIG)
        
        # Store results
        result_entry = {
            "metadata_file": metadata_file,
            "binary_data_id": binary_data_id,
            "ground_truth": metadata.get('annotations', {}),
            "inference_success": inference_result["success"],
            "inference_output": inference_result["output"],
            "inference_error": inference_result["error"]
        }
        
        if inference_result["success"]:
            result_entry["parsed_inference"] = parse_inference_output(inference_result["output"])
        
        results.append(result_entry)
    
    return results

# Process all images (uncomment to execute)
# all_results = process_all_images()

# %% [markdown]
# ## Compare Results with Ground Truth

# %%
def compare_results(results: List[Dict[str, Any]]):
    """Compare inference results with ground truth annotations"""
    comparison_data = []
    
    for result in results:
        if not result["inference_success"]:
            continue
            
        # Extract ground truth
        gt_bboxes = result["ground_truth"].get("bboxes", [])
        
        # Extract inference results (you'll need to adapt this based on actual output format)
        inference_data = result.get("parsed_inference", {})
        
        comparison_entry = {
            "file": os.path.basename(result["metadata_file"]),
            "ground_truth_count": len(gt_bboxes),
            "ground_truth_labels": [bbox["label"] for bbox in gt_bboxes],
            "inference_raw": result["inference_output"][:200] + "..." if result["inference_output"] else "",
            # Add more fields as needed based on actual inference output format
        }
        
        comparison_data.append(comparison_entry)
    
    return pd.DataFrame(comparison_data)

# Compare results (uncomment after running inference)
# if 'all_results' in locals():
#     comparison_df = compare_results(all_results)
#     print("\nComparison Results:")
#     print(comparison_df.to_string())

# %% [markdown]
# ## Export Results

# %%
def export_results(results: List[Dict[str, Any]], output_path: str = "inference_results.json"):
    """Export results to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results exported to {output_path}")

# Export results (uncomment after running inference)
# if 'all_results' in locals():
#     export_results(all_results)

# %% [markdown]
# ## Next Steps
# 
# 1. **Create a .env file** with your actual Viam configuration:
#    ```
#    VIAM_MODEL_NAME=your-burner-detection-model
#    VIAM_MODEL_ORG_ID=33604ab0-737c-4481-baa2-b526ccb00362
#    VIAM_MODEL_VERSION=2024-XX-XXTXX-XX-XX
#    VIAM_ORG_ID=33604ab0-737c-4481-baa2-b526ccb00362
#    METADATA_DIR=metadata
#    IMAGES_DIR=images
#    ```
# 2. **Install dependencies** (recommended to use a virtual environment):
#    ```bash
#    python3 -m venv venv
#    source venv/bin/activate  # On Windows: venv\Scripts\activate
#    pip install -r requirements.txt
#    ```
# 3. Run the test_single_inference() function to see the output format
# 4. Adjust the parse_inference_output() function based on actual output format
# 5. Process all images using process_all_images()
# 6. Compare results with ground truth using compare_results()
