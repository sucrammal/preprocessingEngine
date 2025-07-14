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
    "model_org_id": os.getenv("VIAM_ORG_ID", "your-model-org-id"),
    "model_version": os.getenv("VIAM_MODEL_VERSION", "2024-XX-XXTXX-XX-XX"),
}

METADATA_DIR = os.getenv("METADATA_DIR", "metadata")
IMAGES_DIR = os.getenv("IMAGES_DIR", "data")
MODEL_DIR = os.getenv("MODEL_DIR", "models")

MODEL_INPUT_SIZE = None

INFERENCE_CONFIDENCE = float(os.getenv("INFERENCE_CONFIDENCE", "0.5"))
IOU_CONFIDENCE = float(os.getenv("IOU_CONFIDENCE", "0.5"))  # Fixed: was using INFERENCE_CONFIDENCE env var
PREPROCESSING_METHOD = os.getenv("PREPROCESSING_METHOD", "simple")

# LCN Configuration - New configurable parameters
LCN_WINDOW_SIZE = int(os.getenv("LCN_WINDOW_SIZE", "9"))
LCN_EPSILON = float(os.getenv("LCN_EPSILON", "1e-8"))
LCN_NORMALIZATION_TYPE = os.getenv("LCN_NORMALIZATION_TYPE", "divisive")  # Options: divisive, subtractive, adaptive
LCN_WINDOW_SHAPE = os.getenv("LCN_WINDOW_SHAPE", "square")  # Options: square, circular, gaussian
LCN_STATISTICAL_MEASURE = os.getenv("LCN_STATISTICAL_MEASURE", "mean")  # Options: mean, median, percentile
LCN_CONTRAST_BOOST = float(os.getenv("LCN_CONTRAST_BOOST", "1.0"))

# Configuration printing will only happen when script is run directly
def print_configuration_summary():
    """Print configuration summary"""
    print("🔧 CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Model: {VIAM_CONFIG['model_name']} v{VIAM_CONFIG['model_version']}")
    print(f"Data: {len(glob.glob(os.path.join(METADATA_DIR, '*.json')))} metadata files")
    print(f"Preprocessing method: {PREPROCESSING_METHOD}")
    print(f"Inference confidence threshold: {INFERENCE_CONFIDENCE}")
    print(f"IoU evaluation threshold: {IOU_CONFIDENCE}")
    
    # Show LCN configuration if using LCN
    if PREPROCESSING_METHOD == "lcn":
        print(f"\n🎛️  LCN Configuration:")
        print(f"   Window size: {LCN_WINDOW_SIZE}")
        print(f"   Window shape: {LCN_WINDOW_SHAPE}")
        print(f"   Normalization type: {LCN_NORMALIZATION_TYPE}")
        print(f"   Statistical measure: {LCN_STATISTICAL_MEASURE}")
        print(f"   Contrast boost: {LCN_CONTRAST_BOOST}")
        print(f"   Epsilon: {LCN_EPSILON}")
    
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


def load_dataset_dataframe(file_path: str = "burner_dataset_complete.pkl") -> pd.DataFrame:
    """
    Load a saved dataset DataFrame from a local file
    
    Args:
        file_path: Path to the saved DataFrame file (.pkl, .csv, or .json)
        
    Returns:
        Loaded DataFrame, or empty DataFrame if file not found
    """
    print(f"\n📂 LOADING DATASET FROM: {file_path}")
    print("=" * 60)
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        print("💡 Available files in current directory:")
        for f in os.listdir('.'):
            if f.endswith(('.pkl', '.csv', '.json')):
                print(f"   - {f}")
        return pd.DataFrame()
    
    try:
        if file_path.endswith('.pkl'):
            df = pd.read_pickle(file_path)
            print(f"✅ Loaded DataFrame from pickle: {file_path}")
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            print(f"✅ Loaded DataFrame from CSV: {file_path}")
            print("⚠️  Note: CSV format may not preserve all data types (images, bounding boxes)")
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
            print(f"✅ Loaded DataFrame from JSON: {file_path}")
            print("⚠️  Note: JSON format may not preserve all data types (images, bounding boxes)")
        else:
            print(f"❌ Unsupported file format: {file_path}")
            print("Supported formats: .pkl, .csv, .json")
            return pd.DataFrame()
        
        # Print dataset summary
        print(f"\n📊 DATASET SUMMARY")
        print("=" * 40)
        print(f"Total images: {len(df)}")
        
        if 'num_burners' in df.columns:
            print(f"Images with ground truth burners: {len(df[df['has_burners']]) if 'has_burners' in df.columns else 'N/A'}")
            print(f"Total ground truth burners: {df['num_burners'].sum()}")
            print(f"Average burners per image: {df['num_burners'].mean():.2f}")
        
        if 'num_inferred_burners' in df.columns:
            print(f"Images with predicted burners: {len(df[df['has_inferred_burners']]) if 'has_inferred_burners' in df.columns else 'N/A'}")
            print(f"Total predicted burners: {df['num_inferred_burners'].sum()}")
            print(f"Average predicted burners per image: {df['num_inferred_burners'].mean():.2f}")
        
        # Show available columns
        print(f"\n📋 Available columns:")
        for col in df.columns:
            print(f"   - {col}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading DataFrame: {e}")
        return pd.DataFrame()


def save_dataset_dataframe(df: pd.DataFrame, 
                          file_path: str = "burner_dataset_complete.pkl",
                          save_format: str = "pickle",
                          include_images: bool = True) -> bool:
    """
    Save a dataset DataFrame to a local file
    
    Args:
        df: DataFrame to save
        file_path: Path where to save the file
        save_format: Format to save in ('pickle', 'csv', 'json')
        include_images: Whether to include image arrays (only for pickle format)
        
    Returns:
        True if save was successful, False otherwise
    """
    print(f"\n💾 SAVING DATASET TO: {file_path}")
    print("=" * 60)
    
    if df.empty:
        print("❌ DataFrame is empty - nothing to save")
        return False
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
        
        if save_format.lower() == "pickle":
            # Save as pickle (preserves all data types including images)
            df.to_pickle(file_path)
            print(f"✅ Saved DataFrame to pickle: {file_path}")
            print(f"   File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
            
        elif save_format.lower() == "csv":
            # Save as CSV (loses image arrays and complex data types)
            if include_images and 'image' in df.columns:
                print("⚠️  Warning: CSV format cannot preserve image arrays")
                print("   Creating summary DataFrame without images...")
                # Create summary DataFrame without image arrays
                summary_df = df.drop(columns=['image'])
                summary_df.to_csv(file_path, index=False)
            else:
                df.to_csv(file_path, index=False)
            print(f"✅ Saved DataFrame to CSV: {file_path}")
            
        elif save_format.lower() == "json":
            # Save as JSON (loses image arrays and complex data types)
            if include_images and 'image' in df.columns:
                print("⚠️  Warning: JSON format cannot preserve image arrays")
                print("   Creating summary DataFrame without images...")
                # Create summary DataFrame without image arrays
                summary_df = df.drop(columns=['image'])
                summary_df.to_json(file_path, orient='records', indent=2)
            else:
                df.to_json(file_path, orient='records', indent=2)
            print(f"✅ Saved DataFrame to JSON: {file_path}")
            
        else:
            print(f"❌ Unsupported save format: {save_format}")
            print("Supported formats: pickle, csv, json")
            return False
        
        # Print save summary
        print(f"\n📊 SAVE SUMMARY")
        print("=" * 40)
        print(f"Total images: {len(df)}")
        print(f"Columns saved: {len(df.columns)}")
        print(f"Save format: {save_format}")
        
        if 'num_burners' in df.columns:
            print(f"Ground truth burners: {df['num_burners'].sum()}")
        if 'num_inferred_burners' in df.columns:
            print(f"Predicted burners: {df['num_inferred_burners'].sum()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving DataFrame: {e}")
        return False


def save_dataset_summary(df: pd.DataFrame, 
                        summary_file: str = "dataset_summary.csv",
                        detailed_summary: bool = True) -> bool:
    """
    Save a detailed summary of the dataset (without image arrays)
    
    Args:
        df: DataFrame to summarize
        summary_file: Path for the summary file
        detailed_summary: Whether to include detailed statistics
        
    Returns:
        True if save was successful, False otherwise
    """
    print(f"\n📋 SAVING DATASET SUMMARY TO: {summary_file}")
    print("=" * 60)
    
    if df.empty:
        print("❌ DataFrame is empty - nothing to summarize")
        return False
    
    try:
        # Create summary DataFrame with key columns
        summary_columns = ['image_name', 'image_path', 'num_burners', 'has_burners']
        if 'inferred_burner_bboxes' in df.columns:
            summary_columns.extend(['num_inferred_burners', 'has_inferred_burners'])
        
        # Filter to only include columns that exist
        available_columns = [col for col in summary_columns if col in df.columns]
        summary_df = df[available_columns].copy()
        
        # Add detailed statistics if requested
        if detailed_summary:
            # Add metadata about the dataset
            summary_stats = {
                'total_images': len(df),
                'images_with_gt_burners': len(df[df['has_burners']]) if 'has_burners' in df.columns else 0,
                'images_with_pred_burners': len(df[df['has_inferred_burners']]) if 'has_inferred_burners' in df.columns else 0,
                'total_gt_burners': df['num_burners'].sum() if 'num_burners' in df.columns else 0,
                'total_pred_burners': df['num_inferred_burners'].sum() if 'num_inferred_burners' in df.columns else 0,
                'avg_gt_burners_per_image': df['num_burners'].mean() if 'num_burners' in df.columns else 0,
                'avg_pred_burners_per_image': df['num_inferred_burners'].mean() if 'num_inferred_burners' in df.columns else 0
            }
            
            # Create a separate stats file
            stats_file = summary_file.replace('.csv', '_stats.json')
            with open(stats_file, 'w') as f:
                json.dump(summary_stats, f, indent=2)
            print(f"✅ Saved detailed statistics to: {stats_file}")
        
        # Save summary DataFrame
        summary_df.to_csv(summary_file, index=False)
        print(f"✅ Saved summary to: {summary_file}")
        
        # Print summary
        print(f"\n📊 SUMMARY STATISTICS")
        print("=" * 40)
        print(f"Total images: {len(summary_df)}")
        if 'num_burners' in summary_df.columns:
            print(f"Images with ground truth burners: {len(summary_df[summary_df['has_burners']])}")
            print(f"Total ground truth burners: {summary_df['num_burners'].sum()}")
        if 'num_inferred_burners' in summary_df.columns:
            print(f"Images with predicted burners: {len(summary_df[summary_df['has_inferred_burners']])}")
            print(f"Total predicted burners: {summary_df['num_inferred_burners'].sum()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving summary: {e}")
        return False


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

def create_lcn_kernel(window_size: int, shape: str = "square") -> np.ndarray:
    """
    Create different kernel shapes for LCN
    
    Args:
        window_size: Size of the kernel (will be made odd if even)
        shape: 'square', 'circular', or 'gaussian'
    """
    # Ensure odd window size
    if window_size % 2 == 0:
        window_size += 1
    
    if shape == "square":
        return np.ones((window_size, window_size))
    
    elif shape == "circular":
        kernel = np.zeros((window_size, window_size))
        center = window_size // 2
        radius = center
        y, x = np.ogrid[:window_size, :window_size]
        mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
        kernel[mask] = 1
        return kernel
    
    elif shape == "gaussian":
        # Create Gaussian kernel
        kernel = np.zeros((window_size, window_size))
        center = window_size // 2
        sigma = window_size / 6.0  # Standard deviation
        
        for i in range(window_size):
            for j in range(window_size):
                x, y = i - center, j - center
                kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        
        # Normalize kernel
        kernel = kernel / np.sum(kernel)
        return kernel
    
    else:
        raise ValueError(f"Unknown kernel shape: {shape}")


def apply_local_contrast_normalization(image_array: np.ndarray, 
                                     window_size: int = None, 
                                     epsilon: float = None,
                                     normalization_type: str = None,
                                     window_shape: str = None,
                                     statistical_measure: str = None,
                                     contrast_boost: float = None) -> np.ndarray:
    """
    Apply Enhanced Local Contrast Normalization (LCN)
    
    Args:
        image_array: Input image array
        window_size: Size of local window (uses global LCN_WINDOW_SIZE if None)
        epsilon: Small value to prevent division by zero (uses global LCN_EPSILON if None)
        normalization_type: Type of normalization - 'divisive', 'subtractive', 'adaptive'
        window_shape: Shape of window - 'square', 'circular', 'gaussian'
        statistical_measure: Statistical measure - 'mean', 'median', 'percentile'
        contrast_boost: Contrast enhancement factor
    
    Returns:
        Normalized image array
    """
    # Use global config if parameters not provided
    window_size = window_size or LCN_WINDOW_SIZE
    epsilon = epsilon or LCN_EPSILON
    normalization_type = normalization_type or LCN_NORMALIZATION_TYPE
    window_shape = window_shape or LCN_WINDOW_SHAPE
    statistical_measure = statistical_measure or LCN_STATISTICAL_MEASURE
    contrast_boost = contrast_boost or LCN_CONTRAST_BOOST
    
    # Ensure odd window size
    if window_size % 2 == 0:
        window_size += 1
    
    # Validate inputs
    if not isinstance(image_array, np.ndarray) or image_array.ndim != 3:
        raise ValueError("Input must be a 3D numpy array (height, width, channels)")
    
    # Increase epsilon for better numerical stability
    epsilon = max(epsilon, 1e-6)
    
    normalized = np.zeros_like(image_array, dtype=np.float32)
    
    # Create kernel based on shape
    if window_shape == "gaussian":
        kernel = create_lcn_kernel(window_size, window_shape)
    
    for channel in range(image_array.shape[2]):
        channel_data = image_array[:, :, channel].astype(np.float32)
        
        # Check for problematic input data
        if np.all(channel_data == channel_data.flat[0]):
            # Uniform channel - just normalize to 0.5
            normalized[:, :, channel] = 0.5
            continue
        
        # Calculate local statistics based on statistical measure
        try:
            if statistical_measure == "mean":
                if window_shape == "gaussian":
                    local_stat = ndimage.convolve(channel_data, kernel, mode='reflect')
                else:
                    local_stat = ndimage.uniform_filter(channel_data, size=window_size, mode='reflect')
                    
            elif statistical_measure == "median":
                local_stat = ndimage.median_filter(channel_data, size=window_size, mode='reflect')
                
            elif statistical_measure == "percentile":
                # Use 25th percentile for local statistic
                local_stat = ndimage.percentile_filter(channel_data, percentile=25, size=window_size, mode='reflect')
            
            else:
                raise ValueError(f"Unknown statistical measure: {statistical_measure}")
        
        except Exception as e:
            print(f"⚠️ Warning: Error in statistical calculation for channel {channel}: {e}")
            # Fallback to simple uniform filter
            local_stat = ndimage.uniform_filter(channel_data, size=window_size, mode='reflect')
        
        # Calculate local variance/std for divisive normalization
        if normalization_type in ["divisive", "adaptive"]:
            try:
                if window_shape == "gaussian":
                    local_variance = ndimage.convolve(channel_data**2, kernel, mode='reflect') - local_stat**2
                else:
                    local_variance = ndimage.uniform_filter(channel_data**2, size=window_size, mode='reflect') - local_stat**2
                    
                # Ensure non-negative variance and add epsilon
                local_variance = np.maximum(local_variance, 0)
                local_std = np.sqrt(local_variance) + epsilon
                
            except Exception as e:
                print(f"⚠️ Warning: Error in variance calculation for channel {channel}: {e}")
                # Fallback to a constant std
                local_std = np.ones_like(channel_data) * epsilon
        
        # Apply normalization based on type
        try:
            if normalization_type == "divisive":
                # Classic divisive normalization: (x - local_mean) / local_std
                channel_normalized = (channel_data - local_stat) / local_std
                
            elif normalization_type == "subtractive":
                # Subtractive normalization: x - local_mean
                channel_normalized = channel_data - local_stat
                
            elif normalization_type == "adaptive":
                # Adaptive normalization: stronger normalization in low-variance regions
                # Cap the variance weight to prevent extreme values
                variance_weight = np.clip(1.0 / local_std, 0, 100)  # Cap at 100x amplification
                channel_normalized = (channel_data - local_stat) * variance_weight * contrast_boost
                
            else:
                raise ValueError(f"Unknown normalization type: {normalization_type}")
        
        except Exception as e:
            print(f"⚠️ Warning: Error in normalization for channel {channel}: {e}")
            # Fallback to simple normalization
            channel_normalized = (channel_data - channel_data.mean()) / (channel_data.std() + epsilon)
        
        # Apply contrast boost (with safety limits)
        if contrast_boost != 1.0 and normalization_type != "adaptive":
            # Limit contrast boost to reasonable range
            safe_boost = np.clip(contrast_boost, 0.1, 5.0)
            channel_normalized = channel_normalized * safe_boost
        
        # Check for NaN or inf values
        if np.any(~np.isfinite(channel_normalized)):
            print(f"⚠️ Warning: Non-finite values detected in channel {channel}, replacing with zeros")
            channel_normalized = np.nan_to_num(channel_normalized, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Rescale to 0-1 range with improved stability
        channel_min = np.percentile(channel_normalized, 1)  # Use 1st percentile instead of min
        channel_max = np.percentile(channel_normalized, 99)  # Use 99th percentile instead of max
        
        if channel_max > channel_min + epsilon:
            # Normal rescaling
            channel_normalized = (channel_normalized - channel_min) / (channel_max - channel_min)
            # Clip to [0, 1] range
            channel_normalized = np.clip(channel_normalized, 0, 1)
        else:
            # Nearly uniform channel - map to middle value
            print(f"⚠️ Warning: Channel {channel} has very low dynamic range, setting to 0.5")
            channel_normalized = np.full_like(channel_normalized, 0.5)
        
        # Final safety check
        if np.any(channel_normalized < 0) or np.any(channel_normalized > 1):
            print(f"⚠️ Warning: Channel {channel} values outside [0,1] range, clipping")
            channel_normalized = np.clip(channel_normalized, 0, 1)
        
        normalized[:, :, channel] = channel_normalized
    
    return normalized


def diagnose_lcn_issues(image_array: np.ndarray, 
                       window_size: int = 9,
                       normalization_type: str = "divisive") -> Dict[str, Any]:
    """
    Diagnose potential issues with LCN processing
    
    Args:
        image_array: Input image array
        window_size: Window size for LCN
        normalization_type: Type of normalization
    
    Returns:
        Dictionary with diagnostic information
    """
    print(f"\n🔍 DIAGNOSING LCN ISSUES")
    print("=" * 50)
    
    diagnostics = {}
    
    # Basic image info
    diagnostics['input_shape'] = image_array.shape
    diagnostics['input_dtype'] = image_array.dtype
    diagnostics['input_range'] = (image_array.min(), image_array.max())
    
    print(f"Input image shape: {image_array.shape}")
    print(f"Input image dtype: {image_array.dtype}")
    print(f"Input image range: [{image_array.min():.3f}, {image_array.max():.3f}]")
    
    # Check each channel
    print(f"\nChannel analysis:")
    for channel in range(image_array.shape[2]):
        channel_data = image_array[:, :, channel]
        channel_mean = channel_data.mean()
        channel_std = channel_data.std()
        channel_min = channel_data.min()
        channel_max = channel_data.max()
        
        print(f"  Channel {channel}: mean={channel_mean:.3f}, std={channel_std:.3f}, range=[{channel_min:.3f}, {channel_max:.3f}]")
        
        # Check for problematic conditions
        if channel_std < 1e-6:
            print(f"    ⚠️ Channel {channel} has very low variance (uniform)")
        if channel_min == channel_max:
            print(f"    ⚠️ Channel {channel} is completely uniform")
        
        diagnostics[f'channel_{channel}'] = {
            'mean': channel_mean,
            'std': channel_std,
            'min': channel_min,
            'max': channel_max
        }
    
    # Test LCN processing
    print(f"\nTesting LCN with window_size={window_size}, type={normalization_type}:")
    
    try:
        result = apply_local_contrast_normalization(
            image_array, 
            window_size=window_size,
            normalization_type=normalization_type,
            window_shape="square"
        )
        
        diagnostics['lcn_success'] = True
        diagnostics['output_shape'] = result.shape
        diagnostics['output_range'] = (result.min(), result.max())
        
        print(f"✅ LCN processing successful")
        print(f"Output shape: {result.shape}")
        print(f"Output range: [{result.min():.3f}, {result.max():.3f}]")
        
        # Check output channels
        print(f"\nOutput channel analysis:")
        for channel in range(result.shape[2]):
            channel_data = result[:, :, channel]
            channel_mean = channel_data.mean()
            channel_std = channel_data.std()
            
            print(f"  Channel {channel}: mean={channel_mean:.3f}, std={channel_std:.3f}")
            
            # Check for problematic output
            if channel_mean < 0.01 or channel_mean > 0.99:
                print(f"    ⚠️ Channel {channel} mean is extreme: {channel_mean:.3f}")
            if channel_std < 0.01:
                print(f"    ⚠️ Channel {channel} has very low output variance")
                
            diagnostics[f'output_channel_{channel}'] = {
                'mean': channel_mean,
                'std': channel_std
            }
        
        # Check for color artifacts
        if result.shape[2] == 3:  # RGB image
            r_mean, g_mean, b_mean = [result[:,:,i].mean() for i in range(3)]
            
            # Check for extreme channel imbalances
            channel_means = [r_mean, g_mean, b_mean]
            max_mean, min_mean = max(channel_means), min(channel_means)
            
            if max_mean - min_mean > 0.5:
                print(f"    ⚠️ Large channel imbalance detected!")
                print(f"    R: {r_mean:.3f}, G: {g_mean:.3f}, B: {b_mean:.3f}")
                diagnostics['color_imbalance'] = True
            else:
                diagnostics['color_imbalance'] = False
        
    except Exception as e:
        print(f"❌ LCN processing failed: {e}")
        diagnostics['lcn_success'] = False
        diagnostics['error'] = str(e)
    
    return diagnostics


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


def test_lcn_parameters(image_array: np.ndarray, 
                       window_sizes: List[int] = [5, 9, 15, 25],
                       normalization_types: List[str] = ["divisive", "subtractive", "adaptive"],
                       window_shapes: List[str] = ["square", "circular", "gaussian"],
                       crop_size: int = 150):
    """
    Test different LCN parameter combinations and visualize the effects
    
    Args:
        image_array: Input image array
        window_sizes: List of window sizes to test
        normalization_types: List of normalization types to test
        window_shapes: List of window shapes to test
        crop_size: Size of cropped region for detail view
    """
    print(f"\n🧪 TESTING LCN PARAMETERS")
    print("=" * 60)
    
    # Create a center crop for detailed analysis
    h, w = image_array.shape[:2]
    center_y, center_x = h // 2, w // 2
    y1 = max(0, center_y - crop_size // 2)
    y2 = min(h, center_y + crop_size // 2)
    x1 = max(0, center_x - crop_size // 2)
    x2 = min(w, center_x + crop_size // 2)
    
    # Test 1: Window Size Effects
    print(f"\n1️⃣ Testing Window Sizes: {window_sizes}")
    fig, axes = plt.subplots(2, len(window_sizes), figsize=(4 * len(window_sizes), 8))
    
    for i, window_size in enumerate(window_sizes):
        # Apply LCN with different window sizes
        lcn_result = apply_local_contrast_normalization(
            image_array, 
            window_size=window_size,
            normalization_type="divisive",
            window_shape="square"
        )
        
        # Show full image
        axes[0, i].imshow((lcn_result * 255).astype(np.uint8))
        axes[0, i].set_title(f"Window Size: {window_size}")
        axes[0, i].axis('off')
        
        # Show cropped detail
        cropped = lcn_result[y1:y2, x1:x2]
        axes[1, i].imshow((cropped * 255).astype(np.uint8))
        axes[1, i].set_title(f"Detail: {window_size}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Test 2: Normalization Type Effects
    print(f"\n2️⃣ Testing Normalization Types: {normalization_types}")
    fig, axes = plt.subplots(2, len(normalization_types), figsize=(4 * len(normalization_types), 8))
    
    for i, norm_type in enumerate(normalization_types):
        # Apply LCN with different normalization types
        lcn_result = apply_local_contrast_normalization(
            image_array, 
            window_size=9,
            normalization_type=norm_type,
            window_shape="square"
        )
        
        # Show full image
        axes[0, i].imshow((lcn_result * 255).astype(np.uint8))
        axes[0, i].set_title(f"Type: {norm_type}")
        axes[0, i].axis('off')
        
        # Show cropped detail
        cropped = lcn_result[y1:y2, x1:x2]
        axes[1, i].imshow((cropped * 255).astype(np.uint8))
        axes[1, i].set_title(f"Detail: {norm_type}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Test 3: Window Shape Effects
    print(f"\n3️⃣ Testing Window Shapes: {window_shapes}")
    fig, axes = plt.subplots(2, len(window_shapes), figsize=(4 * len(window_shapes), 8))
    
    for i, shape in enumerate(window_shapes):
        # Apply LCN with different window shapes
        lcn_result = apply_local_contrast_normalization(
            image_array, 
            window_size=9,
            normalization_type="divisive",
            window_shape=shape
        )
        
        # Show full image
        axes[0, i].imshow((lcn_result * 255).astype(np.uint8))
        axes[0, i].set_title(f"Shape: {shape}")
        axes[0, i].axis('off')
        
        # Show cropped detail
        cropped = lcn_result[y1:y2, x1:x2]
        axes[1, i].imshow((cropped * 255).astype(np.uint8))
        axes[1, i].set_title(f"Detail: {shape}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Show current global configuration
    print(f"\n🎛️ Current Global LCN Configuration:")
    print(f"   Window Size: {LCN_WINDOW_SIZE}")
    print(f"   Window Shape: {LCN_WINDOW_SHAPE}")
    print(f"   Normalization Type: {LCN_NORMALIZATION_TYPE}")
    print(f"   Statistical Measure: {LCN_STATISTICAL_MEASURE}")
    print(f"   Contrast Boost: {LCN_CONTRAST_BOOST}")
    print(f"   Epsilon: {LCN_EPSILON}")


def compare_lcn_configurations(image_array: np.ndarray, 
                              configs: List[Dict],
                              config_names: List[str] = None):
    """
    Compare different LCN configurations side by side
    
    Args:
        image_array: Input image array
        configs: List of configuration dictionaries
        config_names: Optional names for each configuration
    """
    if config_names is None:
        config_names = [f"Config {i+1}" for i in range(len(configs))]
    
    print(f"\n🔍 COMPARING LCN CONFIGURATIONS")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, len(configs), figsize=(4 * len(configs), 8))
    if len(configs) == 1:
        axes = axes.reshape(2, 1)
    
    for i, (config, name) in enumerate(zip(configs, config_names)):
        # Apply LCN with specific configuration
        lcn_result = apply_local_contrast_normalization(image_array, **config)
        
        # Show full image
        axes[0, i].imshow((lcn_result * 255).astype(np.uint8))
        axes[0, i].set_title(f"{name}")
        axes[0, i].axis('off')
        
        # Show histogram
        axes[1, i].hist(lcn_result.flatten(), bins=50, alpha=0.7, color=f'C{i}')
        axes[1, i].set_title(f"{name} Histogram")
        axes[1, i].set_xlabel("Pixel Value")
        axes[1, i].set_ylabel("Frequency")
        axes[1, i].set_xlim(0, 1)
        
        # Print configuration details
        print(f"\n{name}:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    plt.tight_layout()
    plt.show()


print("✅ Enhanced preprocessing functions defined")
print("💡 New LCN parameters available:")
print("   - LCN_WINDOW_SIZE: Window size for local statistics")
print("   - LCN_NORMALIZATION_TYPE: divisive, subtractive, adaptive")
print("   - LCN_WINDOW_SHAPE: square, circular, gaussian")
print("   - LCN_STATISTICAL_MEASURE: mean, median, percentile")
print("   - LCN_CONTRAST_BOOST: Contrast enhancement factor")
print("   - LCN_EPSILON: Small value to prevent division by zero")


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


def run_inference_on_dataframe(df: pd.DataFrame, interpreter, 
                              normalization_method: str = "simple",
                              confidence_threshold: float = 0.5,
                              debug: bool = False,
                              auto_save: bool = True,
                              save_path: str = "burner_dataset_complete.pkl") -> pd.DataFrame:
    """
    Run inference on all images in the DataFrame
    
    Args:
        df: DataFrame with images
        interpreter: TFLite interpreter
        normalization_method: Preprocessing method ('simple', 'gcn', 'lcn')
        confidence_threshold: Minimum confidence score for detections
        debug: Whether to show detailed debug information
        auto_save: Whether to automatically save the DataFrame after inference
        save_path: Path to save the DataFrame (if auto_save is True)
        
    Returns:
        DataFrame with inference results added
    """
    
    print(f"\n🎯 RUNNING INFERENCE ON {len(df)} IMAGES")
    print("=" * 60)
    print(f"Using {normalization_method} normalization")
    print(f"Confidence threshold: {confidence_threshold}")
    if auto_save:
        print(f"Auto-save enabled: {save_path}")
    
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
    
    # Auto-save if enabled
    if auto_save:
        print(f"\n💾 Auto-saving results...")
        save_success = save_dataset_dataframe(df_with_inference, save_path, "pickle", True)
        if save_success:
            print(f"✅ Results saved to: {save_path}")
        else:
            print(f"⚠️  Failed to save results to: {save_path}")
    
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


def analyze_failure_cases(df: pd.DataFrame, num_samples: int = 5, random_seed: int = 42) -> pd.DataFrame:
    """
    Analyze and visualize failure cases where presence/absence predictions don't match ground truth
    
    Args:
        df: DataFrame with inference results (must have 'has_burners' and 'has_inferred_burners' columns)
        num_samples: Number of random failure cases to display
        random_seed: Random seed for reproducible sampling
        
    Returns:
        DataFrame containing only the failure cases
    """
    print(f"\n🔍 ANALYZING FAILURE CASES")
    print("=" * 60)
    
    # Check if inference results are available
    if 'has_inferred_burners' not in df.columns:
        print("❌ No inference results found in DataFrame")
        print("Please run run_inference_on_dataframe() first")
        return pd.DataFrame()
    
    # Identify failure cases (presence/absence mismatches)
    false_positives = df[(~df['has_burners']) & (df['has_inferred_burners'])]  # No GT, but predicted
    false_negatives = df[(df['has_burners']) & (~df['has_inferred_burners'])]  # Has GT, but not predicted
    
    # Combine all failure cases
    failure_cases = pd.concat([false_positives, false_negatives], ignore_index=True)
    
    print(f"📊 Failure Case Analysis:")
    print(f"   Total images: {len(df)}")
    print(f"   False Positives: {len(false_positives)} (predicted burners, but no ground truth)")
    print(f"   False Negatives: {len(false_negatives)} (missed burners that exist in ground truth)")
    print(f"   Total failure cases: {len(failure_cases)}")
    
    if len(failure_cases) == 0:
        print("✅ No failure cases found! Perfect presence/absence accuracy.")
        return pd.DataFrame()
    
    # Calculate failure rate
    failure_rate = len(failure_cases) / len(df) * 100
    print(f"   Failure rate: {failure_rate:.1f}%")
    
    # Sample random failure cases for visualization
    np.random.seed(random_seed)
    sample_size = min(num_samples, len(failure_cases))
    sampled_failures = failure_cases.sample(n=sample_size, random_state=random_seed)
    
    print(f"\n📸 Visualizing {sample_size} random failure cases:")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Pad with empty plots if we have fewer samples than subplots
    for i in range(len(axes)):
        if i < len(sampled_failures):
            row = sampled_failures.iloc[i]
            
            # Display image
            axes[i].imshow(row['image'])
            
            # Create detailed title
            failure_type = "False Positive" if not row['has_burners'] else "False Negative"
            color = "red" if failure_type == "False Positive" else "orange"
            
            title = f"{row['image_name']}\n{failure_type}\nGT: {row['num_burners']}, Pred: {row['num_inferred_burners']}"
            axes[i].set_title(title, color=color, fontsize=10)
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
        else:
            # Hide empty subplots
            axes[i].set_visible(False)
    
    # Add legend
    import matplotlib.lines as mlines
    legend_elements = [
        mlines.Line2D([], [], color='green', linewidth=2, label='Ground Truth'),
        mlines.Line2D([], [], color='red', linewidth=2, linestyle='--', label='Predictions'),
        mlines.Line2D([], [], color='red', linewidth=2, label='False Positive'),
        mlines.Line2D([], [], color='orange', linewidth=2, label='False Negative')
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1.0))
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed analysis of sampled cases
    print(f"\n📋 Detailed Analysis of Sampled Cases:")
    print("-" * 60)
    for i, (_, row) in enumerate(sampled_failures.iterrows()):
        failure_type = "False Positive" if not row['has_burners'] else "False Negative"
        print(f"\n{i+1}. {row['image_name']} - {failure_type}")
        print(f"   Ground truth burners: {row['num_burners']}")
        print(f"   Predicted burners: {row['num_inferred_burners']}")
        
        if failure_type == "False Positive":
            print(f"   Issue: Model predicted {row['num_inferred_burners']} burners but none exist")
            print(f"   Predictions: {[f'{conf:.2f}' for *_, conf in row['inferred_burner_bboxes']]}")
        else:
            print(f"   Issue: Model missed {row['num_burners']} existing burners")
            if row['num_inferred_burners'] > 0:
                print(f"   Partial predictions: {[f'{conf:.2f}' for *_, conf in row['inferred_burner_bboxes']]}")
    
    # Provide suggestions for improvement
    print(f"\n💡 Suggestions for Improvement:")
    print("-" * 60)
    
    if len(false_positives) > 0:
        avg_fp_confidence = np.mean([
            np.mean([conf for *_, conf in row['inferred_burner_bboxes']]) 
            for _, row in false_positives.iterrows() if len(row['inferred_burner_bboxes']) > 0
        ])
        print(f"   False Positives ({len(false_positives)} cases):")
        print(f"   - Average confidence: {avg_fp_confidence:.3f}")
        print(f"   - Consider raising confidence threshold above {avg_fp_confidence:.3f}")
        print(f"   - Review images with reflections, shadows, or burner-like objects")
    
    if len(false_negatives) > 0:
        print(f"   False Negatives ({len(false_negatives)} cases):")
        print(f"   - Model is missing existing burners")
        print(f"   - Consider lowering confidence threshold")
        print(f"   - Review preprocessing method effectiveness")
        print(f"   - Check if burners are too small, occluded, or poorly lit")
    
    return failure_cases


def debug_single_image_pipeline(metadata_file: str, 
                              preprocessing_type: str = "simple",
                              model_path: str = None,
                              confidence_threshold: float = 0.5,
                              debug: bool = False) -> None:
    """
    Debug function to visualize the complete pipeline for a single image
    
    Args:
        metadata_file: Path to the metadata JSON file
        preprocessing_type: Type of preprocessing ('simple', 'gcn', 'lcn')
        model_path: Path to the .tflite model file
        confidence_threshold: Confidence threshold for inference
        debug: Whether to show detailed debug information
    """
    print(f"\n🔍 DEBUGGING SINGLE IMAGE PIPELINE")
    print("=" * 60)
    print(f"Metadata file: {metadata_file}")
    print(f"Preprocessing type: {preprocessing_type}")
    print(f"Model path: {model_path}")
    print(f"Confidence threshold: {confidence_threshold}")
    print("=" * 60)
    
    try:
        # Step 1: Load metadata
        if not os.path.exists(metadata_file):
            print(f"❌ Metadata file not found: {metadata_file}")
            return
            
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"✅ Loaded metadata: {os.path.basename(metadata_file)}")
        
        # Step 2: Find corresponding image
        image_path = find_image_for_metadata(metadata_file)
        if not image_path:
            print(f"❌ No corresponding image found for metadata file")
            return
        
        print(f"✅ Found corresponding image: {os.path.basename(image_path)}")
        
        # Step 3: Load image
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        print(f"✅ Loaded image: {image_array.shape} (H x W x C)")
        
        # Step 4: Extract ground truth bounding boxes
        gt_bboxes = extract_burner_bounding_boxes(metadata)
        print(f"✅ Found {len(gt_bboxes)} ground truth burner bounding boxes")
        
        # Step 5: Apply preprocessing
        preprocessed_image = preprocess_image(image_array, None, preprocessing_type)
        print(f"✅ Applied {preprocessing_type} preprocessing")
        
        # Step 6: Load model and run inference (if model path provided)
        inferred_bboxes = []
        if model_path:
            if not os.path.exists(model_path):
                print(f"❌ Model file not found: {model_path}")
                model_path = None
            else:
                try:
                    interpreter = load_model(model_path)
                    print(f"✅ Loaded model: {os.path.basename(model_path)}")
                    
                    # Run inference
                    inferred_bboxes = run_single_inference(
                        image_array, 
                        interpreter, 
                        preprocessing_type, 
                        confidence_threshold,
                        debug=debug
                    )
                    print(f"✅ Inference complete: {len(inferred_bboxes)} detections")
                    
                    # Show detection details
                    if inferred_bboxes:
                        print(f"📊 Detection details:")
                        for i, (xmin, ymin, xmax, ymax, conf) in enumerate(inferred_bboxes):
                            print(f"  Detection {i+1}: confidence={conf:.3f}, "
                                  f"box=({xmin:.3f}, {ymin:.3f}, {xmax:.3f}, {ymax:.3f})")
                    
                except Exception as e:
                    print(f"❌ Error loading model or running inference: {e}")
                    model_path = None
        
        # Step 7: Create visualization
        print(f"\n📊 Creating visualization...")
        
        # Determine number of subplots
        num_plots = 3 if model_path else 2
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
        
        # Ensure axes is always a list
        if num_plots == 1:
            axes = [axes]
        elif num_plots == 2:
            axes = [axes[0], axes[1]]
        
        h, w = image_array.shape[:2]
        
        # Plot 1: Original image with ground truth bounding boxes
        axes[0].imshow(image_array)
        axes[0].set_title(f"Original Image\n{len(gt_bboxes)} Ground Truth Burners")
        axes[0].axis('off')
        
        # Draw ground truth bounding boxes (green)
        for bbox in gt_bboxes:
            xmin, ymin, xmax, ymax = bbox
            x1, y1 = int(xmin * w), int(ymin * h)
            x2, y2 = int(xmax * w), int(ymax * h)
            
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                   linewidth=2, edgecolor='green', facecolor='none')
            axes[0].add_patch(rect)
            
            # Add label
            axes[0].text(x1, y1 - 5, 'GT', color='green', fontsize=10, weight='bold')
        
        # Plot 2: Preprocessed image
        # Convert preprocessed image to displayable format
        display_preprocessed = (preprocessed_image * 255).astype(np.uint8)
        axes[1].imshow(display_preprocessed)
        axes[1].set_title(f"Preprocessed Image\n({preprocessing_type.upper()})")
        axes[1].axis('off')
        
        # Plot 3: Preprocessed image with inferred bounding boxes (if model available)
        if model_path and num_plots == 3:
            axes[2].imshow(display_preprocessed)
            axes[2].set_title(f"Inference Results\n{len(inferred_bboxes)} Detections")
            axes[2].axis('off')
            
            # Draw inferred bounding boxes (red)
            for bbox in inferred_bboxes:
                xmin, ymin, xmax, ymax, confidence = bbox
                x1, y1 = int(xmin * w), int(ymin * h)
                x2, y2 = int(xmax * w), int(ymax * h)
                
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                       linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
                axes[2].add_patch(rect)
                
                # Add confidence score
                axes[2].text(x1, y1 - 5, f'{confidence:.2f}', 
                           color='red', fontsize=10, weight='bold')
        
        # Add legend
        if model_path:
            import matplotlib.lines as mlines
            legend_elements = [
                mlines.Line2D([], [], color='green', linewidth=2, label='Ground Truth'),
                mlines.Line2D([], [], color='red', linewidth=2, linestyle='--', label='Predictions')
            ]
            plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1.0))
        
        plt.tight_layout()
        plt.show()
        
        # Step 8: Print summary
        print(f"\n📋 PIPELINE SUMMARY")
        print("=" * 40)
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Original size: {image_array.shape[:2]} (H x W)")
        print(f"Preprocessing: {preprocessing_type}")
        print(f"Ground truth burners: {len(gt_bboxes)}")
        
        if model_path:
            print(f"Model: {os.path.basename(model_path)}")
            print(f"Predicted burners: {len(inferred_bboxes)}")
            print(f"Confidence threshold: {confidence_threshold}")
            
            # Basic accuracy check
            if len(gt_bboxes) > 0 and len(inferred_bboxes) > 0:
                print(f"✅ Both ground truth and predictions found")
            elif len(gt_bboxes) > 0 and len(inferred_bboxes) == 0:
                print(f"⚠️  Ground truth found but no predictions")
            elif len(gt_bboxes) == 0 and len(inferred_bboxes) > 0:
                print(f"⚠️  Predictions found but no ground truth")
            else:
                print(f"ℹ️  No ground truth or predictions")
        
        print(f"✅ Debug visualization complete!")
        
    except Exception as e:
        print(f"❌ Error in debug pipeline: {e}")
        import traceback
        traceback.print_exc()


def find_metadata_for_image(image_file: str) -> str:
    """
    Find corresponding metadata file for an image file
    
    Args:
        image_file: Path to the image file
        
    Returns:
        Path to the corresponding metadata file, or None if not found
    """
    if not os.path.exists(image_file):
        return None
    
    # Get image filename and binary ID from the path
    image_basename = os.path.basename(image_file)
    image_name_without_ext = os.path.splitext(image_basename)[0]
    
    # Search through all metadata files
    metadata_files = glob.glob(os.path.join(METADATA_DIR, "*.json"))
    
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Method 1: Try direct filename match
            metadata_filename = metadata.get('fileName', '')
            metadata_fileext = metadata.get('fileExt', '')
            full_metadata_filename = metadata_filename + metadata_fileext
            
            if full_metadata_filename == image_basename:
                return metadata_file
            
            # Method 2: Try binary ID match
            binary_id = metadata.get('id', '')
            if binary_id and binary_id in image_basename:
                return metadata_file
                
            # Method 3: Try filename without extension match
            if metadata_filename == image_name_without_ext:
                return metadata_file
                
        except Exception as e:
            # Skip problematic metadata files
            continue
    
    return None


def debug_image_pipeline(file_path: str, 
                        preprocessing_type: str = "simple",
                        model_path: str = None,
                        confidence_threshold: float = 0.5,
                        debug: bool = False) -> None:
    """
    Debug function that works with either metadata files or image files
    
    This is an overloaded version of debug_single_image_pipeline that automatically
    detects whether the input is a metadata file or image file and handles accordingly.
    
    Args:
        file_path: Path to either metadata JSON file or image file
        preprocessing_type: Type of preprocessing ('simple', 'gcn', 'lcn')
        model_path: Path to the .tflite model file
        confidence_threshold: Confidence threshold for inference
        debug: Whether to show detailed debug information
    """
    print(f"\n🔍 AUTO-DETECTING FILE TYPE AND RUNNING DEBUG PIPELINE")
    print("=" * 60)
    print(f"Input file: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    # Determine file type and get metadata file path
    metadata_file = None
    
    # Check if it's a metadata file (JSON in metadata directory)
    if (file_path.endswith('.json') and 
        (METADATA_DIR in file_path or os.path.dirname(file_path) == METADATA_DIR)):
        metadata_file = file_path
        print(f"✅ Detected metadata file")
        
    # Check if it's an image file
    elif file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        print(f"✅ Detected image file")
        print(f"🔍 Searching for corresponding metadata...")
        
        metadata_file = find_metadata_for_image(file_path)
        if metadata_file:
            print(f"✅ Found metadata: {os.path.basename(metadata_file)}")
        else:
            print(f"❌ No corresponding metadata found for image: {os.path.basename(file_path)}")
            return
    
    # Unknown file type
    else:
        print(f"❌ Unknown file type. Expected .json metadata file or image file (.jpg, .png, etc.)")
        return
    
    # Call the original debug function with metadata file
    print(f"🚀 Running debug pipeline with metadata: {os.path.basename(metadata_file)}")
    
    debug_single_image_pipeline(
        metadata_file=metadata_file,
        preprocessing_type=preprocessing_type,
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        debug=debug
    )


def main(load_existing: bool = False, existing_file: str = "burner_dataset_complete.pkl"):
    """
    Main function to run the preprocessing pipeline
    
    Args:
        load_existing: Whether to load existing dataset instead of creating new one
        existing_file: Path to existing dataset file to load
    """
    # Print configuration summary
    print_configuration_summary()
    
    # Step 1: Load or Create Dataset
    print("\n🚀 STEP 1: DATASET LOADING/CREATION")
    print("=" * 60)
    
    if load_existing:
        print(f"📂 Attempting to load existing dataset from: {existing_file}")
        dataset_df = load_dataset_dataframe(existing_file)
        
        if dataset_df.empty:
            print("❌ Failed to load existing dataset, creating new one...")
            dataset_df = create_dataset_dataframe()
        else:
            print("✅ Successfully loaded existing dataset!")
    else:
        print("🆕 Creating new dataset...")
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

    # Model loading
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

    # Test inference on single image
    if model_interpreter is not None:
        test_inference_on_single_image(
            dataset_df, 
            model_interpreter, 
            image_index=0,
            confidence_threshold=INFERENCE_CONFIDENCE,  # Use global variable
            debug=True
        )

    # Run inference if model is available and we don't already have inference results
    if model_interpreter is not None:
        # Check if we already have inference results
        if 'inferred_burner_bboxes' in dataset_df.columns:
            print("\n✅ Inference results already available in loaded dataset")
            print("   Skipping inference step...")
        else:
            print("\n🚀 STEP 3: INFERENCE")
            print("=" * 60)

            dataset_df = run_inference_on_dataframe(
                dataset_df, 
                model_interpreter, 
                PREPROCESSING_METHOD, 
                INFERENCE_CONFIDENCE,
                auto_save=True,
                save_path="burner_dataset_complete.pkl"
            )
        
        # Visualize results
        if not dataset_df.empty and 'inferred_burner_bboxes' in dataset_df.columns:
            visualize_inference_results(dataset_df)
    else:
        print("\n⚠️  SKIPPING STEP 3: No model available")

    # Run evaluation if we have inference results
    if 'inferred_burner_bboxes' in dataset_df.columns:
        print("\n🚀 STEP 4: EVALUATION")
        print("=" * 60)
        
        # Method 1: Presence/Absence
        presence_results = evaluate_presence_absence(dataset_df)
        
        # Method 2: IoU Matching
        iou_results = evaluate_iou_matching(dataset_df, iou_threshold=IOU_CONFIDENCE)
        
        # Visualize results
        visualize_evaluation_results(presence_results, iou_results)
        
        # Step 5: Analyze failure cases
        print("\n🚀 STEP 5: FAILURE CASE ANALYSIS")
        print("=" * 60)
        
        failure_cases = analyze_failure_cases(dataset_df, num_samples=5, random_seed=42)
        
        # Save results
        results_summary = {
            'presence_absence': presence_results,
            'iou_matching': iou_results,
            'dataset_info': {
                'total_images': len(dataset_df),
                'images_with_burners': len(dataset_df[dataset_df['has_burners']]),
                'total_gt_burners': dataset_df['num_burners'].sum(),
                'total_predicted_burners': dataset_df['num_inferred_burners'].sum()
            },
            'failure_analysis': {
                'total_failure_cases': len(failure_cases),
                'failure_rate_percent': len(failure_cases) / len(dataset_df) * 100 if len(dataset_df) > 0 else 0,
                'false_positives': len(dataset_df[(~dataset_df['has_burners']) & (dataset_df['has_inferred_burners'])]),
                'false_negatives': len(dataset_df[(dataset_df['has_burners']) & (~dataset_df['has_inferred_burners'])])
            }
        }
        
        # Convert numpy types to Python native types for JSON serialization
        results_summary_serializable = convert_numpy_types(results_summary)
        
        with open('evaluation_results.json', 'w') as f:
            json.dump(results_summary_serializable, f, indent=2)
        
        print(f"\n💾 Results saved to 'evaluation_results.json'")
        
    else:
        print("\n⚠️  SKIPPING STEP 5: No inference results available")

    # Final Summary
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
        
        # Save complete dataset using new function
        print(f"\n💾 Saving complete dataset...")
        save_success = save_dataset_dataframe(dataset_df, "burner_dataset_complete.pkl", "pickle", True)
        
        # Save summary using new function
        print(f"\n📋 Saving dataset summary...")
        summary_success = save_dataset_summary(dataset_df, "dataset_summary.csv", True)
        
        if save_success and summary_success:
            print(f"✅ All files saved successfully!")
        else:
            print(f"⚠️  Some files may not have been saved properly")
        
    else:
        print("❌ No dataset created")

    print(f"\n✅ All steps completed successfully!")
    print(f"📁 Output files:")
    print(f"   - burner_dataset_complete.pkl (complete dataset)")
    print(f"   - dataset_summary.csv (summary statistics)")
    if 'inferred_burner_bboxes' in dataset_df.columns:
        print(f"   - evaluation_results.json (evaluation metrics)")
        print(f"   - dataset_summary_stats.json (detailed statistics)")


def main_load_existing():
    """Convenience function to run main with existing dataset loading"""
    main(load_existing=True)


def main_create_new():
    """Convenience function to run main with new dataset creation"""
    main(load_existing=False)


if __name__ == "__main__":
    main()

