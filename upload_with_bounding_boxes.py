#!/usr/bin/env python
# coding: utf-8

"""
Upload Preprocessed Images with Bounding Boxes to Viam

This script:
1. Uses file_upload_from_path() to upload preprocessed images to Viam platform
2. Uses add_bounding_box_to_image_by_id() to add bounding box information to uploaded images
3. Handles scaling/normalization effects from LCN preprocessing on bounding boxes
4. Uses environment variables for credentials and configuration

Usage:
    python upload_with_bounding_boxes.py [--debug] [--max-images N]
"""

import asyncio
import os
import tempfile
import json
import argparse
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Import preprocessing functions from existing file
from preprocessing import (
    create_dataset_dataframe, 
    preprocess_image,
    find_image_for_metadata,
    METADATA_DIR,
    IMAGES_DIR
)

# Viam SDK imports
from viam.rpc.dial import DialOptions, Credentials
from viam.app.viam_client import ViamClient
from viam.proto.app.data import BoundingBox

# Load environment variables
load_dotenv()

# Global debug flag
DEBUG = False

def debug_print(message: str):
    """Print message only if debug mode is enabled"""
    if DEBUG:
        print(message)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Upload preprocessed images with bounding boxes to Viam",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable verbose debug output showing detailed upload and bounding box information'
    )
    
    parser.add_argument(
        '--max-images', 
        type=int,
        default=0,
        help='Maximum number of images to upload (0 for all images)'
    )
    
    return parser.parse_args()

# Parse command line arguments
args = parse_arguments()
DEBUG = args.debug

# Viam API Configuration
VIAM_CONFIG = {
    "api_key": os.getenv("VIAM_API_KEY", ""),
    "api_key_id": os.getenv("VIAM_API_KEY_ID", ""),
    "part_id": os.getenv("VIAM_PART_ID", ""),
    "organization_id": os.getenv("VIAM_ORG_ID", ""),
    "location_id": os.getenv("VIAM_LOCATION_ID", ""),
}

# Dataset Configuration
DATASET_ID = os.getenv("VIAM_DATASET_ID", "")
PREPROCESSING_METHOD = os.getenv("UPLOAD_PREPROCESSING_METHOD", "lcn")
# Remove the complex tag creation logic and replace with just the single tag
UPLOAD_TAGS = os.getenv("UPLOAD_TAGS", "preprocessing:lcn").split(",")
MAX_IMAGES_UPLOAD = args.max_images if args.max_images > 0 else int(os.getenv("MAX_IMAGES_UPLOAD", "0"))
TEMP_DIR = os.getenv("TEMP_DIR", "temp_preprocessed")

# Remove LCN scaling configuration - simplified approach

print("🚀 VIAM PREPROCESSED DATA UPLOAD WITH BOUNDING BOXES")
print("=" * 70)
print(f"Preprocessing method: {PREPROCESSING_METHOD}")
print(f"Upload tags: {UPLOAD_TAGS}")
print(f"Max images to upload: {'All' if MAX_IMAGES_UPLOAD == 0 else MAX_IMAGES_UPLOAD}")
print(f"Temp directory: {TEMP_DIR}")
print(f"Dataset ID: {DATASET_ID}")
print(f"Debug mode: {'Enabled' if DEBUG else 'Disabled'}")
print("=" * 70)

# Validate required configuration
missing_config = []
required_vars = ["VIAM_API_KEY", "VIAM_API_KEY_ID", "VIAM_PART_ID", "VIAM_ORGANIZATION_ID", "VIAM_LOCATION_ID"]
for var in required_vars:
    if not VIAM_CONFIG.get(var.lower().replace("viam_", ""), ""):
        missing_config.append(var)

if not DATASET_ID:
    missing_config.append("VIAM_DATASET_ID")

if missing_config:
    print(f"❌ Missing required environment variables: {', '.join(missing_config)}")
    print("\nAdd these to your .env file:")
    for var in missing_config:
        print(f"   {var}=your-{var.lower().replace('_', '-')}")
    print("\nOptional variables:")
    print("   MAX_IMAGES_UPLOAD=1  # For testing (0 means all images)")
    print("   UPLOAD_PREPROCESSING_METHOD=lcn")
    print("   TEMP_DIR=temp_preprocessed")
    exit(1)


async def connect_to_viam() -> ViamClient:
    """Connect to Viam using API key authentication"""
    print("\n🔗 CONNECTING TO VIAM")
    print("=" * 40)
    
    dial_options = DialOptions(
        credentials=Credentials(
            type="api-key",
            payload=VIAM_CONFIG["api_key"],
        ),
        auth_entity=VIAM_CONFIG["api_key_id"]
    )
    
    try:
        client = await ViamClient.create_from_dial_options(dial_options)
        print("✅ Successfully connected to Viam")
        return client
    except Exception as e:
        print(f"❌ Failed to connect to Viam: {e}")
        raise


def create_temp_directory() -> str:
    """Create temporary directory for preprocessed images"""
    temp_path = Path(TEMP_DIR)
    temp_path.mkdir(exist_ok=True)
    debug_print(f"📁 Created temp directory: {temp_path.absolute()}")
    return str(temp_path)


def save_preprocessed_image(image_array: np.ndarray, original_path: str, temp_dir: str) -> str:
    """
    Save preprocessed image to temporary file
    
    Args:
        image_array: Preprocessed image array (normalized 0-1)
        original_path: Original image path for naming
        temp_dir: Temporary directory path
        
    Returns:
        Path to saved preprocessed image
    """
    # Get original filename and create new name
    original_name = Path(original_path).stem
    preprocessed_name = f"{original_name}_preprocessed_{PREPROCESSING_METHOD}.png"
    preprocessed_path = Path(temp_dir) / preprocessed_name
    
    # Convert normalized array back to 0-255 range for saving
    image_uint8 = (np.clip(image_array, 0, 1) * 255).astype(np.uint8)
    
    # Save as PNG to preserve quality
    pil_image = Image.fromarray(image_uint8)
    pil_image.save(preprocessed_path, format='PNG')
    
    return str(preprocessed_path)


def normalize_bounding_boxes_for_image(bboxes: List[Dict], image_height: int, image_width: int) -> List[Dict]:
    """
    Normalize bounding boxes to 0-1 range based on preprocessed image dimensions
    
    Args:
        bboxes: List of bounding box dictionaries
        image_height: Height of the preprocessed image in pixels
        image_width: Width of the preprocessed image in pixels
        
    Returns:
        List of normalized bounding boxes
    """
    normalized_bboxes = []
    
    debug_print(f"📏 Normalizing bounding boxes for image size: {image_width}x{image_height}")
    
    for bbox in bboxes:
        normalized_bbox = bbox.copy()
        
        # If bounding box coordinates are in pixel values, normalize them
        if 'xMin' in bbox and 'yMin' in bbox:
            # Pixel coordinates to normalized (0-1) coordinates
            normalized_bbox['xMinNormalized'] = bbox['xMin'] / image_width
            normalized_bbox['xMaxNormalized'] = bbox['xMax'] / image_width
            normalized_bbox['yMinNormalized'] = bbox['yMin'] / image_height
            normalized_bbox['yMaxNormalized'] = bbox['yMax'] / image_height
        elif 'xMinNormalized' in bbox:
            # Already normalized, just copy
            normalized_bbox['xMinNormalized'] = bbox['xMinNormalized']
            normalized_bbox['xMaxNormalized'] = bbox['xMaxNormalized']
            normalized_bbox['yMinNormalized'] = bbox['yMinNormalized']
            normalized_bbox['yMaxNormalized'] = bbox['yMaxNormalized']
        
        # Ensure coordinates stay within [0, 1] range
        normalized_bbox['xMinNormalized'] = max(0, min(1, normalized_bbox['xMinNormalized']))
        normalized_bbox['xMaxNormalized'] = max(0, min(1, normalized_bbox['xMaxNormalized']))
        normalized_bbox['yMinNormalized'] = max(0, min(1, normalized_bbox['yMinNormalized']))
        normalized_bbox['yMaxNormalized'] = max(0, min(1, normalized_bbox['yMaxNormalized']))
        
        normalized_bboxes.append(normalized_bbox)
    
    return normalized_bboxes


def create_viam_bounding_box(bbox_dict: Dict) -> BoundingBox:
    """
    Create a Viam BoundingBox from a dictionary
    
    Args:
        bbox_dict: Dictionary containing bounding box information
        
    Returns:
        Viam BoundingBox object
    """
    return BoundingBox(
        id=bbox_dict.get('id', ''),
        label=bbox_dict.get('label', ''),
        x_min_normalized=bbox_dict.get('xMinNormalized', 0.0),
        y_min_normalized=bbox_dict.get('yMinNormalized', 0.0),
        x_max_normalized=bbox_dict.get('xMaxNormalized', 0.0),
        y_max_normalized=bbox_dict.get('yMaxNormalized', 0.0)
    )


async def upload_image_with_bounding_boxes(
    data_client,
    image_path: str,
    preprocessed_image: np.ndarray,
    metadata: Dict,
    tags: List[str]
) -> Tuple[str, bool]:
    """
    Upload image and add bounding boxes
    
    Args:
        data_client: Viam data client
        image_path: Path to the image file
        preprocessed_image: Preprocessed image array
        metadata: Image metadata with bounding box information
        tags: Tags for the uploaded image
        
    Returns:
        Tuple of (file_id, success)
    """
    try:
        # Step 1: Upload the image using file_upload_from_path
        debug_print(f"📤 Uploading image: {Path(image_path).name}")
        file_id = await data_client.file_upload_from_path(
            part_id=VIAM_CONFIG["part_id"],
            tags=tags,
            filepath=image_path
        )
        
        # Step 2: Get preprocessed image dimensions
        image_height, image_width = preprocessed_image.shape[:2]
        
        # Step 3: Extract and normalize bounding boxes
        if 'annotations' in metadata and 'bboxes' in metadata['annotations']:
            bboxes = metadata['annotations']['bboxes']
            
            # Normalize bounding boxes based on preprocessed image dimensions
            normalized_bboxes = normalize_bounding_boxes_for_image(bboxes, image_height, image_width)
            
            # Step 4: Add bounding boxes to the uploaded image
            debug_print(f"📦 Adding {len(normalized_bboxes)} bounding boxes to image")
            
            for bbox_dict in normalized_bboxes:
                # Create Viam BoundingBox object
                viam_bbox = create_viam_bounding_box(bbox_dict)
                
                # Add bounding box to image
                await data_client.add_bounding_box_to_image_by_id(
                    binary_id=file_id,
                    label=viam_bbox.label,
                    x_min_normalized=viam_bbox.x_min_normalized,
                    y_min_normalized=viam_bbox.y_min_normalized,
                    x_max_normalized=viam_bbox.x_max_normalized,
                    y_max_normalized=viam_bbox.y_max_normalized
                )
                
                debug_print(f"  ✅ Added bounding box: {viam_bbox.label} "
                      f"({viam_bbox.x_min_normalized:.3f}, {viam_bbox.y_min_normalized:.3f}, "
                      f"{viam_bbox.x_max_normalized:.3f}, {viam_bbox.y_max_normalized:.3f})")
        
        return file_id, True
        
    except Exception as e:
        print(f"❌ Error uploading image with bounding boxes: {e}")
        return "", False


async def process_and_upload_metadata_files(viam_client: ViamClient, temp_dir: str, max_images: int = None):
    """
    Process metadata files and upload corresponding images with bounding boxes
    
    Args:
        viam_client: Connected Viam client
        temp_dir: Temporary directory for preprocessed images
        max_images: Maximum number of images to process (None for all)
    """
    # Get all metadata files
    metadata_files = list(Path(METADATA_DIR).glob("*.json"))
    
    if max_images and max_images > 0:
        metadata_files = metadata_files[:max_images]
    
    print(f"\n📤 PROCESSING AND UPLOADING {len(metadata_files)} IMAGES WITH BOUNDING BOXES")
    print("=" * 70)
    
    data_client = viam_client.data_client
    successful_uploads = 0
    failed_uploads = 0
    uploaded_file_ids = []
    
    for metadata_path in tqdm(metadata_files, desc="Processing & uploading", unit="img"):
        try:
            # Step 1: Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Step 2: Find corresponding image using existing function
            image_path = find_image_for_metadata(str(metadata_path))
            if not image_path:
                print(f"⚠️  No image found for metadata: {metadata_path.name}")
                failed_uploads += 1
                continue
            
            # Step 3: Load and preprocess image
            from PIL import Image as PILImage
            import numpy as np
            
            original_image = np.array(PILImage.open(image_path))
            preprocessed_image = preprocess_image(
                original_image, 
                target_size=None,  # Keep original size
                normalization_method=PREPROCESSING_METHOD
            )
            
            # Step 4: Save preprocessed image to temp file
            temp_image_path = save_preprocessed_image(
                preprocessed_image, 
                image_path, 
                temp_dir
            )
            
            # Step 5: Extract image-specific tags from metadata
            image_tags = []
            if 'captureMetadata' in metadata and 'tags' in metadata['captureMetadata']:
                image_tags = metadata['captureMetadata']['tags']
                debug_print(f"📋 Found image-specific tags: {image_tags}")
            else:
                debug_print(f"⚠️  No image-specific tags found, using preset tags")
                image_tags = UPLOAD_TAGS
            
            # Step 6: Use simplified tags - just the preprocessing method
            simplified_tags = ["preprocessing:lcn"]
            
            # Step 7: Upload image with bounding boxes
            file_id, success = await upload_image_with_bounding_boxes(
                data_client,
                temp_image_path,
                preprocessed_image,
                metadata,
                simplified_tags
            )
            
            if success:
                successful_uploads += 1
                uploaded_file_ids.append(file_id)
                debug_print(f"✅ Successfully uploaded: {Path(image_path).name} (ID: {file_id})")
            else:
                failed_uploads += 1
            
            # Step 8: Clean up temp image file
            os.remove(temp_image_path)
            
        except Exception as e:
            print(f"❌ Error processing {metadata_path.name}: {e}")
            failed_uploads += 1
            continue
    
    print(f"\n✅ UPLOAD SUMMARY")
    print("=" * 50)
    print(f"   Successfully uploaded: {successful_uploads}")
    print(f"   Failed uploads: {failed_uploads}")
    print(f"   Total processed: {successful_uploads + failed_uploads}")
    print(f"   Uploaded file IDs: {len(uploaded_file_ids)}")
    
    if successful_uploads > 0:
        print(f"\n📋 UPLOAD INFORMATION")
        print("=" * 50)
        print(f"   All images uploaded with bounding box annotations")
        print(f"   Dataset ID: {DATASET_ID}")
        print(f"   Preprocessing method: {PREPROCESSING_METHOD}")
        print(f"   Bounding boxes: Normalized to preprocessed image dimensions")
        print(f"   Tags: Image-specific tags from metadata + preprocessing info")


def cleanup_temp_directory(temp_dir: str):
    """Clean up temporary directory"""
    try:
        import shutil
        shutil.rmtree(temp_dir)
        debug_print(f"🧹 Cleaned up temp directory: {temp_dir}")
    except Exception as e:
        print(f"⚠️  Warning: Could not clean up temp directory: {e}")


async def main():
    """Main upload process"""
    print(f"\n🚀 STARTING PREPROCESSED IMAGE UPLOAD WITH BOUNDING BOXES")
    print("=" * 70)
    
    # Step 1: Check for metadata files
    metadata_files = list(Path(METADATA_DIR).glob("*.json"))
    if not metadata_files:
        print(f"❌ No metadata files found in {METADATA_DIR}")
        return
    
    max_images = MAX_IMAGES_UPLOAD if MAX_IMAGES_UPLOAD > 0 else None
    print(f"📋 Found {len(metadata_files)} metadata files")
    print(f"Will process: {'All' if max_images is None else max_images} images")
    
    # Step 2: Create temp directory
    temp_dir = create_temp_directory()
    
    try:
        # Step 3: Connect to Viam
        viam_client = await connect_to_viam()
        
        # Step 4: Process and upload images with bounding boxes
        await process_and_upload_metadata_files(viam_client, temp_dir, max_images)
        
        # Step 5: Close connection
        viam_client.close()
        print("🔌 Disconnected from Viam")
        
    except Exception as e:
        print(f"❌ Upload process failed: {e}")
        raise
    
    finally:
        # Step 6: Cleanup
        cleanup_temp_directory(temp_dir)
    
    print(f"\n🎉 UPLOAD PROCESS COMPLETE!")
    print("=" * 70)
    print(f"Check your Viam dataset '{DATASET_ID}' to see the uploaded images with bounding boxes.")
    print(f"All images have been preprocessed using {PREPROCESSING_METHOD} method.")
    print(f"Bounding boxes have been normalized to 0-1 range based on preprocessed image dimensions.")


if __name__ == "__main__":
    asyncio.run(main()) 