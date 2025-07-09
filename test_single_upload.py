#!/usr/bin/env python
# coding: utf-8

"""
Test Upload Single Image

Quick test script to upload just one image with bounding boxes
"""

import os
import subprocess
import sys

def test_single_upload():
    """Test upload of a single image"""
    print("🧪 TESTING SINGLE IMAGE UPLOAD")
    print("=" * 50)
    
    # Run the main upload script with debug flag and max-images limit
    try:
        result = subprocess.run([
            sys.executable, 
            'upload_with_bounding_boxes.py',
            '--debug',           # Enable verbose debug output
            '--max-images', '1'  # Upload only 1 image for testing
        ], capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n✅ Single image test upload completed successfully!")
            print("Check your Viam dataset to verify the upload worked correctly.")
            print("\nTo upload more images:")
            print("  python upload_with_bounding_boxes.py --max-images 10")
            print("  python upload_with_bounding_boxes.py --debug  # All images with verbose output")
            print("  python upload_with_bounding_boxes.py          # All images, minimal output")
        else:
            print(f"\n❌ Test upload failed with return code: {result.returncode}")
            
    except Exception as e:
        print(f"❌ Error running test upload: {e}")

if __name__ == "__main__":
    test_single_upload() 