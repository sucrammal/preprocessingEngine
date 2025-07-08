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
    
    # Set environment variable for single image upload
    os.environ['MAX_IMAGES_UPLOAD'] = '1'
    
    # Run the main upload script
    try:
        result = subprocess.run([sys.executable, 'upload_with_bounding_boxes.py'], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n✅ Single image test upload completed successfully!")
            print("Check your Viam dataset to verify the upload worked correctly.")
        else:
            print(f"\n❌ Test upload failed with return code: {result.returncode}")
            
    except Exception as e:
        print(f"❌ Error running test upload: {e}")

if __name__ == "__main__":
    test_single_upload() 