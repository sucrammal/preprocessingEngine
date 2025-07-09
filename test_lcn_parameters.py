#!/usr/bin/env python
# coding: utf-8

"""
Test LCN Parameters

This script demonstrates how to test different Local Contrast Normalization (LCN) 
parameters and compare their effects on image preprocessing.

Usage:
    python test_lcn_parameters.py
"""

import os
import numpy as np
from preprocessing import (
    create_dataset_dataframe, 
    test_lcn_parameters, 
    compare_lcn_configurations,
    apply_local_contrast_normalization,
    diagnose_lcn_issues
)

def demonstrate_lcn_parameters():
    """Demonstrate different LCN parameter configurations"""
    print("🧪 LCN PARAMETER TESTING DEMONSTRATION")
    print("=" * 60)
    
    # Load a sample image
    print("Loading sample image...")
    df = create_dataset_dataframe(max_images=1)
    if df.empty:
        print("❌ No images found! Make sure you have images in the data directory.")
        return
    
    sample_image = df.iloc[0]['image']
    print(f"✅ Loaded sample image: {df.iloc[0]['image_name']}")
    print(f"Image shape: {sample_image.shape}")
    
    # Test 1: Compare different window sizes
    print("\n" + "="*60)
    print("TEST 1: Window Size Effects")
    print("="*60)
    
    window_size_configs = [
        {"window_size": 5, "normalization_type": "divisive"},
        {"window_size": 9, "normalization_type": "divisive"},
        {"window_size": 15, "normalization_type": "divisive"},
        {"window_size": 25, "normalization_type": "divisive"}
    ]
    
    compare_lcn_configurations(
        sample_image, 
        window_size_configs, 
        ["Small (5px)", "Medium (9px)", "Large (15px)", "XLarge (25px)"]
    )
    
    # Test 2: Compare different normalization types
    print("\n" + "="*60)
    print("TEST 2: Normalization Type Effects")
    print("="*60)
    
    normalization_configs = [
        {"window_size": 9, "normalization_type": "divisive"},
        {"window_size": 9, "normalization_type": "subtractive"},
        {"window_size": 9, "normalization_type": "adaptive", "contrast_boost": 1.0},
        {"window_size": 9, "normalization_type": "adaptive", "contrast_boost": 1.5}
    ]
    
    compare_lcn_configurations(
        sample_image, 
        normalization_configs, 
        ["Divisive", "Subtractive", "Adaptive (1.0)", "Adaptive (1.5)"]
    )
    
    # Test 3: Compare different window shapes
    print("\n" + "="*60)
    print("TEST 3: Window Shape Effects")
    print("="*60)
    
    shape_configs = [
        {"window_size": 9, "window_shape": "square"},
        {"window_size": 9, "window_shape": "circular"},
        {"window_size": 9, "window_shape": "gaussian"}
    ]
    
    compare_lcn_configurations(
        sample_image, 
        shape_configs, 
        ["Square Window", "Circular Window", "Gaussian Window"]
    )
    
    # Test 4: Recommended configurations for different scenarios
    print("\n" + "="*60)
    print("TEST 4: Recommended Configurations")
    print("="*60)
    
    recommended_configs = [
        {
            "window_size": 9,
            "normalization_type": "divisive",
            "window_shape": "square",
            "statistical_measure": "mean",
            "contrast_boost": 1.0
        },
        {
            "window_size": 15,
            "normalization_type": "adaptive",
            "window_shape": "gaussian",
            "statistical_measure": "mean",
            "contrast_boost": 1.5
        },
        {
            "window_size": 5,
            "normalization_type": "divisive",
            "window_shape": "square",
            "statistical_measure": "mean",
            "contrast_boost": 1.2
        },
        {
            "window_size": 9,
            "normalization_type": "divisive",
            "window_shape": "circular",
            "statistical_measure": "median",
            "contrast_boost": 1.0
        }
    ]
    
    compare_lcn_configurations(
        sample_image, 
        recommended_configs, 
        ["Standard LCN", "Strong Adaptive", "Fine Detail", "Robust Median"]
    )
    
    # Test 5: Full parameter exploration
    print("\n" + "="*60)
    print("TEST 5: Full Parameter Exploration")
    print("="*60)
    
    # Run the comprehensive parameter test
    test_lcn_parameters(sample_image)
    
    print("\n✅ LCN parameter testing complete!")
    print("\n💡 Tips for choosing LCN parameters:")
    print("   - Use smaller window sizes (5-9) for fine detail enhancement")
    print("   - Use larger window sizes (15-25) for global lighting correction")
    print("   - Use 'adaptive' normalization for images with strong shadows")
    print("   - Use 'gaussian' window shape for smoother results")
    print("   - Use 'median' statistical measure for noisy images")
    print("   - Adjust contrast_boost between 0.8-1.5 for optimal results")


def test_specific_configuration():
    """Test a specific LCN configuration"""
    print("\n🎯 TESTING SPECIFIC LCN CONFIGURATION")
    print("=" * 60)
    
    # Load sample image
    df = create_dataset_dataframe(max_images=1)
    if df.empty:
        print("❌ No images found!")
        return
    
    sample_image = df.iloc[0]['image']
    
    # Test specific configuration
    config = {
        "window_size": 15,
        "normalization_type": "adaptive",
        "window_shape": "gaussian",
        "statistical_measure": "mean",
        "contrast_boost": 1.3,
        "epsilon": 1e-8
    }
    
    print("Testing configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Apply LCN with this configuration
    result = apply_local_contrast_normalization(sample_image, **config)
    
    print(f"\n✅ LCN applied successfully!")
    print(f"Input image shape: {sample_image.shape}")
    print(f"Output image shape: {result.shape}")
    print(f"Output value range: [{result.min():.3f}, {result.max():.3f}]")
    print(f"Output mean: {result.mean():.3f}")
    print(f"Output std: {result.std():.3f}")


def diagnose_green_cyan_issues():
    """Diagnose and demonstrate fixes for green/cyan color artifacts"""
    print("\n🩺 DIAGNOSING GREEN/CYAN COLOR ARTIFACTS")
    print("=" * 60)
    
    # Load sample image
    df = create_dataset_dataframe(max_images=1)
    if df.empty:
        print("❌ No images found!")
        return
    
    sample_image = df.iloc[0]['image']
    print(f"Testing with image: {df.iloc[0]['image_name']}")
    
    # Run diagnostics on the original image
    print("\n1️⃣ DIAGNOSING INPUT IMAGE")
    diagnostics = diagnose_lcn_issues(sample_image, window_size=9, normalization_type="divisive")
    
    # Test problematic configurations that might cause green/cyan artifacts
    print("\n2️⃣ TESTING PROBLEMATIC CONFIGURATIONS")
    
    problematic_configs = [
        {
            "name": "Extreme Adaptive",
            "config": {
                "window_size": 5,
                "normalization_type": "adaptive",
                "contrast_boost": 3.0,
                "epsilon": 1e-10
            }
        },
        {
            "name": "Very Large Window",
            "config": {
                "window_size": 51,
                "normalization_type": "divisive",
                "contrast_boost": 2.0
            }
        },
        {
            "name": "High Contrast Boost",
            "config": {
                "window_size": 9,
                "normalization_type": "divisive",
                "contrast_boost": 5.0
            }
        }
    ]
    
    for test_case in problematic_configs:
        print(f"\nTesting: {test_case['name']}")
        print("-" * 40)
        
        try:
            result = apply_local_contrast_normalization(sample_image, **test_case['config'])
            
            # Check for color artifacts
            if result.shape[2] == 3:  # RGB
                r_mean, g_mean, b_mean = [result[:,:,i].mean() for i in range(3)]
                channel_imbalance = max(r_mean, g_mean, b_mean) - min(r_mean, g_mean, b_mean)
                
                print(f"  Channel means - R: {r_mean:.3f}, G: {g_mean:.3f}, B: {b_mean:.3f}")
                print(f"  Channel imbalance: {channel_imbalance:.3f}")
                
                if channel_imbalance > 0.3:
                    print(f"  ⚠️  HIGH CHANNEL IMBALANCE - Likely color artifacts!")
                elif r_mean < 0.05 or g_mean < 0.05 or b_mean < 0.05:
                    print(f"  ⚠️  EXTREME LOW CHANNEL - Likely color artifacts!")
                elif r_mean > 0.95 or g_mean > 0.95 or b_mean > 0.95:
                    print(f"  ⚠️  EXTREME HIGH CHANNEL - Likely color artifacts!")
                else:
                    print(f"  ✅ Channels look balanced")
            
        except Exception as e:
            print(f"  ❌ Configuration failed: {e}")
    
    # Show safe configurations
    print("\n3️⃣ RECOMMENDED SAFE CONFIGURATIONS")
    
    safe_configs = [
        {
            "name": "Conservative",
            "config": {
                "window_size": 9,
                "normalization_type": "divisive",
                "window_shape": "square",
                "contrast_boost": 1.0
            }
        },
        {
            "name": "Moderate Enhancement",
            "config": {
                "window_size": 11,
                "normalization_type": "divisive",
                "window_shape": "circular",
                "contrast_boost": 1.2
            }
        },
        {
            "name": "Safe Adaptive",
            "config": {
                "window_size": 15,
                "normalization_type": "adaptive",
                "window_shape": "gaussian",
                "contrast_boost": 1.1
            }
        }
    ]
    
    print("\nTesting safe configurations:")
    for test_case in safe_configs:
        print(f"\n{test_case['name']}:")
        
        try:
            result = apply_local_contrast_normalization(sample_image, **test_case['config'])
            
            if result.shape[2] == 3:  # RGB
                r_mean, g_mean, b_mean = [result[:,:,i].mean() for i in range(3)]
                channel_imbalance = max(r_mean, g_mean, b_mean) - min(r_mean, g_mean, b_mean)
                
                print(f"  Channel means - R: {r_mean:.3f}, G: {g_mean:.3f}, B: {b_mean:.3f}")
                print(f"  Channel imbalance: {channel_imbalance:.3f}")
                print(f"  ✅ Safe configuration")
            
        except Exception as e:
            print(f"  ❌ Unexpected error: {e}")
    
    print(f"\n💡 TIPS TO AVOID GREEN/CYAN ARTIFACTS:")
    print(f"   1. Keep contrast_boost between 0.8 and 1.5")
    print(f"   2. Use window_size between 7 and 21 (odd numbers)")
    print(f"   3. For 'adaptive' normalization, use larger window_size (15+)")
    print(f"   4. If you see artifacts, try 'divisive' instead of 'adaptive'")
    print(f"   5. Use 'gaussian' or 'circular' window_shape for smoother results")
    print(f"   6. Run diagnose_lcn_issues() to check for problems")


if __name__ == "__main__":
    # Check if we have the required functions
    try:
        print("🧪 RUNNING LCN PARAMETER TESTS")
        print("=" * 80)
        
        # Run diagnostics first to check for issues
        diagnose_green_cyan_issues()
        
        # Then run the full parameter demonstration
        demonstrate_lcn_parameters()
        
        # Test specific configuration
        test_specific_configuration()
        
    except Exception as e:
        print(f"❌ Error running LCN parameter tests: {e}")
        print("Make sure you have:")
        print("1. Images in the 'data' directory")
        print("2. Metadata files in the 'metadata' directory")
        print("3. All required dependencies installed")
        print("\nIf you're seeing green/cyan artifacts, try running:")
        print("python -c \"from test_lcn_parameters import diagnose_green_cyan_issues; diagnose_green_cyan_issues()\"") 