#!/usr/bin/env python
# coding: utf-8

"""
LCN Diagnostic Tool

Quick diagnostic tool to identify and fix green/cyan color artifacts 
in Local Contrast Normalization (LCN) preprocessing.

Usage:
    python diagnose_lcn.py
"""

import os
import sys
from preprocessing import create_dataset_dataframe, diagnose_lcn_issues, apply_local_contrast_normalization

def quick_lcn_diagnosis():
    """Quick diagnosis of LCN issues"""
    print("🩺 LCN COLOR ARTIFACT DIAGNOSTIC TOOL")
    print("=" * 60)
    
    # Load a sample image
    try:
        df = create_dataset_dataframe(max_images=1)
        if df.empty:
            print("❌ No images found in the data directory!")
            print("Make sure you have images in the 'data' directory and metadata in 'metadata' directory.")
            return False
        
        sample_image = df.iloc[0]['image']
        image_name = df.iloc[0]['image_name']
        
        print(f"✅ Testing with image: {image_name}")
        print(f"Image shape: {sample_image.shape}")
        
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return False
    
    # Get current LCN configuration from environment
    current_config = {
        "window_size": int(os.getenv("LCN_WINDOW_SIZE", "9")),
        "normalization_type": os.getenv("LCN_NORMALIZATION_TYPE", "divisive"),
        "window_shape": os.getenv("LCN_WINDOW_SHAPE", "square"),
        "statistical_measure": os.getenv("LCN_STATISTICAL_MEASURE", "mean"),
        "contrast_boost": float(os.getenv("LCN_CONTRAST_BOOST", "1.0")),
        "epsilon": float(os.getenv("LCN_EPSILON", "1e-8"))
    }
    
    print(f"\n🎛️ CURRENT LCN CONFIGURATION:")
    for key, value in current_config.items():
        print(f"   {key}: {value}")
    
    # Test current configuration
    print(f"\n🧪 TESTING CURRENT CONFIGURATION")
    print("-" * 40)
    
    try:
        result = apply_local_contrast_normalization(sample_image, **current_config)
        
        # Check for color artifacts
        if result.shape[2] == 3:  # RGB
            r_mean, g_mean, b_mean = [result[:,:,i].mean() for i in range(3)]
            channel_imbalance = max(r_mean, g_mean, b_mean) - min(r_mean, g_mean, b_mean)
            
            print(f"Output channel means:")
            print(f"  Red: {r_mean:.3f}")
            print(f"  Green: {g_mean:.3f}")  
            print(f"  Blue: {b_mean:.3f}")
            print(f"  Channel imbalance: {channel_imbalance:.3f}")
            
            # Diagnose issues
            issues_found = []
            
            if channel_imbalance > 0.3:
                issues_found.append("HIGH CHANNEL IMBALANCE")
            
            if r_mean < 0.05 or g_mean < 0.05 or b_mean < 0.05:
                issues_found.append("EXTREME LOW CHANNEL VALUES")
            
            if r_mean > 0.95 or g_mean > 0.95 or b_mean > 0.95:
                issues_found.append("EXTREME HIGH CHANNEL VALUES")
            
            # Check for green/cyan artifacts specifically
            if g_mean > r_mean + 0.3 or g_mean > b_mean + 0.3:
                issues_found.append("GREEN COLOR ARTIFACT")
            
            if (g_mean > r_mean + 0.2 and b_mean > r_mean + 0.2):
                issues_found.append("CYAN COLOR ARTIFACT")
            
            if issues_found:
                print(f"\n⚠️  ISSUES DETECTED:")
                for issue in issues_found:
                    print(f"   - {issue}")
                
                print(f"\n💡 RECOMMENDED FIXES:")
                
                if "HIGH CHANNEL IMBALANCE" in issues_found:
                    print(f"   - Reduce contrast_boost (try 1.0 or lower)")
                    print(f"   - Use larger window_size (try 15 or higher)")
                
                if "EXTREME LOW CHANNEL VALUES" in issues_found:
                    print(f"   - Increase epsilon to 1e-6 or higher")
                    print(f"   - Try 'divisive' normalization instead of 'adaptive'")
                
                if "EXTREME HIGH CHANNEL VALUES" in issues_found:
                    print(f"   - Reduce contrast_boost")
                    print(f"   - Use 'subtractive' normalization")
                
                if "GREEN COLOR ARTIFACT" in issues_found or "CYAN COLOR ARTIFACT" in issues_found:
                    print(f"   - Use conservative settings:")
                    print(f"     export LCN_WINDOW_SIZE=9")
                    print(f"     export LCN_NORMALIZATION_TYPE=divisive")
                    print(f"     export LCN_CONTRAST_BOOST=1.0")
                
                # Test recommended safe configuration
                print(f"\n🔧 TESTING SAFE CONFIGURATION:")
                safe_config = {
                    "window_size": 9,
                    "normalization_type": "divisive",
                    "window_shape": "square",
                    "contrast_boost": 1.0,
                    "epsilon": 1e-6
                }
                
                safe_result = apply_local_contrast_normalization(sample_image, **safe_config)
                
                if safe_result.shape[2] == 3:
                    r_safe, g_safe, b_safe = [safe_result[:,:,i].mean() for i in range(3)]
                    safe_imbalance = max(r_safe, g_safe, b_safe) - min(r_safe, g_safe, b_safe)
                    
                    print(f"Safe config channel means - R: {r_safe:.3f}, G: {g_safe:.3f}, B: {b_safe:.3f}")
                    print(f"Safe config imbalance: {safe_imbalance:.3f}")
                    
                    if safe_imbalance < 0.2:
                        print(f"✅ Safe configuration works well!")
                        print(f"\nTo use this configuration, run:")
                        print(f"export LCN_WINDOW_SIZE=9")
                        print(f"export LCN_NORMALIZATION_TYPE=divisive")
                        print(f"export LCN_WINDOW_SHAPE=square")
                        print(f"export LCN_CONTRAST_BOOST=1.0")
                        print(f"export LCN_EPSILON=1e-6")
                    else:
                        print(f"⚠️ Even safe configuration has issues - may be image-specific")
                
            else:
                print(f"\n✅ NO COLOR ARTIFACTS DETECTED")
                print(f"Current configuration appears to work well!")
        
    except Exception as e:
        print(f"❌ Error testing LCN configuration: {e}")
        print(f"This suggests a serious configuration issue.")
        return False
    
    return True

def interactive_lcn_tuning():
    """Interactive LCN parameter tuning"""
    print(f"\n🎚️ INTERACTIVE LCN TUNING")
    print("=" * 40)
    
    try:
        df = create_dataset_dataframe(max_images=1)
        sample_image = df.iloc[0]['image']
    except:
        print("❌ Could not load sample image")
        return
    
    # Start with current configuration
    config = {
        "window_size": int(os.getenv("LCN_WINDOW_SIZE", "9")),
        "normalization_type": os.getenv("LCN_NORMALIZATION_TYPE", "divisive"),
        "window_shape": os.getenv("LCN_WINDOW_SHAPE", "square"),
        "statistical_measure": os.getenv("LCN_STATISTICAL_MEASURE", "mean"),
        "contrast_boost": float(os.getenv("LCN_CONTRAST_BOOST", "1.0")),
        "epsilon": float(os.getenv("LCN_EPSILON", "1e-8"))
    }
    
    print("Current configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Test a few variations automatically
    test_configs = [
        {"name": "Conservative", "window_size": 9, "normalization_type": "divisive", "contrast_boost": 1.0},
        {"name": "Moderate", "window_size": 11, "normalization_type": "divisive", "contrast_boost": 1.2},
        {"name": "Enhanced", "window_size": 15, "normalization_type": "adaptive", "contrast_boost": 1.1}
    ]
    
    print(f"\nTesting variations:")
    best_config = None
    best_score = float('inf')
    
    for test_config in test_configs:
        try:
            # Merge with base config (exclude 'name' from function call)
            full_config = config.copy()
            test_config_for_function = {k: v for k, v in test_config.items() if k != 'name'}
            full_config.update(test_config_for_function)
            
            result = apply_local_contrast_normalization(sample_image, **full_config)
            
            if result.shape[2] == 3:
                r_mean, g_mean, b_mean = [result[:,:,i].mean() for i in range(3)]
                imbalance = max(r_mean, g_mean, b_mean) - min(r_mean, g_mean, b_mean)
                
                print(f"  {test_config['name']}: imbalance={imbalance:.3f}")
                
                if imbalance < best_score:
                    best_score = imbalance
                    best_config = full_config.copy()
                    best_config["name"] = test_config["name"]
        
        except Exception as e:
            print(f"  {test_config['name']}: ❌ Failed ({e})")
    
    if best_config:
        print(f"\n🏆 BEST CONFIGURATION: {best_config['name']} (imbalance: {best_score:.3f})")
        print(f"Recommended environment variables:")
        print(f"export LCN_WINDOW_SIZE={best_config['window_size']}")
        print(f"export LCN_NORMALIZATION_TYPE={best_config['normalization_type']}")
        print(f"export LCN_WINDOW_SHAPE={best_config['window_shape']}")
        print(f"export LCN_CONTRAST_BOOST={best_config['contrast_boost']}")

if __name__ == "__main__":
    print("Starting LCN diagnostic tool...")
    
    success = quick_lcn_diagnosis()
    
    if success:
        interactive_lcn_tuning()
    
    print(f"\n📚 For more detailed testing, run:")
    print(f"   python test_lcn_parameters.py")
    print(f"\n📖 For complete documentation, see:")
    print(f"   README.md - Advanced LCN Configuration section") 