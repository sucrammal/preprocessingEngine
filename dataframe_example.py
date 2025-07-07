#!/usr/bin/env python3
"""
Simple example script showing how to work with the burner dataset DataFrame

This script demonstrates common operations you can perform on your burner dataset
after it has been saved as a pandas DataFrame.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

def load_burner_dataset(filename: str = "burner_dataset.pkl") -> pd.DataFrame:
    """Load the burner dataset DataFrame"""
    try:
        df = pd.read_pickle(filename)
        print(f"✅ Loaded dataset: {df.shape[0]} images, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"❌ File {filename} not found. Please run the preprocessing pipeline first.")
        return pd.DataFrame()

def explore_dataset(df: pd.DataFrame):
    """Basic dataset exploration"""
    print("\n📊 DATASET OVERVIEW")
    print("=" * 50)
    
    print(f"Total images: {len(df)}")
    print(f"Images with burners: {len(df[df['num_burners'] > 0])}")
    print(f"Images without burners: {len(df[df['num_burners'] == 0])}")
    print(f"Total burner objects: {df['num_burners'].sum()}")
    print(f"Average burners per image: {df['num_burners'].mean():.2f}")
    
    # Distribution of burner counts
    print(f"\nBurner count distribution:")
    burner_counts = df['num_burners'].value_counts().sort_index()
    for count, num_images in burner_counts.items():
        print(f"  {count} burners: {num_images} images")

def get_images_with_burners(df: pd.DataFrame, min_burners: int = 1) -> pd.DataFrame:
    """Get all images that have at least min_burners burners"""
    return df[df['num_burners'] >= min_burners]

def get_burner_boxes_for_image(df: pd.DataFrame, image_name: str) -> List[Dict]:
    """Get all burner bounding boxes for a specific image"""
    image_row = df[df['image_name'] == image_name]
    if image_row.empty:
        print(f"Image {image_name} not found in dataset")
        return []
    
    return image_row.iloc[0]['burner_bboxes']

def convert_to_yolo_format(bbox: Dict) -> tuple:
    """Convert bounding box to YOLO format (center_x, center_y, width, height)"""
    center_x = (bbox['xMinNormalized'] + bbox['xMaxNormalized']) / 2
    center_y = (bbox['yMinNormalized'] + bbox['yMaxNormalized']) / 2
    width = bbox['xMaxNormalized'] - bbox['xMinNormalized']
    height = bbox['yMaxNormalized'] - bbox['yMinNormalized']
    return center_x, center_y, width, height

def convert_to_pixel_coords(bbox: Dict, img_width: int, img_height: int) -> tuple:
    """Convert normalized bounding box to pixel coordinates"""
    x_min = int(bbox['xMinNormalized'] * img_width)
    y_min = int(bbox['yMinNormalized'] * img_height)
    x_max = int(bbox['xMaxNormalized'] * img_width)
    y_max = int(bbox['yMaxNormalized'] * img_height)
    return x_min, y_min, x_max, y_max

def create_train_test_split(df: pd.DataFrame, test_size: float = 0.2) -> tuple:
    """Create train/test split while maintaining burner distribution"""
    
    # Split by burner count to maintain distribution
    train_dfs = []
    test_dfs = []
    
    for burner_count in df['num_burners'].unique():
        subset = df[df['num_burners'] == burner_count]
        n_test = int(len(subset) * test_size)
        
        # Shuffle and split
        subset_shuffled = subset.sample(frac=1, random_state=42)
        test_subset = subset_shuffled[:n_test]
        train_subset = subset_shuffled[n_test:]
        
        train_dfs.append(train_subset)
        test_dfs.append(test_subset)
    
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    # Shuffle the final sets
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"✅ Train/Test Split Created:")
    print(f"   Train set: {len(train_df)} images")
    print(f"   Test set: {len(test_df)} images")
    print(f"   Split ratio: {len(test_df)/(len(train_df)+len(test_df)):.2%}")
    
    return train_df, test_df

def export_to_yolo_format(df: pd.DataFrame, output_dir: str = "yolo_dataset"):
    """Export dataset to YOLO format"""
    import os
    
    # Create output directories
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/labels", exist_ok=True)
    
    # Create classes.txt file
    with open(f"{output_dir}/classes.txt", "w") as f:
        f.write("burner\n")
    
    print(f"📁 Exporting {len(df)} images to YOLO format in {output_dir}/")
    
    for _, row in df.iterrows():
        image_name = row['image_name']
        image_path = row['image_path']
        
        # Copy image (you might want to do this with shutil.copy)
        print(f"   Image: {image_name}")
        
        # Create label file
        label_name = image_name.replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = f"{output_dir}/labels/{label_name}"
        
        with open(label_path, "w") as f:
            for bbox in row['burner_bboxes']:
                center_x, center_y, width, height = convert_to_yolo_format(bbox)
                f.write(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
    
    print(f"✅ YOLO export complete!")

def plot_dataset_statistics(df: pd.DataFrame):
    """Create visualization of dataset statistics"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Distribution of burner counts
    burner_counts = df['num_burners'].value_counts().sort_index()
    axes[0, 0].bar(burner_counts.index, burner_counts.values)
    axes[0, 0].set_title('Distribution of Burner Counts')
    axes[0, 0].set_xlabel('Number of Burners')
    axes[0, 0].set_ylabel('Number of Images')
    
    # 2. Histogram of burner counts
    axes[0, 1].hist(df['num_burners'], bins=max(1, df['num_burners'].max()), alpha=0.7)
    axes[0, 1].set_title('Histogram of Burner Counts')
    axes[0, 1].set_xlabel('Number of Burners')
    axes[0, 1].set_ylabel('Frequency')
    
    # 3. Pie chart of burner presence
    has_burners = len(df[df['num_burners'] > 0])
    no_burners = len(df[df['num_burners'] == 0])
    axes[1, 0].pie([has_burners, no_burners], labels=['Has Burners', 'No Burners'], autopct='%1.1f%%')
    axes[1, 0].set_title('Images with/without Burners')
    
    # 4. If predictions are available, show comparison
    if 'num_predicted_burners' in df.columns:
        axes[1, 1].scatter(df['num_burners'], df['num_predicted_burners'], alpha=0.5)
        axes[1, 1].plot([0, df['num_burners'].max()], [0, df['num_burners'].max()], 'r--')
        axes[1, 1].set_title('Ground Truth vs Predictions')
        axes[1, 1].set_xlabel('Ground Truth Burners')
        axes[1, 1].set_ylabel('Predicted Burners')
    else:
        axes[1, 1].text(0.5, 0.5, 'No predictions available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Predictions (Not Available)')
    
    plt.tight_layout()
    plt.savefig('dataset_statistics.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("📊 Dataset visualization saved as 'dataset_statistics.png'")

def main():
    """Main example function"""
    print("🔥 BURNER DATASET DATAFRAME EXAMPLE")
    print("=" * 60)
    
    # Load the dataset
    df = load_burner_dataset("burner_dataset.pkl")
    
    if df.empty:
        print("Please run the preprocessing pipeline first to create the dataset.")
        return
    
    # Explore the dataset
    explore_dataset(df)
    
    # Example 1: Get images with burners
    print("\n🔍 Example 1: Images with burners")
    images_with_burners = get_images_with_burners(df)
    print(f"Found {len(images_with_burners)} images with burners")
    
    # Example 2: Get burner boxes for a specific image
    if not images_with_burners.empty:
        sample_image = images_with_burners.iloc[0]['image_name']
        print(f"\n🔍 Example 2: Burner boxes for {sample_image}")
        boxes = get_burner_boxes_for_image(df, sample_image)
        print(f"Found {len(boxes)} burner boxes:")
        for i, bbox in enumerate(boxes):
            print(f"  Burner {i+1}: ({bbox['xMinNormalized']:.3f}, {bbox['yMinNormalized']:.3f}, "
                  f"{bbox['xMaxNormalized']:.3f}, {bbox['yMaxNormalized']:.3f})")
            
            # Convert to different formats
            yolo_format = convert_to_yolo_format(bbox)
            pixel_coords = convert_to_pixel_coords(bbox, 640, 640)
            print(f"    YOLO format: {yolo_format}")
            print(f"    Pixel coords (640x640): {pixel_coords}")
    
    # Example 3: Create train/test split
    print(f"\n🔍 Example 3: Train/Test Split")
    train_df, test_df = create_train_test_split(df, test_size=0.2)
    
    # Example 4: Plot statistics
    print(f"\n🔍 Example 4: Dataset Visualization")
    plot_dataset_statistics(df)
    
    # Example 5: Export to YOLO format (commented out to avoid creating files)
    # print(f"\n🔍 Example 5: Export to YOLO format")
    # export_to_yolo_format(df, "yolo_dataset")
    
    print("\n✅ All examples completed!")

if __name__ == "__main__":
    main() 