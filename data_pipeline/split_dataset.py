"""
BIM Sign Language Dataset Splitter
Stratified 80-10-10 Split

This script:
1. Takes augmented images (exactly 1000 per class)
2. Performs stratified 80-10-10 split
3. Organizes into train_data/, val_data/, test_data/
"""

import os
import shutil
import random
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # Paths
    "input_dir": "augmented_data",    # Augmented images (1000 per class)
    "train_dir": "train_data",
    "val_dir": "val_data", 
    "test_dir": "test_data",
    
    # Split ratios (must sum to 1.0)
    "train_ratio": 0.80,              # 800 images per class
    "val_ratio": 0.10,                # 100 images per class
    "test_ratio": 0.10,               # 100 images per class
    
    # Random seed for reproducibility
    "seed": 42
}

# =============================================================================
# FUNCTIONS
# =============================================================================
def get_image_files(directory):
    """Get all image files in a directory."""
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    return [f for f in os.listdir(directory) if f.lower().endswith(extensions)]

def split_dataset(config):
    """Split dataset with stratification."""
    
    input_dir = config["input_dir"]
    
    # Set random seed
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    
    # Check input directory
    if not os.path.exists(input_dir):
        print(f"ERROR: Input directory '{input_dir}' not found!")
        print(f"\nPlease run augment_offline.py first.")
        return None
    
    # Get all classes
    classes = sorted([d for d in os.listdir(input_dir) 
                     if os.path.isdir(os.path.join(input_dir, d))])
    
    if not classes:
        print(f"ERROR: No class folders found in '{input_dir}'!")
        return None
    
    # Count images per class
    print(f"\n{'='*60}")
    print("CLASS DISTRIBUTION")
    print(f"{'='*60}")
    
    class_counts = {}
    for class_name in classes:
        class_dir = os.path.join(input_dir, class_name)
        images = get_image_files(class_dir)
        class_counts[class_name] = len(images)
        print(f"  {class_name}: {len(images)} images")
    
    print(f"{'='*60}\n")
    
    # Clear output directories
    for dir_name in [config["train_dir"], config["val_dir"], config["test_dir"]]:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        os.makedirs(dir_name)
    
    # Target samples per class
    TARGET_PER_CLASS = 1000
    
    # Statistics
    stats = {"total": {"train": 0, "val": 0, "test": 0}}
    
    print(f"{'='*60}")
    print(f"SAMPLING {TARGET_PER_CLASS} PER CLASS, THEN SPLITTING (80-10-10)")
    print(f"{'='*60}")
    
    # Process each class
    for class_name in classes:
        class_input_dir = os.path.join(input_dir, class_name)
        images = get_image_files(class_input_dir)
        
        # Sample exactly 1000 images (or all if less than 1000)
        if len(images) > TARGET_PER_CLASS:
            sampled = random.sample(images, TARGET_PER_CLASS)
        else:
            sampled = images.copy()
            if len(sampled) < TARGET_PER_CLASS:
                print(f"  ⚠️ {class_name}: Only {len(sampled)} images (need {TARGET_PER_CLASS})")
        
        # Shuffle sampled images
        random.shuffle(sampled)
        
        # Calculate split sizes from sampled images
        n = len(sampled)
        train_count = int(n * config["train_ratio"])
        val_count = int(n * config["val_ratio"])
        
        # Split
        train_images = sampled[:train_count]
        val_images = sampled[train_count:train_count + val_count]
        test_images = sampled[train_count + val_count:]
        
        # Create output directories
        for dir_name in [config["train_dir"], config["val_dir"], config["test_dir"]]:
            os.makedirs(os.path.join(dir_name, class_name), exist_ok=True)
        
        # Copy files (use copyfile to copy data only, no metadata - avoids WSL2 permission issues)
        for img in train_images:
            shutil.copyfile(
                os.path.join(class_input_dir, img),
                os.path.join(config["train_dir"], class_name, img)
            )
        
        for img in val_images:
            shutil.copyfile(
                os.path.join(class_input_dir, img),
                os.path.join(config["val_dir"], class_name, img)
            )
        
        for img in test_images:
            shutil.copyfile(
                os.path.join(class_input_dir, img),
                os.path.join(config["test_dir"], class_name, img)
            )
        
        stats["total"]["train"] += len(train_images)
        stats["total"]["val"] += len(val_images)
        stats["total"]["test"] += len(test_images)
        
        print(f"  {class_name}: {len(train_images)} / {len(val_images)} / {len(test_images)}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SPLIT COMPLETE")
    print(f"{'='*60}")
    print(f"Training:   {stats['total']['train']} images → {config['train_dir']}/")
    print(f"Validation: {stats['total']['val']} images → {config['val_dir']}/")
    print(f"Test:       {stats['total']['test']} images → {config['test_dir']}/")
    print(f"{'='*60}\n")
    
    return stats

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("STRATIFIED 80-10-10 DATASET SPLIT")
    print("="*60)
    
    stats = split_dataset(CONFIG)
    
    if stats:
        print("Next step: Run training")
        print("  python train.py")
