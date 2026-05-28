"""
BIM Sign Language Dataset - Offline Augmentation
Augments each class to exactly 1000 images

This script:
1. Counts images per class
2. For classes with < 1000 images, generates augmented versions
3. Uses rotation, brightness, zoom, and horizontal flip
4. Saves augmented images as new files
"""

import os
import cv2
import numpy as np
import random
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # Paths
    "input_dir": "processed_data",     # ROI-extracted images
    "output_dir": "augmented_data",    # Output with exactly 1000 per class
    
    # Target
    "target_per_class": 1000,
    
    # Augmentation settings (matching your training plan)
    "rotation_range": 15,              # ±15 degrees
    "brightness_range": (0.8, 1.2),    # 80% to 120% brightness
    "zoom_range": 0.1,                 # ±10% zoom
    "horizontal_flip": True,           # Mirror images
    
    # Random seed
    "seed": 42
}

# =============================================================================
# AUGMENTATION FUNCTIONS
# =============================================================================
def rotate_image(image, angle):
    """Rotate image by given angle."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return rotated

def adjust_brightness(image, factor):
    """Adjust image brightness."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def zoom_image(image, factor):
    """Zoom image by given factor (>1 zooms in, <1 zooms out)."""
    h, w = image.shape[:2]
    
    # Calculate new dimensions
    new_h, new_w = int(h * factor), int(w * factor)
    
    if factor > 1:
        # Zoom in - resize larger then crop center
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        start_x = (new_w - w) // 2
        start_y = (new_h - h) // 2
        zoomed = resized[start_y:start_y+h, start_x:start_x+w]
    else:
        # Zoom out - resize smaller then pad
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        zoomed = np.zeros((h, w, 3), dtype=np.uint8)
        start_x = (w - new_w) // 2
        start_y = (h - new_h) // 2
        zoomed[start_y:start_y+new_h, start_x:start_x+new_w] = resized
    
    return zoomed

def horizontal_flip(image):
    """Flip image horizontally."""
    return cv2.flip(image, 1)

def augment_image(image, config):
    """Apply random augmentation to image."""
    augmented = image.copy()
    
    # Random rotation
    angle = random.uniform(-config["rotation_range"], config["rotation_range"])
    augmented = rotate_image(augmented, angle)
    
    # Random brightness
    brightness = random.uniform(*config["brightness_range"])
    augmented = adjust_brightness(augmented, brightness)
    
    # Random zoom
    zoom = random.uniform(1 - config["zoom_range"], 1 + config["zoom_range"])
    augmented = zoom_image(augmented, zoom)
    
    # Random horizontal flip (50% chance)
    if config["horizontal_flip"] and random.random() > 0.5:
        augmented = horizontal_flip(augmented)
    
    return augmented

# =============================================================================
# MAIN FUNCTIONS
# =============================================================================
def get_image_files(directory):
    """Get all image files in a directory."""
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    return [f for f in os.listdir(directory) if f.lower().endswith(extensions)]

def augment_dataset(config):
    """Augment dataset to reach target count per class."""
    
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    target = config["target_per_class"]
    
    # Set seed
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    
    # Check input
    if not os.path.exists(input_dir):
        print(f"ERROR: Input directory '{input_dir}' not found!")
        return None
    
    # Get classes
    classes = sorted([d for d in os.listdir(input_dir) 
                     if os.path.isdir(os.path.join(input_dir, d))])
    
    if not classes:
        print(f"ERROR: No class folders found!")
        return None
    
    print(f"\n{'='*60}")
    print(f"OFFLINE AUGMENTATION TO {target} IMAGES PER CLASS")
    print(f"{'='*60}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Classes: {len(classes)}")
    print(f"{'='*60}\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Statistics
    stats = {}
    
    # Process each class
    for class_name in classes:
        input_class_dir = os.path.join(input_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)
        
        # Get existing images
        images = get_image_files(input_class_dir)
        original_count = len(images)
        
        # Copy original images
        for img_name in images:
            src = os.path.join(input_class_dir, img_name)
            dst = os.path.join(output_class_dir, img_name)
            img = cv2.imread(src)
            if img is not None:
                cv2.imwrite(dst, img)
        
        # Check if augmentation needed
        if original_count >= target:
            print(f"  {class_name}: {original_count} images (no augmentation needed)")
            stats[class_name] = {"original": original_count, "augmented": 0, "total": original_count}
            continue
        
        # Calculate deficit
        deficit = target - original_count
        
        print(f"  {class_name}: {original_count} → {target} (generating {deficit} augmented)")
        
        # Generate augmented images
        augmented_count = 0
        pbar = tqdm(total=deficit, desc=f"    Augmenting {class_name}", leave=False)
        
        while augmented_count < deficit:
            # Pick random source image
            src_name = random.choice(images)
            src_path = os.path.join(input_class_dir, src_name)
            
            # Load image
            img = cv2.imread(src_path)
            if img is None:
                continue
            
            # Apply augmentation
            aug_img = augment_image(img, config)
            
            # Generate unique filename
            base_name = os.path.splitext(src_name)[0]
            ext = os.path.splitext(src_name)[1]
            aug_name = f"{base_name}_aug{augmented_count:04d}{ext}"
            aug_path = os.path.join(output_class_dir, aug_name)
            
            # Save augmented image
            cv2.imwrite(aug_path, aug_img)
            augmented_count += 1
            pbar.update(1)
        
        pbar.close()
        stats[class_name] = {"original": original_count, "augmented": deficit, "total": target}
    
    # Summary
    print(f"\n{'='*60}")
    print("AUGMENTATION COMPLETE")
    print(f"{'='*60}")
    
    total_original = sum(s["original"] for s in stats.values())
    total_augmented = sum(s["augmented"] for s in stats.values())
    total_final = sum(s["total"] for s in stats.values())
    
    print(f"Original images:  {total_original}")
    print(f"Augmented images: {total_augmented}")
    print(f"Final total:      {total_final}")
    print(f"\nOutput saved to: {output_dir}/")
    print(f"{'='*60}\n")
    
    return stats

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("OFFLINE AUGMENTATION SCRIPT")
    print("="*60)
    
    stats = augment_dataset(CONFIG)
    
    if stats:
        print("Next step: Run split_dataset.py")
        print("  Update split_dataset.py to use 'augmented_data' as input")
        print("  python split_dataset.py")
