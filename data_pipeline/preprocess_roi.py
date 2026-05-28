"""
BIM Sign Language Dataset Preprocessing
ROI Extraction with 15% Buffer + Resize to 224×224

This script:
1. Detects hands using MediaPipe
2. Extracts ROI with 15% buffer margin
3. Resizes to 224×224
4. Saves processed images maintaining folder structure
"""

import os
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
import shutil

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # Input/Output paths
    "input_dir": "cleaned_data",      # Your MediaPipe-cleaned images
    "output_dir": "processed_data",   # Where processed images will be saved
    
    # Processing settings
    "target_size": (224, 224),        # Final image size
    "roi_buffer": 0.15,               # 15% buffer around hand
    
    # MediaPipe settings
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
}

# =============================================================================
# MEDIAPIPE SETUP
# =============================================================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def get_hand_roi(image, landmarks, buffer=0.15):
    """
    Extract ROI around hand landmarks with buffer margin.
    
    Args:
        image: Input image (BGR)
        landmarks: MediaPipe hand landmarks
        buffer: Buffer margin as percentage (0.15 = 15%)
    
    Returns:
        Cropped ROI image or None if failed
    """
    h, w, _ = image.shape
    
    # Get all landmark coordinates
    x_coords = [lm.x * w for lm in landmarks.landmark]
    y_coords = [lm.y * h for lm in landmarks.landmark]
    
    # Calculate bounding box
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Calculate box dimensions
    box_width = x_max - x_min
    box_height = y_max - y_min
    
    # Add buffer margin (15%)
    buffer_x = box_width * buffer
    buffer_y = box_height * buffer
    
    # Expand bounding box with buffer
    x_min = max(0, int(x_min - buffer_x))
    x_max = min(w, int(x_max + buffer_x))
    y_min = max(0, int(y_min - buffer_y))
    y_max = min(h, int(y_max + buffer_y))
    
    # Ensure valid ROI
    if x_max <= x_min or y_max <= y_min:
        return None
    
    # Crop ROI
    roi = image[y_min:y_max, x_min:x_max]
    
    return roi

def process_image(image_path, hands_detector, config):
    """
    Process a single image: detect hand, extract ROI, resize.
    
    Args:
        image_path: Path to input image
        hands_detector: MediaPipe Hands object
        config: Configuration dictionary
    
    Returns:
        Processed image or None if hand not detected
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # Convert to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect hands
    results = hands_detector.process(image_rgb)
    
    if not results.multi_hand_landmarks:
        return None
    
    # Use first detected hand
    hand_landmarks = results.multi_hand_landmarks[0]
    
    # Extract ROI with buffer
    roi = get_hand_roi(image, hand_landmarks, config["roi_buffer"])
    
    if roi is None or roi.size == 0:
        return None
    
    # Resize to target size
    resized = cv2.resize(roi, config["target_size"], interpolation=cv2.INTER_LANCZOS4)
    
    return resized

def process_dataset(config):
    """
    Process entire dataset: ROI extraction + resize for all images.
    """
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    
    # Check input directory exists
    if not os.path.exists(input_dir):
        print(f"ERROR: Input directory '{input_dir}' not found!")
        print(f"Please place your cleaned images in '{input_dir}/' with subfolders for each class (A, B, ..., Z, 0, 1, ..., 9)")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all class folders
    classes = sorted([d for d in os.listdir(input_dir) 
                     if os.path.isdir(os.path.join(input_dir, d))])
    
    if not classes:
        print(f"ERROR: No class folders found in '{input_dir}'!")
        return
    
    print(f"\n{'='*60}")
    print("BIM SIGN LANGUAGE DATASET PREPROCESSING")
    print(f"{'='*60}")
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target size:      {config['target_size']}")
    print(f"ROI buffer:       {config['roi_buffer']*100:.0f}%")
    print(f"Classes found:    {len(classes)}")
    print(f"{'='*60}\n")
    
    # Statistics
    stats = {
        "total_processed": 0,
        "total_failed": 0,
        "per_class": {}
    }
    
    # Initialize MediaPipe Hands
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=config["min_detection_confidence"]
    ) as hands:
        
        # Process each class
        for class_name in classes:
            input_class_dir = os.path.join(input_dir, class_name)
            output_class_dir = os.path.join(output_dir, class_name)
            os.makedirs(output_class_dir, exist_ok=True)
            
            # Get all images in class
            image_files = [f for f in os.listdir(input_class_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            processed = 0
            failed = 0
            
            # Process each image with progress bar
            for filename in tqdm(image_files, desc=f"Processing {class_name}", leave=False):
                input_path = os.path.join(input_class_dir, filename)
                output_path = os.path.join(output_class_dir, filename)
                
                # Process image
                result = process_image(input_path, hands, config)
                
                if result is not None:
                    cv2.imwrite(output_path, result)
                    processed += 1
                else:
                    failed += 1
            
            # Update stats
            stats["total_processed"] += processed
            stats["total_failed"] += failed
            stats["per_class"][class_name] = {"processed": processed, "failed": failed}
            
            print(f"  {class_name}: {processed} processed, {failed} failed")
    
    # Print summary
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total processed: {stats['total_processed']}")
    print(f"Total failed:    {stats['total_failed']}")
    print(f"Output saved to: {output_dir}/")
    print(f"{'='*60}\n")
    
    # Warn about imbalanced classes
    counts = [stats["per_class"][c]["processed"] for c in classes]
    if max(counts) - min(counts) > 100:
        print("⚠️  WARNING: Class imbalance detected!")
        print("   Consider balancing your dataset before training.\n")
    
    return stats

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("ROI EXTRACTION WITH 15% BUFFER")
    print("="*60 + "\n")
    
    # Check if input directory exists
    if not os.path.exists(CONFIG["input_dir"]):
        print(f"Input directory '{CONFIG['input_dir']}' not found!")
        print("\nPlease:")
        print(f"1. Create folder: {CONFIG['input_dir']}/")
        print("2. Add class subfolders: A/, B/, ..., Z/, 0/, 1/, ..., 9/")
        print("3. Place your cleaned images in each class folder")
        print("4. Run this script again")
    else:
        process_dataset(CONFIG)
