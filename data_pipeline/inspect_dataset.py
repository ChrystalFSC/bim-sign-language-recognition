"""
Visual inspection of dataset - check weak classes for quality issues
"""
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Classes to inspect (weak performers)
WEAK_CLASSES = ['V', '2', 'R', 'U', 'W']
ALL_CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
               'U', 'V', 'W', 'X', 'Y', 'Z']

DATA_DIR = "train_data"
SAMPLES_PER_CLASS = 8

def inspect_class(class_name, data_dir, num_samples=8):
    """Display random samples from a class."""
    class_path = os.path.join(data_dir, class_name)
    
    if not os.path.exists(class_path):
        print(f"Class {class_name} not found at {class_path}")
        return
    
    all_images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if len(all_images) < num_samples:
        samples = all_images
    else:
        samples = random.sample(all_images, num_samples)
    
    # Create grid
    cols = 4
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    fig.suptitle(f"Class: {class_name} (Random Samples from {len(all_images)} images)", fontsize=14)
    
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for i, ax in enumerate(axes):
        if i < len(samples):
            img_path = os.path.join(class_path, samples[i])
            try:
                img = Image.open(img_path)
                ax.imshow(img)
                ax.set_title(samples[i][:15], fontsize=8)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"output/inspect_{class_name}.png", dpi=100)
    plt.show()
    print(f"Saved: output/inspect_{class_name}.png")

def compare_confusing_classes():
    """Side-by-side comparison of commonly confused classes."""
    confusions = [
        ('V', '2'),
        ('V', 'U'),
        ('R', 'U'),
        ('2', '5'),
        ('W', 'V'),
    ]
    
    for cls1, cls2 in confusions:
        path1 = os.path.join(DATA_DIR, cls1)
        path2 = os.path.join(DATA_DIR, cls2)
        
        if not os.path.exists(path1) or not os.path.exists(path2):
            continue
        
        images1 = random.sample(os.listdir(path1), min(4, len(os.listdir(path1))))
        images2 = random.sample(os.listdir(path2), min(4, len(os.listdir(path2))))
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f"Comparison: {cls1} vs {cls2} (Are they similar?)", fontsize=14)
        
        for i, img_name in enumerate(images1):
            img = Image.open(os.path.join(path1, img_name))
            axes[0, i].imshow(img)
            axes[0, i].set_title(f"{cls1}", fontsize=12, color='blue')
            axes[0, i].axis('off')
        
        for i, img_name in enumerate(images2):
            img = Image.open(os.path.join(path2, img_name))
            axes[1, i].imshow(img)
            axes[1, i].set_title(f"{cls2}", fontsize=12, color='red')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"output/compare_{cls1}_vs_{cls2}.png", dpi=100)
        plt.show()
        print(f"Saved: output/compare_{cls1}_vs_{cls2}.png")

def check_image_stats(class_name):
    """Check image statistics for a class."""
    class_path = os.path.join(DATA_DIR, class_name)
    
    if not os.path.exists(class_path):
        return
    
    images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    sizes = []
    brightnesses = []
    
    for img_name in images[:100]:  # Sample 100
        img = Image.open(os.path.join(class_path, img_name))
        sizes.append(img.size)
        
        img_array = np.array(img)
        brightness = np.mean(img_array)
        brightnesses.append(brightness)
    
    unique_sizes = set(sizes)
    avg_brightness = np.mean(brightnesses)
    
    print(f"\nClass {class_name}:")
    print(f"  Total images: {len(images)}")
    print(f"  Unique sizes: {len(unique_sizes)}")
    print(f"  Common sizes: {list(unique_sizes)[:3]}")
    print(f"  Avg brightness: {avg_brightness:.1f}")

def main():
    print("="*60)
    print("DATASET INSPECTION - WEAK CLASSES")
    print("="*60)
    
    # Create output dir if needed
    os.makedirs("output", exist_ok=True)
    
    # 1. Inspect weak classes
    print("\n1. Inspecting weak classes...")
    for cls in WEAK_CLASSES:
        print(f"\n--- Class: {cls} ---")
        inspect_class(cls, DATA_DIR, SAMPLES_PER_CLASS)
        check_image_stats(cls)
    
    # 2. Compare confusing pairs
    print("\n" + "="*60)
    print("2. Comparing commonly confused class pairs...")
    print("="*60)
    compare_confusing_classes()
    
    print("\n" + "="*60)
    print("INSPECTION COMPLETE!")
    print("="*60)
    print("\nCheck the output/ folder for saved images.")
    print("\nLook for:")
    print("  - Mislabeled images (wrong sign)")
    print("  - Poor quality (blurry, dark)")
    print("  - Inconsistent hand positions")
    print("  - Classes that look too similar")

if __name__ == "__main__":
    main()
