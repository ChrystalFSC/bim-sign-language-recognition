"""
Classify all images in my_test_images folder using Stage 3.1 Keras model
Automatically processes all .jpg, .jpeg, and .png files
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os
from pathlib import Path

# Paths
MODEL_PATH = 'stage_training/output/best_model_stage3_1.keras'
CLASSES_FILE = 'bim_sign_app/assets/classes.txt'
TEST_FOLDER = 'my_test_images'

# Focal loss for loading model
class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma=5.0, alpha=0.25, name='focal_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
    
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        y_true = tf.cast(y_true, tf.int32)
        num_classes = tf.shape(y_pred)[-1]
        y_true_one_hot = tf.one_hot(y_true, depth=num_classes)
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        weight = self.alpha * tf.pow((1 - y_pred), self.gamma)
        focal_loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))
    
    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha})
        return config

def focal_loss_fn(y_true, y_pred, gamma=5.0, alpha=0.25):
    """Focal loss function for model loading"""
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    y_true = tf.cast(y_true, tf.int32)
    num_classes = tf.shape(y_pred)[-1]
    y_true_one_hot = tf.one_hot(y_true, depth=num_classes)
    cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
    weight = alpha * tf.pow((1 - y_pred), gamma)
    focal_loss = weight * cross_entropy
    return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))

# Check if test folder exists
if not os.path.exists(TEST_FOLDER):
    print(f"Error: Folder '{TEST_FOLDER}' not found!")
    print(f"Please create the folder and add test images.")
    exit(1)

# Get all image files
image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
image_files = []
for ext in image_extensions:
    image_files.extend(Path(TEST_FOLDER).glob(f'*{ext}'))

if len(image_files) == 0:
    print(f"No images found in '{TEST_FOLDER}' folder!")
    print(f"Supported formats: {', '.join(image_extensions)}")
    exit(1)

print(f"Found {len(image_files)} images in '{TEST_FOLDER}'")

# Load class labels
print("\nLoading class labels...")
with open(CLASSES_FILE, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Load model
print(f"Loading model from {MODEL_PATH}...")
model = keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        'FocalLoss': FocalLoss,
        'focal_loss_fn': focal_loss_fn,
        'loss': FocalLoss()
    }
)
print("✓ Model loaded successfully\n")

# Process each image
print("="*60)
print("CLASSIFICATION RESULTS")
print("="*60)

results = []

for idx, image_path in enumerate(sorted(image_files), 1):
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32)
    input_data = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(input_data, verbose=0)[0]
    
    # Get top prediction
    top_idx = np.argmax(predictions)
    predicted_class = classes[top_idx]
    confidence = predictions[top_idx] * 100
    
    # Store result
    results.append({
        'filename': image_path.name,
        'prediction': predicted_class,
        'confidence': confidence
    })
    
    # Print result
    print(f"{idx:2d}. {image_path.name:30s} → {predicted_class:3s} ({confidence:5.2f}%)")

print("="*60)
print(f"\nProcessed {len(results)} images")

# Summary statistics
if results:
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    print(f"Average confidence: {avg_confidence:.2f}%")
    
    high_conf = sum(1 for r in results if r['confidence'] >= 90)
    print(f"High confidence (≥90%): {high_conf}/{len(results)} ({high_conf/len(results)*100:.1f}%)")
