"""
Generate confusion matrix for Stage 3.1 model
Shows which signs the model confuses with each other
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

# Paths
MODEL_PATH = 'experimental_models/mobilenetv3_small/output/best_model_stage3_1.keras'
TEST_DATA_PATH = 'test_data'
CLASSES_FILE = 'bim_sign_app/assets/classes.txt'
OUTPUT_DIR = 'experimental_models/mobilenetv3_small/output'

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

print("="*60)
print("CONFUSION MATRIX GENERATOR - Stage 3.1 Model")
print("="*60)

# Load class names
print("\nLoading class labels...")
with open(CLASSES_FILE, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]
print(f"Classes: {len(class_names)}")

# Load model
print(f"\nLoading model from {MODEL_PATH}...")
model = keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        'FocalLoss': FocalLoss,
        'focal_loss_fn': focal_loss_fn,
        'loss': FocalLoss()
    }
)
print("✓ Model loaded successfully")

# Load test dataset
print(f"\nLoading test data from {TEST_DATA_PATH}...")
test_ds = keras.preprocessing.image_dataset_from_directory(
    TEST_DATA_PATH,
    image_size=(224, 224),
    batch_size=32,
    shuffle=False,
    label_mode='int'
)

# Add rescaling (model has internal rescaling layer, but let's be consistent)
print("Dataset loaded successfully")
print(f"Total test batches: {len(test_ds)}")

# Get predictions
print("\nGenerating predictions on test set...")
y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print(f"✓ Predictions complete ({len(y_true)} samples)")

# Calculate accuracy
accuracy = np.mean(y_true == y_pred) * 100
print(f"\nOverall Accuracy: {accuracy:.2f}%")

# Generate confusion matrix
print("\nGenerating confusion matrix...")
cm = confusion_matrix(y_true, y_pred)

# Create figure with larger size for 36 classes
plt.figure(figsize=(20, 18))

# Plot confusion matrix
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names,
    cbar_kws={'label': 'Number of Predictions'},
    square=True
)

plt.title(f'Confusion Matrix - Stage 3.1 Model\nAccuracy: {accuracy:.2f}%', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Class', fontsize=14, fontweight='bold')
plt.ylabel('True Class', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Save figure
output_file = os.path.join(OUTPUT_DIR, 'confusion_matrix_stage3_1.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Confusion matrix saved to: {output_file}")

# Generate normalized confusion matrix (percentages)
plt.figure(figsize=(20, 18))
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

sns.heatmap(
    cm_normalized,
    annot=True,
    fmt='.2f',
    cmap='RdYlGn',
    xticklabels=class_names,
    yticklabels=class_names,
    cbar_kws={'label': 'Percentage'},
    square=True,
    vmin=0,
    vmax=1
)

plt.title(f'Normalized Confusion Matrix - Stage 3.1 Model\nAccuracy: {accuracy:.2f}%', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Class', fontsize=14, fontweight='bold')
plt.ylabel('True Class', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Save normalized version
output_file_norm = os.path.join(OUTPUT_DIR, 'confusion_matrix_normalized_stage3_1.png')
plt.savefig(output_file_norm, dpi=300, bbox_inches='tight')
print(f"✓ Normalized confusion matrix saved to: {output_file_norm}")

# Classification report
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print(report)

# Save classification report
report_file = os.path.join(OUTPUT_DIR, 'classification_report_stage3_1.txt')
with open(report_file, 'w') as f:
    f.write("Classification Report - Stage 3.1 Model\n")
    f.write("="*60 + "\n")
    f.write(f"Overall Accuracy: {accuracy:.2f}%\n\n")
    f.write(report)
print(f"✓ Classification report saved to: {report_file}")

# Find most confused pairs
print("\n" + "="*60)
print("TOP 10 MOST CONFUSED PAIRS")
print("="*60)

confused_pairs = []
for i in range(len(class_names)):
    for j in range(len(class_names)):
        if i != j and cm[i][j] > 0:
            confused_pairs.append({
                'true': class_names[i],
                'predicted': class_names[j],
                'count': cm[i][j],
                'percentage': cm[i][j] / cm[i].sum() * 100
            })

confused_pairs = sorted(confused_pairs, key=lambda x: x['count'], reverse=True)[:10]

for idx, pair in enumerate(confused_pairs, 1):
    print(f"{idx:2d}. '{pair['true']}' → '{pair['predicted']}': "
          f"{pair['count']:3d} times ({pair['percentage']:5.1f}%)")

# Per-class accuracy
print("\n" + "="*60)
print("PER-CLASS ACCURACY")
print("="*60)

class_accuracies = []
for i in range(len(class_names)):
    if cm[i].sum() > 0:
        acc = cm[i][i] / cm[i].sum() * 100
        class_accuracies.append({'class': class_names[i], 'accuracy': acc, 'samples': cm[i].sum()})

# Sort by accuracy
class_accuracies = sorted(class_accuracies, key=lambda x: x['accuracy'])

print("\nWorst 10 Classes:")
for idx, item in enumerate(class_accuracies[:10], 1):
    print(f"{idx:2d}. {item['class']:3s}: {item['accuracy']:5.1f}% ({item['samples']} samples)")

print("\nBest 10 Classes:")
for idx, item in enumerate(class_accuracies[-10:][::-1], 1):
    print(f"{idx:2d}. {item['class']:3s}: {item['accuracy']:5.1f}% ({item['samples']} samples)")

print("\n" + "="*60)
print("COMPLETE!")
print("="*60)
print(f"\nGenerated files:")
print(f"1. {output_file}")
print(f"2. {output_file_norm}")
print(f"3. {report_file}")
