import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# 1. Configuration
MODEL_PATH = 'experimental_models/mobilenetv3_small/output/best_model_stage3_1.keras'
TEST_DIR = 'test_data'
IMG_SIZE = (224, 224)

# 2. Load Data (Ensuring we don't shuffle so labels match predictions)
print("Loading test data...")
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=32,
    label_mode='categorical',
    shuffle=False  # CRITICAL: Do not shuffle for the confusion matrix!
)

# Get class names BEFORE mapping (or it disappears!)
class_names = test_ds.class_names

# Normalize test data
def normalize(images, labels):
    return tf.cast(images, tf.float32) / 255.0, labels

test_ds = test_ds.map(normalize)

# 3. Load Model and Predict
print("Loading model and making predictions...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Get true labels and predictions
y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

# 4. Generate Classification Report (Precision, Recall, F1)
print("\n--- CLASSIFICATION REPORT ---")
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)

# Save report to text file for your thesis appendix
with open('experimental_models/mobilenetv3_small/output/classification_report.txt', 'w') as f:
    f.write(report)

# 5. Generate Confusion Matrix
print("\nGenerating Confusion Matrix...")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(18, 14))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('BIM Recognition - Confusion Matrix (Test Set)')
plt.ylabel('Actual Sign')
plt.xlabel('Predicted Sign')

# Save the plot as a high-res image for Chapter 4
plt.savefig('experimental_models/mobilenetv3_small/output/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Saved: experimental_models/mobilenetv3_small/output/confusion_matrix.png")

print("✅ Evaluation complete. Check the 'experimental_models/mobilenetv3_small/output' folder for your visuals!")
