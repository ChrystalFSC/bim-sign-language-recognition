import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

# --- 1. SETTINGS ---
MODEL_PATH = 'experimental_models/mobilenetv3_small/output/best_model_stage3_1.keras' # Update to your path
TEST_DIR = 'test_data'                          # Update to your test folder
IMG_SIZE = (224, 224)

# --- 2. LOAD MODEL WITH CUSTOM OBJECTS ---
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = alpha * tf.math.pow(1.0 - y_pred, gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)
    return focal_loss_fixed

print("="*70)
print("FINAL MODEL EVALUATION - Stage 3.1 Keras Model")
print("="*70)
print(f"\n[1/6] Loading model: {MODEL_PATH}")

try:
    model = tf.keras.models.load_model(MODEL_PATH, 
                                       custom_objects={'focal_loss_fixed': focal_loss()})
    print("   Model loaded successfully!")
except Exception as e:
    print(f"   ERROR loading model: {e}")
    print(f"\n   Attempting alternative loading method...")
    # Try loading without custom objects and recompile
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(
        optimizer='adam',
        loss=focal_loss(),
        metrics=['accuracy']
    )
    print("   Model loaded and recompiled successfully!")

# Get class names from directory
CLASS_NAMES = sorted(os.listdir(TEST_DIR))

# ---  3. LOAD TEST DATA ---
print(f"\n[2/6] Loading test data from: {TEST_DIR}")
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

print(f"   Found {test_data.samples} images across {len(CLASS_NAMES)} classes")

# --- 4. RUN INFERENCE ---
print("\n[3/6] Running inference on test set...")
predictions = model.predict(test_data)
y_pred = np.argmax(predictions, axis=1)
y_true = test_data.classes

# --- 5. CALCULATE MACRO METRICS ---
print("\n[4/6] Calculating metrics...")
accuracy = np.mean(y_pred == y_true)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

# Calculate Top-5 Accuracy
top_5_correct = 0
for i in range(len(y_true)):
    top_5_indices = np.argsort(predictions[i])[-5:]
    if y_true[i] in top_5_indices:
        top_5_correct += 1
top_5_acc = top_5_correct / len(y_true)

print("\n" + "="*70)
print("SECTION 4.3 DATA FOR THESIS")
print("="*70)
print(f"Final Test Accuracy: {accuracy*100:.2f}%")
print(f"Macro-Precision:    {precision:.4f}")
print(f"Macro-Recall:       {recall:.4f}")
print(f"Macro-F1 Score:     {f1:.4f}")
print(f"Top-5 Accuracy:     {top_5_acc*100:.2f}%")

# --- 6. GENERATE CONFUSION MATRIX ---
print("\n[5/6] Generating confusion matrix...")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(18, 15))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('BIM Alphanumeric Confusion Matrix (Stage 3.1)', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=14)
plt.xlabel('Predicted Label', fontsize=14)
plt.tight_layout()
plt.savefig('confusion_matrix_stage3_1.png', dpi=300, bbox_inches='tight')
print("   SECTION 4.4 DATA")
print("   Confusion Matrix saved as 'confusion_matrix_stage3_1.png'")

# --- 7. PER-CLASS REPORT (For Section 4.5) ---
print("\n[6/6] Generating per-class performance report...")
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
with open('per_class_performance.txt', 'w') as f:
    f.write("SECTION 4.5: DETAILED PER-CLASS PERFORMANCE\n")
    f.write("="*70 + "\n\n")
    f.write(report)
    f.write("\n\nSUMMARY FOR THESIS\n")
    f.write("="*70 + "\n")
    f.write(f"Top-1 Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    f.write(f"Top-5 Accuracy:  {top_5_acc:.4f} ({top_5_acc*100:.2f}%)\n")
    f.write(f"Macro-Precision: {precision:.4f}\n")
    f.write(f"Macro-Recall:    {recall:.4f}\n")
    f.write(f"Macro-F1 Score:  {f1:.4f}\n")

print("   Detailed per-class report saved to 'per_class_performance.txt'")

print("\n" + "="*70)
print("EVALUATION COMPLETE!")
print("="*70)
print("\nGenerated files for thesis:")
print("  - confusion_matrix_stage3_1.png (Section 4.4)")
print("  - per_class_performance.txt (Section 4.5)")
