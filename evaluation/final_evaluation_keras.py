import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

# --- 1. SETTINGS ---
MODEL_PATH = 'experimental_models/mobilenetv3_small/output/best_model_stage3_1.keras'
TEST_DIR = 'test_data'  # Your test folder with 3,600 images
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
print(f"\n[1/6] Loading model from: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH, 
                                   custom_objects={'focal_loss_fixed': focal_loss()})
print("   Model loaded successfully!")

# --- 3. LOAD TEST DATA ---
print(f"\n[2/6] Loading test data from: {TEST_DIR}")
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

CLASS_NAMES = list(test_data.class_indices.keys())
print(f"   Found {test_data.samples} images across {len(CLASS_NAMES)} classes")
print(f"   Classes: {CLASS_NAMES}")

# --- 4. RUN INFERENCE ---
print("\n[3/6] Running inference on test set...")
predictions = model.predict(test_data, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = test_data.classes

# --- 5. CALCULATE MACRO METRICS ---
print("\n[4/6] Calculating metrics...")
accuracy = np.mean(y_pred == y_true)
precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

# Calculate Top-5 Accuracy
top_5_correct = 0
for i in range(len(y_true)):
    top_5_indices = np.argsort(predictions[i])[-5:]
    if y_true[i] in top_5_indices:
        top_5_correct += 1
top_5_acc = top_5_correct / len(y_true)

# Print Results
print("\n" + "="*70)
print("FINAL EVALUATION RESULTS")
print("="*70)

print("\n--- SECTION 4.3: COMPARATIVE ACCURACY BENCHMARKS ---")
print(f"Final Test Accuracy (Top-1): {accuracy*100:.2f}%")
print(f"Top-5 Accuracy:              {top_5_acc*100:.2f}%")
print(f"Macro-Precision:             {precision:.4f} ({precision*100:.2f}%)")
print(f"Macro-Recall:                {recall:.4f} ({recall*100:.2f}%)")
print(f"Macro-F1 Score:              {f1:.4f} ({f1*100:.2f}%)")

print("\n" + "="*70)
print("THESIS SUMMARY TABLE")
print("="*70)
print("\nMetric                | Value")
print("-" * 70)
print(f"Top-1 Accuracy        | {accuracy*100:.2f}%")
print(f"Top-5 Accuracy        | {top_5_acc*100:.2f}%")
print(f"Macro-Precision       | {precision*100:.2f}%")
print(f"Macro-Recall          | {recall*100:.2f}%")
print(f"Macro-F1 Score        | {f1*100:.2f}%")
print(f"Test Set Size         | {len(y_true)} images")
print(f"Number of Classes     | {len(CLASS_NAMES)} classes")

# --- 6. GENERATE CONFUSION MATRIX ---
print("\n[5/6] Generating confusion matrix...")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(18, 15))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            cbar_kws={'label': 'Count'})
plt.title('BIM Alphanumeric Confusion Matrix (Stage 3.1)', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix_stage3_1.png', dpi=300, bbox_inches='tight')
print("   Confusion Matrix saved as 'confusion_matrix_stage3_1.png'")

# --- 7. PER-CLASS REPORT (For Section 4.5) ---
print("\n[6/6] Generating per-class performance report...")
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0)

with open('per_class_performance.txt', 'w') as f:
    f.write("DETAILED PER-CLASS CLASSIFICATION REPORT\n")
    f.write("="*70 + "\n\n")
    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Test Set: {TEST_DIR}\n")
    f.write(f"Total Images: {len(y_true)}\n\n")
    f.write(report)
    f.write("\n\nSUMMARY METRICS\n")
    f.write("-"*70 + "\n")
    f.write(f"Top-1 Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    f.write(f"Top-5 Accuracy:     {top_5_acc:.4f} ({top_5_acc*100:.2f}%)\n")
    f.write(f"Macro-Precision:    {precision:.4f} ({precision*100:.2f}%)\n")
    f.write(f"Macro-Recall:       {recall:.4f} ({recall*100:.2f}%)\n")
    f.write(f"Macro-F1 Score:     {f1:.4f} ({f1*100:.2f}%)\n")

print("   Per-class report saved to 'per_class_performance.txt'")

# --- 8. SAVE RESULTS TO JSON ---
import json
results = {
    'model': MODEL_PATH,
    'test_set_size': int(len(y_true)),
    'num_classes': len(CLASS_NAMES),
    'top1_accuracy': float(accuracy),
    'top5_accuracy': float(top_5_acc),
    'macro_precision': float(precision),
    'macro_recall': float(recall),
    'macro_f1': float(f1),
}

with open('final_evaluation_keras.json', 'w') as f:
    json.dump(results, f, indent=2)

print("   Results saved to 'final_evaluation_keras.json'")

print("\n" + "="*70)
print("EVALUATION COMPLETE!")
print("="*70)
print("\nFiles generated:")
print("  1. confusion_matrix_stage3_1.png - Confusion matrix (Section 4.4)")
print("  2. per_class_performance.txt - Detailed per-class metrics (Section 4.5)")
print("  3. final_evaluation_keras.json - Machine-readable results")
print("\nUse these results for your thesis!")
