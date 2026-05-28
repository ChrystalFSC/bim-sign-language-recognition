import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading model and test data...")
model = keras.models.load_model('output/best_optimized.keras', compile=False)

test_ds = tf.keras.utils.image_dataset_from_directory(
    'test_data',
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical',
    shuffle=False
)
test_ds = test_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
               'U', 'V', 'W', 'X', 'Y', 'Z']

print("Getting predictions...")
y_true = []
y_pred = []
for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(predictions, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Calculate per-class accuracy
per_class_acc = []
for i in range(len(class_names)):
    class_mask = (y_true == i)
    if class_mask.sum() > 0:
        acc = np.mean(y_pred[class_mask] == y_true[class_mask])
        per_class_acc.append(acc * 100)

# Build confusion matrix
cm = np.zeros((len(class_names), len(class_names)), dtype=int)
for true_idx, pred_idx in zip(y_true, y_pred):
    cm[true_idx, pred_idx] += 1

# Create visualizations
fig = plt.figure(figsize=(18, 10))

# 1. Per-class accuracy bar chart
ax1 = plt.subplot(2, 2, 1)
colors = ['#2ecc71' if acc >= 95 else '#f39c12' if acc >= 90 else '#e74c3c' 
          for acc in per_class_acc]
bars = ax1.bar(class_names, per_class_acc, color=colors, edgecolor='black', linewidth=0.5)
ax1.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='90% threshold')
ax1.axhline(y=95, color='green', linestyle='--', alpha=0.5, label='95% threshold')
ax1.set_xlabel('Sign Class', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Per-Class Test Accuracy (Focal Loss Model)', fontsize=14, fontweight='bold')
ax1.set_ylim([85, 101])
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 2. Confusion matrix heatmap (normalized)
ax2 = plt.subplot(2, 2, 2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Proportion'}, ax=ax2)
ax2.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax2.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax2.set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')

# 3. Accuracy distribution
ax3 = plt.subplot(2, 2, 3)
ax3.hist(per_class_acc, bins=10, color='#3498db', edgecolor='black', alpha=0.7)
ax3.axvline(x=np.mean(per_class_acc), color='red', linestyle='--', 
            linewidth=2, label=f'Mean: {np.mean(per_class_acc):.1f}%')
ax3.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Number of Classes', fontsize=12, fontweight='bold')
ax3.set_title('Distribution of Class Accuracies', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# 4. Summary statistics
ax4 = plt.subplot(2, 2, 4)
ax4.axis('off')
overall_acc = np.mean(y_true == y_pred) * 100
summary_text = f"""
MODEL PERFORMANCE SUMMARY
{"="*40}

Overall Test Accuracy:  {overall_acc:.2f}%
Average Class Accuracy: {np.mean(per_class_acc):.2f}%

Classes >= 95%:  {sum(1 for acc in per_class_acc if acc >= 95)}/36
Classes >= 90%:  {sum(1 for acc in per_class_acc if acc >= 90)}/36
Classes >= 70%:  {sum(1 for acc in per_class_acc if acc >= 70)}/36

Total Test Samples:     3,600
Correct Predictions:    {np.sum(y_true == y_pred)}
Incorrect Predictions:  {np.sum(y_true != y_pred)}

Best Classes (100%):
  {', '.join([class_names[i] for i, acc in enumerate(per_class_acc) if acc == 100][:10])}
  ... and {sum(1 for acc in per_class_acc if acc == 100) - 10} more

Weakest Classes:
"""
# Add weakest classes
weak_classes = sorted([(class_names[i], acc) for i, acc in enumerate(per_class_acc)], 
                     key=lambda x: x[1])[:3]
for name, acc in weak_classes:
    summary_text += f"  {name}: {acc:.1f}%\n"

ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', 
         facecolor='wheat', alpha=0.3))

plt.suptitle('Focal Loss Model - Test Set Evaluation Results', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save
output_path = 'output/evaluation_results.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Saved evaluation graphs to: {output_path}")

# Also create a detailed confusion matrix for problematic pairs
fig2, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'}, ax=ax, linewidths=0.5)
ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
ax.set_title('Detailed Confusion Matrix (Raw Counts)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('output/confusion_matrix_detailed.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved detailed confusion matrix to: output/confusion_matrix_detailed.png")

print("\n🎉 All visualizations generated successfully!")
