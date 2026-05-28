"""
Generate Confusion Matrix Visualizations for Thesis
- Full 36x36 heatmap
- Zoomed cluster comparison (Stage 2 vs Stage 3.1)
- Error rate reduction calculations
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# Configuration
STAGE2_MODEL = 'stage_training/output/best_model_stage2.keras'
STAGE3_1_MODEL = 'stage_training/output/best_model_stage3_1.keras'
TEST_DATA_DIR = 'test_data'  # FINAL TEST SET - 3,600 images (100 per class)
IMG_SIZE = (224, 224)
CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y', 'Z']

# Focal loss definition
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
print("CONFUSION MATRIX VISUALIZATION FOR THESIS")
print("="*70)

# Load TEST data (FINAL EVALUATION - COMPLETELY UNSEEN)
print("\n[1/6] Loading TEST data (3,600 unseen images)...")
test_gen = tf.keras.preprocessing.image.ImageDataGenerator()  # NO RESCALE - model has it!
test_data = test_gen.flow_from_directory(
    TEST_DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)
print(f"   Found {test_data.samples} TEST images ({test_data.samples // 36} per class)")

# Load Stage 2 model
print("\n[2/6] Loading Stage 2 model...")
try:
    model_stage2 = tf.keras.models.load_model(
        STAGE2_MODEL,
        custom_objects={'focal_loss_fixed': focal_loss()},
        compile=False
    )
    print("   Stage 2 model loaded!")
except Exception as e:
    print(f"   ERROR: {e}")
    print("   Skipping Stage 2...")
    model_stage2 = None

# Load Stage 3.1 model
print("\n[3/6] Loading Stage 3.1 model...")
try:
    model_stage3_1 = tf.keras.models.load_model(
        STAGE3_1_MODEL,
        custom_objects={'focal_loss_fixed': focal_loss()},
        compile=False
    )
    print("   Stage 3.1 model loaded!")
except Exception as e:
    print(f"   ERROR: {e}")
    exit(1)

y_true = test_data.classes

# Get predictions from both models  
print("\n[4/6] Running inference on TEST SET (completely unseen data)...")

if model_stage2:
    print("   Stage 2 predictions...")
    preds_stage2 = model_stage2.predict(test_data, verbose=0)
    y_pred_stage2 = np.argmax(preds_stage2, axis=1)
    cm_stage2 = confusion_matrix(y_true, y_pred_stage2)
else:
    cm_stage2 = None

print("   Stage 3.1 predictions...")
preds_stage3_1 = model_stage3_1.predict(test_data, verbose=0)
y_pred_stage3_1 = np.argmax(preds_stage3_1, axis=1)
cm_stage3_1 = confusion_matrix(y_true, y_pred_stage3_1)

# Calculate Top-5 Accuracy
print("\n   Calculating Top-5 accuracy...")
top5_correct = 0
for i in range(len(y_true)):
    top5_indices = np.argsort(preds_stage3_1[i])[-5:][::-1]  # Get top 5 predictions
    if y_true[i] in top5_indices:
        top5_correct += 1
top5_accuracy = top5_correct / len(y_true)

# Calculate Top-1 Accuracy
top1_accuracy = np.mean(y_pred_stage3_1 == y_true)

print(f"\n   ✓ Top-1 Accuracy: {top1_accuracy*100:.2f}%")
print(f"   ✓ Top-5 Accuracy: {top5_accuracy*100:.2f}%")

# ============================================================================
# FIGURE 1: Full 36x36 Confusion Matrix (Stage 3.1)
# ============================================================================
print("\n[5/6] Generating Figure 1: Full 36x36 Confusion Matrix...")

plt.figure(figsize=(20, 18))
sns.heatmap(cm_stage3_1, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASSES, yticklabels=CLASSES,
            cbar_kws={'label': 'Count'},
            square=True)
plt.title('Confusion Matrix - Stage 3.1 Final Model (36 Classes)', 
          fontsize=18, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix_full_36x36.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: confusion_matrix_full_36x36.png")
plt.close()

# ============================================================================
# FIGURE 2: Zoomed Cluster Comparison (U, V, 2, R)
# ============================================================================
print("\n[6/6] Generating Figure 2: Zoomed Cluster Comparison...")

# Classes of interest: U, V, 2, R
cluster_classes = ['2', 'R', 'U', 'V']
cluster_indices = [CLASSES.index(c) for c in cluster_classes]

# Extract sub-matrices
cm_stage3_1_cluster = cm_stage3_1[np.ix_(cluster_indices, cluster_indices)]

if cm_stage2 is not None:
    cm_stage2_cluster = cm_stage2[np.ix_(cluster_indices, cluster_indices)]
    
    # Create side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Stage 2
    sns.heatmap(cm_stage2_cluster, annot=True, fmt='d', cmap='Oranges',
                xticklabels=cluster_classes, yticklabels=cluster_classes,
                cbar_kws={'label': 'Count'}, ax=ax1, square=True,
                vmin=0, vmax=cm_stage3_1_cluster.max())
    ax1.set_title('Stage 2: Confusing Classes', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    # Stage 3.1
    sns.heatmap(cm_stage3_1_cluster, annot=True, fmt='d', cmap='Greens',
                xticklabels=cluster_classes, yticklabels=cluster_classes,
                cbar_kws={'label': 'Count'}, ax=ax2, square=True,
                vmin=0, vmax=cm_stage3_1_cluster.max())
    ax2.set_title('Stage 3.1: After Fine-Tuning', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    plt.suptitle('Confusion Reduction: High-Similarity Classes (2, R, U, V)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('confusion_cluster_comparison.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: confusion_cluster_comparison.png")
    plt.close()
    
    # Calculate error reduction
    print("\n" + "="*70)
    print("ERROR RATE REDUCTION ANALYSIS")
    print("="*70)
    
    # Calculate off-diagonal errors for each pair
    def get_misclassification_rate(cm, i, j):
        """Get misclassification rate between class i and j"""
        total_i = cm[i, :].sum()
        if total_i == 0:
            return 0
        return cm[i, j] / total_i * 100
    
    pairs = [
        ('U', 'V'),
        ('V', 'U'),
        ('2', 'R'),
        ('R', '2'),
        ('2', 'U'),
        ('U', '2'),
    ]
    
    print("\nMisclassification Rates:")
    print("-" * 70)
    print(f"{'Pair':<10} {'Stage 2':<15} {'Stage 3.1':<15} {'Reduction':<15}")
    print("-" * 70)
    
    for true_class, pred_class in pairs:
        i_cluster = cluster_classes.index(true_class)
        j_cluster = cluster_classes.index(pred_class)
        
        rate_stage2 = get_misclassification_rate(cm_stage2_cluster, i_cluster, j_cluster)
        rate_stage3_1 = get_misclassification_rate(cm_stage3_1_cluster, i_cluster, j_cluster)
        reduction = rate_stage2 - rate_stage3_1
        
        print(f"{true_class}→{pred_class:<8} {rate_stage2:>6.2f}%{'':<8} {rate_stage3_1:>6.2f}%{'':<8} {reduction:>+6.2f}%")
    
    # Overall diagonal accuracy for cluster
    diag_stage2 = np.diag(cm_stage2_cluster).sum() / cm_stage2_cluster.sum() * 100
    diag_stage3_1 = np.diag(cm_stage3_1_cluster).sum() / cm_stage3_1_cluster.sum() * 100
    
    print("-" * 70)
    print(f"{'Overall Cluster Accuracy':<10}")
    print(f"  Stage 2:   {diag_stage2:.2f}%")
    print(f"  Stage 3.1: {diag_stage3_1:.2f}%")
    print(f"  Improvement: +{diag_stage3_1 - diag_stage2:.2f}%")
    
else:
    # Just show Stage 3.1 cluster
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_stage3_1_cluster, annot=True, fmt='d', cmap='Greens',
                xticklabels=cluster_classes, yticklabels=cluster_classes,
                cbar_kws={'label': 'Count'}, square=True)
    plt.title('Confusing Classes Performance (2, R, U, V) - Stage 3.1',
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('confusion_cluster_stage3_1.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: confusion_cluster_stage3_1.png")
    plt.close()

# ============================================================================
# Generate Per-Class Report
# ============================================================================
print("\n" + "="*70)
print("PER-CLASS PERFORMANCE (Stage 3.1)")
print("="*70)

report = classification_report(y_true, y_pred_stage3_1, 
                               target_names=CLASSES,
                               digits=4)
print(report)

# Save report
with open('per_class_performance_stage3_1.txt', 'w') as f:
    f.write("STAGE 3.1 - DETAILED PER-CLASS PERFORMANCE\n")
    f.write("="*70 + "\n\n")
    f.write("OVERALL METRICS\n")
    f.write("-"*70 + "\n")
    f.write(f"Top-1 Accuracy: {top1_accuracy:.4f} ({top1_accuracy*100:.2f}%)\n")
    f.write(f"Top-5 Accuracy: {top5_accuracy:.4f} ({top5_accuracy*100:.2f}%)\n")
    f.write(f"Test Set Size:  {len(y_true)} images\n")
    f.write(f"Num Classes:    {len(CLASSES)}\n\n")
    f.write(report)

print("\n" + "="*70)
print("VISUALIZATION COMPLETE!")
print("="*70)
print("\nGenerated files for thesis:")
print("  1. confusion_matrix_full_36x36.png - Full heatmap")
if cm_stage2 is not None:
    print("  2. confusion_cluster_comparison.png - Before/After comparison")
else:
    print("  2. confusion_cluster_stage3_1.png - Stage 3.1 cluster only")
print("  3. per_class_performance_stage3_1.txt - Detailed metrics")
