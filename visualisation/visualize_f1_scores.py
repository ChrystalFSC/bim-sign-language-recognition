"""
Visualize Per-Class F1-Scores from Classification Report
Automatically highlights "Hard Examples" (F1 < 0.90) in red
"""

import re
import matplotlib.pyplot as plt
import numpy as np

# Configuration
REPORT_FILE = 'per_class_performance_stage3_1.txt'
OUTPUT_FILE = 'f1_scores_by_class.png'
THRESHOLD = 0.90  # Classes below this are "hard examples"

print("="*70)
print("F1-SCORE VISUALIZATION")
print("="*70)

# Parse the classification report
print(f"\n[1/3] Reading report: {REPORT_FILE}")

classes = []
f1_scores = []

with open(REPORT_FILE, 'r') as f:
    lines = f.readlines()
    
    # Find the classification report section
    in_report = False
    for line in lines:
        # Skip header lines and summary lines
        if 'precision' in line and 'recall' in line:
            in_report = True
            continue
        
        if not in_report:
            continue
            
        # Stop at accuracy line
        if 'accuracy' in line or 'macro avg' in line or 'weighted avg' in line:
            break
        
        # Parse class performance line
        # Format: "     0     0.9876    0.9900    0.9888       100"
        parts = line.split()
        if len(parts) >= 4:
            try:
                class_name = parts[0]
                precision = float(parts[1])
                recall = float(parts[2])
                f1 = float(parts[3])
                
                # Skip if invalid
                if f1 > 0:  # Valid F1 score
                    classes.append(class_name)
                    f1_scores.append(f1)
            except (ValueError, IndexError):
                continue

print(f"   Found {len(classes)} classes")

# Identify hard examples
hard_classes = [i for i, f1 in enumerate(f1_scores) if f1 < THRESHOLD]
easy_classes = [i for i, f1 in enumerate(f1_scores) if f1 >= THRESHOLD]

print(f"\n[2/3] Analysis:")
print(f"   Easy classes (F1 >= {THRESHOLD}): {len(easy_classes)}")
print(f"   Hard classes (F1 < {THRESHOLD}):  {len(hard_classes)}")

if hard_classes:
    print(f"\n   Hard Examples:")
    for idx in hard_classes:
        print(f"     {classes[idx]}: F1 = {f1_scores[idx]:.4f}")

# Create visualization
print(f"\n[3/3] Generating visualization...")

fig, ax = plt.subplots(figsize=(18, 8))

# Create color array (red for hard, green for easy)
colors = ['#e74c3c' if f1 < THRESHOLD else '#27ae60' for f1 in f1_scores]

# Create bar chart
bars = ax.bar(range(len(classes)), f1_scores, color=colors, edgecolor='black', linewidth=1)

# Customize plot
ax.set_xlabel('Class', fontsize=14, fontweight='bold')
ax.set_ylabel('F1-Score', fontsize=14, fontweight='bold')
ax.set_title('Per-Class F1-Scores (Stage 3.1 Test Set)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(range(len(classes)))
ax.set_xticklabels(classes, rotation=0, ha='center', fontsize=11)
ax.set_ylim([0, 1.05])
ax.axhline(y=THRESHOLD, color='red', linestyle='--', linewidth=2, 
           label=f'Hard Example Threshold (F1 < {THRESHOLD})', alpha=0.7)
ax.axhline(y=1.0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for i, (bar, f1) in enumerate(zip(bars, f1_scores)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{f1:.3f}',
            ha='center', va='bottom', fontsize=8, rotation=90)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#27ae60', edgecolor='black', label=f'Easy (F1 ≥ {THRESHOLD})'),
    Patch(facecolor='#e74c3c', edgecolor='black', label=f'Hard (F1 < {THRESHOLD})')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

# Stats box - moved to bottom right to avoid hiding values
stats_text = f'Total Classes: {len(classes)}\n'
stats_text += f'Easy: {len(easy_classes)} ({len(easy_classes)/len(classes)*100:.1f}%)\n'
stats_text += f'Hard: {len(hard_classes)} ({len(hard_classes)/len(classes)*100:.1f}%)\n'
stats_text += f'Mean F1: {np.mean(f1_scores):.4f}'

ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='bottom', horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: {OUTPUT_FILE}")

# Summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
print(f"\nOverall Performance:")
print(f"  Mean F1-Score:   {np.mean(f1_scores):.4f}")
print(f"  Median F1-Score: {np.median(f1_scores):.4f}")
print(f"  Min F1-Score:    {np.min(f1_scores):.4f} (Class: {classes[np.argmin(f1_scores)]})")
print(f"  Max F1-Score:    {np.max(f1_scores):.4f} (Class: {classes[np.argmax(f1_scores)]})")
print(f"  Std Dev:         {np.std(f1_scores):.4f}")

print(f"\nClass Distribution:")
print(f"  Easy Classes: {len(easy_classes)}/{len(classes)} ({len(easy_classes)/len(classes)*100:.1f}%)")
print(f"  Hard Classes: {len(hard_classes)}/{len(classes)} ({len(hard_classes)/len(classes)*100:.1f}%)")

if hard_classes:
    print(f"\nHard Examples (F1 < {THRESHOLD}):")
    for idx in sorted(hard_classes, key=lambda i: f1_scores[i]):
        print(f"  {classes[idx]:>3}: {f1_scores[idx]:.4f}")

print("\n" + "="*70)
print("VISUALIZATION COMPLETE!")
print("="*70)
print(f"\nGenerated: {OUTPUT_FILE}")
print("Use this figure in your thesis to show per-class performance.")
