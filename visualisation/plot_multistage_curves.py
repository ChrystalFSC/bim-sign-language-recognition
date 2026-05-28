import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load all four stage training logs
stage1 = pd.read_csv('stage_training/output/training_log_stage1.csv')
stage2 = pd.read_csv('stage_training/output/training_log_stage2.csv')
stage3 = pd.read_csv('stage_training/output/training_log_stage3.csv')
stage3_1 = pd.read_csv('stage_training/output/training_log_stage3_1.csv')

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

# Calculate cumulative epochs
stage1_epochs = len(stage1)
stage2_epochs = len(stage2)
stage3_epochs = len(stage3)
stage3_1_epochs = len(stage3_1)

stage1['cumulative_epoch'] = stage1['epoch'] + 1
stage2['cumulative_epoch'] = stage2['epoch'] + stage1_epochs + 1
stage3['cumulative_epoch'] = stage3['epoch'] + stage1_epochs + stage2_epochs + 1
stage3_1['cumulative_epoch'] = stage3_1['epoch'] + stage1_epochs + stage2_epochs + stage3_epochs + 1

# Plot 1: Accuracy
ax1.plot(stage1['cumulative_epoch'], stage1['accuracy'], 'b-', linewidth=2, label='Stage 1 - Train Acc')
ax1.plot(stage1['cumulative_epoch'], stage1['val_accuracy'], 'b--', linewidth=2, label='Stage 1 - Val Acc')

ax1.plot(stage2['cumulative_epoch'], stage2['accuracy'], 'g-', linewidth=2, label='Stage 2 - Train Acc')
ax1.plot(stage2['cumulative_epoch'], stage2['val_accuracy'], 'g--', linewidth=2, label='Stage 2 - Val Acc')

ax1.plot(stage3['cumulative_epoch'], stage3['accuracy'], 'r-', linewidth=2, label='Stage 3 - Train Acc')
ax1.plot(stage3['cumulative_epoch'], stage3['val_accuracy'], 'r--', linewidth=2, label='Stage 3 - Val Acc')

ax1.plot(stage3_1['cumulative_epoch'], stage3_1['accuracy'], 'm-', linewidth=2, label='Stage 3.1 - Train Acc')
ax1.plot(stage3_1['cumulative_epoch'], stage3_1['val_accuracy'], 'm--', linewidth=2, label='Stage 3.1 - Val Acc')

# Add vertical lines to separate stages
ax1.axvline(x=stage1_epochs, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
ax1.axvline(x=stage1_epochs + stage2_epochs, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
ax1.axvline(x=stage1_epochs + stage2_epochs + stage3_epochs, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

# Add stage labels
ax1.text(stage1_epochs/2, 0.95, 'Stage 1\n(Top Layers)', 
         ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
ax1.text(stage1_epochs + stage2_epochs/2, 0.95, 'Stage 2\n(Mid Layers)', 
         ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
ax1.text(stage1_epochs + stage2_epochs + stage3_epochs/2, 0.95, 'Stage 3\n(Full Model)', 
         ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
ax1.text(stage1_epochs + stage2_epochs + stage3_epochs + stage3_1_epochs/2, 0.95, 'Stage 3.1\n(Fine-tune)', 
         ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='plum', alpha=0.5))

ax1.set_xlabel('Cumulative Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Multi-Stage Training: Accuracy Progress', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right', fontsize=8, ncol=2)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.5, 1.05])

# Plot 2: Loss
ax2.plot(stage1['cumulative_epoch'], stage1['loss'], 'b-', linewidth=2, label='Stage 1 - Train Loss')
ax2.plot(stage1['cumulative_epoch'], stage1['val_loss'], 'b--', linewidth=2, label='Stage 1 - Val Loss')

ax2.plot(stage2['cumulative_epoch'], stage2['loss'], 'g-', linewidth=2, label='Stage 2 - Train Loss')
ax2.plot(stage2['cumulative_epoch'], stage2['val_loss'], 'g--', linewidth=2, label='Stage 2 - Val Loss')

ax2.plot(stage3['cumulative_epoch'], stage3['loss'], 'r-', linewidth=2, label='Stage 3 - Train Loss')
ax2.plot(stage3['cumulative_epoch'], stage3['val_loss'], 'r--', linewidth=2, label='Stage 3 - Val Loss')

ax2.plot(stage3_1['cumulative_epoch'], stage3_1['loss'], 'm-', linewidth=2, label='Stage 3.1 - Train Loss')
ax2.plot(stage3_1['cumulative_epoch'], stage3_1['val_loss'], 'm--', linewidth=2, label='Stage 3.1 - Val Loss')

# Add vertical lines to separate stages
ax2.axvline(x=stage1_epochs, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
ax2.axvline(x=stage1_epochs + stage2_epochs, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
ax2.axvline(x=stage1_epochs + stage2_epochs + stage3_epochs, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

# Add stage labels
ax2.text(stage1_epochs/2, ax2.get_ylim()[1]*0.95, 'Stage 1\n(Top Layers)', 
         ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
ax2.text(stage1_epochs + stage2_epochs/2, ax2.get_ylim()[1]*0.95, 'Stage 2\n(Mid Layers)', 
         ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
ax2.text(stage1_epochs + stage2_epochs + stage3_epochs/2, ax2.get_ylim()[1]*0.95, 'Stage 3\n(Full Model)', 
         ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
ax2.text(stage1_epochs + stage2_epochs + stage3_epochs + stage3_1_epochs/2, ax2.get_ylim()[1]*0.95, 'Stage 3.1\n(Fine-tune)', 
         ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='plum', alpha=0.5))

ax2.set_xlabel('Cumulative Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax2.set_title('Multi-Stage Training: Loss Progress', fontsize=14, fontweight='bold')
ax2.legend(loc='center right', fontsize=8, ncol=1)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('multistage_learning_curves.png', dpi=300, bbox_inches='tight')
print("SUCCESS: Multi-stage learning curves saved to: multistage_learning_curves.png")

# Print summary statistics
print("\n" + "="*70)
print("MULTI-STAGE TRAINING SUMMARY (4 Stages)")
print("="*70)

print(f"\nStage 1 (Epochs 1-{stage1_epochs}):")
print(f"  Initial Val Acc: {stage1['val_accuracy'].iloc[0]:.4f}")
print(f"  Final Val Acc:   {stage1['val_accuracy'].iloc[-1]:.4f}")
print(f"  Improvement:     +{(stage1['val_accuracy'].iloc[-1] - stage1['val_accuracy'].iloc[0]):.4f}")

print(f"\nStage 2 (Epochs {stage1_epochs+1}-{stage1_epochs+stage2_epochs}):")
print(f"  Initial Val Acc: {stage2['val_accuracy'].iloc[0]:.4f}")
print(f"  Final Val Acc:   {stage2['val_accuracy'].iloc[-1]:.4f}")
print(f"  Improvement:     +{(stage2['val_accuracy'].iloc[-1] - stage2['val_accuracy'].iloc[0]):.4f}")

print(f"\nStage 3 (Epochs {stage1_epochs+stage2_epochs+1}-{stage1_epochs+stage2_epochs+stage3_epochs}):")
print(f"  Initial Val Acc: {stage3['val_accuracy'].iloc[0]:.4f}")
print(f"  Final Val Acc:   {stage3['val_accuracy'].iloc[-1]:.4f}")
print(f"  Improvement:     +{(stage3['val_accuracy'].iloc[-1] - stage3['val_accuracy'].iloc[0]):.4f}")

print(f"\nStage 3.1 (Epochs {stage1_epochs+stage2_epochs+stage3_epochs+1}-{stage1_epochs+stage2_epochs+stage3_epochs+stage3_1_epochs}):")
print(f"  Initial Val Acc: {stage3_1['val_accuracy'].iloc[0]:.4f}")
print(f"  Final Val Acc:   {stage3_1['val_accuracy'].iloc[-1]:.4f}")
print(f"  Best Val Acc:    {stage3_1['val_accuracy'].max():.4f}")
print(f"  Improvement:     +{(stage3_1['val_accuracy'].iloc[-1] - stage3_1['val_accuracy'].iloc[0]):.4f}")

print(f"\nOverall Progress:")
print(f"  Starting Val Acc:  {stage1['val_accuracy'].iloc[0]:.4f}")
print(f"  Final Val Acc:     {stage3_1['val_accuracy'].iloc[-1]:.4f}")
print(f"  Best Val Acc:      {stage3_1['val_accuracy'].max():.4f}")
print(f"  Total Improvement: +{(stage3_1['val_accuracy'].iloc[-1] - stage1['val_accuracy'].iloc[0]):.4f}")
print(f"  Total Epochs:      {stage1_epochs + stage2_epochs + stage3_epochs + stage3_1_epochs}")

plt.show()
