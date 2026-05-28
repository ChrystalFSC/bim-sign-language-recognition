"""
Generate all thesis figures for FYP 2 comparative study.
Saves publication-ready PNG figures to 'thesis_figures/' directory.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

# ============================================================
# DATA FROM ALL 9 MODELS
# ============================================================
MODELS = [
    'AlexNet', 'VGG16', 'VGG19', 'ResNet18', 'DenseNet201',
    'MobileNetV2', 'MobileNetV3\nSmall', 'MobileNetV3\nLarge', 'EfficientNetV2\nM'
]
MODELS_FLAT = [
    'AlexNet', 'VGG16', 'VGG19', 'ResNet18', 'DenseNet201',
    'MobileNetV2', 'MobileNetV3-Small', 'MobileNetV3-Large', 'EfficientNetV2-M'
]

TEST_TOP1 = [99.36, 86.42, 79.92, 99.44, 97.81, 97.42, 96.06, 97.75, 94.72]
TEST_TOP5 = [99.94, 97.36, 95.03, 99.94, 99.86, 99.81, 99.64, 99.89, 99.47]
MACRO_F1  = [0.9936, 0.8638, 0.7950, 0.9945, 0.9781, 0.9743, 0.9605, 0.9774, 0.9472]
MODEL_SIZE_MB = [14.5, 56.6, 76.8, 43.3, 73.3, 10.0, 4.6, 12.8, 205.8]
PARAMS_M  = [3.8, 138, 144, 11, 20, 3.4, 1.5, 5.4, 54]
PRETRAINED = [False, True, True, False, True, True, True, True, True]

# Stage-by-stage best validation accuracy
S1_VAL = [98.69, 85.28, 77.89, 31.14, 97.67, 97.25, 95.39, 97.69, 93.11]
S2_VAL = [99.22, 85.11, 78.11, 99.53, 97.89, 97.53, 95.42, 97.78, 92.83]
S31_VAL = [99.22, 85.36, 79.39, 99.56, 97.86, 97.61, 95.75, 97.81, 93.14]

# Hard class F1 scores (U, V, 2, R)
HARD_F1 = {
    'AlexNet':           [0.9950, 0.9400, 0.9505, 0.9950],
    'VGG16':             [0.6595, 0.5833, 0.6557, 0.8358],
    'VGG19':             [0.6243, 0.3008, 0.5758, 0.7265],
    'ResNet18':          [1.0000, 0.9347, 0.9360, 1.0000],
    'DenseNet201':       [0.9340, 0.9045, 0.9082, 0.9543],
    'MobileNetV2':       [0.9505, 0.8800, 0.8750, 0.9340],
    'MobileNetV3-Small': [0.8912, 0.8627, 0.8776, 0.9490],
    'MobileNetV3-Large': [0.9436, 0.9100, 0.9400, 0.9700],
    'EfficientNetV2-M':  [0.8824, 0.8700, 0.8557, 0.9043],
}

# Output directory
OUT_DIR = 'thesis_figures'
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# STYLE SETUP
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'figure.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

# Color scheme
C_PRETRAINED = '#4C78A8'   # Blue for pretrained
C_SCRATCH    = '#E45756'   # Red/orange for scratch-trained
C_HIGHLIGHT  = '#54A24B'   # Green for MobileNetV3-Small
C_FROZEN     = '#F58518'   # Orange for frozen base

def get_colors():
    """Color each model: green=MNv3S, red=scratch, blue=pretrained"""
    colors = []
    for i, m in enumerate(MODELS_FLAT):
        if 'MobileNetV3-Small' in m:
            colors.append(C_HIGHLIGHT)
        elif not PRETRAINED[i]:
            colors.append(C_SCRATCH)
        else:
            colors.append(C_PRETRAINED)
    return colors

# ============================================================
# FIGURE 4.1: BAR CHART — Final Test Accuracy
# ============================================================
def fig_accuracy_bars():
    order = np.argsort(TEST_TOP1)[::-1]
    colors = get_colors()

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(9), [TEST_TOP1[i] for i in order],
                  color=[colors[i] for i in order], edgecolor='white', linewidth=0.5)

    for bar, i in zip(bars, order):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{TEST_TOP1[i]:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(range(9))
    ax.set_xticklabels([MODELS[i] for i in order], fontsize=9)
    ax.set_ylabel('Test Top-1 Accuracy (%)')
    ax.set_title('Figure 4.1: Final Test Accuracy Comparison Across 9 CNN Architectures')
    ax.set_ylim(70, 102)
    ax.axhline(y=95, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.text(8.5, 95.3, '95% threshold', ha='right', fontsize=8, color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend_items = [Patch(color=C_PRETRAINED, label='Pretrained (ImageNet)'),
                    Patch(color=C_SCRATCH, label='Trained from scratch'),
                    Patch(color=C_HIGHLIGHT, label='MobileNetV3-Small (proposed)')]
    ax.legend(handles=legend_items, loc='lower right', fontsize=9)

    plt.savefig(os.path.join(OUT_DIR, 'fig4_1_accuracy_bars.png'))
    plt.close()
    print("  [OK] fig4_1_accuracy_bars.png")

# ============================================================
# FIGURE 4.2: SCATTER — Accuracy vs Model Size
# ============================================================
def fig_accuracy_vs_size():
    colors = get_colors()
    fig, ax = plt.subplots(figsize=(10, 7))

    for i in range(9):
        size = max(PARAMS_M[i] * 3, 40)
        ax.scatter(MODEL_SIZE_MB[i], TEST_TOP1[i], s=size, c=colors[i],
                   edgecolors='black', linewidth=0.6, zorder=5, alpha=0.85)

        offset_x, offset_y = 3, 0.3
        if MODELS_FLAT[i] == 'EfficientNetV2-M':
            offset_x = -15
        elif MODELS_FLAT[i] == 'DenseNet201':
            offset_y = -1.0
        elif MODELS_FLAT[i] == 'MobileNetV3-Large':
            offset_x = 2
            offset_y = -0.8

        ax.annotate(MODELS_FLAT[i], (MODEL_SIZE_MB[i], TEST_TOP1[i]),
                    textcoords="offset points", xytext=(offset_x, offset_y),
                    fontsize=8, fontweight='bold' if i == 6 else 'normal')

    # Highlight MobileNetV3-Small with a box
    ax.annotate('Best accuracy-to-size\nratio', xy=(4.6, 96.06),
                xytext=(30, -35), textcoords='offset points',
                fontsize=9, color=C_HIGHLIGHT, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_HIGHLIGHT, lw=1.5))

    ax.set_xlabel('Model Size (MB)')
    ax.set_ylabel('Test Top-1 Accuracy (%)')
    ax.set_title('Figure 4.2: Accuracy vs Model Size Trade-off')
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}'))
    ax.set_ylim(75, 101)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2)

    from matplotlib.patches import Patch
    legend_items = [Patch(color=C_PRETRAINED, label='Pretrained'),
                    Patch(color=C_SCRATCH, label='From scratch'),
                    Patch(color=C_HIGHLIGHT, label='MobileNetV3-Small')]
    ax.legend(handles=legend_items, loc='lower left', fontsize=9)

    plt.savefig(os.path.join(OUT_DIR, 'fig4_2_accuracy_vs_size.png'))
    plt.close()
    print("  [OK] fig4_2_accuracy_vs_size.png")

# ============================================================
# FIGURE 4.3: LINE CHART — Stage-by-Stage Progression
# ============================================================
def fig_stage_progression():
    fig, ax = plt.subplots(figsize=(10, 7))
    stages = ['Stage 1\n(Frozen)', 'Stage 2\n(Fine-tune)', 'Stage 3.1\n(Focal Loss)']
    x = [0, 1, 2]
    markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X', 'h']
    cmap = plt.cm.tab10

    for i in range(9):
        vals = [S1_VAL[i], S2_VAL[i], S31_VAL[i]]
        color = cmap(i / 9)
        lw = 2.5 if i == 6 else 1.3  # Bold MobileNetV3-Small
        ax.plot(x, vals, marker=markers[i], label=MODELS_FLAT[i],
                color=color, linewidth=lw, markersize=7, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel('Best Validation Accuracy (%)')
    ax.set_title('Figure 4.3: Stage-by-Stage Accuracy Progression')
    ax.set_ylim(25, 102)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', alpha=0.2)

    # Annotate ResNet18 jump
    ax.annotate('ResNet18: 31% → 99.5%\n(layers unfrozen)',
                xy=(1, 99.53), xytext=(-0.3, 55),
                fontsize=8, color='gray',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1))

    plt.savefig(os.path.join(OUT_DIR, 'fig4_3_stage_progression.png'))
    plt.close()
    print("  [OK] fig4_3_stage_progression.png")

# ============================================================
# FIGURE 4.4: PARAMS vs ACCURACY (dual info)
# ============================================================
def fig_params_vs_accuracy():
    order = np.argsort(PARAMS_M)
    colors = get_colors()

    fig, ax1 = plt.subplots(figsize=(12, 6))
    bars = ax1.bar(range(9), [PARAMS_M[i] for i in order],
                   color=[colors[i] for i in order], alpha=0.7, edgecolor='white')
    ax1.set_ylabel('Parameters (Millions)', color=C_PRETRAINED)
    ax1.set_xticks(range(9))
    ax1.set_xticklabels([MODELS[i] for i in order], fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(range(9), [TEST_TOP1[i] for i in order], 'ko-', linewidth=2, markersize=8, label='Test Accuracy')
    ax2.set_ylabel('Test Top-1 Accuracy (%)')
    ax2.set_ylim(70, 102)

    for j, i in enumerate(order):
        ax2.annotate(f'{TEST_TOP1[i]:.1f}%', (j, TEST_TOP1[i]),
                     textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8)

    ax1.set_title('Figure 4.4: Parameters vs Accuracy — Complexity Does Not Guarantee Performance')
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    plt.savefig(os.path.join(OUT_DIR, 'fig4_4_params_vs_accuracy.png'))
    plt.close()
    print("  [OK] fig4_4_params_vs_accuracy.png")

# ============================================================
# FIGURE 4.5: GROUPED BAR — Hard Class F1 Scores
# ============================================================
def fig_hard_class_f1():
    classes = ['U', 'V', '2', 'R']
    models_list = list(HARD_F1.keys())
    n_models = len(models_list)
    n_classes = len(classes)
    x = np.arange(n_classes)
    width = 0.08
    cmap = plt.cm.tab10

    fig, ax = plt.subplots(figsize=(14, 7))
    for j, model in enumerate(models_list):
        offset = (j - n_models/2 + 0.5) * width
        vals = HARD_F1[model]
        ax.bar(x + offset, vals, width, label=model, color=cmap(j/n_models),
               edgecolor='white', linewidth=0.3)

    ax.axhline(y=0.90, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.text(3.5, 0.905, 'F1 = 0.90 threshold', ha='right', fontsize=8, color='red')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Score')
    ax.set_title('Figure 4.5: Hard Class Performance (U, V, 2, R) Across All Architectures')
    ax.set_ylim(0.2, 1.08)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(os.path.join(OUT_DIR, 'fig4_5_hard_class_f1.png'))
    plt.close()
    print("  [OK] fig4_5_hard_class_f1.png")

# ============================================================
# FIGURE 4.6: FROZEN BASE CONTROL COMPARISON
# ============================================================
def fig_frozen_base():
    labels = ['AlexNet', 'ResNet18']
    trainable = [99.36, 99.44]
    frozen = [18.11, 16.89]
    x = np.arange(2)
    width = 0.3

    fig, ax = plt.subplots(figsize=(8, 6))
    b1 = ax.bar(x - width/2, trainable, width, label='All Layers Trainable', color=C_SCRATCH, edgecolor='white')
    b2 = ax.bar(x + width/2, frozen, width, label='Frozen Base (Random Weights)', color=C_FROZEN, edgecolor='white')

    for bar, val in zip(b1, trainable):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val}%',
                ha='center', fontweight='bold', fontsize=11)
    for bar, val in zip(b2, frozen):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val}%',
                ha='center', fontweight='bold', fontsize=11, color=C_FROZEN)

    # Delta annotations
    for i in range(2):
        delta = trainable[i] - frozen[i]
        ax.annotate(f'Δ = {delta:.1f}%', xy=(x[i], 55), ha='center',
                    fontsize=10, color='gray', fontstyle='italic')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Test Top-1 Accuracy (%)')
    ax.set_title('Table 4.2 (Visual): Transfer Learning Validation\nFreezing Random Weights = Near-Random Performance')
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(os.path.join(OUT_DIR, 'fig4_6_frozen_base_control.png'))
    plt.close()
    print("  [OK] fig4_6_frozen_base_control.png")

# ============================================================
# FIGURE 4.7: FOCAL LOSS IMPACT (S2 vs S3.1)
# ============================================================
def fig_focal_loss_impact():
    deltas = [S31_VAL[i] - S2_VAL[i] for i in range(9)]
    order = np.argsort(deltas)[::-1]
    colors_delta = ['green' if d >= 0 else 'red' for d in [deltas[i] for i in order]]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(9), [deltas[i] for i in order], color=colors_delta, edgecolor='white', alpha=0.8)
    ax.set_yticks(range(9))
    ax.set_yticklabels([MODELS_FLAT[i] for i in order], fontsize=10)
    ax.set_xlabel('Accuracy Change (percentage points)')
    ax.set_title('Figure 4.7: Impact of Focal Loss (Stage 2 → Stage 3.1)')
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, i in zip(bars, order):
        val = deltas[i]
        ax.text(bar.get_width() + 0.02 * (1 if val >= 0 else -1), bar.get_y() + bar.get_height()/2,
                f'{val:+.2f}%', va='center', fontsize=9, fontweight='bold')

    plt.savefig(os.path.join(OUT_DIR, 'fig4_7_focal_loss_impact.png'))
    plt.close()
    print("  [OK] fig4_7_focal_loss_impact.png")

# ============================================================
# FIGURE 5.1: RADAR CHART — Multi-Criteria Comparison (Top 5 models)
# ============================================================
def fig_radar():
    sel = ['MobileNetV3-Small', 'MobileNetV3-Large', 'MobileNetV2', 'DenseNet201', 'EfficientNetV2-M']
    sel_idx = [MODELS_FLAT.index(m) for m in sel]
    categories = ['Accuracy', 'Top-5 Acc', 'Hard Class\nF1 (avg)', 'Size\nEfficiency', 'Param\nEfficiency']

    def normalize_inv(val, all_vals):
        """Invert: smaller = better (for size/params)"""
        return 1 - (val - min(all_vals)) / (max(all_vals) - min(all_vals) + 1e-9)

    data = {}
    for i, m in zip(sel_idx, sel):
        acc_norm = (TEST_TOP1[i] - 79) / (100 - 79)
        top5_norm = (TEST_TOP5[i] - 95) / (100 - 95)
        hard_avg = np.mean(HARD_F1[m])
        size_norm = normalize_inv(MODEL_SIZE_MB[i], [MODEL_SIZE_MB[j] for j in sel_idx])
        param_norm = normalize_inv(PARAMS_M[i], [PARAMS_M[j] for j in sel_idx])
        data[m] = [acc_norm, top5_norm, hard_avg, size_norm, param_norm]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors_radar = ['#54A24B', '#4C78A8', '#F58518', '#E45756', '#72B7B2']

    for (model, vals), color in zip(data.items(), colors_radar):
        vals_closed = vals + vals[:1]
        ax.plot(angles, vals_closed, 'o-', linewidth=2, label=model, color=color)
        ax.fill(angles, vals_closed, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title('Figure 5.1: Multi-Criteria Model Comparison\n(Higher = Better on All Axes)', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)

    plt.savefig(os.path.join(OUT_DIR, 'fig5_1_radar_comparison.png'))
    plt.close()
    print("  [OK] fig5_1_radar_comparison.png")

# ============================================================
# RUN ALL
# ============================================================
if __name__ == '__main__':
    print(f"\nGenerating thesis figures to '{OUT_DIR}/'...\n")
    fig_accuracy_bars()
    fig_accuracy_vs_size()
    fig_stage_progression()
    fig_params_vs_accuracy()
    fig_hard_class_f1()
    fig_frozen_base()
    fig_focal_loss_impact()
    fig_radar()
    print(f"\nDone! All figures saved to '{OUT_DIR}/'")
    print(f"Total: {len(os.listdir(OUT_DIR))} files generated.")
