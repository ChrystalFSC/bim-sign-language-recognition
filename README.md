# BIM Sign Language Recognition — Deep Learning Benchmarking & Mobile Deployment

A systematic comparative study of **9 CNN architectures** for mobile-based Bahasa Isyarat Malaysia (BIM) fingerspelling recognition, with mobile deployment via a two-stage MediaPipe + MobileNetV3-Small inference pipeline.

---

## Key Results 🏆

### Proposed Model — MobileNetV3-Small

| Metric | Value |
|:---|:---|
| Test Top-1 Accuracy | **96.06%** |
| Test Top-5 Accuracy | **99.64%** |
| Macro F1-Score | 0.9605 |
| Model Parameters | 959,892 |
| Model Size (Keras) | 4.6 MB |
| Model Size (TFLite Float16) | **1.87 MB** |
| Mobile Inference Latency | **174 ms** (Snapdragon 845) |
| GPU Delegation | 217 / 217 ops |

### 9-Model Benchmark Comparison

| Architecture | Top-1 Acc (%) | Top-5 Acc (%) | Params | Size (MB) |
|:---|:---:|:---:|:---:|:---:|
| ResNet18 *(scratch)* | **99.44** | 99.94 | ~11M | 43.3 |
| AlexNet *(scratch)* | 99.36 | 99.94 | ~3.8M | 14.5 |
| DenseNet201 | 97.81 | 99.86 | ~20M | 73.3 |
| MobileNetV3-Large | 97.75 | 99.89 | ~5.4M | 12.8 |
| MobileNetV2 | 97.42 | 99.81 | ~3.4M | 10.0 |
| **MobileNetV3-Small** ⭐ | **96.06** | **99.64** | **959K** | **4.6** |
| EfficientNetV2-M | 94.72 | 99.47 | ~54M | 205.8 |
| VGG16 | 86.42 | 97.36 | ~138M | 56.6 |
| VGG19 | 79.92 | 95.03 | ~144M | 76.8 |

> ⭐ **MobileNetV3-Small** selected for mobile deployment: best accuracy-to-size ratio with 11.7× fewer parameters than ResNet18.
>
> **Note:** The main benchmark compares 9 CNN architectures. The `alexnet_frozen_base/` and `resnet18_frozen_base/` folders are control experiments used to validate the effect of training configuration and weight initialisation.

---

## System Architecture 🔧 

### Two-Stage Mobile Inference Pipeline

```
Camera Frame
    │
    ▼
┌─────────────────────────────┐
│  STAGE 1: Hand Detection    │
│  MediaPipe 21-Point         │
│  Hand Landmark Detector     │
│  (1.98 MB TFLite)           │
│  → Rectangular ROI Crop     │
└─────────────┬───────────────┘
              │ 224×224 (bilinear resize)
              ▼
┌─────────────────────────────┐
│  STAGE 2: Classification    │
│  MobileNetV3-Small          │
│  (1.87 MB TFLite Float16)   │
│  GPU Delegated (217/217)    │
│  → Predicted BIM Sign       │
└─────────────────────────────┘
```

### Four-Stage Transfer Learning Pipeline

| Stage | Backbone | LR | Loss | Purpose |
|:---:|:---|:---:|:---|:---|
| 1 | Frozen | 1×10⁻³ | Categorical CE | Train classifier head only |
| 2 | Partial unfreeze | 5×10⁻⁵ | Categorical CE | Fine-tune last 50% of layers |
| 3 | Full unfreeze | 1×10⁻⁵ | Focal Loss (γ=2.0) | Focus on hard-to-classify signs |
| 3.1 | Full unfreeze | 1×10⁻⁶ | Focal Loss (γ=5.0) | Terminal learning rate squeeze |

---

## Project Structure 📁

```
bim-sign-language-recognition/
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Safe monorepo exclusions (includes Flutter build caches)
├── README.md                          # Comprehensive unified documentation
│
├── dataset/
│   └── README.md                     # Dataset characteristics & documentation
│
├── experimental_models/              # Unified comparative benchmarking (all 9 models)
│   ├── mobilenetv3_small/            # Proposed Deployment Model (⭐)
│   │   ├── train_stage1.py ... train_stage3_1.py
│   │   └── output/                   # CSVs, confusion matrices, reports for MNv3-Small
│   │
│   ├── alexnet/                      # AlexNet scratch-trained baseline
│   ├── alexnet_frozen_base/          # Transfer learning control baseline
│   ├── resnet18/                     # ResNet18 scratch-trained baseline
│   ├── resnet18_frozen_base/         # Transfer learning control baseline
│   ├── densenet201/                  # DenseNet201 comparative baseline
│   ├── mobilenetv2/                  # MobileNetV2 comparative baseline
│   ├── mobilenetv3_large/            # MobileNetV3-Large comparative baseline
│   ├── efficientnetv2_m/             # EfficientNetV2-M comparative baseline
│   ├── vgg16/                        # VGG16 comparative baseline
│   └── vgg19/                        # VGG19 comparative baseline
│
├── data_pipeline/                    # Dataset preparation scripts
│   ├── preprocess_roi.py             # Crop frames based on MediaPipe coordinates
│   ├── augment_offline.py            # Apply geometric/photometric augmentations
│   ├── split_dataset.py              # Divide images into train/val/test splits
│   └── inspect_dataset.py            # Log dataset composition and counts
│
├── evaluation/                       # Unified evaluation scripts
│   ├── evaluate_thesis.py            # Standard test accuracy calculator
│   ├── evaluate_detailed.py          # Generate detailed per-class report & CM
│   ├── check_model_params.py         # Analyse parameter count & layers
│   ├── final_evaluation_keras.py     # Main final Keras evaluation routine
│   ├── generate_confusion_matrix.py  # Create high-res confusion matrix heatmaps
│   └── get_report_metrics.py         # Automated metrics compiler for thesis report
│
├── visualisation/                    # Thesis figure generation scripts
│   ├── generate_thesis_figures.py    # Generates final thesis visual charts
│   └── visualize_training.py         # Plots training curves from CSV logs
│
├── conversion/                       # Keras → TFLite conversion
│   └── convert_flutter.py            # Fresh build (float32 clean TFLite exporter)
│
├── demo/                             # Inference demos for local testing
│   ├── classify_keras.py             # Single-image classifier script
│   ├── test_my_image.py              # Run inference on any custom image
│   └── webcam_demo_tflite.py         # Real-time TFLite webcam classifier
│
├── deployment/                       # Final deployable model artefacts
│   ├── README.md                     # Model specification metadata
│   ├── label_map.txt                 # 36-class sign label map
│   └── mobilenetv3_small_float16.tflite # Final compiled 1.87 MB Float16 model
│
├── thesis_outputs/                   # Final report evidence outputs
│   ├── report_metrics/               # Compiled metric TXT files
│   ├── classification_reports/       # Stage-specific classification reports
│   └── confusion_matrices/           # Heatmap PNGs of model predictions
│
├── benchmark_efficiency.py           # FLOPs, parameters, size benchmarking
├── tflite_optimize.py                # TFLite quantisation script (Float16/INT8)
│
└── flutter_app/                      # Flutter mobile app
    ├── lib/                          # App logic & MediaPipe interface
    ├── assets/                       # Embedded assets
    │   ├── model/                    # Hand landmark + MobileNetV3-Small models
    │   └── labels/                   # label_map.txt alphanumeric signs
    ├── pubspec.yaml                  # Flutter package manager configuration
    └── README.md                     # Mobile app quick start instructions
```

---

## Dataset

- **Language**: Bahasa Isyarat Malaysia (BIM)
- **Classes**: 36 — digits 0–9, letters A–Z
- **Total Images**: 36,000 (1,000 per class)
- **Split**: 28,800 train / 3,600 val / 3,600 test (800/100/100 per class)
- **Input Size**: 224 × 224 × 3 (RGB)
- **Source**: Consolidated dataset prepared for this study using selected BIM-compatible static fingerspelling images and controlled augmentation.

---

## 🛡️ Reproducibility Notice

This repository is designed to support academic validation and partial reproducibility of the project results.

**Included in this repository:**
- Complete training scripts for all evaluated architectures
- Evaluation, metrics gathering, and graph plotting scripts
- Final classification reports, confusion matrices, and compiled metrics under `thesis_outputs/`
- The final optimised TensorFlow Lite deployment model under `deployment/`
- Flutter mobile prototype code, configured with the embedded deployment assets

**Not included in this repository:**
- Full raw dataset and augmented image database (due to storage and licensing limitations)
- Large Keras model checkpoint weights (`.keras` / `.h5`) exceeding 100 MB

---

## Model Weights & Releases

The final deployed TensorFlow Lite model is included directly in this repository:
```
deployment/mobilenetv3_small_float16.tflite
```
A copy of this file is also placed under `flutter_app/assets/model/` so that the mobile prototype compiles and runs immediately out-of-the-box. 

Large trained `.keras` model checkpoints for the comparative architectures are excluded from repository commits to avoid excessive Git size. If required, trained checkpoints can be provided upon request or distributed through GitHub Releases.

---

## 🚀 Quick Start

> ⚠️ **Note:** Full model training requires the preprocessed dataset, which is not included in this repository due to size constraints. The included scripts document the pipeline, while final evaluation outputs are provided directly inside the `thesis_outputs/` directory.

### 1. Environment Setup
Install the Python dependencies in a virtual environment:
```bash
pip install -r requirements.txt
```

### 2. Run the Flutter Mobile Prototype
To build and run the real-time sign recognition application on a connected Android device:
```bash
cd flutter_app
flutter pub get
flutter run
```

### 3. Run Python Demo Using the TFLite Model

You can test the optimised model using a local computer webcam:

```bash
python demo/webcam_demo_tflite.py
```

### 4. Retraining Pipeline (If Dataset is Provided)
If the dataset directories are added under `train_data/`, `val_data/`, and `test_data/`, you can retrain the proposed MobileNetV3-Small model sequentially:
```bash
cd experimental_models/mobilenetv3_small
python train_stage1.py
python train_stage2.py
python train_stage3.py
python train_stage3_1.py
```

---

## Flutter Mobile Prototype📱 

The Flutter Android proof-of-concept application is included under:

`flutter_app/`

It integrates the final MobileNetV3-Small Float16 TFLite model for capture-based BIM fingerspelling recognition.

### Run the Flutter app

```bash
cd flutter_app
flutter pub get
flutter run
```

The deployed model is stored in:
```
flutter_app/assets/model/mobilenetv3_small_float16.tflite
```

---

