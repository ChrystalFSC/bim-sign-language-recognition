# BIM Sign Language App — Real-Time Fingerspelling Recognition

A Flutter Android application for real-time Bahasa Isyarat Malaysia (BIM) fingerspelling recognition, powered by a two-stage inference pipeline using Google MediaPipe hand landmark tracking and a MobileNetV3-Small TFLite classifier.

**Final Year Project (FYP2)** — Computer Science, 2025/2026

---

## 📱 Features

- **21-Point Hand Landmark Tracking** — Google MediaPipe detects and localises the hand region in real-time
- **Two-Stage Inference Pipeline** — Landmark-guided rectangular crop → MobileNetV3-Small classification
- **Three Recognition Modes** — Digits only (0–9), Letters only (A–Z), or All 36 classes
- **GPU Acceleration** — Full TFLite GPU delegation (217/217 ops) on Android
- **Auto-Detection Mode** — Continuous real-time sign recognition
- **Front/Back Camera Toggle** — Works with both cameras
- **Top-3 Candidate Display** — Shows ranked predictions with confidence scores

---

## 🏆 Performance

| Metric | Value |
|:---|:---|
| Classification Accuracy (Top-1) | **96.06%** |
| Classification Accuracy (Top-5) | **99.64%** |
| Model Size (TFLite Float16) | **1.87 MB** |
| Hand Landmark Model Size | 1.98 MB |
| Average Inference Latency | **174 ms** |
| GPU Delegation | 217 / 217 ops |
| Test Device | Xiaomi MIX 2S (Snapdragon 845) |

---

## 🔧 Architecture

### Two-Stage Inference Pipeline

```
Camera Frame (JPEG)
        │
        ▼
┌─────────────────────────────────┐
│  STAGE 1: MediaPipe Detection   │
│  hand_landmark_lite.tflite      │
│  21-point hand keypoint detect  │
│  → Normalize by 224.0           │
│  → Rectangular bounding box     │
│  → 15% padded crop              │
└──────────────┬──────────────────┘
               │ Bilinear resize → 224×224
               ▼
┌─────────────────────────────────┐
│  STAGE 2: Classification        │
│  model_stage3_1_float16.tflite  │
│  MobileNetV3-Small              │
│  Input: [0, 255] Float32        │
│  GPU Delegated (Full)           │
│  → Predicted BIM Sign + Conf    │
└─────────────────────────────────┘
```

### Mode Colour Theme

| Mode | Colour | Hex |
|:---|:---|:---|
| Numbers (0–9) | Dark Blue | `#004B87` |
| Letters (A–Z) | Yellow | `#FFD100` |
| All Classes | Teal | `#007E9A` |

---

## 📁 Project Structure

```
bim_sign_app/
├── lib/
│   ├── main.dart                   # App entry point, camera pipeline & UI
│   ├── sign_classifier.dart        # TFLite classification wrapper
│   ├── mediapipe_hand_detector.dart# MediaPipe 21-point landmark detector
│   └── hand_landmark_painter.dart  # Canvas overlay for skeleton visualisation
├── assets/
│   ├── model_stage3_1_float16.tflite  # Final MobileNetV3-Small classifier
│   ├── hand_landmark_lite.tflite      # MediaPipe hand landmark detector
│   └── classes.txt                    # 36-class label file (0–9, A–Z)
├── android/
│   └── app/build.gradle.kts        # Android build configuration
├── pubspec.yaml                    # Flutter dependencies
└── README.md
```

---

## 🚀 Setup & Running

### Prerequisites
- Flutter SDK ≥ 3.0
- Android device with Android 8.0+ (API 26+)
- USB debugging enabled on device

### 1. Install Flutter dependencies

```bash
cd bim_sign_app
flutter pub get
```

### 2. Connect Android device and run

```bash
flutter run
```

### 3. Build release APK (optional)

```bash
flutter build apk --release
```

The APK will be at `build/app/outputs/flutter-apk/app-release.apk`.

---

## 📦 Key Dependencies

| Package | Purpose |
|:---|:---|
| `camera` | CameraX-based Android camera preview |
| `tflite_flutter` | TFLite runtime with GPU delegation |
| `image` | CPU-side image decoding & bilinear resize |
| `permission_handler` | Runtime camera permission |

See `pubspec.yaml` for exact versions.

---

## ⚙️ Technical Notes

- **Input Range**: Model expects pixel values in **[0, 255]** (not normalized). The model contains an internal Keras `Rescaling(1./255)` layer.
- **Landmark Normalization**: Raw MediaPipe landmark coordinates are in `[0, 224]` (landmark model input size). Must be divided by `224.0` before clamping to `[0, 1]`.
- **Aspect Ratio**: Crops are rectangular (not forced square) to match the trained data geometry, then bilinearly stretched to 224×224.
- **INT8 Quantization**: Not supported — MobileNetV3's hard-swish activations cause accuracy collapse to ~29.97% under INT8. Float16 is used instead.

---

## 🔗 Related Repository

- **[bim-sign-language-recognition](../bim-sign-language-recognition)** — Training pipeline, 9-model benchmarking, evaluation scripts, and TFLite conversion for the deep learning models used in this app.
