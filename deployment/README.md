# 🚀 Deployment Models

This directory contains the final optimised deep learning model checkpoints compiled for mobile and edge deployment.

## 📂 Model Deliverables

### Proposed Model: MobileNetV3-Small (Float16 Quantised)
- **Filename**: `mobilenetv3_small_float16.tflite`
- **Model Size**: **1.87 MB** (1,960,908 bytes)
- **Top-1 Accuracy**: **96.06%**
- **GPU Compatibility**: Fully delegated (217 / 217 operations compatible with Android NNAPI / GPU delegates)
- **Input Dimensions**: `1 × 224 × 224 × 3` (RGB Float32 raw pixel range [0.0, 255.0]). *Note: The model incorporates an internal Keras Rescaling(1./255) layer, so input values should remain in [0.0, 255.0] without manual normalisation.*
- **Output Classes**: `36` (Bahasa Isyarat Malaysia alphanumeric digits 0-9 and letters A-Z)

---

## 🛠️ Verification & Pipeline Integration

The model resides in two locations within this unified workspace:
1. `deployment/mobilenetv3_small_float16.tflite` - Central release checkpoint for benchmarks.
2. `flutter_app/assets/model/mobilenetv3_small_float16.tflite` - Loaded directly by the mobile app's inference handler.

### Model Conversion Pipeline
To regenerate this optimised asset from your Keras checkpoint, run:
```bash
python tflite_optimize.py
```
This will run full integer (INT8) and float16 post-training quantisation, writing outputs to `tflite_models/`.
