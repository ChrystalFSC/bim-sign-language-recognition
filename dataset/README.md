# 📊 Bahasa Isyarat Malaysia (BIM) Dataset

This directory contains the documentation and characteristics of the fingerspelling dataset constructed for this study. The actual image database is excluded from version control due to storage size and licensing constraints.

---

## 🔍 Dataset Characteristics

- **Language**: Bahasa Isyarat Malaysia (BIM)
- **Classes**: 36 — Digits 0–9, Letters A–Z
- **Total Unified Images**: 36,000 (1,000 samples per class)
- **Split Ratio**: 80% Training / 10% Validation / 10% Testing
  - **Training Set**: 28,800 images (800 per class)
  - **Validation Set**: 3,600 images (100 per class)
  - **Testing Set**: 3,600 images (100 per class)
- **Input Geometry**: 224 × 224 pixels (RGB, 3 channels)

---

## 🛠️ Data Acquisition & Processing Pipeline

1. **Source Compilation**: Consolidated BIM-compatible static fingerspelling images compiled from available open-source databases and supplementary project-specific captures.
2. **Region of Interest (ROI) Extraction**:
   - MediaPipe Hand Landmark Tracking was used to identify hand coordinates.
   - Images were cropped with a 15% bounding-box padding ratio to retain structural hand geometry.
3. **Normalisation**: Cropped regions resized to a uniform `224×224×3` resolution.
4. **Offline Augmentation**: Geometric and photometric augmentations applied strictly to the training split to enforce environmental robustness (lighting, contrast, rotations, translation shifts).
