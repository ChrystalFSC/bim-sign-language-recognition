# Landmark Classifier Workflow

This adds a second classifier path that can learn from the 21 hand landmarks directly.

## 1. Collect Landmark CSV On Phone

1. Install and open the app.
2. Tap the dataset icon in the app bar.
3. Select the current sign.
4. Hold the hand clearly inside the camera view.
5. Tap `SAVE LANDMARKS`.
6. Collect roughly 50-100 samples per class, more for confusing classes.

The phone saves:

```text
/sdcard/Android/data/com.example.bim_sign_app/files/BIMLandmarkData/landmarks.csv
```

## 2. Pull CSV To The Project

```powershell
adb pull /sdcard/Android/data/com.example.bim_sign_app/files/BIMLandmarkData/landmarks.csv D:\Private\bim_sign_app\data\landmarks.csv
```

## 3. Train Landmark Model

Create a Python environment with TensorFlow installed, then run:

```powershell
python scripts\train_landmark_classifier.py --csv data\landmarks.csv --output assets\landmark_classifier.tflite
```

The script trains a small dense neural network:

```text
63 landmark features -> Dense 128 -> Dense 64 -> 36 classes
```

## 4. App Integration

After `assets/landmark_classifier.tflite` exists, add it to `pubspec.yaml`, then add a `LandmarkClassifier` in Flutter and combine it with the image classifier:

```text
if image result and landmark result agree:
  trust prediction
if they disagree:
  show uncertain or use the higher-confidence result
```

This hybrid approach is more credible than image-only prediction because it uses both visual appearance and structured hand geometry.
