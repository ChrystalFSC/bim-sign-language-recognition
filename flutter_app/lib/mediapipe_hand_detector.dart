import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'dart:math' as math;
import 'package:flutter/material.dart';

/// MediaPipe Hand Detector with 21-point landmarks
/// Two-stage pipeline: Palm Detection → Hand Landmark Detection
class MediaPipeHandDetector {
  Interpreter? _palmDetector;
  Interpreter? _landmarkDetector;
  bool _isLoaded = false;
  
  // Hand landmark model input  
  static const int landmarkInputSize = 224;
  static const int numLandmarks = 21;
  static const double minPresenceConfidence = 0.35;
  
  bool get isLoaded => _isLoaded;
  
  /// Load both MediaPipe models
  Future<void> load() async {
    try {
      if (kDebugMode) debugPrint('Loading MediaPipe hand detection models...');
      
      // Load hand landmark model (we'll use ROI-guided approach)
      final landmarkOptions = InterpreterOptions();
      landmarkOptions.threads = 2;
      
      _landmarkDetector = await Interpreter.fromAsset(
        'assets/hand_landmark_lite.tflite',
        options: landmarkOptions,
      );
      
      _isLoaded = true;
      if (kDebugMode) {
        debugPrint('MediaPipe hand landmark model loaded');
        debugPrint('  Input: ${_landmarkDetector!.getInputTensors()}');
        debugPrint('  Output: ${_landmarkDetector!.getOutputTensors()}');
      }
      
    } catch (e) {
      debugPrint('Error loading MediaPipe models: $e');
      _isLoaded = false;
    }
  }
  
  /// Detect 21 hand landmarks from a cropped hand region
  /// The input image should be roughly cropped to the hand area (e.g., from ROI box)
  /// Returns HandDetectionResult with landmarks and bounding box, or null if no hand
  Future<HandDetectionResult?> detect(img.Image handRegionImage) async {
    if (!_isLoaded || _landmarkDetector == null) return null;
    
    try {
      // Resize to model input
      final resized = img.copyResize(
        handRegionImage,
        width: landmarkInputSize,
        height: landmarkInputSize,
        interpolation: img.Interpolation.linear,
      );
      
      // Normalize to [0, 1]
      final inputData = Float32List(landmarkInputSize * landmarkInputSize * 3);
      int idx = 0;
      for (int y = 0; y < landmarkInputSize; y++) {
        for (int x = 0; x < landmarkInputSize; x++) {
          final pixel = resized.getPixel(x, y);
          final r = pixel.r.toDouble();
          final g = pixel.g.toDouble();
          final b = pixel.b.toDouble();
          inputData[idx++] = r / 255.0;
          inputData[idx++] = g / 255.0;
          inputData[idx++] = b / 255.0;
        }
      }
      
      final input = inputData.reshape([1, landmarkInputSize, landmarkInputSize, 3]);
      
      // MediaPipe hand landmark model outputs:
      // Output 0: landmarks (1, 63) - 21 landmarks × 3 (x, y, z)
      // Output 1: hand presence score (1, 1)
      // Output 2: handedness (1, 1)  
      // Output 3: world landmarks (1, 63)
      final landmarkOutput = List.generate(1, (_) => List.filled(63, 0.0));
      final presenceOutput = List.generate(1, (_) => List.filled(1, 0.0));
      final handednessOutput = List.generate(1, (_) => List.filled(1, 0.0));
      final worldOutput = List.generate(1, (_) => List.filled(63, 0.0));
      
      final outputs = {
        0: landmarkOutput,
        1: presenceOutput,
        2: handednessOutput,
        3: worldOutput,
      };
      
      _landmarkDetector!.runForMultipleInputs([input], outputs);
      
      // Check hand presence
      final presenceScore = presenceOutput[0][0];
      if (presenceScore < minPresenceConfidence) {
        return null;
      }
      
      // Parse 21 landmarks (normalized 0-1 coordinates)
      final landmarks = <Offset>[];
      final rawLandmarks = List<double>.filled(numLandmarks * 3, 0.0);
      double minX = 1.0, minY = 1.0, maxX = 0.0, maxY = 0.0;
      
      for (int i = 0; i < numLandmarks; i++) {
        final x = (landmarkOutput[0][i * 3] / landmarkInputSize).clamp(0.0, 1.0);
        final y = (landmarkOutput[0][i * 3 + 1] / landmarkInputSize).clamp(0.0, 1.0);
        final z = landmarkOutput[0][i * 3 + 2] / landmarkInputSize;
        rawLandmarks[i * 3] = x;
        rawLandmarks[i * 3 + 1] = y;
        rawLandmarks[i * 3 + 2] = z;
        
        landmarks.add(Offset(
          x * handRegionImage.width,
          y * handRegionImage.height,
        ));
        
        minX = math.min(minX, x);
        minY = math.min(minY, y);
        maxX = math.max(maxX, x);
        maxY = math.max(maxY, y);
      }
      
      // Calculate tight bounding box from landmarks with padding
      const padding = 0.15;
      final boxWidth = maxX - minX;
      final boxHeight = maxY - minY;
      
      final bbox = [
        math.max(0.0, minX - boxWidth * padding),
        math.max(0.0, minY - boxHeight * padding),
        math.min(1.0, maxX + boxWidth * padding),
        math.min(1.0, maxY + boxHeight * padding),
      ];
      
      return HandDetectionResult(
        landmarks: landmarks,
        features: _normalizeLandmarks(rawLandmarks),
        boundingBox: bbox,
        confidence: presenceScore,
        imageWidth: handRegionImage.width,
        imageHeight: handRegionImage.height,
      );
      
    } catch (e) {
      debugPrint('MediaPipe detection error: $e');
      return null;
    }
  }

  List<double> _normalizeLandmarks(List<double> rawLandmarks) {
    if (rawLandmarks.length != numLandmarks * 3) {
      return List<double>.filled(numLandmarks * 3, 0.0);
    }

    final wristX = rawLandmarks[0];
    final wristY = rawLandmarks[1];
    final wristZ = rawLandmarks[2];
    double scale = 0.0;

    for (int i = 0; i < numLandmarks; i++) {
      final dx = rawLandmarks[i * 3] - wristX;
      final dy = rawLandmarks[i * 3 + 1] - wristY;
      scale = math.max(scale, math.sqrt(dx * dx + dy * dy));
    }
    if (scale < 1e-6) scale = 1.0;

    final features = List<double>.filled(numLandmarks * 3, 0.0);
    for (int i = 0; i < numLandmarks; i++) {
      features[i * 3] = (rawLandmarks[i * 3] - wristX) / scale;
      features[i * 3 + 1] = (rawLandmarks[i * 3 + 1] - wristY) / scale;
      features[i * 3 + 2] = (rawLandmarks[i * 3 + 2] - wristZ) / scale;
    }
    return features;
  }
  
  /// Crop around detected hand landmarks while keeping enough context for the classifier.
  img.Image cropToHandLandmarks(
    img.Image image,
    List<double> bbox, {
    double padding = 0.35,
  }) {
    final width = image.width;
    final height = image.height;
    
    // Convert normalized bbox to pixel coordinates
    double left = bbox[0] * width;
    double top = bbox[1] * height;
    double right = bbox[2] * width;
    double bottom = bbox[3] * height;
    final boxW = right - left;
    final boxH = bottom - top;
    final minCropSize = math.min(width, height) * 0.45;

    left = math.max(0.0, left - boxW * padding);
    top = math.max(0.0, top - boxH * padding);
    right = math.min(width.toDouble(), right + boxW * padding);
    bottom = math.min(height.toDouble(), bottom + boxH * padding);

    if (right - left < minCropSize) {
      final centerX = (left + right) / 2;
      left = math.max(0.0, centerX - minCropSize / 2);
      right = math.min(width.toDouble(), centerX + minCropSize / 2);
    }

    if (bottom - top < minCropSize) {
      final centerY = (top + bottom) / 2;
      top = math.max(0.0, centerY - minCropSize / 2);
      bottom = math.min(height.toDouble(), centerY + minCropSize / 2);
    }
    
    // Ensure valid dimensions
    final cropW = (right - left).toInt().clamp(10, width);
    final cropH = (bottom - top).toInt().clamp(10, height);
    
    return img.copyCrop(
      image,
      x: left.toInt().clamp(0, width - cropW),
      y: top.toInt().clamp(0, height - cropH),
      width: cropW,
      height: cropH,
    );
  }
  
  void dispose() {
    _palmDetector?.close();
    _landmarkDetector?.close();
    _isLoaded = false;
  }
}

/// Result from MediaPipe hand detection
class HandDetectionResult {
  final List<Offset> landmarks;    // 21 landmark points in pixel coords
  final List<double> features;      // 63 normalized x/y/z features for classification
  final List<double> boundingBox;  // [left, top, right, bottom] normalized 0-1
  final double confidence;          // Hand presence confidence
  final int imageWidth;
  final int imageHeight;
  
  HandDetectionResult({
    required this.landmarks,
    required this.features,
    required this.boundingBox,
    required this.confidence,
    required this.imageWidth,
    required this.imageHeight,
  });
}
