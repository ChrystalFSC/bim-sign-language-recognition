import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'dart:typed_data';
import 'dart:math' as math;
import 'package:flutter/material.dart';

/// MediaPipe Hand Detector with 21-point landmarks
/// Two-stage pipeline: Palm Detection → Hand Landmark Detection
class MediaPipeHandDetector {
  Interpreter? _palmDetector;
  Interpreter? _landmarkDetector;
  bool _isLoaded = false;
  
  // Palm detection model input
  static const int PALM_INPUT_SIZE = 192;
  
  // Hand landmark model input  
  static const int LANDMARK_INPUT_SIZE = 224;
  static const int NUM_LANDMARKS = 21;
  
  bool get isLoaded => _isLoaded;
  
  /// Load both MediaPipe models
  Future<void> load() async {
    try {
      print('Loading MediaPipe hand detection models...');
      
      // Load hand landmark model (we'll use ROI-guided approach)
      final landmarkOptions = InterpreterOptions();
      landmarkOptions.threads = 2;
      
      _landmarkDetector = await Interpreter.fromAsset(
        'assets/model/hand_landmark_lite.tflite',
        options: landmarkOptions,
      );
      
      _isLoaded = true;
      print('✓ MediaPipe hand landmark model loaded');
      print('  Input: ${_landmarkDetector!.getInputTensors()}');
      print('  Output: ${_landmarkDetector!.getOutputTensors()}');
      
    } catch (e) {
      print('Error loading MediaPipe models: $e');
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
        width: LANDMARK_INPUT_SIZE,
        height: LANDMARK_INPUT_SIZE,
        interpolation: img.Interpolation.linear,
      );
      
      // Normalize to [0, 1]
      final inputData = Float32List(LANDMARK_INPUT_SIZE * LANDMARK_INPUT_SIZE * 3);
      int idx = 0;
      for (int y = 0; y < LANDMARK_INPUT_SIZE; y++) {
        for (int x = 0; x < LANDMARK_INPUT_SIZE; x++) {
          final pixel = resized.getPixel(x, y);
          final r = pixel.r.toDouble();
          final g = pixel.g.toDouble();
          final b = pixel.b.toDouble();
          inputData[idx++] = (r > 1.0 ? r : r * 255.0) / 255.0;
          inputData[idx++] = (g > 1.0 ? g : g * 255.0) / 255.0;
          inputData[idx++] = (b > 1.0 ? b : b * 255.0) / 255.0;
        }
      }
      
      final input = inputData.reshape([1, LANDMARK_INPUT_SIZE, LANDMARK_INPUT_SIZE, 3]);
      
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
      if (presenceScore < 0.5) {
        return null;
      }
      
      // Parse 21 landmarks (normalized 0-1 coordinates)
      final landmarks = <Offset>[];
      double minX = 1.0, minY = 1.0, maxX = 0.0, maxY = 0.0;
      
      for (int i = 0; i < NUM_LANDMARKS; i++) {
        final x = (landmarkOutput[0][i * 3] / LANDMARK_INPUT_SIZE).clamp(0.0, 1.0);
        final y = (landmarkOutput[0][i * 3 + 1] / LANDMARK_INPUT_SIZE).clamp(0.0, 1.0);
        
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
        boundingBox: bbox,
        confidence: presenceScore,
        imageWidth: handRegionImage.width,
        imageHeight: handRegionImage.height,
      );
      
    } catch (e) {
      print('MediaPipe detection error: $e');
      return null;
    }
  }
  
  /// Crop image tightly around detected hand landmarks
  /// Returns a rectangular crop of the hand region
  img.Image cropToHandLandmarks(img.Image image, List<double> bbox) {
    final width = image.width;
    final height = image.height;
    
    // Convert normalized bbox to pixel coordinates
    double left = bbox[0] * width;
    double top = bbox[1] * height;
    double right = bbox[2] * width;
    double bottom = bbox[3] * height;
    
    print('DEBUG: cropToHandLandmarks - image size: ${width}x${height}, bbox: [${bbox[0].toStringAsFixed(3)}, ${bbox[1].toStringAsFixed(3)}, ${bbox[2].toStringAsFixed(3)}, ${bbox[3].toStringAsFixed(3)}], pixel coordinates: left=${left.toStringAsFixed(1)}, top=${top.toStringAsFixed(1)}, right=${right.toStringAsFixed(1)}, bottom=${bottom.toStringAsFixed(1)}');
    
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
  final List<double> boundingBox;  // [left, top, right, bottom] normalized 0-1
  final double confidence;          // Hand presence confidence
  final int imageWidth;
  final int imageHeight;
  
  HandDetectionResult({
    required this.landmarks,
    required this.boundingBox,
    required this.confidence,
    required this.imageWidth,
    required this.imageHeight,
  });
}
