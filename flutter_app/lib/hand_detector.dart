import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'dart:math';

/// Simple hand detector using color-based approach
/// Detects hand region by finding skin tones and largest contiguous region
class HandDetector {
  bool _isLoaded = true; // No model to load, always ready
  
  bool get isLoaded => _isLoaded;
  
  Future<void> load() async {
    // No model to load for color-based detection
    if (kDebugMode) debugPrint('Hand detector ready (color-based)');
    _isLoaded = true;
  }
  
  /// Detect hand bounding box using skin color detection
  /// Returns [left, top, right, bottom] in normalized coords (0.0 to 1.0)
  /// Returns null if no hand detected
  Future<List<double>?> detectHand(img.Image image) async {
    try {
      // Downsample for faster processing
      final small = img.copyResize(image, width: 160);
      final width = small.width;
      final height = small.height;
      
      // Find all pixels that match skin tone
      List<Point<int>> skinPixels = [];
      
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          final pixel = small.getPixel(x, y);
          if (_isSkinTone(pixel.r.toInt(), pixel.g.toInt(), pixel.b.toInt())) {
            skinPixels.add(Point(x, y));
          }
        }
      }
      
      if (skinPixels.isEmpty) {
        return null; // No skin detected
      }
      
      // Find bounding box of all skin pixels
      int minX = width;
      int minY = height;
      int maxX = 0;
      int maxY = 0;
      
      for (final point in skinPixels) {
        if (point.x < minX) minX = point.x;
        if (point.y < minY) minY = point.y;
        if (point.x > maxX) maxX = point.x;
        if (point.y > maxY) maxY = point.y;
      }
      
      // Check if detected region is reasonable size
      final boxWidth = maxX - minX;
      final boxHeight = maxY - minY;
      final boxArea = boxWidth * boxHeight;
      final imageArea = width * height;
      
      // Hand should be at least 5% and at most 80% of image
      if (boxArea < imageArea * 0.05 || boxArea > imageArea * 0.80) {
        return null;
      }
      
      // Return normalized bounding box
      return [
        minX / width,
        minY / height,
        maxX / width,
        maxY / height,
      ];
      
    } catch (e) {
      debugPrint('Hand detection error: $e');
      return null;
    }
  }
  
  /// Check if RGB values match skin tone
  /// Uses multiple color space checks for robustness
  bool _isSkinTone(int r, int g, int b) {
    // Rule 1: Simple RGB rule
    bool rgbRule = (r > 95 && g > 40 && b > 20) &&
                   (max(max(r, g), b) - min(min(r, g), b) > 15) &&
                   (r > g && r > b);
    
    if (!rgbRule) return false;
    
    // Rule 2: YCbCr color space (better for skin detection)
    double cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b;
    double cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b;
    
    bool ycbcrRule = (cb >= 77 && cb <= 127) && (cr >= 133 && cr <= 173);
    
    return ycbcrRule;
  }
  
  /// Crop image to hand region with padding
  /// bbox: [left, top, right, bottom] in normalized coords (0.0 to 1.0)
  /// padding: percentage to add around box (0.2 = 20%)
  img.Image cropToHand(img.Image image, List<double> bbox, {double padding = 0.2}) {
    final width = image.width;
    final height = image.height;
    
    // Convert normalized coords to pixels
    double left = bbox[0] * width;
    double top = bbox[1] * height;
    double right = bbox[2] * width;
    double bottom = bbox[3] * height;
    
    // Calculate box dimensions
    double boxWidth = right - left;
    double boxHeight = bottom - top;
    
    // Add padding
    left = max(0.0, left - boxWidth * padding);
    top = max(0.0, top - boxHeight * padding);
    right = min(width.toDouble(), right + boxWidth * padding);
    bottom = min(height.toDouble(), bottom + boxHeight * padding);
    
    // Make it square (use larger dimension)
    double size = max(right - left, bottom - top);
    double centerX = (left + right) / 2;
    double centerY = (top + bottom) / 2;
    
    left = max(0.0, centerX - size / 2);
    top = max(0.0, centerY - size / 2);
    right = min(width.toDouble(), centerX + size / 2);
    bottom = min(height.toDouble(), centerY + size / 2);
    
    // Crop
    return img.copyCrop(
      image,
      x: left.toInt(),
      y: top.toInt(),
      width: (right - left).toInt(),
      height: (bottom - top).toInt(),
    );
  }
  
  void dispose() {
    // Nothing to dispose
  }
}
