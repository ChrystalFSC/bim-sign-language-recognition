import 'package:flutter/material.dart';

/// MediaPipe 21-point hand landmark painter
/// Draws the detected hand skeleton with landmark connections
class HandLandmarkPainter extends CustomPainter {
  final List<Offset>? landmarks;   // 21 landmark points in image coords
  final List<double>? boundingBox; // [left, top, right, bottom] normalized
  final Size imageSize;
  final double confidence;
  
  HandLandmarkPainter({
    this.landmarks,
    this.boundingBox,
    required this.imageSize,
    this.confidence = 0.0,
  });
  
  // MediaPipe hand landmark connections (finger bones)
  static const List<List<int>> connections = [
    // Thumb
    [0, 1], [1, 2], [2, 3], [3, 4],
    // Index finger
    [0, 5], [5, 6], [6, 7], [7, 8],
    // Middle finger
    [5, 9], [9, 10], [10, 11], [11, 12],
    // Ring finger
    [9, 13], [13, 14], [14, 15], [15, 16],
    // Pinky
    [13, 17], [17, 18], [18, 19], [19, 20],
    // Palm base
    [0, 17],
  ];
  
  @override
  void paint(Canvas canvas, Size size) {
    if (landmarks == null || landmarks!.length < 21) return;
    
    // Scale from image coordinates to canvas coordinates
    final scaleX = size.width / imageSize.width;
    final scaleY = size.height / imageSize.height;
    
    // Scale landmarks to canvas
    final scaledLandmarks = landmarks!.map((lm) => 
      Offset(lm.dx * scaleX, lm.dy * scaleY)
    ).toList();
    
    // Draw bounding box
    if (boundingBox != null) {
      final topLeft = Offset(
        boundingBox![0] * size.width,
        boundingBox![1] * size.height,
      );
      final bottomRight = Offset(
        boundingBox![2] * size.width,
        boundingBox![3] * size.height,
      );
      
      final rect = Rect.fromPoints(topLeft, bottomRight);
      
      // Semi-transparent fill
      final fillPaint = Paint()
        ..color = Colors.green.withOpacity(0.1)
        ..style = PaintingStyle.fill;
      canvas.drawRRect(
        RRect.fromRectAndRadius(rect, const Radius.circular(8)),
        fillPaint,
      );
      
      // Border
      final borderPaint = Paint()
        ..color = Colors.greenAccent.withOpacity(0.7)
        ..strokeWidth = 2.0
        ..style = PaintingStyle.stroke;
      canvas.drawRRect(
        RRect.fromRectAndRadius(rect, const Radius.circular(8)),
        borderPaint,
      );
      
    }
    
    // Draw connections (skeleton lines)
    final linePaint = Paint()
      ..color = Colors.greenAccent.withOpacity(0.8)
      ..strokeWidth = 2.0
      ..strokeCap = StrokeCap.round;
    
    for (final conn in connections) {
      if (conn[0] < scaledLandmarks.length && conn[1] < scaledLandmarks.length) {
        canvas.drawLine(scaledLandmarks[conn[0]], scaledLandmarks[conn[1]], linePaint);
      }
    }
    
    // Draw landmark points
    for (int i = 0; i < scaledLandmarks.length; i++) {
      final point = scaledLandmarks[i];
      
      // Fingertips (4, 8, 12, 16, 20) are larger and highlighted
      final isFingertip = [4, 8, 12, 16, 20].contains(i);
      final isWrist = i == 0;
      
      // Outer glow
      final glowPaint = Paint()
        ..color = (isFingertip ? Colors.cyanAccent : Colors.greenAccent).withOpacity(0.3)
        ..style = PaintingStyle.fill;
      canvas.drawCircle(point, isFingertip ? 8 : 5, glowPaint);
      
      // Inner dot
      final dotPaint = Paint()
        ..color = isFingertip ? Colors.cyanAccent : (isWrist ? Colors.yellowAccent : Colors.greenAccent)
        ..style = PaintingStyle.fill;
      canvas.drawCircle(point, isFingertip ? 5 : 3, dotPaint);
      
      // White center
      final centerPaint = Paint()
        ..color = Colors.white
        ..style = PaintingStyle.fill;
      canvas.drawCircle(point, isFingertip ? 2 : 1.5, centerPaint);
    }
  }
  
  @override
  bool shouldRepaint(HandLandmarkPainter oldDelegate) {
    return landmarks != oldDelegate.landmarks || 
           boundingBox != oldDelegate.boundingBox ||
           imageSize != oldDelegate.imageSize;
  }
}
