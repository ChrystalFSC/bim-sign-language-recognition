import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:io';

/// Data Collection Mode for capturing phone camera training data
class CollectionMode extends StatefulWidget {
  final CameraController cameraController;
  
  const CollectionMode({super.key, required this.cameraController});
  
  @override
  State<CollectionMode> createState() => _CollectionModeState();
}

class _CollectionModeState extends State<CollectionMode> {
  String _currentSign = '0';
  int _imageCount = 0;
  final int _targetPerSign = 20;
  String _statusMessage = '';
  
  // All classes in order
  final List<String> _allClasses = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
  ];
  
  Future<void> _captureImage() async {
    try {
      // Get external storage directory
      final Directory? extDir = await getExternalStorageDirectory();
      if (extDir == null) {
        setState(() => _statusMessage = 'Cannot access storage');
        return;
      }
      
      // Create SignData directory structure
      final String basePath = '/storage/emulated/0/SignData';
      final Directory signDir = Directory('$basePath/$_currentSign');
      
      if (!await signDir.exists()) {
        await signDir.create(recursive: true);
      }
      
      // Generate filename
      final String timestamp = DateTime.now().millisecondsSinceEpoch.toString();
      final String filename = 'cam_$timestamp.jpg';
      final String fullPath = '${signDir.path}/$filename';
      
      // Capture image
      final XFile image = await widget.cameraController.takePicture();
      await File(image.path).copy(fullPath);
      
      setState(() {
        _imageCount++;
        _statusMessage = 'Captured $_imageCount/$_targetPerSign for $_currentSign';
      });
      
    } catch (e) {
      setState(() => _statusMessage = 'Error: $e');
    }
  }
  
  void _nextSign() {
    final currentIndex = _allClasses.indexOf(_currentSign);
    if (currentIndex < _allClasses.length - 1) {
      setState(() {
        _currentSign = _allClasses[currentIndex + 1];
        _imageCount = 0;
        _statusMessage = 'Ready to collect $_currentSign';
      });
    } else {
      setState(() => _statusMessage = 'All signs collected!');
    }
  }
  
  void _previousSign() {
    final currentIndex = _allClasses.indexOf(_currentSign);
    if (currentIndex > 0) {
      setState(() {
        _currentSign = _allClasses[currentIndex - 1];
        _imageCount = 0;
        _statusMessage = 'Ready to collect $_currentSign';
      });
    }
  }
  
  @override
  Widget build(BuildContext context) {
    final progress = _imageCount / _targetPerSign;
    
    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: Column(
          children: [
            // Header
            Container(
              padding: const EdgeInsets.all(16),
              color: Colors.orange.shade700,
              child: Column(
                children: [
                  const Text(
                    'DATA COLLECTION MODE',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Saving to: /storage/emulated/0/SignData',
                    style: TextStyle(color: Colors.white70, fontSize: 11),
                  ),
                ],
              ),
            ),
            
            // Current Sign Info
            Container(
              padding: const EdgeInsets.all(20),
              color: Colors.grey.shade900,
              child: Column(
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      IconButton(
                        onPressed: _previousSign,
                        icon: const Icon(Icons.arrow_back, color: Colors.white),
                      ),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 30, vertical: 10),
                        decoration: BoxDecoration(
                          color: Colors.orange,
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Text(
                          _currentSign,
                          style: const TextStyle(
                            fontSize: 48,
                            fontWeight: FontWeight.bold,
                            color: Colors.white,
                          ),
                        ),
                      ),
                      IconButton(
                        onPressed: _nextSign,
                        icon: const Icon(Icons.arrow_forward, color: Colors.white),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  Text(
                    '$_imageCount / $_targetPerSign images',
                    style: const TextStyle(color: Colors.white, fontSize: 18),
                  ),
                  const SizedBox(height: 8),
                  LinearProgressIndicator(
                    value: progress,
                    backgroundColor: Colors.grey.shade700,
                    valueColor: const AlwaysStoppedAnimation<Color>(Colors.orange),
                    minHeight: 8,
                  ),
                  if (_statusMessage.isNotEmpty) ...[
                    const SizedBox(height: 8),
                    Text(
                      _statusMessage,
                      style: const TextStyle(color: Colors.orange, fontSize: 12),
                    ),
                  ],
                ],
              ),
            ),
            
            // Camera Preview
            Expanded(
              child: widget.cameraController.value.isInitialized
                  ? CameraPreview(widget.cameraController)
                  : const Center(child: CircularProgressIndicator()),
            ),
            
            // Controls
            Container(
              padding: const EdgeInsets.all(20),
              color: Colors.grey.shade900,
              child: Column(
                children: [
                  // Instruction
                  Container(
                    padding: const EdgeInsets.all(12),
                    margin: const EdgeInsets.only(bottom: 16),
                    decoration: BoxDecoration(
                      color: Colors.blue.shade900.withOpacity(0.3),
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(color: Colors.blue.shade700),
                    ),
                    child: Text(
                      'Position your hand showing sign "$_currentSign"\nTry different angles, backgrounds, and lighting',
                      textAlign: TextAlign.center,
                      style: const TextStyle(color: Colors.white70, fontSize: 13),
                    ),
                  ),
                  
                  // Capture Button
                  SizedBox(
                    width: double.infinity,
                    height: 60,
                    child: ElevatedButton.icon(
                      onPressed: _captureImage,
                      icon: const Icon(Icons.camera_alt, size: 28),
                      label: const Text('CAPTURE IMAGE', style: TextStyle(fontSize: 18)),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.orange,
                        foregroundColor: Colors.white,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                    ),
                  ),
                  
                  const SizedBox(height: 12),
                  
                  // Action Buttons Row
                  Row(
                    children: [
                      Expanded(
                        child: OutlinedButton.icon(
                          onPressed: _nextSign,
                          icon: const Icon(Icons.skip_next, size: 20),
                          label: const Text('NEXT SIGN'),
                          style: OutlinedButton.styleFrom(
                            foregroundColor: Colors.white,
                            side: const BorderSide(color: Colors.white54),
                          ),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: OutlinedButton.icon(
                          onPressed: () => Navigator.pop(context),
                          icon: const Icon(Icons.exit_to_app, size: 20),
                          label: const Text('EXIT'),
                          style: OutlinedButton.styleFrom(
                            foregroundColor: Colors.red.shade300,
                            side: BorderSide(color: Colors.red.shade300),
                          ),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
