import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'sign_classifier.dart';
import 'mediapipe_hand_detector.dart';
import 'hand_landmark_painter.dart';
import 'dart:math' as math;

List<CameraDescription> cameras = [];

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  try {
    cameras = await availableCameras();
  } catch (e) {
    print('Error getting cameras: $e');
  }
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'BIM Sign Language',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.blue,
        useMaterial3: true,
      ),
      home: const CameraPage(),
    );
  }
}

class CameraPage extends StatefulWidget {
  const CameraPage({super.key});

  @override
  State<CameraPage> createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> with WidgetsBindingObserver {
  CameraController? _cameraController;
  SignClassifier? _classifier;
  MediaPipeHandDetector? _handDetector;
  
  String _prediction = '-';
  double _confidence = 0.0;
  int _inferenceTime = 0;
  int _totalPipelineTime = 0;
  List<Map<String, dynamic>> _top3 = [];
  bool _isModelLoaded = false;
  bool _isCameraReady = false;
  bool _isProcessing = false;
  String _statusMessage = 'Initializing...';
  bool _isDisposed = false;
  bool _isFrontCamera = false;
  
  // Mode: 'numbers', 'letters', or 'all'
  String _mode = 'all';
  
  // Auto-detection mode
  bool _isAutoDetecting = false;
  
  // MediaPipe landmarks for visualization
  List<Offset>? _landmarks;
  List<double>? _handBbox;
  double _handConfidence = 0.0;
  bool _showLandmarks = true;
  bool _handDetected = false;
  
  // Prediction smoothing
  List<String> _recentPredictions = [];
  static const int _smoothingWindow = 3;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initialize();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (_cameraController == null || !_cameraController!.value.isInitialized) return;
    if (state == AppLifecycleState.inactive) {
      _cameraController?.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _initCamera();
    }
  }

  Future<void> _initialize() async {
    await _requestPermissions();
    await _loadModels();
    await _initCamera();
  }

  Future<void> _requestPermissions() async {
    final status = await Permission.camera.request();
    if (!status.isGranted) {
      setState(() => _statusMessage = 'Camera permission denied');
    }
  }

  Future<void> _loadModels() async {
    try {
      setState(() => _statusMessage = 'Loading models...');
      
      // Load sign classifier
      _classifier = SignClassifier();
      await _classifier!.load();
      
      // Load MediaPipe hand detector
      _handDetector = MediaPipeHandDetector();
      await _handDetector!.load();
      
      if (!_isDisposed) {
        setState(() {
          _isModelLoaded = true;
          _statusMessage = _handDetector!.isLoaded 
            ? 'Ready! MediaPipe + MobileNetV3'
            : 'Ready! (hand detection unavailable)';
        });
        print('DEBUG: Models loaded! Classifier=${_classifier!.isLoaded} MediaPipe=${_handDetector!.isLoaded}');
      }
    } catch (e) {
      if (!_isDisposed) {
        setState(() => _statusMessage = 'Model error: $e');
      }
    }
  }

  Future<void> _initCamera() async {
    if (cameras.isEmpty) {
      setState(() => _statusMessage = 'No cameras available');
      return;
    }

    CameraDescription? selectedCamera;
    final targetDirection = _isFrontCamera 
        ? CameraLensDirection.front 
        : CameraLensDirection.back;
    
    for (var cam in cameras) {
      if (cam.lensDirection == targetDirection) {
        selectedCamera = cam;
        break;
      }
    }
    selectedCamera ??= cameras.first;

    await _cameraController?.dispose();

    _cameraController = CameraController(
      selectedCamera,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );

    try {
      await _cameraController!.initialize();
      if (!_isDisposed) {
        setState(() {
          _isCameraReady = true;
          _statusMessage = 'Ready! Position hand in box';
        });
        if (_isAutoDetecting) _startAutoDetection();
      }
    } catch (e) {
      if (!_isDisposed) {
        setState(() => _statusMessage = 'Camera error: $e');
      }
    }
  }
  
  void _toggleAutoDetection() {
    setState(() {
      _isAutoDetecting = !_isAutoDetecting;
      if (_isAutoDetecting) {
        _statusMessage = 'Auto-detecting...';
        _recentPredictions.clear();
        _startAutoDetection();
      } else {
        _statusMessage = 'Auto-detection OFF';
      }
    });
  }
  
  void _startAutoDetection() async {
    while (_isAutoDetecting && _isCameraReady && !_isDisposed) {
      if (!_isProcessing) {
        await _captureAndClassify();
      }
      await Future.delayed(const Duration(milliseconds: 200));
    }
  }

  Future<void> _switchCamera() async {
    setState(() {
      _isCameraReady = false;
      _isFrontCamera = !_isFrontCamera;
    });
    await _initCamera();
  }

  Float32List _preprocessImage(img.Image image) {
    // 1. Resize to 224x224 directly (matches Keras stretching/squashing rectangular crops)
    final resized = img.copyResize(
      image,
      width: 224,
      height: 224,
      interpolation: img.Interpolation.linear,
    );
    
    // 3. Adaptive range detection
    double maxVal = 0.0;
    for (int y = 0; y < 224; y++) {
      for (int x = 0; x < 224; x++) {
        final pixel = resized.getPixel(x, y);
        if (pixel.r > maxVal) maxVal = pixel.r.toDouble();
        if (pixel.g > maxVal) maxVal = pixel.g.toDouble();
        if (pixel.b > maxVal) maxVal = pixel.b.toDouble();
      }
    }
    
    // If the image is normalized in [0, 1], scale it to [0, 255]
    final double scale = maxVal <= 1.01 ? 255.0 : 1.0;
    print('DEBUG: Preprocessing range check: maxVal=${maxVal.toStringAsFixed(3)}, scaleFactor=$scale');
    
    // 4. Convert to Float32List in 0-255 range (training used no rescale)
    final Float32List inputData = Float32List(224 * 224 * 3);
    int idx = 0;
    double sumR = 0, sumG = 0, sumB = 0;
    for (int y = 0; y < 224; y++) {
      for (int x = 0; x < 224; x++) {
        final pixel = resized.getPixel(x, y);
        final r = pixel.r.toDouble() * scale;
        final g = pixel.g.toDouble() * scale;
        final b = pixel.b.toDouble() * scale;
        inputData[idx++] = r;
        inputData[idx++] = g;
        inputData[idx++] = b;
        sumR += r;
        sumG += g;
        sumB += b;
      }
    }
    final avgR = sumR / (224 * 224);
    final avgG = sumG / (224 * 224);
    final avgB = sumB / (224 * 224);
    print('DEBUG: Preprocessed image averages: R=${avgR.toStringAsFixed(1)}, G=${avgG.toStringAsFixed(1)}, B=${avgB.toStringAsFixed(1)}');
    return inputData;
  }

  /// Temporal smoothing: majority vote over recent predictions
  String _getSmoothedPrediction(String label) {
    _recentPredictions.add(label);
    if (_recentPredictions.length > _smoothingWindow) {
      _recentPredictions.removeAt(0);
    }
    if (_recentPredictions.length >= _smoothingWindow) {
      Map<String, int> counts = {};
      for (var pred in _recentPredictions) {
        counts[pred] = (counts[pred] ?? 0) + 1;
      }
      var sorted = counts.entries.toList()
        ..sort((a, b) => b.value.compareTo(a.value));
      return sorted.first.key;
    }
    return label;
  }

  Future<void> _captureAndClassify() async {
    if (!_isCameraReady || !_isModelLoaded || _isProcessing) return;
    if (_cameraController == null || !_cameraController!.value.isInitialized) return;

    _isProcessing = true;
    if (!_isAutoDetecting) {
      setState(() => _statusMessage = 'Processing...');
    }

    final pipelineStart = DateTime.now();

    try {
      // Step 1: Capture image
      final XFile imageFile = await _cameraController!.takePicture();
      final bytes = await imageFile.readAsBytes();
      var image = img.decodeImage(bytes);
      if (image == null) throw Exception('Failed to decode');
      
      if (_isFrontCamera) {
        image = img.flipHorizontal(image);
      }
      
      // Step 2: Initial ROI crop (centre 75% of frame)
      final roiPercent = 0.75;
      final roiW = (image.width * roiPercent).toInt();
      final roiH = (image.height * roiPercent).toInt();
      final roiX = ((image.width - roiW) / 2).toInt();
      final roiY = ((image.height - roiH) / 2).toInt();
      var roiImage = img.copyCrop(image, x: roiX, y: roiY, width: roiW, height: roiH);
      
      // Step 3: MediaPipe hand detection on ROI
      img.Image classificationInput = roiImage;
      bool handFound = false;
      
      if (_handDetector != null && _handDetector!.isLoaded) {
        final result = await _handDetector!.detect(roiImage);
        
        if (result != null) {
          handFound = true;
          
          // Crop tightly around detected hand landmarks
          classificationInput = _handDetector!.cropToHandLandmarks(roiImage, result.boundingBox);
          
          // Scale landmarks back to full image coordinates for visualization
          if (_showLandmarks && !_isDisposed) {
            final scaledLandmarks = result.landmarks.map((lm) => Offset(
              lm.dx + roiX,
              lm.dy + roiY,
            )).toList();
            
            setState(() {
              _landmarks = scaledLandmarks;
              _handBbox = [
                result.boundingBox[0] * roiW / image!.width + roiX / image.width,
                result.boundingBox[1] * roiH / image.height + roiY / image.height,
                result.boundingBox[2] * roiW / image.width + roiX / image.width,
                result.boundingBox[3] * roiH / image.height + roiY / image.height,
              ];
              _handConfidence = result.confidence;
              _handDetected = true;
            });
          }
          
          print('MediaPipe: Hand detected (${(result.confidence * 100).toStringAsFixed(0)}%) with 21 landmarks');
        } else {
          if (!_isDisposed) {
            setState(() {
              _landmarks = null;
              _handBbox = null;
              _handDetected = false;
            });
          }
          print('MediaPipe: No hand detected');
        }
      }
      
      // Only classify if hand was detected — prevents false "G"/"1" predictions
      if (!handFound && _handDetector != null && _handDetector!.isLoaded) {
        if (!_isDisposed) {
          setState(() {
            _prediction = '-';
            _confidence = 0.0;
            _top3 = [];
            _statusMessage = 'No hand detected. Position hand in box.';
            _isProcessing = false;
          });
        }
        return;
      }
      
      // Step 4: Preprocess for classifier
      final inputData = _preprocessImage(classificationInput);
      
      // Step 5: Classify
      final result = _classifier!.classifyImage(inputData, mode: _mode);
      
      // Apply smoothing in auto mode
      String displayLabel;
      if (_isAutoDetecting) {
        displayLabel = _getSmoothedPrediction(result['label'] ?? '?');
      } else {
        displayLabel = result['label'] ?? '?';
        _recentPredictions.clear();
      }
      
      if (!_isDisposed) {
        setState(() {
          _prediction = displayLabel;
          _confidence = result['confidence'] ?? 0.0;
          _inferenceTime = result['timeMs'] ?? 0;
          _top3 = List<Map<String, dynamic>>.from(result['top3'] ?? []);
          _statusMessage = _isAutoDetecting 
            ? 'Auto-detecting...'
            : 'Tap CAPTURE to try again';
          _isProcessing = false;
        });
      }
    } catch (e) {
      print('Error: $e');
      if (!_isDisposed) {
        setState(() {
          _statusMessage = 'Error: $e';
          _isProcessing = false;
        });
      }
    }
  }

  @override
  void dispose() {
    _isDisposed = true;
    WidgetsBinding.instance.removeObserver(this);
    _cameraController?.dispose();
    _classifier?.dispose();
    _handDetector?.dispose();
    super.dispose();
  }

  Color _getModeColor() {
    switch (_mode) {
      case 'numbers': return const Color(0xFF004B87);
      case 'letters': return const Color(0xFFFFD100);
      default: return const Color(0xFF007E9A);
    }
  }
  
  String _getModeLabel() {
    switch (_mode) {
      case 'numbers': return '0-9 ONLY';
      case 'letters': return 'A-Z ONLY';
      default: return 'ALL';
    }
  }

  @override
  Widget build(BuildContext context) {
    final screenWidth = MediaQuery.of(context).size.width;
    final boxSize = screenWidth * 0.65;
    final modeColor = _getModeColor();
    
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: Text('BIM Sign - ${_getModeLabel()}'),
        backgroundColor: modeColor,
        foregroundColor: Colors.white,
        centerTitle: true,
        actions: [
          // Auto-detection toggle
          IconButton(
            onPressed: _isCameraReady && _isModelLoaded ? _toggleAutoDetection : null,
            icon: Icon(_isAutoDetecting ? Icons.play_circle : Icons.play_circle_outline),
            tooltip: _isAutoDetecting ? 'Auto ON' : 'Auto OFF',
            color: _isAutoDetecting ? Colors.greenAccent : Colors.white,
          ),
          // Landmark toggle
          IconButton(
            onPressed: () => setState(() => _showLandmarks = !_showLandmarks),
            icon: Icon(_showLandmarks ? Icons.visibility : Icons.visibility_off),
            tooltip: _showLandmarks ? 'Landmarks ON' : 'Landmarks OFF',
            color: _showLandmarks ? Colors.cyanAccent : Colors.white,
          ),
          // Camera switch
          IconButton(
            onPressed: _isCameraReady && !_isProcessing ? _switchCamera : null,
            icon: Icon(_isFrontCamera ? Icons.camera_rear : Icons.camera_front),
          ),
        ],
      ),
      body: SafeArea(
        child: Column(
          children: [
            // Mode selector
            Container(
              padding: const EdgeInsets.symmetric(vertical: 8),
              color: Colors.grey[900],
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  _buildModeButton('numbers', '0-9', Icons.pin, const Color(0xFF004B87)),
                  _buildModeButton('letters', 'A-Z', Icons.abc, const Color(0xFFFFD100)),
                  _buildModeButton('all', 'ALL', Icons.apps, const Color(0xFF007E9A)),
                ],
              ),
            ),
            
            // Camera preview
            Expanded(
              flex: 2,
              child: Container(
                margin: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(16),
                  color: Colors.grey[900],
                ),
                clipBehavior: Clip.hardEdge,
                child: _isCameraReady && _cameraController != null
                    ? Stack(
                        fit: StackFit.expand,
                        children: [
                          Transform(
                            alignment: Alignment.center,
                            transform: Matrix4.identity()
                              ..scale(_isFrontCamera ? -1.0 : 1.0, 1.0, 1.0),
                            child: CameraPreview(_cameraController!),
                          ),
                          
                          // MediaPipe 21-point hand landmarks overlay
                          if (_showLandmarks && _landmarks != null && _cameraController != null)
                            LayoutBuilder(
                              builder: (context, constraints) {
                                return CustomPaint(
                                  size: Size(constraints.maxWidth, constraints.maxHeight),
                                  painter: HandLandmarkPainter(
                                    landmarks: _landmarks,
                                    boundingBox: _handBbox,
                                    confidence: _handConfidence,
                                    imageSize: Size(
                                      _cameraController!.value.previewSize!.height,
                                      _cameraController!.value.previewSize!.width,
                                    ),
                                  ),
                                );
                              },
                            ),
                          
                          // ROI Box
                          Center(
                            child: Container(
                              width: boxSize,
                              height: boxSize,
                              decoration: BoxDecoration(
                                border: Border.all(
                                  color: _handDetected 
                                    ? Colors.greenAccent 
                                    : (_isProcessing ? Colors.orange : modeColor.withOpacity(0.5)),
                                  width: _handDetected ? 3 : 2,
                                ),
                                borderRadius: BorderRadius.circular(12),
                              ),
                              child: !_handDetected && _prediction == '-'
                                ? Center(
                                    child: Container(
                                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                                      decoration: BoxDecoration(
                                        color: Colors.black54,
                                        borderRadius: BorderRadius.circular(8),
                                      ),
                                      child: const Text(
                                        'Place hand here',
                                        style: TextStyle(color: Colors.white70, fontSize: 14),
                                      ),
                                    ),
                                  )
                                : null,
                            ),
                          ),
                          
                          // Mode indicator
                          Positioned(
                            top: 8, left: 8,
                            child: Container(
                              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                              decoration: BoxDecoration(
                                color: modeColor,
                                borderRadius: BorderRadius.circular(12),
                              ),
                              child: Text(
                                _getModeLabel(),
                                style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 11),
                              ),
                            ),
                          ),
                        ],
                      )
                    : const Center(child: CircularProgressIndicator(color: Colors.blue)),
              ),
            ),
            
            // Result panel
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.grey[900],
                borderRadius: const BorderRadius.vertical(top: Radius.circular(20)),
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  // Prediction
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 12),
                        decoration: BoxDecoration(
                          color: _confidence > 0.7 ? Colors.green : modeColor,
                          borderRadius: BorderRadius.circular(16),
                        ),
                        child: Text(
                          _prediction,
                          style: TextStyle(
                            fontSize: 48,
                            fontWeight: FontWeight.bold,
                            // Use black text on the yellow letters mode for contrast
                            color: _mode == 'letters' ? Colors.black : Colors.white,
                          ),
                        ),
                      ),
                      const SizedBox(width: 16),
                      Text(
                        '${(_confidence * 100).toStringAsFixed(0)}%',
                        style: TextStyle(
                          fontSize: 24,
                          fontWeight: FontWeight.bold,
                          color: _confidence > 0.7 ? Colors.greenAccent : Colors.white70,
                        ),
                      ),
                    ],
                  ),
                  
                  // Top 3
                  if (_top3.isNotEmpty) ...[
                    const SizedBox(height: 8),
                    Wrap(
                      spacing: 8,
                      children: _top3.map((p) => Chip(
                        label: Text('${p['label']}: ${((p['confidence'] ?? 0) * 100).toStringAsFixed(0)}%'),
                        backgroundColor: Colors.grey[800],
                        labelStyle: const TextStyle(color: Colors.white70, fontSize: 12),
                      )).toList(),
                    ),
                  ],
                  
                  const SizedBox(height: 6),
                  Text(_statusMessage, style: TextStyle(color: Colors.grey[500], fontSize: 10)),
                  const SizedBox(height: 8),
                  
                  // Capture button
                  SizedBox(
                    width: double.infinity,
                    height: 48,
                    child: ElevatedButton.icon(
                      onPressed: (_isCameraReady && _isModelLoaded && !_isProcessing)
                          ? _captureAndClassify : null,
                      icon: _isProcessing 
                          ? const SizedBox(width: 18, height: 18, 
                              child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                          : const Icon(Icons.camera_alt),
                      label: Text(_isProcessing ? 'Processing...' : 'CAPTURE'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: modeColor,
                        foregroundColor: Colors.white,
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                      ),
                    ),
                  ),
                  
                  // Auto-detection indicator
                  if (_isAutoDetecting)
                    Padding(
                      padding: const EdgeInsets.only(top: 8),
                      child: Container(
                        width: double.infinity,
                        padding: const EdgeInsets.all(8),
                        decoration: BoxDecoration(
                          color: Colors.greenAccent.withOpacity(0.15),
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(color: Colors.greenAccent.withOpacity(0.5), width: 1),
                        ),
                        child: const Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(Icons.play_circle, color: Colors.greenAccent, size: 16),
                            SizedBox(width: 6),
                            Text('AUTO-DETECTING', style: TextStyle(color: Colors.greenAccent, fontWeight: FontWeight.bold, fontSize: 12)),
                          ],
                        ),
                      ),
                    ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
  
  Widget _buildModeButton(String mode, String label, IconData icon, Color color) {
    final isSelected = _mode == mode;
    return GestureDetector(
      onTap: () {
        setState(() {
          _mode = mode;
          _prediction = '-';
          _confidence = 0.0;
          _top3 = [];
          _recentPredictions.clear();
        });
      },
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        decoration: BoxDecoration(
          color: isSelected ? color : Colors.grey[800],
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: isSelected ? color : Colors.grey[700]!,
            width: 2,
          ),
        ),
        child: Row(
          children: [
            Icon(icon, color: Colors.white, size: 18),
            const SizedBox(width: 6),
            Text(
              label,
              style: TextStyle(
                color: Colors.white,
                fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
