import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

/// Sign Language Classifier with mode filtering
class SignClassifier {
  Interpreter? _interpreter;
  List<String> _labels = [];
  bool _isLoaded = false;
  
  bool get isLoaded => _isLoaded;
  List<String> get labels => _labels;

  Future<void> load() async {
    try {
      if (kDebugMode) debugPrint('Loading model...');
      
      InterpreterOptions options = InterpreterOptions();
      try {
        final gpuDelegate = GpuDelegateV2();
        options.addDelegate(gpuDelegate);
        if (kDebugMode) debugPrint('GPU enabled');
      } catch (e) {
        if (kDebugMode) debugPrint('Using CPU');
      }
      options.threads = 4;
      
      _interpreter = await Interpreter.fromAsset(
        'assets/model_stage3_1_float16.tflite',  // Stage 3.1: Float16 optimized model
        options: options,
      );
      
      if (kDebugMode) {
        debugPrint('Input: ${_interpreter!.getInputTensors()}');
        debugPrint('Output: ${_interpreter!.getOutputTensors()}');
      }
      
      // Load class labels
      final labelsData = await rootBundle.loadString('assets/classes.txt');
      _labels = labelsData.split('\n').map((l) => l.trim()).where((l) => l.isNotEmpty).toList();
      if (kDebugMode) debugPrint('Loaded ${_labels.length} classes: $_labels');
      
      _isLoaded = true;
    } catch (e) {
      debugPrint('Error: $e');
      rethrow;
    }
  }

  /// Classify image with optional mode filtering
  /// mode: 'all', 'numbers', or 'letters'
  Map<String, dynamic> classifyImage(Float32List inputData, {String mode = 'all'}) {
    if (!_isLoaded || _interpreter == null) {
      return {'label': 'Not loaded', 'confidence': 0.0, 'timeMs': 0};
    }

    final input = inputData.reshape([1, 224, 224, 3]);
    
    // For INT8 quantized model, TFLite handles dequantization automatically
    // if we use the correct output type (List<List<double>>)
    final output = List.generate(1, (_) => List.filled(36, 0.0));
    
    final stopwatch = Stopwatch()..start();
    _interpreter!.run(input, output);
    stopwatch.stop();
    
    final inferenceTime = stopwatch.elapsedMilliseconds;
    final scores = output[0];
    
    // Filter based on mode
    List<MapEntry<int, double>> indexed = [];
    for (int i = 0; i < scores.length; i++) {
      final label = _labels[i];
      bool include = false;
      
      switch (mode) {
        case 'numbers':
          // Include only 0-9
          include = RegExp(r'^[0-9]$').hasMatch(label);
          break;
        case 'letters':
          // Include only A-Z
          include = RegExp(r'^[A-Z]$').hasMatch(label);
          break;
        default:
          include = true;
      }
      
      if (include) {
        indexed.add(MapEntry(i, scores[i]));
      }
    }
    
    // Sort by score
    indexed.sort((a, b) => b.value.compareTo(a.value));
    
    // Renormalize scores within filtered set
    final sumScores = indexed.fold(0.0, (sum, e) => sum + e.value);
    
    final topIdx = indexed.isNotEmpty ? indexed[0].key : 0;
    final topScore = sumScores > 0 ? indexed[0].value / sumScores : 0.0;
    
    // Get top 3
    final top3 = indexed.take(3).map((e) {
      return {
        'label': _labels[e.key],
        'confidence': sumScores > 0 ? e.value / sumScores : 0.0,
      };
    }).toList();
    
    final label = topIdx < _labels.length ? _labels[topIdx] : 'Unknown';
    return {
      'label': label, 
      'confidence': topScore,
      'timeMs': inferenceTime,
      'top3': top3,
    };
  }

  void dispose() {
    _interpreter?.close();
  }
}
