import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

/// Classifies BIM signs from normalized 21-point hand landmark features.
class LandmarkClassifier {
  Interpreter? _interpreter;
  List<String> _labels = [];
  bool _isLoaded = false;

  bool get isLoaded => _isLoaded;

  Future<void> load() async {
    try {
      final options = InterpreterOptions()..threads = 2;
      _interpreter = await Interpreter.fromAsset(
        'assets/landmark_classifier.tflite',
        options: options,
      );

      final labelsData = await rootBundle.loadString('assets/classes.txt');
      _labels = labelsData
          .split('\n')
          .map((label) => label.trim())
          .where((label) => label.isNotEmpty)
          .toList();

      if (kDebugMode) {
        debugPrint('Landmark classifier loaded');
        debugPrint('  Input: ${_interpreter!.getInputTensors()}');
        debugPrint('  Output: ${_interpreter!.getOutputTensors()}');
      }

      _isLoaded = true;
    } catch (e) {
      debugPrint('Landmark classifier error: $e');
      _isLoaded = false;
    }
  }

  Map<String, dynamic> classify(List<double> features, {String mode = 'all'}) {
    if (!_isLoaded || _interpreter == null || features.length != 63) {
      return {
        'label': 'Not loaded',
        'confidence': 0.0,
        'timeMs': 0,
        'top3': <Map<String, dynamic>>[],
        'source': 'landmark',
      };
    }

    final input = Float32List.fromList(features).reshape([1, 63]);
    final output = List.generate(1, (_) => List.filled(36, 0.0));

    final stopwatch = Stopwatch()..start();
    _interpreter!.run(input, output);
    stopwatch.stop();

    final indexed = <MapEntry<int, double>>[];
    final scores = output[0];

    for (int i = 0; i < scores.length && i < _labels.length; i++) {
      final label = _labels[i];
      final include = switch (mode) {
        'numbers' => RegExp(r'^[0-9]$').hasMatch(label),
        'letters' => RegExp(r'^[A-Z]$').hasMatch(label),
        _ => true,
      };

      if (include) {
        indexed.add(MapEntry(i, scores[i]));
      }
    }

    indexed.sort((a, b) => b.value.compareTo(a.value));
    final sumScores = indexed.fold(0.0, (sum, item) => sum + item.value);
    final topIndex = indexed.isNotEmpty ? indexed.first.key : 0;
    final topScore = sumScores > 0 ? indexed.first.value / sumScores : 0.0;

    final top3 = indexed.take(3).map((item) {
      return {
        'label': _labels[item.key],
        'confidence': sumScores > 0 ? item.value / sumScores : 0.0,
      };
    }).toList();

    return {
      'label': topIndex < _labels.length ? _labels[topIndex] : 'Unknown',
      'confidence': topScore,
      'timeMs': stopwatch.elapsedMilliseconds,
      'top3': top3,
      'source': 'landmark',
    };
  }

  void dispose() {
    _interpreter?.close();
  }
}
