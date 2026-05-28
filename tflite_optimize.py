"""
TFLite Optimization Script for MobileNetV3-Small
=================================================
Converts the best Stage 3.1 model into 3 TFLite variants:
  1. Dynamic Range Quantization (default, same as FYP1)
  2. Float16 Quantization
  3. Full INT8 Quantization (with representative dataset)

Then evaluates each variant's accuracy on the test set.
"""
import os
import sys
import time
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras

# ============================================================
# Configuration
# NOTE: Paths are relative to repo root. Run from repo root:
#   python tflite_optimize.py
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, 'stage_training', 'output', 'best_model_stage3_1.keras')
TEST_DIR   = os.path.join(BASE_DIR, 'test_data')
TRAIN_DIR  = os.path.join(BASE_DIR, 'train_data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'tflite_models')
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 64

# Focal loss needed to load the model
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.pow(1 - y_pred, gamma)
        return tf.reduce_sum(weight * cross_entropy, axis=-1)
    return focal_loss_fn

# ============================================================
# Load model
# ============================================================
print("=" * 60)
print("TFLite OPTIMIZATION - MobileNetV3-Small")
print("=" * 60)

print(f"\nLoading model from: {MODEL_PATH}")
model = keras.models.load_model(
    MODEL_PATH,
    custom_objects={'focal_loss_fn': focal_loss(5.0, 0.25)}
)
print(f"Model loaded. Parameters: {model.count_params():,}")

# ============================================================
# Load test data
# ============================================================
print(f"\nLoading test data from: {TEST_DIR}")
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=INPUT_SHAPE[:2],
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Also load some training images for INT8 calibration
print(f"Loading calibration data from: {TRAIN_DIR}")
cal_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
cal_gen = cal_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=INPUT_SHAPE[:2],
    batch_size=1,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

# ============================================================
# Conversion functions
# ============================================================

def convert_dynamic_range(model, output_path):
    """Convert with dynamic range quantization (default)."""
    print("\n--- Converting: Dynamic Range Quantization ---")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Saved: {output_path} ({size_mb:.2f} MB)")
    return tflite_model

def convert_float16(model, output_path):
    """Convert with float16 quantization."""
    print("\n--- Converting: Float16 Quantization ---")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Saved: {output_path} ({size_mb:.2f} MB)")
    return tflite_model

def convert_int8(model, output_path, cal_gen, num_cal_images=200):
    """Convert with full INT8 quantization using representative dataset."""
    print(f"\n--- Converting: Full INT8 Quantization (calibrating with {num_cal_images} images) ---")
    
    # Build representative dataset generator
    cal_images = []
    for i in range(num_cal_images):
        img_batch, _ = next(cal_gen)
        cal_images.append(img_batch)
    cal_images = np.concatenate(cal_images, axis=0)
    print(f"  Calibration images collected: {cal_images.shape[0]}")
    
    def representative_dataset():
        for i in range(len(cal_images)):
            yield [cal_images[i:i+1].astype(np.float32)]
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Saved: {output_path} ({size_mb:.2f} MB)")
    return tflite_model

# ============================================================
# Evaluate TFLite model accuracy
# ============================================================

def evaluate_tflite(tflite_path, test_gen):
    """Evaluate a TFLite model on the test set."""
    print(f"  Evaluating accuracy on {test_gen.samples} test images...")
    
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_dtype = input_details[0]['dtype']
    input_shape = input_details[0]['shape']
    
    # Check if INT8 quantized (uint8 input)
    is_quantized = input_dtype == np.uint8
    if is_quantized:
        input_scale = input_details[0]['quantization'][0]
        input_zero_point = input_details[0]['quantization'][1]
    
    test_gen.reset()
    y_true = test_gen.classes
    y_pred = []
    
    total_batches = len(test_gen)
    for batch_idx in range(total_batches):
        images, _ = test_gen[batch_idx]
        
        for i in range(len(images)):
            img = images[i:i+1]
            
            if is_quantized:
                img = img / input_scale + input_zero_point
                img = np.clip(img, 0, 255).astype(np.uint8)
            else:
                img = img.astype(np.float32)
            
            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            y_pred.append(np.argmax(output[0]))
    
    y_pred = np.array(y_pred[:len(y_true)])
    accuracy = np.mean(y_pred == y_true)
    
    # Top-5
    # Re-run for top-5 (need full output)
    test_gen.reset()
    all_outputs = []
    for batch_idx in range(total_batches):
        images, _ = test_gen[batch_idx]
        for i in range(len(images)):
            img = images[i:i+1]
            if is_quantized:
                img = img / input_scale + input_zero_point
                img = np.clip(img, 0, 255).astype(np.uint8)
            else:
                img = img.astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            all_outputs.append(output[0])
    
    all_outputs = np.array(all_outputs[:len(y_true)])
    top5_indices = np.argsort(all_outputs, axis=1)[:, -5:]
    top5_acc = np.mean([1 if y_true[i] in top5_indices[i] else 0 for i in range(len(y_true))])
    
    # Latency (desktop, TFLite interpreter)
    dummy = np.random.rand(1, *INPUT_SHAPE).astype(np.float32)
    if is_quantized:
        dummy = (dummy / input_scale + input_zero_point).astype(np.uint8)
    
    # Warmup
    for _ in range(10):
        interpreter.set_tensor(input_details[0]['index'], dummy)
        interpreter.invoke()
    
    # Timed
    latencies = []
    for _ in range(100):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], dummy)
        interpreter.invoke()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    avg_latency = np.mean(latencies)
    
    return {
        'top1_accuracy': accuracy,
        'top5_accuracy': top5_acc,
        'avg_latency_ms': avg_latency,
    }

# ============================================================
# Main
# ============================================================

# 1. Dynamic Range
dr_path = os.path.join(OUTPUT_DIR, 'mobilenetv3small_dynamic_range.tflite')
convert_dynamic_range(model, dr_path)
dr_results = evaluate_tflite(dr_path, test_gen)
print(f"  Top-1: {dr_results['top1_accuracy']*100:.2f}%  |  Top-5: {dr_results['top5_accuracy']*100:.2f}%  |  Latency: {dr_results['avg_latency_ms']:.2f} ms")

# 2. Float16
f16_path = os.path.join(OUTPUT_DIR, 'mobilenetv3small_float16.tflite')
convert_float16(model, f16_path)
f16_results = evaluate_tflite(f16_path, test_gen)
print(f"  Top-1: {f16_results['top1_accuracy']*100:.2f}%  |  Top-5: {f16_results['top5_accuracy']*100:.2f}%  |  Latency: {f16_results['avg_latency_ms']:.2f} ms")

# 3. INT8
int8_path = os.path.join(OUTPUT_DIR, 'mobilenetv3small_int8.tflite')
convert_int8(model, int8_path, cal_gen, num_cal_images=200)
int8_results = evaluate_tflite(int8_path, test_gen)
print(f"  Top-1: {int8_results['top1_accuracy']*100:.2f}%  |  Top-5: {int8_results['top5_accuracy']*100:.2f}%  |  Latency: {int8_results['avg_latency_ms']:.2f} ms")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("TFLITE OPTIMIZATION SUMMARY")
print("=" * 70)
print(f"{'Variant':<25} {'Size (MB)':>10} {'Top-1 (%)':>10} {'Top-5 (%)':>10} {'Latency (ms)':>14}")
print("-" * 70)

keras_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
print(f"{'Original (.keras)':<25} {keras_size:>10.2f} {'96.06':>10} {'99.78':>10} {'43.19':>14}")

for name, path, results in [
    ('Dynamic Range', dr_path, dr_results),
    ('Float16', f16_path, f16_results),
    ('INT8', int8_path, int8_results),
]:
    size = os.path.getsize(path) / (1024 * 1024)
    print(f"{name:<25} {size:>10.2f} {results['top1_accuracy']*100:>10.2f} {results['top5_accuracy']*100:>10.2f} {results['avg_latency_ms']:>14.2f}")

# Save summary to file
summary_path = os.path.join(OUTPUT_DIR, 'optimization_summary.txt')
with open(summary_path, 'w') as f:
    f.write("TFLite Optimization Summary - MobileNetV3-Small\n")
    f.write("=" * 70 + "\n")
    f.write(f"Source model: {MODEL_PATH}\n")
    f.write(f"TensorFlow version: {tf.__version__}\n")
    f.write(f"INT8 calibration images: 200\n\n")
    f.write(f"{'Variant':<25} {'Size (MB)':>10} {'Top-1 (%)':>10} {'Top-5 (%)':>10} {'Latency (ms)':>14}\n")
    f.write("-" * 70 + "\n")
    f.write(f"{'Original (.keras)':<25} {keras_size:>10.2f} {'96.06':>10} {'99.78':>10} {'43.19':>14}\n")
    for name, path, results in [
        ('Dynamic Range', dr_path, dr_results),
        ('Float16', f16_path, f16_results),
        ('INT8', int8_path, int8_results),
    ]:
        size = os.path.getsize(path) / (1024 * 1024)
        f.write(f"{name:<25} {size:>10.2f} {results['top1_accuracy']*100:>10.2f} {results['top5_accuracy']*100:>10.2f} {results['avg_latency_ms']:>14.2f}\n")

print(f"\n[SAVED] Summary: {summary_path}")
print(f"[SAVED] TFLite models saved to: {OUTPUT_DIR}/")
print("\nNext step: Copy the .tflite files to your Android app and test on your Xiaomi MIX 2S!")
