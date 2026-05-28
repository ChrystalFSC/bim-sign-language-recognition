"""
Computational Efficiency Benchmark Script
==========================================
Measures for all 9 architectures:
  1. Total Parameters
  2. Trainable Parameters
  3. FLOPs (Floating Point Operations)
  4. Model file size (MB)
  5. Inference latency (ms per image, averaged over 100 images)

Results are saved to a CSV and formatted text file for thesis use.
"""
import os
import sys
import time
import numpy as np
import csv

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras

# ============================================================
# Configuration
# NOTE: Paths are relative to the repo root (bim-sign-language-recognition/).
# Run this script from the repo root: python benchmark_efficiency.py
# Model .keras files must be present locally (not committed to git — see README).
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    'MobileNetV3-Small': os.path.join(BASE_DIR, 'stage_training', 'output', 'best_model_stage3_1.keras'),
    'MobileNetV2':       os.path.join(BASE_DIR, 'experimental_models', 'mobilenetv2', 'output', 'best_model_stage3_1.keras'),
    'MobileNetV3-Large': os.path.join(BASE_DIR, 'experimental_models', 'mobilenetv3_large', 'output', 'best_model_stage3_1.keras'),
    'DenseNet201':       os.path.join(BASE_DIR, 'experimental_models', 'densenet201', 'output', 'best_model_stage3_1.keras'),
    'EfficientNetV2-M':  os.path.join(BASE_DIR, 'experimental_models', 'efficientnetv2_m', 'output', 'best_model_stage3_1.keras'),
    'ResNet18':          os.path.join(BASE_DIR, 'experimental_models', 'resnet18', 'output', 'best_model_stage3_1.keras'),
    'AlexNet':           os.path.join(BASE_DIR, 'experimental_models', 'alexnet', 'output', 'best_model_stage3_1.keras'),
    'VGG16':             os.path.join(BASE_DIR, 'experimental_models', 'vgg16', 'output', 'best_model_stage3_1.keras'),
    'VGG19':             os.path.join(BASE_DIR, 'experimental_models', 'vgg19', 'output', 'best_model_stage3_1.keras'),
}

INPUT_SHAPE = (224, 224, 3)
NUM_WARMUP = 10      # Warmup runs (not counted)
NUM_INFERENCE = 100  # Timed runs

OUTPUT_DIR = os.path.join(BASE_DIR, 'benchmark_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# FLOPs Calculation using tf.profiler
# ============================================================
def get_flops(model):
    """Calculate FLOPs for a Keras model using tf.profiler."""
    try:
        # Method 1: Use concrete function profiling
        input_signature = [tf.TensorSpec(shape=(1,) + INPUT_SHAPE, dtype=tf.float32)]
        
        @tf.function(input_signature=input_signature)
        def inference(x):
            return model(x, training=False)
        
        concrete_func = inference.get_concrete_function()
        
        # Use tf.profiler to count FLOPs
        frozen_func, graph_def = _convert_to_frozen(concrete_func)
        
        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name='')
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(graph, run_meta=run_meta, cmd='op', options=opts)
            return flops.total_float_ops
    except Exception as e1:
        try:
            # Method 2: Fallback - estimate from model layers
            return _estimate_flops_from_layers(model)
        except Exception as e2:
            print(f"    [WARN] FLOPs calculation failed: {e1}")
            return None

def _convert_to_frozen(concrete_func):
    """Convert a concrete function to a frozen graph for profiling."""
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    graph_def = frozen_func.graph.as_graph_def()
    return frozen_func, graph_def

def _estimate_flops_from_layers(model):
    """Estimate FLOPs by analyzing model layers (rough approximation)."""
    total_flops = 0
    for layer in model.layers:
        if isinstance(layer, (keras.layers.Conv2D, keras.layers.DepthwiseConv2D)):
            output_shape = layer.output_shape
            if isinstance(output_shape, list):
                output_shape = output_shape[0]
            
            if isinstance(layer, keras.layers.Conv2D):
                kernel = layer.kernel_size
                in_channels = layer.input_shape[-1] if hasattr(layer, 'input_shape') else 0
                out_channels = layer.filters
                h, w = output_shape[1], output_shape[2]
                # FLOPs = 2 * H * W * K_h * K_w * C_in * C_out
                flops = 2 * h * w * kernel[0] * kernel[1] * in_channels * out_channels
                total_flops += flops
            elif isinstance(layer, keras.layers.DepthwiseConv2D):
                kernel = layer.kernel_size
                channels = output_shape[-1]
                h, w = output_shape[1], output_shape[2]
                flops = 2 * h * w * kernel[0] * kernel[1] * channels
                total_flops += flops
                
        elif isinstance(layer, keras.layers.Dense):
            in_features = layer.input_shape[-1] if hasattr(layer, 'input_shape') else 0
            out_features = layer.units
            total_flops += 2 * in_features * out_features
    
    return total_flops if total_flops > 0 else None

# ============================================================
# Inference Latency Measurement
# ============================================================
def measure_inference_latency(model, num_warmup=NUM_WARMUP, num_runs=NUM_INFERENCE):
    """Measure average inference latency in milliseconds."""
    # Create a random input tensor
    dummy_input = np.random.rand(1, *INPUT_SHAPE).astype(np.float32)
    
    # Warmup runs
    print(f"    Warming up ({num_warmup} runs)...", end='', flush=True)
    for _ in range(num_warmup):
        model.predict(dummy_input, verbose=0)
    print(" done")
    
    # Timed runs
    print(f"    Timing ({num_runs} runs)...", end='', flush=True)
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        model.predict(dummy_input, verbose=0)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms
    print(" done")
    
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    
    return {
        'avg_ms': avg_latency,
        'std_ms': std_latency,
        'min_ms': min_latency,
        'max_ms': max_latency,
    }

# ============================================================
# Main Benchmark Loop
# ============================================================
def main():
    print("=" * 70)
    print("COMPUTATIONAL EFFICIENCY BENCHMARK")
    print(f"Input Shape: {INPUT_SHAPE}")
    print(f"Inference runs: {NUM_INFERENCE} (+ {NUM_WARMUP} warmup)")
    print(f"Device: {tf.config.list_physical_devices()}")
    print("=" * 70)
    
    results = []
    
    # Custom objects for focal loss models
    def focal_loss(gamma=2.0, alpha=0.25):
        def focal_loss_fn(y_true, y_pred):
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
            cross_entropy = -y_true * tf.math.log(y_pred)
            weight = alpha * y_true * tf.pow(1 - y_pred, gamma)
            return tf.reduce_sum(weight * cross_entropy, axis=-1)
        return focal_loss_fn
    
    custom_objects = {'focal_loss_fn': focal_loss(5.0, 0.25)}
    
    for model_name, model_path in MODELS.items():
        print(f"\n{'='*70}")
        print(f"Benchmarking: {model_name}")
        print(f"  Path: {model_path}")
        
        # File size
        file_size_bytes = os.path.getsize(model_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        print(f"  File size: {file_size_mb:.2f} MB")
        
        # Load model
        print(f"  Loading model...", end='', flush=True)
        try:
            model = keras.models.load_model(model_path, custom_objects=custom_objects)
            print(" done")
        except Exception as e:
            print(f" FAILED: {e}")
            continue
        
        # Parameters
        total_params = model.count_params()
        trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")
        
        # FLOPs
        print(f"  Calculating FLOPs...")
        flops = get_flops(model)
        if flops:
            print(f"  FLOPs: {flops:,}")
            gflops = flops / 1e9
            mflops = flops / 1e6
        else:
            print(f"  FLOPs: Could not calculate")
            gflops = None
            mflops = None
        
        # Inference latency
        print(f"  Measuring inference latency...")
        latency = measure_inference_latency(model)
        print(f"  Avg latency: {latency['avg_ms']:.2f} ms ± {latency['std_ms']:.2f} ms")
        print(f"  Min: {latency['min_ms']:.2f} ms | Max: {latency['max_ms']:.2f} ms")
        
        # FPS
        fps = 1000.0 / latency['avg_ms'] if latency['avg_ms'] > 0 else 0
        
        results.append({
            'Model': model_name,
            'Total Parameters': total_params,
            'Trainable Parameters': trainable_params,
            'Non-Trainable Parameters': non_trainable_params,
            'FLOPs': flops,
            'GFLOPs': gflops,
            'MFLOPs': mflops,
            'Model Size (MB)': round(file_size_mb, 2),
            'Avg Latency (ms)': round(latency['avg_ms'], 2),
            'Std Latency (ms)': round(latency['std_ms'], 2),
            'Min Latency (ms)': round(latency['min_ms'], 2),
            'Max Latency (ms)': round(latency['max_ms'], 2),
            'FPS': round(fps, 1),
        })
        
        # Clean up
        del model
        tf.keras.backend.clear_session()
    
    # ============================================================
    # Save Results
    # ============================================================
    
    # CSV output
    csv_path = os.path.join(OUTPUT_DIR, 'efficiency_benchmarks.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[SAVED] CSV: {csv_path}")
    
    # Formatted text output for thesis
    txt_path = os.path.join(OUTPUT_DIR, 'efficiency_benchmarks.txt')
    with open(txt_path, 'w') as f:
        f.write("COMPUTATIONAL EFFICIENCY BENCHMARKS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Input Resolution: {INPUT_SHAPE[0]}x{INPUT_SHAPE[1]}\n")
        f.write(f"Inference Runs: {NUM_INFERENCE} (+ {NUM_WARMUP} warmup)\n")
        f.write(f"Platform: Desktop (see GPU/CPU info below)\n")
        f.write(f"TensorFlow: {tf.__version__}\n")
        f.write(f"Devices: {tf.config.list_physical_devices()}\n")
        f.write("=" * 70 + "\n\n")
        
        # Table format
        header = f"{'Model':<22} {'Params':>12} {'FLOPs':>14} {'Size (MB)':>10} {'Latency (ms)':>14} {'FPS':>8}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        
        for r in sorted(results, key=lambda x: x['Total Parameters']):
            params_str = f"{r['Total Parameters']/1e6:.2f}M"
            flops_str = f"{r['GFLOPs']:.2f}G" if r['GFLOPs'] else "N/A"
            line = f"{r['Model']:<22} {params_str:>12} {flops_str:>14} {r['Model Size (MB)']:>10.2f} {r['Avg Latency (ms)']:>10.2f}±{r['Std Latency (ms)']:.2f} {r['FPS']:>8.1f}"
            f.write(line + "\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("Note: Latency measured on desktop hardware. Mobile latency will differ.\n")
        f.write("FLOPs = total floating-point operations for a single forward pass.\n")
    
    print(f"[SAVED] Text: {txt_path}")
    
    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Model':<22} {'Params':>12} {'FLOPs':>14} {'Size (MB)':>10} {'Latency':>12} {'FPS':>8}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: x['Total Parameters']):
        params_str = f"{r['Total Parameters']/1e6:.2f}M"
        flops_str = f"{r['GFLOPs']:.2f}G" if r['GFLOPs'] else "N/A"
        print(f"{r['Model']:<22} {params_str:>12} {flops_str:>14} {r['Model Size (MB)']:>10.2f} {r['Avg Latency (ms)']:>10.2f} ms {r['FPS']:>8.1f}")

if __name__ == '__main__':
    main()
