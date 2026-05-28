"""
Check Model Parameter Count and Architecture Details
For thesis reporting
"""

import tensorflow as tf

# Focal loss (needed to load model)
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = alpha * tf.math.pow(1.0 - y_pred, gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)
    return focal_loss_fixed

print("="*70)
print("MODEL ARCHITECTURE ANALYSIS")
print("="*70)

# Load model
MODEL_PATH = 'experimental_models/mobilenetv3_small/output/best_model_stage3_1.keras'

print(f"\n[1/3] Loading model: {MODEL_PATH}")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={'focal_loss_fixed': focal_loss()},
    compile=False
)
print("   ✓ Model loaded!")

# Get parameter count
print("\n[2/3] Counting parameters...")

total_params = model.count_params()
trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
non_trainable_params = total_params - trainable_params

print(f"\n   Total Parameters:        {total_params:,}")
print(f"   Trainable Parameters:    {trainable_params:,}")
print(f"   Non-trainable Parameters: {non_trainable_params:,}")

# Model size estimate
model_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32 param
print(f"\n   Estimated Model Size (Float32): {model_size_mb:.2f} MB")

# Get model summary
print("\n[3/3] Model Architecture Summary:")
print("="*70)
model.summary()

# Detailed layer info
print("\n" + "="*70)
print("LAYER DETAILS")
print("="*70)

print(f"\n{'Layer Name':<30} {'Type':<20} {'Trainable':<12} {'Params':<15}")
print("-"*77)

for layer in model.layers:
    layer_params = layer.count_params()
    layer_type = layer.__class__.__name__
    trainable = "Yes" if layer.trainable else "No"
    
    print(f"{layer.name:<30} {layer_type:<20} {trainable:<12} {layer_params:>12,}")

# Summary for thesis
print("\n" + "="*70)
print("THESIS SUMMARY")
print("="*70)

print(f"""
Model: MobileNetV3-Small (Stage 3.1 Fine-tuned)

Architecture Metrics:
  - Total Parameters:      {total_params:,}
  - Trainable Parameters:  {trainable_params:,}
  - Model Size (Float32):  {model_size_mb:.2f} MB
  - Input Shape:           {model.input_shape}
  - Output Classes:        {model.output_shape[-1]}
  - Layers:                {len(model.layers)}

For Thesis Comparison Table:
  - Parameters: ~{total_params/1e6:.2f}M ({total_params:,})
  - Size: {model_size_mb:.2f} MB (uncompressed float32)
  - Quantized Size: ~1.13 MB (INT8 quantization)
  - Compression Ratio: {model_size_mb/1.13:.1f}x
""")

print("="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
