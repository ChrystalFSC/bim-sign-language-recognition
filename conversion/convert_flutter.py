"""
Convert model to TFLite WITHOUT TF Select Ops
By rebuilding the model architecture fresh and copying weights
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

MODEL_PATH = "experimental_models/mobilenetv3_small/output/best_model_stage3_1.keras"
OUTPUT_DIR = "experimental_models/mobilenetv3_small/output"

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.pow(1 - y_pred, gamma)
        return tf.reduce_sum(weight * cross_entropy, axis=-1)
    return focal_loss_fn

def build_mobilenetv3_small(num_classes=36, dropout_rate=0.2):
    """Build clean MobileNetV3-Small without mixed precision."""
    
    # Ensure float32
    tf.keras.mixed_precision.set_global_policy('float32')
    
    # Load base model
    base_model = keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        include_preprocessing=False  # We handle preprocessing in app
    )
    
    # Build model
    inputs = keras.Input(shape=(224, 224, 3), dtype=tf.float32)
    
    # Normalize like ImageNet
    x = inputs  # Preprocessing done in app
    
    # Base model
    x = base_model(x, training=False)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = keras.Model(inputs, outputs)
    return model

def main():
    print("="*60)
    print("FLUTTER-COMPATIBLE TFLITE CONVERSION")
    print("="*60)
    
    # Disable mixed precision
    tf.keras.mixed_precision.set_global_policy('float32')
    
    # Load trained model
    print("\n1. Loading trained model...")
    trained_model = keras.models.load_model(
        MODEL_PATH,
        custom_objects={'focal_loss_fn': focal_loss(5.0, 0.25)}
    )
    print("   Loaded!")
    
    # Build fresh model (float32 only)
    print("\n2. Building fresh float32 model...")
    fresh_model = build_mobilenetv3_small(num_classes=36)
    
    # Copy weights from trained model
    print("\n3. Copying weights...")
    for i, layer in enumerate(fresh_model.layers):
        try:
            if layer.weights:
                # Find matching layer in trained model
                for trained_layer in trained_model.layers:
                    if trained_layer.name == layer.name:
                        layer.set_weights(trained_layer.get_weights())
                        break
        except:
            pass
    
    # Copy final Dense layer weights specifically
    trained_dense = trained_model.layers[-1]
    fresh_dense = fresh_model.layers[-1]
    if trained_dense.weights and fresh_dense.weights:
        try:
            weights = [w.numpy().astype(np.float32) for w in trained_dense.weights]
            fresh_dense.set_weights(weights)
            print("   Dense layer weights copied!")
        except Exception as e:
            print(f"   Warning: {e}")
    
    fresh_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("   Model ready!")
    
    # Convert to TFLite
    print("\n4. Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(fresh_model)
    
    # Use ONLY builtin ops (no SELECT_TF_OPS)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    
    tflite_model = converter.convert()
    
    output_path = os.path.join(OUTPUT_DIR, "model_flutter.tflite")
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"   ✅ Saved: {output_path} ({size_mb:.2f} MB)")
    
    # Verify
    print("\n5. Verifying...")
    interpreter = tf.lite.Interpreter(model_path=output_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"   Input: {input_details[0]['shape']} {input_details[0]['dtype']}")
    print(f"   Output: {output_details[0]['shape']}")
    
    # Test
    test_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    print(f"   ✅ Inference works! Output sum: {output.sum():.4f}")
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print(f"\nCopy this file to Flutter:")
    print(f"  output/model_flutter.tflite")

if __name__ == "__main__":
    main()
