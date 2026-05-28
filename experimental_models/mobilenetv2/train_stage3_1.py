"""
Stage 3.1: Terminal Learning Rate Squeeze - MobileNetV2
More aggressive Focal Loss (gamma=5.0) with very low learning rate (1e-6).
"""
import os, numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

CONFIG = {"input_shape": (224, 224, 3), "batch_size": 64, "epochs": 20, "learning_rate": 1e-6,
    "focal_gamma": 5.0, "focal_alpha": 0.25,
    "model_path": "output/best_model_stage3.keras",
    "train_dir": "../../train_data", "val_dir": "../../val_data", "test_dir": "../../test_data",
    "output_dir": "output", "seed": 42}

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.pow(1 - y_pred, gamma)
        focal_loss = weight * cross_entropy
        return tf.reduce_sum(focal_loss, axis=-1)
    return focal_loss_fn

def load_data():
    train_datagen = ImageDataGenerator(rotation_range=15, horizontal_flip=True, brightness_range=[0.8, 1.2], zoom_range=0.1)
    val_datagen = ImageDataGenerator()
    train_gen = train_datagen.flow_from_directory(CONFIG["train_dir"], target_size=CONFIG["input_shape"][:2], batch_size=CONFIG["batch_size"], class_mode='categorical', shuffle=True, seed=CONFIG["seed"])
    val_gen = val_datagen.flow_from_directory(CONFIG["val_dir"], target_size=CONFIG["input_shape"][:2], batch_size=CONFIG["batch_size"], class_mode='categorical', shuffle=False)
    test_gen = val_datagen.flow_from_directory(CONFIG["test_dir"], target_size=CONFIG["input_shape"][:2], batch_size=CONFIG["batch_size"], class_mode='categorical', shuffle=False)
    return train_gen, val_gen, test_gen

def main():
    print("\n" + "="*60 + "\nSTAGE 3.1: THE LAST SQUEEZE - MobileNetV2\nAggressive Focal Loss (gamma=5.0)\n" + "="*60 + "\n")
    np.random.seed(CONFIG["seed"]); tf.random.set_seed(CONFIG["seed"])
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("Mixed Precision: Enabled")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs: {gpus}")
    print(f"Loading model from: {CONFIG['model_path']}")
    model = keras.models.load_model(CONFIG["model_path"],
        custom_objects={'focal_loss_fn': focal_loss(2.0, 0.25)})
    print("Model loaded!")
    print(f"Recompiling with AGGRESSIVE Focal Loss: gamma={CONFIG['focal_gamma']}, lr={CONFIG['learning_rate']}")
    optimizer = keras.optimizers.Adam(learning_rate=CONFIG["learning_rate"])
    model.compile(optimizer=optimizer, loss=focal_loss(CONFIG["focal_gamma"], CONFIG["focal_alpha"]), metrics=['accuracy'])
    train_gen, val_gen, test_gen = load_data()
    callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=os.path.join(CONFIG["output_dir"], 'best_model_stage3_1.keras'), monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-9, verbose=1),
        CSVLogger(os.path.join(CONFIG["output_dir"], 'training_log_stage3_1.csv'))]
    history = model.fit(train_gen, validation_data=val_gen, epochs=CONFIG["epochs"], callbacks=callbacks, verbose=1)
    model = keras.models.load_model(os.path.join(CONFIG["output_dir"], 'best_model_stage3_1.keras'),
        custom_objects={'focal_loss_fn': focal_loss(CONFIG["focal_gamma"], CONFIG["focal_alpha"])})
    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes
    class_names = list(test_gen.class_indices.keys())
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    accuracy = np.mean(y_pred == y_true)
    top_5_indices = np.argsort(predictions, axis=1)[:, -5:]
    top_5_acc = np.mean([1 if y_true[i] in top_5_indices[i] else 0 for i in range(len(y_true))])
    print(f"\nFINAL TEST ACCURACY: {accuracy*100:.2f}%%")
    print(f"FINAL TOP-5 ACCURACY: {top_5_acc*100:.2f}%%")
    # Save per-class performance
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    with open(os.path.join(CONFIG["output_dir"], 'per_class_performance_stage3_1.txt'), 'w') as f:
        f.write(f"STAGE 3.1 - DETAILED PER-CLASS PERFORMANCE - MobileNetV2\n")
        f.write("="*70 + "\n\nOVERALL METRICS\n" + "-"*70 + "\n")
        f.write(f"Top-1 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%%)\n")
        f.write(f"Top-5 Accuracy: {top_5_acc:.4f} ({top_5_acc*100:.2f}%%)\n")
        f.write(f"Test Set Size:  {len(y_true)} images\nNum Classes:    {len(class_names)}\n\n")
        f.write(report + "\n")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Stage 3.1 (Aggressive Focal Loss) - MobileNetV2')
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], 'confusion_matrix_stage3_1.png'), dpi=150); plt.close()
    print(f"\nSTAGE 3.1 COMPLETE! - MobileNetV2 | Model saved: output/best_model_stage3_1.keras")

if __name__ == "__main__":
    main()
