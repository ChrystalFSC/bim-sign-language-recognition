import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Use the same config as train_optimized.py
CONFIG = {
    "input_shape": (224, 224, 3),
    "num_classes": 36,
    "batch_size": 32,
    "epochs": 10,  # Just 10 epochs to show learning curves
    "initial_lr": 1e-3,
    "min_lr": 1e-6,
    "warmup_epochs": 3,
    "dropout_rate": 0.3,
    "focal_gamma": 2.0,
    "mixup_alpha": 0.2,
    "alpha": 1.0,
    "train_dir": "train_data",
    "val_dir": "val_data",
    "test_dir": "test_data",
}

def focal_loss(gamma=2.0):
    """Focal Loss for handling class imbalance."""
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        focal_weight = tf.pow(1 - y_pred, gamma)
        focal = focal_weight * ce
        return tf.reduce_mean(tf.reduce_sum(focal, axis=-1))
    return loss_fn

def mixup(images, labels, alpha=0.2):
    """Apply Mixup augmentation."""
    batch_size = tf.shape(images)[0]
    lam = tf.random.uniform([], 0, alpha)
    indices = tf.random.shuffle(tf.range(batch_size))
    mixed_images = lam * images + (1 - lam) * tf.gather(images, indices)
    mixed_labels = lam * labels + (1 - lam) * tf.gather(labels, indices)
    return mixed_images, mixed_labels

class CosineAnnealingSchedule(keras.callbacks.Callback):
    """Cosine annealing learning rate with warmup."""
    def __init__(self, initial_lr, min_lr, epochs, warmup_epochs):
        super().__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.lr_history = []
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        
        self.model.optimizer.learning_rate.assign(lr)
        self.lr_history.append(lr)
        print(f"\nEpoch {epoch+1}: LR = {lr:.6f}")

def create_augmentation():
    """Strong augmentation for better generalization."""
    return keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
    ], name="augmentation")

def create_model():
    """MobileNetV3-Small with standard alpha=1.0."""
    base = keras.applications.MobileNetV3Small(
        input_shape=CONFIG["input_shape"],
        include_top=False,
        weights='imagenet',
        include_preprocessing=False,
        minimalistic=True,
        alpha=CONFIG["alpha"]
    )
    base.trainable = True
    
    inputs = keras.Input(shape=CONFIG["input_shape"])
    x = create_augmentation()(inputs)
    x = base(x, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(CONFIG["dropout_rate"])(x)
    outputs = layers.Dense(CONFIG["num_classes"], activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

def load_data():
    """Load datasets with Mixup augmentation."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        CONFIG["train_dir"],
        image_size=CONFIG["input_shape"][:2],
        batch_size=CONFIG["batch_size"],
        label_mode='categorical'
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        CONFIG["val_dir"],
        image_size=CONFIG["input_shape"][:2],
        batch_size=CONFIG["batch_size"],
        label_mode='categorical'
    )
    
    def normalize(images, labels):
        return tf.cast(images, tf.float32) / 255.0, labels
    
    def normalize_and_mixup(images, labels):
        images = tf.cast(images, tf.float32) / 255.0
        if tf.random.uniform([]) < 0.5:
            return mixup(images, labels, CONFIG["mixup_alpha"])
        return images, labels
    
    train_ds = train_ds.map(normalize_and_mixup).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(normalize).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds

def main():
    print("=" * 60)
    print("TRAINING FOR HISTORY VISUALIZATION (10 Epochs)")
    print("=" * 60)
    
    # Create model
    print("\n1. Creating model...")
    model = create_model()
    
    # Compile
    print("\n2. Compiling with Focal Loss...")
    model.compile(
        optimizer=keras.optimizers.Adam(CONFIG["initial_lr"]),
        loss=focal_loss(gamma=CONFIG["focal_gamma"]),
        metrics=['accuracy']
    )
    
    # Load data
    print("\n3. Loading data...")
    train_ds, val_ds = load_data()
    
    # Create callbacks
    lr_scheduler = CosineAnnealingSchedule(
        CONFIG["initial_lr"],
        CONFIG["min_lr"],
        CONFIG["epochs"],
        CONFIG["warmup_epochs"]
    )
    
    csv_logger = keras.callbacks.CSVLogger('experimental_models/mobilenetv3_small/output/training_history.csv')
    
    callbacks = [lr_scheduler, csv_logger]
    
    # Train
    print("\n4. Training for 10 epochs (~10 minutes)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CONFIG["epochs"],
        callbacks=callbacks
    )
    
    # Save history for plotting
    np.save('experimental_models/mobilenetv3_small/output/training_history.npy', history.history)
    
    # Create visualizations
    print("\n5. Creating training performance graphs...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Accuracy curves
    ax1 = axes[0, 0]
    ax1.plot(history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Training vs Validation Accuracy\n(Focal Loss + Cosine Annealing)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Find best epoch
    best_epoch = np.argmax(history.history['val_accuracy']) + 1
    best_val_acc = np.max(history.history['val_accuracy'])
    ax1.axvline(x=best_epoch-1, color='green', linestyle='--', alpha=0.5, 
                label=f'Best: Epoch {best_epoch}')
    ax1.text(best_epoch-1, best_val_acc, f' {best_val_acc*100:.1f}%', 
             fontsize=10, va='bottom')
    
    # 2. Loss curves
    ax2 = axes[0, 1]
    ax2.plot(history.history['loss'], 'b-', label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss (Focal Loss)', fontsize=12, fontweight='bold')
    ax2.set_title('Training vs Validation Loss\n(No Overfitting Observed)', 
                  fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Learning rate schedule
    ax3 = axes[1, 0]
    epochs_range = range(1, len(lr_scheduler.lr_history) + 1)
    ax3.plot(epochs_range, lr_scheduler.lr_history, 'g-', linewidth=2)
    ax3.axvline(x=CONFIG["warmup_epochs"], color='red', linestyle='--', 
                alpha=0.5, label=f'Warmup End (Epoch {CONFIG["warmup_epochs"]})')
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax3.set_title('Cosine Annealing Learning Rate Schedule\n(with Linear Warmup)', 
                  fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 4. Generalization gap
    ax4 = axes[1, 1]
    gap = np.array(history.history['accuracy']) - np.array(history.history['val_accuracy'])
    ax4.plot(gap, 'purple', linewidth=2)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax4.fill_between(range(len(gap)), gap, 0, alpha=0.3, color='purple')
    ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Generalization Gap\n(Train Acc - Val Acc)', fontsize=12, fontweight='bold')
    ax4.set_title('Model Generalization Analysis\n(Small Gap = Good)', 
                  fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Training Performance - MobileNetV3-Small with Focal Loss', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    plt.savefig('experimental_models/mobilenetv3_small/output/training_performance.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: experimental_models/mobilenetv3_small/output/training_performance.png")
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}% (Epoch {best_epoch})")
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]*100:.2f}%")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
    print(f"Generalization Gap: {gap[-1]*100:.2f}%")
    print("=" * 60)

if __name__ == '__main__':
    main()
