"""
Stage 1: Full Training - ResNet18
All layers trainable from the start (no pretrained weights available).
Hyperparameter Optimisation: Unlike pretrained models, ResNet18 requires
all layers unfrozen from Stage 1 to ensure proper convergence.
"""
import os, numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator

CONFIG = {
    "input_shape": (224, 224, 3),
    "batch_size": 64,
    "epochs": 30,
    "learning_rate": 1e-3,
    "train_dir": "../../train_data",
    "val_dir": "../../val_data",
    "test_dir": "../../test_data",
    "output_dir": "output",
    "seed": 42
}

def load_data():
    """Load data with augmentation"""
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        horizontal_flip=True,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.8, 1.2]
    )
    val_datagen = ImageDataGenerator()

    train_gen = train_datagen.flow_from_directory(
        CONFIG["train_dir"], target_size=CONFIG["input_shape"][:2],
        batch_size=CONFIG["batch_size"], class_mode='categorical',
        shuffle=True, seed=CONFIG["seed"])
    val_gen = val_datagen.flow_from_directory(
        CONFIG["val_dir"], target_size=CONFIG["input_shape"][:2],
        batch_size=CONFIG["batch_size"], class_mode='categorical',
        shuffle=False)
    return train_gen, val_gen

def basic_block(x, filters, stride=1):
    """ResNet BasicBlock with skip connection"""
    identity = x

    out = keras.layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False)(x)
    out = keras.layers.BatchNormalization()(out)
    out = keras.layers.ReLU()(out)

    out = keras.layers.Conv2D(filters, 3, strides=1, padding='same', use_bias=False)(out)
    out = keras.layers.BatchNormalization()(out)

    if stride != 1 or int(identity.shape[-1]) != filters:
        identity = keras.layers.Conv2D(filters, 1, strides=stride, use_bias=False)(identity)
        identity = keras.layers.BatchNormalization()(identity)

    out = keras.layers.Add()([out, identity])
    out = keras.layers.ReLU()(out)
    return out

def create_model():
    """Create ResNet18 model with ALL layers trainable.
    ResNet18 architecture built manually (not available in tf.keras.applications).
    No pretrained weights — all layers must be trainable from the start.

    Hyperparameter Optimisation Note:
    Unlike pretrained models where Stage 1 freezes the base to preserve
    learned features, ResNet18 has no pretrained weights. Freezing random
    weights would prevent convergence. Therefore, all layers are trainable.
    """
    # Build ResNet18 base
    base_input = keras.Input(shape=CONFIG["input_shape"])
    x = keras.layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False)(base_input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # Layer 1: 2 BasicBlocks, 64 filters
    x = basic_block(x, 64)
    x = basic_block(x, 64)

    # Layer 2: 2 BasicBlocks, 128 filters, stride 2
    x = basic_block(x, 128, stride=2)
    x = basic_block(x, 128)

    # Layer 3: 2 BasicBlocks, 256 filters, stride 2
    x = basic_block(x, 256, stride=2)
    x = basic_block(x, 256)

    # Layer 4: 2 BasicBlocks, 512 filters, stride 2
    x = basic_block(x, 512, stride=2)
    x = basic_block(x, 512)

    base = keras.Model(base_input, x, name='resnet18_base')

    # ALL layers trainable (no pretrained weights to preserve)
    base.trainable = True

    inputs = keras.Input(shape=CONFIG["input_shape"])
    x = keras.layers.Rescaling(1./255)(inputs)
    x = base(x, training=True)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(36, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model

def main():
    print("\n" + "="*60)
    print("STAGE 1: FULL TRAINING - ResNet18")
    print("ALL layers trainable (no pretrained weights)")
    print("Hyperparameter Optimisation: LR=1e-3, 30 epochs")
    print("="*60 + "\n")

    np.random.seed(CONFIG["seed"])
    tf.random.set_seed(CONFIG["seed"])

    print("Creating ResNet18 model (all layers trainable)...")
    model = create_model()
    print(f"Total parameters: {model.count_params():,}")
    print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

    model.compile(
        optimizer=keras.optimizers.Adam(CONFIG["learning_rate"]),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nLoading data...")
    train_gen, val_gen = load_data()

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=os.path.join(CONFIG["output_dir"], 'best_model_stage1.keras'),
            monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        CSVLogger(os.path.join(CONFIG["output_dir"], 'training_log_stage1.csv'))
    ]

    print("\n" + "="*60)
    print("Starting Stage 1 Training - ResNet18 (All Layers Unfrozen)")
    print("="*60 + "\n")

    history = model.fit(train_gen, validation_data=val_gen, epochs=CONFIG["epochs"],
        callbacks=callbacks, verbose=1)

    best_val_acc = max(history.history['val_accuracy'])
    print("\n" + "="*60)
    print(f"STAGE 1 COMPLETE! - ResNet18")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"Model saved: output/best_model_stage1.keras")
    print("\nNext: Run train_stage2.py")
    print("="*60)

if __name__ == "__main__":
    main()
