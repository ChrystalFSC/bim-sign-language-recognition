"""
Stage 1: Feature Extraction Training - ResNet18 (Standardised Frozen Base)
Only train the top classification layers while keeping ResNet18 base frozen.
Note: ResNet18 has no pretrained Keras weights, so base is randomly initialized.
This follows the SAME frozen-base methodology as all other models.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator

CONFIG = {
    "input_shape": (224, 224, 3),
    "batch_size": 64,
    "epochs": 20,
    "learning_rate": 1e-3,
    "train_dir": "../../train_data",
    "val_dir": "../../val_data",
    "test_dir": "../../test_data",
    "output_dir": "output",
    "seed": 42
}

def load_data():
    train_datagen = ImageDataGenerator(rotation_range=10, horizontal_flip=True, zoom_range=0.1)
    val_datagen = ImageDataGenerator()
    train_gen = train_datagen.flow_from_directory(CONFIG["train_dir"], target_size=CONFIG["input_shape"][:2],
        batch_size=CONFIG["batch_size"], class_mode='categorical', shuffle=True, seed=CONFIG["seed"])
    val_gen = val_datagen.flow_from_directory(CONFIG["val_dir"], target_size=CONFIG["input_shape"][:2],
        batch_size=CONFIG["batch_size"], class_mode='categorical', shuffle=False)
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
    """Create ResNet18 model with FROZEN base for feature extraction.
    No pretrained weights — base is randomly initialized.
    """
    base_input = keras.Input(shape=CONFIG["input_shape"])
    x = keras.layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False)(base_input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)
    x = basic_block(x, 64)
    x = basic_block(x, 64)
    x = basic_block(x, 128, stride=2)
    x = basic_block(x, 128)
    x = basic_block(x, 256, stride=2)
    x = basic_block(x, 256)
    x = basic_block(x, 512, stride=2)
    x = basic_block(x, 512)
    base = keras.Model(base_input, x, name='resnet18_base')

    # FREEZE the base model (standardised methodology)
    base.trainable = False

    inputs = keras.Input(shape=CONFIG["input_shape"])
    x = keras.layers.Rescaling(1./255)(inputs)
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(36, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model

def main():
    print("\n" + "="*60)
    print("STAGE 1: FEATURE EXTRACTION - ResNet18 (Frozen Base)")
    print("Base model FROZEN - Only training top layers")
    print("Note: No pretrained weights (random initialization)")
    print("="*60 + "\n")

    np.random.seed(CONFIG["seed"])
    tf.random.set_seed(CONFIG["seed"])

    model = create_model()
    print(f"Total params: {model.count_params():,}")
    print(f"Trainable params: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

    model.compile(optimizer=keras.optimizers.Adam(CONFIG["learning_rate"]),
        loss='categorical_crossentropy', metrics=['accuracy'])

    train_gen, val_gen = load_data()

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=os.path.join(CONFIG["output_dir"], 'best_model_stage1.keras'),
            monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        CSVLogger(os.path.join(CONFIG["output_dir"], 'training_log_stage1.csv'))
    ]

    history = model.fit(train_gen, validation_data=val_gen, epochs=CONFIG["epochs"],
        callbacks=callbacks, verbose=1)

    best_val_acc = max(history.history['val_accuracy'])
    print(f"\nSTAGE 1 COMPLETE! - ResNet18 (Frozen Base)")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print("Next: Run train_stage2.py")

if __name__ == "__main__":
    main()
