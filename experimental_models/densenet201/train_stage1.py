"""
Stage 1: Feature Extraction Training - DenseNet201
Only train the top classification layers while keeping DenseNet201 base frozen.
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
    train_gen = train_datagen.flow_from_directory(CONFIG["train_dir"], target_size=CONFIG["input_shape"][:2], batch_size=CONFIG["batch_size"], class_mode='categorical', shuffle=True, seed=CONFIG["seed"])
    val_gen = val_datagen.flow_from_directory(CONFIG["val_dir"], target_size=CONFIG["input_shape"][:2], batch_size=CONFIG["batch_size"], class_mode='categorical', shuffle=False)
    return train_gen, val_gen

def create_model():
    base = keras.applications.DenseNet201(input_shape=CONFIG["input_shape"], include_top=False, weights='imagenet')
    base.trainable = False
    inputs = keras.Input(shape=CONFIG["input_shape"])
    x = keras.layers.Rescaling(1./255)(inputs)
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(36, activation='softmax')(x)
    return keras.Model(inputs, outputs)

def main():
    print("\n" + "="*60)
    print("STAGE 1: FEATURE EXTRACTION TRAINING - DenseNet201")
    print("Base model FROZEN - Only training top layers")
    print("="*60 + "\n")
    np.random.seed(CONFIG["seed"])
    tf.random.set_seed(CONFIG["seed"])
    model = create_model()
    print(f"Total parameters: {model.count_params():,}")
    print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    model.compile(optimizer=keras.optimizers.Adam(CONFIG["learning_rate"]), loss='categorical_crossentropy', metrics=['accuracy'])
    train_gen, val_gen = load_data()
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=os.path.join(CONFIG["output_dir"], 'best_model_stage1.keras'), monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        CSVLogger(os.path.join(CONFIG["output_dir"], 'training_log_stage1.csv'))
    ]
    history = model.fit(train_gen, validation_data=val_gen, epochs=CONFIG["epochs"], callbacks=callbacks, verbose=1)
    best_val_acc = max(history.history['val_accuracy'])
    print(f"\nSTAGE 1 COMPLETE! - DenseNet201 | Best Val Accuracy: {best_val_acc*100:.2f}%")
    print("Next: Run train_stage2.py")

if __name__ == "__main__":
    main()
