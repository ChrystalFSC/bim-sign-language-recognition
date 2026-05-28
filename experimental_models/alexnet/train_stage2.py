"""
Stage 2: Fine-Tuning Training - AlexNet
Unfreeze the last 50% of layers and fine-tune with lower learning rate.
"""
import os, numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator

CONFIG = {"input_shape": (224, 224, 3), "batch_size": 64, "epochs": 30, "learning_rate": 5e-5,
    "model_path": "output/best_model_stage1.keras",
    "train_dir": "../../train_data", "val_dir": "../../val_data", "test_dir": "../../test_data",
    "output_dir": "output", "seed": 42}

def load_data():
    train_datagen = ImageDataGenerator(rotation_range=15, horizontal_flip=True, brightness_range=[0.8, 1.2], zoom_range=0.15, width_shift_range=0.1, height_shift_range=0.1)
    val_datagen = ImageDataGenerator()
    train_gen = train_datagen.flow_from_directory(CONFIG["train_dir"], target_size=CONFIG["input_shape"][:2], batch_size=CONFIG["batch_size"], class_mode='categorical', shuffle=True, seed=CONFIG["seed"])
    val_gen = val_datagen.flow_from_directory(CONFIG["val_dir"], target_size=CONFIG["input_shape"][:2], batch_size=CONFIG["batch_size"], class_mode='categorical', shuffle=False)
    test_gen = val_datagen.flow_from_directory(CONFIG["test_dir"], target_size=CONFIG["input_shape"][:2], batch_size=CONFIG["batch_size"], class_mode='categorical', shuffle=False)
    return train_gen, val_gen, test_gen

def main():
    print("\n" + "="*60 + "\nSTAGE 2: FINE-TUNING - AlexNet\nUnfreezing last 50%% of layers\n" + "="*60 + "\n")
    np.random.seed(CONFIG["seed"]); tf.random.set_seed(CONFIG["seed"])
    print(f"Loading model from: {CONFIG['model_path']}")
    model = keras.models.load_model(CONFIG["model_path"])
    print("Model loaded!")
    # Partial unfreezing - last 50%%
    unfreeze_limit = len(model.layers) // 2
    for i, layer in enumerate(model.layers):
        layer.trainable = True if i >= unfreeze_limit else False
    trainable_count = sum([1 for layer in model.layers if layer.trainable])
    print(f"Unfrozen {trainable_count}/{len(model.layers)} layers (last 50%%)")
    print(f"Total params: {model.count_params():,}")
    print(f"Trainable params: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    model.compile(optimizer=keras.optimizers.Adam(CONFIG["learning_rate"]), loss='categorical_crossentropy', metrics=['accuracy'])
    train_gen, val_gen, test_gen = load_data()
    callbacks = [EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=os.path.join(CONFIG["output_dir"], 'best_model_stage2.keras'), monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-8, verbose=1),
        CSVLogger(os.path.join(CONFIG["output_dir"], 'training_log_stage2.csv'))]
    print("\nStarting Stage 2 Training (Fine-Tuning) - AlexNet\n")
    history = model.fit(train_gen, validation_data=val_gen, epochs=CONFIG["epochs"], callbacks=callbacks, verbose=1)
    model = keras.models.load_model(os.path.join(CONFIG["output_dir"], 'best_model_stage2.keras'))
    test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    best_val_acc = max(history.history['val_accuracy'])
    print(f"\nSTAGE 2 COMPLETE! - AlexNet")
    print(f"Best Val Acc: {best_val_acc*100:.2f}%% | Test Acc: {test_acc*100:.2f}%%")
    print("Next: Run train_stage3.py")

if __name__ == "__main__":
    main()
