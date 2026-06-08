import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

# TensorFlow Lite conversion may try to import optional JAX when it is installed.
# Some Python environments have a JAX/ml_dtypes version mismatch, which crashes
# TensorFlow import even though this training script does not use JAX.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.modules.setdefault("jax", None)
sys.modules.setdefault("jaxlib", None)

import tensorflow as tf


def load_labels(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_dataset(csv_path: Path, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    label_to_index = {label: index for index, label in enumerate(labels)}
    rows: list[list[float]] = []
    targets: list[int] = []

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        feature_names = [f"f{i}" for i in range(63)]
        missing = [name for name in ["label", *feature_names] if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"CSV is missing columns: {', '.join(missing)}")

        for row in reader:
            label = row["label"].strip()
            if label not in label_to_index:
                continue
            rows.append([float(row[name]) for name in feature_names])
            targets.append(label_to_index[label])

    if not rows:
        raise ValueError("No usable landmark rows found.")

    return np.asarray(rows, dtype=np.float32), np.asarray(targets, dtype=np.int64)


def split_dataset(
    x: np.ndarray,
    y: np.ndarray,
    validation_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for label in np.unique(y):
        indices = np.where(y == label)[0]
        rng.shuffle(indices)
        val_count = max(1, int(round(len(indices) * validation_ratio))) if len(indices) > 1 else 0
        val_indices.extend(indices[:val_count].tolist())
        train_indices.extend(indices[val_count:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return x[train_indices], y[train_indices], x[val_indices], y[val_indices]


def build_model(num_classes: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(63,), name="landmarks"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def export_tflite(model: tf.keras.Model, output_path: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BIM landmark classifier from CSV.")
    parser.add_argument("--csv", required=True, type=Path, help="Path to landmarks.csv from the phone.")
    parser.add_argument("--labels", default=Path("assets/classes.txt"), type=Path)
    parser.add_argument("--output", default=Path("assets/landmark_classifier.tflite"), type=Path)
    parser.add_argument("--metrics", default=Path("landmark_training_metrics.json"), type=Path)
    parser.add_argument("--epochs", default=120, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    labels = load_labels(args.labels)
    x, y = load_dataset(args.csv, labels)
    x_train, y_train, x_val, y_val = split_dataset(x, y, validation_ratio=0.2, seed=args.seed)

    if len(x_val) == 0:
        raise ValueError("Need at least two samples for at least one class to create validation data.")

    model = build_model(num_classes=len(labels))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=15,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=2,
    )

    val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
    export_tflite(model, args.output)

    metrics = {
        "samples": int(len(x)),
        "train_samples": int(len(x_train)),
        "validation_samples": int(len(x_val)),
        "classes": len(labels),
        "validation_loss": float(val_loss),
        "validation_accuracy": float(val_accuracy),
        "epochs_ran": len(history.history["loss"]),
        "output": str(args.output),
    }
    args.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
