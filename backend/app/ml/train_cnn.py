"""
Baseline: a small CNN trained from scratch on the disease image dataset.

Purpose: gives an honest lower-bound baseline to compare the
transfer-learning model against (required for the "experimental
comparison" section of the report). This is NOT the model used in
production inference (see train_transfer.py for that) but its metrics
must be reported alongside the transfer-learning model's metrics.

Usage:
    python -m app.ml.train_cnn --data_dir ../../data/processed --epochs 15
"""
import argparse
import json
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from app.ml.labels import CLASS_NAMES, IMG_SIZE, NUM_CLASSES


def build_cnn_baseline(num_classes=NUM_CLASSES, img_size=IMG_SIZE):
    model = models.Sequential([
        layers.Input(shape=(*img_size, 3)),
        layers.Rescaling(1.0 / 255),

        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ], name="cnn_baseline")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_augmentation_layer():
    """Basic augmentation: random flip/rotation/zoom/contrast.
    Applied only to the training set."""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.15),
    ], name="augmentation")


def load_datasets(data_dir, img_size=IMG_SIZE, batch_size=32):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "train"),
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "val"),
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "test"),
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_ds, val_ds, test_ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../../data/processed")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out", default="../../backend/storage/models/cnn_baseline.keras")
    args = parser.parse_args()

    train_ds, val_ds, test_ds = load_datasets(args.data_dir, batch_size=args.batch_size)

    augment = get_augmentation_layer()
    train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y))

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    model = build_cnn_baseline()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    test_loss, test_acc = model.evaluate(test_ds)
    print(f"CNN baseline — test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.save(args.out)

    history_path = os.path.splitext(args.out)[0] + "_history.json"
    with open(history_path, "w") as f:
        json.dump(history.history, f, indent=2)

    print(f"Saved model to {args.out}")


if __name__ == "__main__":
    main()
