"""
Production disease-classification model: transfer learning on
MobileNetV2 (ImageNet weights), fine-tuned on the crop-disease dataset.

Why MobileNetV2 was chosen over EfficientNet/ResNet for this MVP
(see docs/methodology.md for the full comparison table):
- ~3.5M params vs 11M+ (ResNet18) / 5M+ (EfficientNetB0) -> trains
  faster on a laptop CPU/GPU, which matters for 4 students sharing
  limited compute.
- Well supported, small model file (~14MB), fast inference -> good
  for a live demo and for deploying inside a Docker image without a
  large image size.
- Accuracy on PlantVillage-style leaf datasets is consistently close
  to larger backbones because the classification task (leaf disease
  patterns) is visually simpler than ImageNet-scale classification.

Two-phase training strategy:
  Phase 1: freeze the MobileNetV2 base, train only the new classification
           head (fast, prevents destroying pretrained features early).
  Phase 2: unfreeze the top N layers of the base and fine-tune with a
           much lower learning rate (squeezes out extra accuracy).

Class imbalance: the expanded dataset has classes ranging from ~40 to
~1,300 training images, so both phases use square-root-dampened class
weights (see compute_class_weights below).

Usage:
    python -m app.ml.train_transfer --data_dir ../../data/processed --epochs 10 --fine_tune_epochs 5
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from app.ml.labels import CLASS_NAMES, IMG_SIZE, NUM_CLASSES
from app.ml.train_cnn import load_datasets, get_augmentation_layer


def build_transfer_model(num_classes=NUM_CLASSES, img_size=IMG_SIZE, weights="imagenet"):
    """
    weights: "imagenet" (default, recommended for real training) or
    None (random initialization). `None` exists so this script — and
    the Grad-CAM test suite / CI pipeline that reuses it — can build a
    structurally-identical model without requiring network access to
    download ImageNet weights (useful in sandboxed/offline CI runners;
    see docs/limitations_and_future_work.md for why this matters in
    this project's dev environment). Real accuracy claims in the report
    must always come from a `weights="imagenet"` run.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*img_size, 3),
        include_top=False,
        weights=weights,
    )
    base_model.trainable = False  # Phase 1: freeze

    inputs = layers.Input(shape=(*img_size, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="mobilenetv2_transfer")
    return model, base_model


def compute_class_weights(train_dir):
    """
    Per-class loss weights so rare classes (e.g. ~40 images) are not
    drowned out by large ones (~1,300 images). Weight = sqrt(mean/count):
    a gentler version of the usual inverse-frequency weighting, which
    would over-weight the smallest classes and destabilise training.
    Returns (class_weight dict keyed by class index, counts list).
    """
    train_dir = Path(train_dir)
    counts = [len(list((train_dir / name).glob("*"))) for name in CLASS_NAMES]
    mean_count = float(np.mean(counts))
    class_weight = {i: float((mean_count / max(n, 1)) ** 0.5) for i, n in enumerate(counts)}
    return class_weight, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../../data/processed")
    parser.add_argument("--epochs", type=int, default=10, help="Phase 1 (frozen base) epochs")
    parser.add_argument("--fine_tune_epochs", type=int, default=5, help="Phase 2 (fine-tune) epochs")
    parser.add_argument("--fine_tune_at", type=int, default=100, help="Unfreeze layers from this index onward")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out", default="../../backend/storage/models/disease_model.keras")
    parser.add_argument("--weights", default="imagenet", choices=["imagenet", "none"],
                         help="Use 'none' only for offline/CI smoke tests — see build_transfer_model docstring.")
    parser.add_argument("--no_class_weights", action="store_true",
                         help="Disable class weighting (default: enabled).")
    args = parser.parse_args()

    weights = None if args.weights == "none" else "imagenet"

    train_ds, val_ds, test_ds = load_datasets(args.data_dir, batch_size=args.batch_size)

    if args.no_class_weights:
        class_weight = None
    else:
        class_weight, counts = compute_class_weights(os.path.join(args.data_dir, "train"))
        print("Train images per class:", dict(zip(CLASS_NAMES, counts)))
        print("Class weights:", {CLASS_NAMES[i]: round(w, 2) for i, w in class_weight.items()})

    augment = get_augmentation_layer()
    train_ds_aug = train_ds.map(lambda x, y: (augment(x, training=True), y))

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds_aug = train_ds_aug.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    model, base_model = build_transfer_model(weights=weights)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    ]

    print("\n=== Phase 1: training classification head (frozen base) ===")
    history1 = model.fit(train_ds_aug, validation_data=val_ds, epochs=args.epochs,
                         callbacks=callbacks, class_weight=class_weight)

    print("\n=== Phase 2: fine-tuning top layers of MobileNetV2 ===")
    base_model.trainable = True
    for layer in base_model.layers[:args.fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # much lower LR for fine-tuning
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history2 = model.fit(
        train_ds_aug,
        validation_data=val_ds,
        epochs=args.fine_tune_epochs,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Transfer model (MobileNetV2) — test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.save(args.out)

    # Save label map alongside the model so inference code never has to
    # hardcode class order (keeps ml code and API code loosely coupled).
    label_map_path = os.path.join(os.path.dirname(args.out), "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump({i: name for i, name in enumerate(CLASS_NAMES)}, f, indent=2)

    combined_history = {
        "phase1": history1.history,
        "phase2": history2.history,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
    }
    history_path = os.path.splitext(args.out)[0] + "_history.json"
    with open(history_path, "w") as f:
        json.dump(combined_history, f, indent=2)

    print(f"Saved production model to {args.out}")
    print(f"Saved label map to {label_map_path}")


if __name__ == "__main__":
    main()