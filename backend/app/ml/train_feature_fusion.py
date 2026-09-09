"""
Feature-level (early) fusion — EXPERIMENTAL, compared against the
baseline weighted late-fusion approach in app/services/fusion.py.

WHAT THIS IS: instead of computing the image model's prediction and the
field-context model's risk assessment separately and combining their
final SCORES (late fusion), this trains a single joint classifier on
the CONCATENATION of:
  - the image model's 128-d penultimate embedding (see
    app/services/image_model.py::extract_image_embedding)
  - the field-context model's processed feature vector (one-hot
    crop/growth-stage + scaled temperature/humidity/rainfall/soil-moisture)
and predicts the disease class directly from that joint representation.

HONESTY NOTE (read before citing any numbers from this script): a real
feature-level fusion model needs a dataset where each image is PAIRED
with the field conditions present when it was actually photographed,
labeled with the true disease outcome. No such public dataset exists
for this project (this is the same limitation documented for the
field-context Random Forest in app/ml/train_field_model.py). To still
implement, train, and honestly evaluate real feature-level fusion CODE
(not just describe it), this script pairs each image in our dataset
with a SYNTHETICALLY GENERATED, class-conditional field-context vector:
non-healthy classes are paired with more disease-favorable conditions
(high humidity, moderate temp, recent rainfall) more often than not,
and healthy classes are paired with more neutral/unfavorable conditions
more often than not — with substantial random noise so the pairing is
informative but not a trivial lookup. This creates a genuine (if
synthetic) joint learning signal so the training/evaluation code can be
exercised end-to-end and REAL numbers can be measured and reported
(see docs/evaluation_plan.md "Sandbox Smoke-Test Results" and
app/ml/compare_fusion_approaches.py). These numbers describe how well
this architecture learns a synthetic pairing rule — they are a code/
concept validation, not a real-world accuracy claim, and must never be
presented as such in the project report.

Usage (run AFTER training the image model, e.g. via train_transfer.py):
    python -m app.ml.train_feature_fusion \
        --image_model ../../backend/storage/models/disease_model.keras \
        --data_dir ../../data/processed \
        --out ../../backend/storage/models/feature_fusion_model.keras
"""
import argparse
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

from app.ml.labels import CLASS_NAMES, CLASS_TO_CROP, IMG_SIZE

GROWTH_STAGES = ["seedling", "vegetative", "flowering", "fruiting", "maturity"]
NUMERIC_FEATURES = ["temperature_c", "humidity_pct", "rainfall_mm", "soil_moisture_pct"]
CATEGORICAL_FEATURES = ["crop_type", "growth_stage"]


def _synthesize_paired_field_context(class_name: str, rng: random.Random) -> dict:
    """Class-conditional synthetic field-context pairing — see module
    docstring for the honesty disclosure on why and how this is used."""
    is_healthy = "Healthy" in class_name
    crop_type = CLASS_TO_CROP[class_name]
    growth_stage = rng.choice(GROWTH_STAGES)

    if is_healthy:
        # bias toward conditions LESS favorable to disease, with noise
        temperature_c = rng.gauss(30, 6)
        humidity_pct = rng.gauss(45, 20)
        rainfall_mm = max(0, rng.gauss(3, 5))
        soil_moisture_pct = rng.gauss(40, 15)
    else:
        # bias toward conditions MORE favorable to disease, with noise
        temperature_c = rng.gauss(23, 5)
        humidity_pct = rng.gauss(80, 12)
        rainfall_mm = max(0, rng.gauss(15, 8))
        soil_moisture_pct = rng.gauss(60, 15)

    return {
        "crop_type": crop_type,
        "growth_stage": growth_stage,
        "temperature_c": float(np.clip(temperature_c, -5, 45)),
        "humidity_pct": float(np.clip(humidity_pct, 0, 100)),
        "rainfall_mm": float(np.clip(rainfall_mm, 0, 200)),
        "soil_moisture_pct": float(np.clip(soil_moisture_pct, 0, 100)),
    }


def _load_image_paths_and_labels(data_dir: str, split: str):
    paths, labels = [], []
    split_dir = os.path.join(data_dir, split)
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            paths.append(os.path.join(cls_dir, fname))
            labels.append(cls)
    return paths, labels


def _build_embedding_extractor(image_model_path: str):
    model = tf.keras.models.load_model(image_model_path)
    dense_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
    embedding_layer = dense_layers[0]
    return tf.keras.Model(inputs=model.inputs, outputs=embedding_layer.output)


def _extract_embeddings(embedding_model, image_paths, batch_size=16):
    from PIL import Image
    embeddings = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_arrs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB").resize(IMG_SIZE)
            batch_arrs.append(np.array(img, dtype=np.float32))
        batch_arrs = np.stack(batch_arrs)
        batch_arrs = tf.keras.applications.mobilenet_v2.preprocess_input(batch_arrs)
        batch_embeddings = embedding_model.predict(batch_arrs, verbose=0)
        embeddings.append(batch_embeddings)
    return np.concatenate(embeddings, axis=0)


def build_fusion_dataset(data_dir: str, image_model_path: str, split: str, seed: int = 42):
    rng = random.Random(seed)
    paths, labels = _load_image_paths_and_labels(data_dir, split)
    embedding_model = _build_embedding_extractor(image_model_path)
    embeddings = _extract_embeddings(embedding_model, paths)

    field_rows = [_synthesize_paired_field_context(lbl, rng) for lbl in labels]
    field_df = pd.DataFrame(field_rows)

    return embeddings, field_df, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_model", required=True, help="Path to a trained image model (.keras)")
    parser.add_argument("--data_dir", default="../../data/processed")
    parser.add_argument("--out", default="../../backend/storage/models/feature_fusion_model.keras")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    print("Building training set (image embeddings + synthetic paired field context)...")
    train_emb, train_field_df, train_labels = build_fusion_dataset(args.data_dir, args.image_model, "train")
    val_emb, val_field_df, val_labels = build_fusion_dataset(args.data_dir, args.image_model, "val")
    test_emb, test_field_df, test_labels = build_fusion_dataset(args.data_dir, args.image_model, "test")

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    train_field_feat = preprocessor.fit_transform(train_field_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    val_field_feat = preprocessor.transform(val_field_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    test_field_feat = preprocessor.transform(test_field_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])

    train_field_feat = np.asarray(train_field_feat)
    val_field_feat = np.asarray(val_field_feat)
    test_field_feat = np.asarray(test_field_feat)

    label_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    y_train = np.array([label_to_idx[l] for l in train_labels])
    y_val = np.array([label_to_idx[l] for l in val_labels])
    y_test = np.array([label_to_idx[l] for l in test_labels])

    X_train = np.concatenate([train_emb, train_field_feat], axis=1)
    X_val = np.concatenate([val_emb, val_field_feat], axis=1)
    X_test = np.concatenate([test_emb, test_field_feat], axis=1)

    print(f"Joint feature vector size: {X_train.shape[1]} "
          f"(image embedding: {train_emb.shape[1]} + field features: {train_field_feat.shape[1]})")

    joint_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax"),
    ], name="feature_level_fusion")

    joint_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                         loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    joint_model.summary()

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)]
    joint_model.fit(X_train, y_train, validation_data=(X_val, y_val),
                     epochs=args.epochs, batch_size=8, callbacks=callbacks, verbose=2)

    test_loss, test_acc = joint_model.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(joint_model.predict(X_test, verbose=0), axis=1)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    print(f"\nFeature-level fusion — test_acc={test_acc:.4f} weighted_f1={f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joint_model.save(args.out)
    preprocessor_path = os.path.splitext(args.out)[0] + "_field_preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)

    print(f"Saved joint fusion model to {args.out}")
    print(f"Saved field preprocessor to {preprocessor_path}")

    return {"test_accuracy": test_acc, "test_f1_weighted": f1}


if __name__ == "__main__":
    main()
