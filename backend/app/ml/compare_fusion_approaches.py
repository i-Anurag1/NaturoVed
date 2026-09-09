"""
Experimental comparison: Image-only vs. Weighted Late Fusion vs.
Feature-Level Fusion.

HONESTY NOTE (read before citing any numbers from this script in your
report): this comparison runs on the SYNTHETIC benchmark dataset (see
app/ml/synthetic_benchmark.py) using a MobileNetV2 backbone initialized
with RANDOM weights (no ImageNet pretraining), because this project's
sandboxed development environment has no network access to download
either PlantVillage or Keras' pretrained ImageNet weights (both hosts
are outside the environment's network allowlist — see
docs/limitations_and_future_work.md). The purpose of this script is to
prove the three approaches' CODE is correct and produces genuine,
measured, non-fabricated numbers end-to-end — not to claim real-world
disease-classification accuracy. Re-run this exact script (unmodified)
after training on real PlantVillage data with real ImageNet weights
(`--weights imagenet` in train_transfer.py) to get numbers suitable
for the academic report's "Real PlantVillage Results" section.

WHAT IS COMPARED:
  1. Image-only: MobileNetV2 top-1 prediction, no field context at all.
  2. Weighted late fusion (production default, app/services/fusion.py):
     by construction this uses the SAME top-1 label as image-only (it
     only adjusts the confidence score) — so its "accuracy" is
     identical to image-only by design. What we DO measure and report
     is a calibration metric: the average final_confidence gap between
     correct and incorrect predictions. A well-calibrated fusion should
     widen this gap (i.e. be more confident when right, less confident
     when wrong) compared to the raw image-only confidence.
  3. Feature-level fusion (experimental, app/ml/train_feature_fusion.py):
     a SEPARATELY TRAINED joint classifier on [image embedding + field
     features] — this CAN have different top-1 accuracy than image-only,
     and we measure it directly.

Usage:
    python -m app.ml.compare_fusion_approaches \
        --image_model ../../backend/storage/models/disease_model_synthtest.keras \
        --feature_fusion_model ../../backend/storage/models/feature_fusion_synthtest.keras \
        --field_preprocessor ../../backend/storage/models/feature_fusion_synthtest_field_preprocessor.joblib \
        --data_dir ../../data/synthetic_benchmark
"""
import argparse
import json
import os
import random
import numpy as np
import tensorflow as tf
import joblib
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

from app.ml.labels import CLASS_NAMES
from app.ml.train_feature_fusion import _synthesize_paired_field_context, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from app.services import fusion as fusion_service


def _load_test_set(data_dir):
    paths, labels = [], []
    test_dir = os.path.join(data_dir, "test")
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            paths.append(os.path.join(cls_dir, fname))
            labels.append(cls)
    return paths, labels


def _predict_image_only(model, paths):
    predictions, confidences = [], []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((224, 224))
        arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
        probs = model.predict(arr, verbose=0)[0]
        top_idx = int(np.argmax(probs))
        predictions.append(CLASS_NAMES[top_idx])
        confidences.append(float(probs[top_idx]))
    return predictions, confidences


def _extract_embeddings(embedding_model, paths, batch_size=16):
    embeddings = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        batch_arrs = [np.array(Image.open(p).convert("RGB").resize((224, 224)), dtype=np.float32) for p in batch_paths]
        batch_arrs = tf.keras.applications.mobilenet_v2.preprocess_input(np.stack(batch_arrs))
        embeddings.append(embedding_model.predict(batch_arrs, verbose=0))
    return np.concatenate(embeddings, axis=0)


def run_comparison(image_model_path, feature_fusion_model_path, field_preprocessor_path, data_dir, seed=123):
    rng = random.Random(seed)
    paths, true_labels = _load_test_set(data_dir)
    if not paths:
        raise RuntimeError(f"No test images found under {data_dir}/test — generate the synthetic "
                            "benchmark first: python -m app.ml.synthetic_benchmark")

    print(f"Loaded {len(paths)} test images across {len(set(true_labels))} classes.\n")

    # --- 1. Image-only ---
    image_model = tf.keras.models.load_model(image_model_path)
    image_preds, image_confidences = _predict_image_only(image_model, paths)
    image_only_acc = accuracy_score(true_labels, image_preds)
    image_only_f1 = f1_score(true_labels, image_preds, average="weighted", zero_division=0)

    # --- 2. Weighted late fusion (calibration-only comparison) ---
    field_rows = [_synthesize_paired_field_context(lbl, rng) for lbl in true_labels]
    late_fusion_confidences = []
    from app.services import field_context
    for pred_label, img_conf, field_row in zip(image_preds, image_confidences, field_rows):
        risk_level, risk_score = "Medium", 0.5
        try:
            result = field_context.assess_field_risk(**field_row)
            risk_level, risk_score = result["risk_level"], result["risk_score"]
        except Exception:
            pass
        fused = fusion_service.fuse_predictions(pred_label, img_conf, risk_level, risk_score)
        late_fusion_confidences.append(fused["final_confidence"])

    late_fusion_acc = image_only_acc  # identical by construction — see module docstring
    correct_mask = np.array([p == t for p, t in zip(image_preds, true_labels)])
    image_conf_arr = np.array(image_confidences)
    late_conf_arr = np.array(late_fusion_confidences)

    def _gap(conf_arr):
        if correct_mask.sum() == 0 or (~correct_mask).sum() == 0:
            return None
        return float(conf_arr[correct_mask].mean() - conf_arr[~correct_mask].mean())

    image_only_calibration_gap = _gap(image_conf_arr)
    late_fusion_calibration_gap = _gap(late_conf_arr)

    # --- 3. Feature-level fusion ---
    dense_layers = [l for l in image_model.layers if isinstance(l, tf.keras.layers.Dense)]
    embedding_model = tf.keras.Model(inputs=image_model.inputs, outputs=dense_layers[0].output)
    embeddings = _extract_embeddings(embedding_model, paths)

    field_preprocessor = joblib.load(field_preprocessor_path)
    import pandas as pd
    field_df = pd.DataFrame(field_rows)
    field_feat = np.asarray(field_preprocessor.transform(field_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]))

    joint_input = np.concatenate([embeddings, field_feat], axis=1)
    fusion_model = tf.keras.models.load_model(feature_fusion_model_path)
    fusion_probs = fusion_model.predict(joint_input, verbose=0)
    fusion_preds = [CLASS_NAMES[i] for i in np.argmax(fusion_probs, axis=1)]

    feature_fusion_acc = accuracy_score(true_labels, fusion_preds)
    feature_fusion_f1 = f1_score(true_labels, fusion_preds, average="weighted", zero_division=0)

    print("=" * 70)
    print("RESULTS (synthetic benchmark, random-init backbone — see module docstring)")
    print("=" * 70)
    print(f"{'Approach':<28}{'Accuracy':<12}{'Weighted F1':<14}{'Confidence calibration gap':<28}")
    print(f"{'Image-only':<28}{image_only_acc:<12.4f}{image_only_f1:<14.4f}"
          f"{'' if image_only_calibration_gap is None else f'{image_only_calibration_gap:+.4f}':<28}")
    print(f"{'Weighted late fusion':<28}{late_fusion_acc:<12.4f}{image_only_f1:<14.4f}"
          f"{'' if late_fusion_calibration_gap is None else f'{late_fusion_calibration_gap:+.4f}':<28}")
    print(f"{'Feature-level fusion':<28}{feature_fusion_acc:<12.4f}{feature_fusion_f1:<14.4f}{'n/a':<28}")
    print()
    print("Note: 'Confidence calibration gap' = mean confidence on CORRECT predictions minus "
          "mean confidence on INCORRECT predictions. A larger positive gap means the model is "
          "more confident when right than when wrong (better calibrated), independent of accuracy.")
    print()
    print("Late fusion's accuracy is IDENTICAL to image-only by construction (see fusion.py docstring — "
          "it is a confidence-adjustment, decision-level fusion, not a label-changing fusion).")

    results = {
        "image_only": {"accuracy": image_only_acc, "f1_weighted": image_only_f1,
                        "calibration_gap": image_only_calibration_gap},
        "late_fusion": {"accuracy": late_fusion_acc, "f1_weighted": image_only_f1,
                         "calibration_gap": late_fusion_calibration_gap},
        "feature_fusion": {"accuracy": feature_fusion_acc, "f1_weighted": feature_fusion_f1},
        "note": "Synthetic benchmark, random-init MobileNetV2 backbone (no ImageNet weights available "
                "in the sandboxed dev environment). Re-run on real PlantVillage + ImageNet weights "
                "before citing in the academic report.",
    }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_model", required=True)
    parser.add_argument("--feature_fusion_model", required=True)
    parser.add_argument("--field_preprocessor", required=True)
    parser.add_argument("--data_dir", default="../data/synthetic_benchmark",
                         help="Relative to backend/ when run via `python -m app.ml.compare_fusion_approaches`.")
    parser.add_argument("--out_json", default=None)
    args = parser.parse_args()

    results = run_comparison(args.image_model, args.feature_fusion_model, args.field_preprocessor, args.data_dir)

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {args.out_json}")
