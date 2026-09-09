"""
Evaluate a trained model on the held-out test set and produce:
  - Accuracy, macro/weighted Precision, Recall, F1-score
  - Full sklearn classification report (per-class)
  - Confusion matrix (saved as CSV and PNG)

Usage:
    python -m app.ml.evaluate --model ../../backend/storage/models/disease_model.keras \
                               --data_dir ../../data/processed
"""
import argparse
import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support,
)
from app.ml.labels import CLASS_NAMES, IMG_SIZE


def evaluate_model(model_path: str, data_dir: str, split: str = "test", out_dir: str = "./eval_results"):
    os.makedirs(out_dir, exist_ok=True)
    model = tf.keras.models.load_model(model_path)

    ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, split),
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=32,
        shuffle=False,
    )

    y_true, y_pred = [], []
    for images, labels in ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1).tolist())
        y_true.extend(labels.numpy().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted):    {recall:.4f}")
    print(f"F1-score (weighted):  {f1:.4f}")
    print("\nPer-class report:\n", report)

    # Save artifacts for the project report
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"accuracy": acc, "precision_weighted": precision,
                    "recall_weighted": recall, "f1_weighted": f1}, f, indent=2)

    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    np.savetxt(os.path.join(out_dir, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=90)
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
        print(f"Saved confusion matrix plot to {out_dir}/confusion_matrix.png")
    except ImportError:
        print("matplotlib not installed — skipped confusion matrix PNG (CSV still saved).")

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "confusion_matrix": cm}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data_dir", default="../../data/processed")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out_dir", default="./eval_results")
    args = parser.parse_args()
    evaluate_model(args.model, args.data_dir, args.split, args.out_dir)
