"""
Quick sanity check of the trained image model on a folder of sample leaves.

Name each image after its TRUE class, e.g. Tomato___Early_Blight.jpg, so the
script can tell whether the prediction is right.

Usage (from anywhere):
    python scripts/quick_check.py                      # uses Downloads/leaftest
    python scripts/quick_check.py C:/path/to/folder

NOTE: images copied from the raw training data may have been seen during
training, so this proves the model loads and behaves sensibly. It is NOT a
measure of real accuracy - use app.ml.evaluate (held-out test split) for that.
"""
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

MODELS_DIR = Path(__file__).resolve().parents[1] / "backend" / "storage" / "models"
folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads" / "leaftest"

if not folder.exists():
    sys.exit(f"Folder not found: {folder}")
files = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
if not files:
    sys.exit(f"No .jpg/.jpeg/.png images found in {folder}")

model = tf.keras.models.load_model(MODELS_DIR / "disease_model.keras")
with open(MODELS_DIR / "label_map.json") as fh:
    labels = {int(k): v for k, v in json.load(fh).items()}

right = 0
for f in files:
    img = tf.keras.utils.load_img(f, target_size=(224, 224))
    x = np.expand_dims(tf.keras.utils.img_to_array(img), 0)
    probs = model.predict(x, verbose=0)[0]
    top = int(np.argmax(probs))
    ok = labels[top] == f.stem
    right += ok
    print(("OK    " if ok else "WRONG ") + f"{f.stem:36s} -> {labels[top]} ({probs[top]:.2f})")

print(f"\n{right}/{len(files)} correct")