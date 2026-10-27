import json
from pathlib import Path
from app.ml.labels import CLASS_NAMES, NUM_CLASSES, IMG_SIZE
from app.ml.train_transfer import build_transfer_model

model_dir = Path("storage/models")
model_dir.mkdir(parents=True, exist_ok=True)

model_path = model_dir / "disease_model.keras"
label_path = model_dir / "label_map.json"
warning_path = model_dir / "DEMO_ONLY_WARNING.txt"

if model_path.exists():
    raise SystemExit(
        f"STOP: {model_path} already exists. "
        "This script will not overwrite an existing model."
    )

print("Building demo-only MobileNetV2 with random weights...")
model, _ = build_transfer_model(
    num_classes=NUM_CLASSES,
    img_size=IMG_SIZE,
    weights=None
)

model.save(model_path)
label_path.write_text(
    json.dumps({i: name for i, name in enumerate(CLASS_NAMES)}, indent=2),
    encoding="utf-8"
)
warning_path.write_text(
    "DEMO ONLY: This model was initialized with random weights and was NOT trained "
    "on real crop-disease images. Predictions and Grad-CAM heatmaps are not valid "
    "disease diagnoses. Replace this model with a properly trained and evaluated "
    "model before real-world use.\n",
    encoding="utf-8"
)

print("\nDemo artifacts created:")
print(f"Model:     {model_path.resolve()}")
print(f"Label map: {label_path.resolve()}")
print(f"Warning:   {warning_path.resolve()}")
print(f"Output classes: {NUM_CLASSES}")

# Verify that the saved model can be reloaded and accepts an image tensor.
import numpy as np
import tensorflow as tf

loaded = tf.keras.models.load_model(model_path)
sample = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
output = loaded(sample, training=False).numpy()

assert output.shape == (1, NUM_CLASSES), f"Unexpected output shape: {output.shape}"
assert np.isfinite(output).all(), "Model returned non-finite values"

print("Reload test passed.")
print("Output shape:", output.shape)
print("\nDEMO MODEL READY FOR PIPELINE TESTING — NOT FOR DIAGNOSIS.")
