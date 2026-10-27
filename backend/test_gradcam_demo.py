import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from app.services.image_model import get_loaded_model
from app.services.gradcam import generate_gradcam_overlay_base64

out_dir = Path("storage/demo")
out_dir.mkdir(parents=True, exist_ok=True)

# Create a synthetic leaf-shaped image for software testing only.
image = Image.new("RGB", (224, 224), (235, 235, 225))
draw = ImageDraw.Draw(image)
draw.ellipse((45, 25, 180, 205), fill=(55, 130, 55), outline=(25, 80, 30), width=3)
draw.line((65, 190, 160, 45), fill=(210, 220, 150), width=4)
for x, y in [(85, 130), (105, 95), (125, 75), (95, 155), (140, 110)]:
    draw.ellipse((x, y, x + 12, y + 8), fill=(130, 90, 50))
image_path = out_dir / "synthetic_leaf_demo.png"
image.save(image_path)

model = get_loaded_model()
if model is None:
    raise RuntimeError("No Keras model was loaded by image_model.py")

print("Loaded model:", model.name)
print("Input image:", image_path.resolve())

overlay_uri = generate_gradcam_overlay_base64(
    model=model,
    image_path=str(image_path),
    class_idx=0
)

if not overlay_uri.startswith("data:image/png;base64,"):
    raise RuntimeError("Grad-CAM did not return a PNG data URI")

encoded = overlay_uri.split(",", 1)[1]
overlay_bytes = base64.b64decode(encoded, validate=True)
overlay_path = out_dir / "gradcam_demo_overlay.png"
Image.open(io.BytesIO(overlay_bytes)).convert("RGB").save(overlay_path)

print("\nGRAD-CAM PIPELINE TEST PASSED")
print("Synthetic input:", image_path.resolve())
print("Grad-CAM overlay:", overlay_path.resolve())
