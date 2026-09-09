"""
Synthetic leaf-image benchmark generator.

PURPOSE & HONESTY NOTE (read this before citing any numbers produced
by this module in your academic report): this project's real image
training data is PlantVillage (see data/README.md), which must be
downloaded separately and is NOT bundled in this repository. In the
sandboxed development/CI environment used to build and test this
codebase, there is no network access to either Kaggle (to download
PlantVillage) or storage.googleapis.com (to download Keras'
pretrained ImageNet weights) — both are blocked by the environment's
network allowlist.

To still be able to (a) exercise every line of the training / Grad-CAM
/ fusion-comparison code paths end-to-end and (b) report REAL, MEASURED
(not fabricated) numbers for things like "does Grad-CAM run correctly"
or "does feature-level fusion code train without crashing", this module
generates a small, fully synthetic, programmatically labeled image
dataset: green "leaf" backgrounds with class-specific colored blob
patterns standing in for lesions. It is a controlled unit-test fixture,
NOT a substitute for PlantVillage, and results produced from it must
never be reported as real-world disease-classification accuracy in
your project report — they are reported in docs/evaluation_plan.md
under a clearly labeled "Sandbox Smoke-Test Results" section, separate
from the "Real PlantVillage Results" section you fill in after running
real training with real data and (outside this sandbox) real ImageNet
weights.

Each class gets a distinct, deterministic blob color/pattern so a
model CAN, in principle, learn to separate them — this lets us
sanity-check that the training/eval/Grad-CAM/fusion code is behaving
correctly (loss decreases, accuracy > random chance, Grad-CAM heatmap
concentrates on the blobs) without needing real data.
"""
import os
import random
import numpy as np
from PIL import Image, ImageDraw
from app.ml.labels import CLASS_NAMES

# Deterministic distinct colors per class (RGB) so a model can separate them.
_CLASS_COLORS = {
    "Tomato___Healthy": (34, 139, 34),
    "Tomato___Early_Blight": (139, 90, 43),
    "Tomato___Late_Blight": (60, 40, 30),
    "Tomato___Leaf_Mold": (170, 160, 60),
    "Potato___Healthy": (40, 130, 40),
    "Potato___Early_Blight": (150, 100, 50),
    "Potato___Late_Blight": (70, 50, 40),
    "Corn___Healthy": (50, 145, 50),
    "Corn___Common_Rust": (180, 90, 60),
    "Corn___Gray_Leaf_Spot": (130, 120, 110),
}


def _make_synthetic_image(class_name: str, seed: int, size=(224, 224)) -> Image.Image:
    rng = random.Random(seed)
    img = Image.new("RGB", size, (34, 139, 34))  # green leaf base
    draw = ImageDraw.Draw(img)

    if "Healthy" not in class_name:
        blob_color = _CLASS_COLORS[class_name]
        n_blobs = rng.randint(4, 10)
        for _ in range(n_blobs):
            cx, cy = rng.randint(20, size[0] - 20), rng.randint(20, size[1] - 20)
            r = rng.randint(8, 22)
            jitter = tuple(max(0, min(255, c + rng.randint(-15, 15))) for c in blob_color)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=jitter)
    else:
        # subtle texture noise so "healthy" isn't a single flat color
        for _ in range(rng.randint(2, 5)):
            cx, cy = rng.randint(20, size[0] - 20), rng.randint(20, size[1] - 20)
            r = rng.randint(5, 12)
            jitter = tuple(max(0, min(255, c + rng.randint(-10, 10))) for c in (34, 139, 34))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=jitter)

    return img


def generate_synthetic_image_dataset(output_dir: str, images_per_class: int = 20, seed: int = 42):
    """
    Creates output_dir/{train,val,test}/<ClassName>/*.jpg using a 60/20/20
    split, matching the folder structure expected by
    app.ml.dataset_utils / image_dataset_from_directory.
    """
    rng = random.Random(seed)
    counts = {"train": int(images_per_class * 0.6), "val": int(images_per_class * 0.2)}
    counts["test"] = images_per_class - counts["train"] - counts["val"]

    summary = {}
    global_idx = 0
    for cls in CLASS_NAMES:
        summary[cls] = {}
        for split, n in counts.items():
            split_dir = os.path.join(output_dir, split, cls)
            os.makedirs(split_dir, exist_ok=True)
            for i in range(n):
                img = _make_synthetic_image(cls, seed=seed * 10000 + global_idx)
                img.save(os.path.join(split_dir, f"{cls}_{split}_{i}.jpg"))
                global_idx += 1
            summary[cls][split] = n
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate a small synthetic image benchmark (NOT real disease data — see module docstring)")
    parser.add_argument("--output_dir", default="../data/synthetic_benchmark",
                         help="NOTE: relative to the backend/ directory when run via "
                              "`python -m app.ml.synthetic_benchmark` (matches the path used "
                              "throughout docs/evaluation_plan.md). Previously defaulted to "
                              "'../../data/synthetic_benchmark' (one level too high) — fixed.")
    parser.add_argument("--images_per_class", type=int, default=20)
    args = parser.parse_args()
    summary = generate_synthetic_image_dataset(args.output_dir, args.images_per_class)
    print("Generated synthetic benchmark dataset:")
    for cls, counts in summary.items():
        print(f"  {cls}: {counts}")
