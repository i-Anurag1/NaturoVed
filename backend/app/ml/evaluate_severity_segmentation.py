"""
Severity-estimation segmentation experiment (v2 candidate).

TASK: "Evaluate whether improved leaf segmentation provides better
affected-area estimation and implement it only if testing demonstrates
reliable improvement."

METHOD: Real PlantVillage images have no pixel-level ground-truth lesion
masks (this is the same limitation documented in severity.py), so we
cannot measure "true" affected-area accuracy on real photos. What we
CAN do honestly is generate a synthetic benchmark where we control the
exact ground-truth affected-area percentage (we know exactly how many
pixels we painted as "lesion" when drawing the image), and use that to
compare two segmentation strategies on equal footing:

  v1 (current production method, see app/services/severity.py):
      Fixed HSV threshold ranges tuned by inspection.

  v2 (candidate): LAB color-space "a*" channel (green-red axis) +
      Otsu's automatic thresholding. Rationale: the "a*" channel
      directly encodes green-vs-red/brown separation (which is exactly
      the healthy-vs-lesion distinction for most foliar diseases), and
      Otsu's method picks the threshold automatically per-image instead
      of relying on fixed HSV cutoffs tuned on a small sample — in
      principle more robust to lighting/color variation across photos.

DECISION RULE: adopt v2 in production only if it reduces Mean Absolute
Error (MAE) against synthetic ground truth by a clear, non-marginal
margin (we use a >=20% relative MAE reduction as the bar) AND does not
regress badly on the "Healthy" (0% affected area) case. Otherwise, keep
v1 and document why v2 was rejected. See the printed result of running
this script for the actual measured outcome — do not assume an
improvement without running it.

Usage:
    python -m app.ml.evaluate_severity_segmentation --n_images 60
"""
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw

from app.services import severity as severity_v1


# ---------------------------------------------------------------------------
# Synthetic ground-truth generator (leaf fills the whole frame, background-free,
# so leaf-area = full image and we only need to get affected-area right)
# ---------------------------------------------------------------------------
def _generate_leaf_with_known_area(target_pct: float, size=(224, 224), seed: int = 0):
    """Paints a green background with brown/necrotic blobs whose combined
    pixel area is measured EXACTLY (via a binary mask), giving us a true
    ground-truth affected-area percentage for this synthetic image."""
    rng = random.Random(seed)
    img = Image.new("RGB", size, (40, 140, 40))
    mask = Image.new("L", size, 0)
    draw_img = ImageDraw.Draw(img)
    draw_mask = ImageDraw.Draw(mask)

    target_px = int(target_pct / 100.0 * size[0] * size[1])
    painted_px = 0
    attempts = 0
    while painted_px < target_px and attempts < 500:
        attempts += 1
        r = rng.randint(6, 20)
        cx, cy = rng.randint(0, size[0]), rng.randint(0, size[1])
        color = (
            rng.randint(70, 130),   # R (brown/red-ish)
            rng.randint(40, 90),    # G
            rng.randint(20, 60),    # B
        )
        bbox = [cx - r, cy - r, cx + r, cy + r]
        draw_img.ellipse(bbox, fill=color)
        draw_mask.ellipse(bbox, fill=255)
        painted_px = int(np.sum(np.array(mask) > 0))

    true_pct = round(painted_px / (size[0] * size[1]) * 100, 2)
    return img, true_pct


def generate_severity_benchmark(n_images: int = 60, seed: int = 42):
    """Generates a spread of target affected-area percentages from 0% to
    ~90% so the benchmark covers Mild/Moderate/Severe ranges evenly."""
    rng = random.Random(seed)
    samples = []
    for i in range(n_images):
        target_pct = rng.uniform(0, 90) if i % 10 != 0 else 0.0  # ~10% exactly healthy (0%)
        img, true_pct = _generate_leaf_with_known_area(target_pct, seed=seed * 1000 + i)
        samples.append((img, true_pct))
    return samples


# ---------------------------------------------------------------------------
# v1: current production method (reuses the exact logic in severity.py)
# ---------------------------------------------------------------------------
def estimate_v1(img_bgr: np.ndarray) -> float:
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    leaf_mask = severity_v1._segment_leaf(img_hsv)
    lesion_mask = severity_v1._segment_lesions(img_hsv, leaf_mask)
    leaf_px = int(np.sum(leaf_mask > 0))
    lesion_px = int(np.sum(lesion_mask > 0))
    if leaf_px == 0:
        return 0.0
    return min(round((lesion_px / leaf_px) * 100, 2), 100.0)


# ---------------------------------------------------------------------------
# v2: candidate improved method (LAB a* channel + Otsu)
# ---------------------------------------------------------------------------
def estimate_v2(img_bgr: np.ndarray) -> float:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    a_channel = lab[:, :, 1]  # green(low) <-> red/brown(high)

    # Otsu automatically picks the threshold that best separates the
    # bimodal green-vs-brown distribution in this specific image.
    _, lesion_mask = cv2.threshold(a_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, kernel)

    total_px = img_bgr.shape[0] * img_bgr.shape[1]
    lesion_px = int(np.sum(lesion_mask > 0))
    return min(round((lesion_px / total_px) * 100, 2), 100.0)


def run_comparison(n_images: int = 60):
    samples = generate_severity_benchmark(n_images)

    errors_v1, errors_v2 = [], []
    healthy_false_positive_v1, healthy_false_positive_v2 = [], []

    for img, true_pct in samples:
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        pred_v1 = estimate_v1(img_bgr)
        pred_v2 = estimate_v2(img_bgr)

        errors_v1.append(abs(pred_v1 - true_pct))
        errors_v2.append(abs(pred_v2 - true_pct))

        if true_pct == 0.0:
            healthy_false_positive_v1.append(pred_v1)
            healthy_false_positive_v2.append(pred_v2)

    mae_v1 = float(np.mean(errors_v1))
    mae_v2 = float(np.mean(errors_v2))
    rmse_v1 = float(np.sqrt(np.mean(np.square(errors_v1))))
    rmse_v2 = float(np.sqrt(np.mean(np.square(errors_v2))))

    relative_improvement = (mae_v1 - mae_v2) / mae_v1 if mae_v1 > 0 else 0.0
    healthy_fp_v1 = float(np.mean(healthy_false_positive_v1)) if healthy_false_positive_v1 else 0.0
    healthy_fp_v2 = float(np.mean(healthy_false_positive_v2)) if healthy_false_positive_v2 else 0.0

    print(f"n_images = {n_images}")
    print(f"v1 (HSV fixed-threshold, PRODUCTION):  MAE={mae_v1:.2f}  RMSE={rmse_v1:.2f}  "
          f"avg-false-positive-on-healthy={healthy_fp_v1:.2f}%")
    print(f"v2 (LAB a* + Otsu, CANDIDATE):          MAE={mae_v2:.2f}  RMSE={rmse_v2:.2f}  "
          f"avg-false-positive-on-healthy={healthy_fp_v2:.2f}%")
    print(f"Relative MAE improvement of v2 over v1: {relative_improvement * 100:.1f}%")

    ADOPTION_THRESHOLD = 0.20  # require >=20% relative MAE reduction
    adopt_v2 = relative_improvement >= ADOPTION_THRESHOLD and healthy_fp_v2 <= max(healthy_fp_v1 * 1.5, 5.0)

    print()
    if adopt_v2:
        print("DECISION: ADOPT v2 — meets the >=20% relative MAE improvement bar "
              "without regressing badly on healthy images.")
    else:
        print("DECISION: KEEP v1 (production) — v2 did not clear the >=20% relative "
              "MAE improvement bar on this synthetic benchmark, or regressed on "
              "healthy-image false positives. See docs/evaluation_plan.md for the "
              "recorded result and rationale.")

    return {
        "mae_v1": mae_v1, "mae_v2": mae_v2,
        "rmse_v1": rmse_v1, "rmse_v2": rmse_v2,
        "relative_improvement": relative_improvement,
        "healthy_fp_v1": healthy_fp_v1, "healthy_fp_v2": healthy_fp_v2,
        "adopt_v2": adopt_v2,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_images", type=int, default=60)
    args = parser.parse_args()
    run_comparison(args.n_images)
