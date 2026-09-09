"""
Severity estimation module.

Approach chosen (and why): PlantVillage-style classification datasets do
NOT provide pixel-level lesion masks or ground-truth affected-area
percentages, so we cannot train a supervised segmentation model with
real labels for this MVP. Full manual pixel annotation is out of scope
for a semester project with 4 students.

Instead we implement a classical, UNSUPERVISED, OpenCV-based
color-segmentation heuristic:
  1. Convert the leaf image to HSV color space.
  2. Segment the leaf from the background using a broad
     green+brown+yellow mask (removes background clutter/soil/etc.).
  3. Within the leaf region, segment "lesion-like" pixels (brown,
     yellow, black, gray tones that deviate from healthy green) using
     HSV thresholds tuned on common PlantVillage leaf disease samples.
  4. affected_area_pct = lesion_pixels / leaf_pixels * 100.
  5. Map the percentage to Mild / Moderate / Severe using the
     thresholds specified in the project brief:
        < 20%  -> Mild
        20-50% -> Moderate
        > 50%  -> Severe

Because this is a heuristic (not a trained/validated segmentation
model), we deliberately keep the thresholds configurable and always
report `affected_area_pct` alongside a `method="opencv_color_heuristic"`
tag so results are never presented as clinically precise.

If the predicted disease is "Healthy", we short-circuit and return
0% / no severity class, since severity is undefined for a healthy leaf.
"""
import cv2
import numpy as np

MILD_MAX = 20.0
MODERATE_MAX = 50.0


def _segment_leaf(img_hsv: np.ndarray) -> np.ndarray:
    """Broad mask that keeps leaf-tissue-like pixels (green + diseased
    tones) and drops obvious background (soil browns are ambiguous so
    we bias toward keeping more pixels — false negatives here just
    slightly under-count leaf area, which is the safer failure mode)."""
    green = cv2.inRange(img_hsv, (25, 30, 30), (95, 255, 255))
    lesion_tones = cv2.inRange(img_hsv, (0, 30, 20), (30, 255, 200))
    dark_necrotic = cv2.inRange(img_hsv, (0, 0, 0), (180, 255, 60))
    leaf_mask = cv2.bitwise_or(green, lesion_tones)
    leaf_mask = cv2.bitwise_or(leaf_mask, dark_necrotic)

    # morphological cleanup: remove tiny noise specks, close small holes
    kernel = np.ones((5, 5), np.uint8)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
    return leaf_mask


def _segment_lesions(img_hsv: np.ndarray, leaf_mask: np.ndarray) -> np.ndarray:
    """Within the leaf region, flag brown/yellow/black necrotic-looking
    pixels as lesions."""
    brown_yellow = cv2.inRange(img_hsv, (8, 40, 40), (35, 255, 220))
    dark_spots = cv2.inRange(img_hsv, (0, 0, 0), (180, 100, 70))
    lesion_mask = cv2.bitwise_or(brown_yellow, dark_spots)
    lesion_mask = cv2.bitwise_and(lesion_mask, leaf_mask)  # only within leaf
    return lesion_mask


def estimate_severity(image_path: str, predicted_disease: str) -> dict:
    """
    Returns:
        {
          "affected_area_pct": float | None,
          "severity_class": "Mild" | "Moderate" | "Severe" | "None",
          "method": str
        }
    """
    if "Healthy" in predicted_disease:
        return {
            "affected_area_pct": 0.0,
            "severity_class": "None",
            "method": "opencv_color_heuristic",
        }

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        # Corrupt/unreadable image — degrade gracefully instead of crashing
        return {
            "affected_area_pct": None,
            "severity_class": "Moderate",  # safe conservative default
            "method": "unreadable_image_default",
        }

    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    leaf_mask = _segment_leaf(img_hsv)
    lesion_mask = _segment_lesions(img_hsv, leaf_mask)

    leaf_px = int(np.sum(leaf_mask > 0))
    lesion_px = int(np.sum(lesion_mask > 0))

    if leaf_px == 0:
        affected_pct = 0.0
    else:
        affected_pct = round((lesion_px / leaf_px) * 100, 2)
        affected_pct = min(affected_pct, 100.0)

    if affected_pct < MILD_MAX:
        severity_class = "Mild"
    elif affected_pct < MODERATE_MAX:
        severity_class = "Moderate"
    else:
        severity_class = "Severe"

    return {
        "affected_area_pct": affected_pct,
        "severity_class": severity_class,
        "method": "opencv_color_heuristic",
    }
