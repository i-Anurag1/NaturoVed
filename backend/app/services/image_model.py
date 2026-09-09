"""
Image disease-classification inference service.

Loading strategy:
  1. If a trained Keras model exists at settings.IMAGE_MODEL_PATH
     (produced by app/ml/train_transfer.py), load and use it — this is
     the real, production inference path.
  2. If no trained model file exists yet (e.g. fresh clone, before
     anyone has run training), fall back to a deterministic HEURISTIC
     demo mode based on simple color statistics, so the full API +
     frontend can be demoed end-to-end without requiring GPU training
     first. Every response produced in fallback mode is clearly
     labeled via `model_mode="fallback_heuristic"` so it is never
     confused with a real trained-model prediction.

This mirrors the requirement: "if a component requires data
unavailable in public datasets [or, here, unavailable pretrained
weights in a fresh checkout], provide a realistic MVP implementation
... and clearly label how it should later be replaced."
"""
import os
import json
import numpy as np
from PIL import Image
from app.config import settings
from app.ml.labels import CLASS_NAMES, CLASS_TO_CROP, IMG_SIZE

_model = None
_label_map = None
_model_mode = "not_loaded"


def _resolve_model_path() -> str:
    """
    Return the model file to load. Prefers the configured
    IMAGE_MODEL_PATH (native Keras v3 `.keras` format as of this
    version — see docs/methodology.md "Model Serialization Fix" for
    why we moved off legacy `.h5`). Falls back to a `.h5` file at the
    same location for backward compatibility with models trained by
    an earlier version of this project.
    """
    if os.path.exists(settings.IMAGE_MODEL_PATH):
        return settings.IMAGE_MODEL_PATH
    legacy_h5_path = os.path.splitext(settings.IMAGE_MODEL_PATH)[0] + ".h5"
    if os.path.exists(legacy_h5_path):
        print(f"[image_model] Found legacy .h5 model at {legacy_h5_path} — "
              "consider retraining to the native .keras format (see docs).")
        return legacy_h5_path
    return settings.IMAGE_MODEL_PATH  # doesn't exist; caller handles fallback


def _try_load_model():
    global _model, _label_map, _model_mode
    if _model is not None:
        return

    resolved_path = _resolve_model_path()
    if os.path.exists(resolved_path):
        try:
            import tensorflow as tf
            _model = tf.keras.models.load_model(resolved_path)
            _model_mode = "trained"
            if os.path.exists(settings.LABEL_MAP_PATH):
                with open(settings.LABEL_MAP_PATH) as f:
                    _label_map = {int(k): v for k, v in json.load(f).items()}
            else:
                _label_map = {i: c for i, c in enumerate(CLASS_NAMES)}
            print(f"[image_model] Loaded trained model from {resolved_path}")
            return
        except Exception as e:
            print(f"[image_model] Failed to load trained model ({e}); using fallback heuristic mode.")

    _model = None
    _model_mode = "fallback_heuristic"
    print("[image_model] No trained model found — running in FALLBACK HEURISTIC mode. "
          "Run `python -m app.ml.train_transfer` to train a real model.")


def is_model_loaded() -> bool:
    _try_load_model()
    return _model_mode == "trained"


def get_model_mode() -> str:
    _try_load_model()
    return _model_mode


def get_loaded_model():
    """Returns the loaded tf.keras.Model, or None if running in fallback
    mode. Used by app/services/gradcam.py so Grad-CAM reuses the exact
    in-memory model instance (no redundant disk load)."""
    _try_load_model()
    return _model


def get_label_map():
    """Returns {index: class_name} for the loaded model, or None in
    fallback mode. Used by the Grad-CAM route to map a disease name
    back to its class index."""
    _try_load_model()
    return _label_map


def extract_image_embedding(image_path: str):
    """
    Returns the 128-d penultimate-layer embedding (the Dense(128, relu)
    layer output, before the final softmax) for the given image, using
    the currently loaded trained model. Returns None in fallback mode.

    This embedding is what app/services/feature_fusion.py concatenates
    with processed field-context features for the experimental
    feature-level fusion approach — see docs/methodology.md Section 6
    for why feature-level fusion needs this and how it differs from the
    baseline weighted late-fusion approach in app/services/fusion.py.
    """
    _try_load_model()
    if _model_mode != "trained" or _model is None:
        return None

    import tensorflow as tf
    dense_layers = [l for l in _model.layers if isinstance(l, tf.keras.layers.Dense)]
    if len(dense_layers) < 2:
        return None
    embedding_layer = dense_layers[0]  # the Dense(128, relu) layer, before the final softmax Dense

    embedding_model = tf.keras.Model(inputs=_model.inputs, outputs=embedding_layer.output)
    x = _preprocess_image(image_path)
    embedding = embedding_model.predict(x, verbose=0)[0]
    return embedding


def generate_gradcam_for_disease(image_path: str, predicted_disease: str):
    """
    Convenience wrapper used by the /predict and /prediction/{id}/gradcam
    routes: resolves `predicted_disease` (e.g. "Tomato___Early_Blight")
    to its class index for the currently loaded model, then delegates to
    app.services.gradcam. Returns None (not an error) when no trained
    model is loaded — Grad-CAM is only meaningful for a real trained
    model, never for the fallback heuristic (see app/services/gradcam.py
    module docstring).
    """
    _try_load_model()
    if _model_mode != "trained" or _model is None or _label_map is None:
        return None

    class_idx = next((idx for idx, name in _label_map.items() if name == predicted_disease), None)
    if class_idx is None:
        return None

    from app.services import gradcam
    try:
        return gradcam.generate_gradcam_overlay_base64(_model, image_path, class_idx)
    except gradcam.GradCamUnavailableError as e:
        print(f"[image_model] Grad-CAM unavailable: {e}")
        return None


def _preprocess_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)  # (1, H, W, 3)


def _fallback_heuristic_predict(image_path: str, crop_type: str) -> dict:
    """
    Deterministic, non-random heuristic used ONLY when no trained model
    is available. Uses simple HSV color statistics: a higher proportion
    of brown/yellow/necrotic pixels relative to healthy green pixels
    nudges the "prediction" toward a disease class for that crop,
    otherwise predicts Healthy. This keeps the API contract identical
    to the trained-model path and lets the rest of the pipeline
    (severity/fusion/risk/recommendations) be fully exercised in a demo.
    """
    import cv2
    img_bgr = cv2.imread(image_path)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    green_mask = cv2.inRange(img_hsv, (35, 40, 40), (85, 255, 255))
    lesion_mask = cv2.inRange(img_hsv, (5, 40, 40), (35, 255, 200))  # brown/yellow range

    total_px = img_hsv.shape[0] * img_hsv.shape[1]
    lesion_ratio = float(np.sum(lesion_mask > 0)) / max(total_px, 1)

    candidates = [c for c in CLASS_NAMES if CLASS_TO_CROP[c] == crop_type]
    healthy_label = next((c for c in candidates if "Healthy" in c), candidates[0])
    disease_candidates = [c for c in candidates if "Healthy" not in c]

    if lesion_ratio < 0.03 or not disease_candidates:
        predicted = healthy_label
        confidence = float(np.clip(0.75 + (0.03 - lesion_ratio) * 4, 0.6, 0.95))
    else:
        # pick disease deterministically from lesion_ratio magnitude
        idx = min(int(lesion_ratio * len(disease_candidates) * 3), len(disease_candidates) - 1)
        predicted = disease_candidates[idx]
        confidence = float(np.clip(0.55 + lesion_ratio, 0.5, 0.9))

    return {
        "predicted_disease": predicted,
        "confidence": round(confidence, 4),
        "lesion_ratio": round(lesion_ratio, 4),
    }


def predict_disease(image_path: str, crop_type: str) -> dict:
    """
    Main entry point used by the /predict route.
    Returns: {predicted_disease, confidence, model_mode}
    """
    _try_load_model()

    if _model_mode == "trained":
        x = _preprocess_image(image_path)
        probs = _model.predict(x, verbose=0)[0]

        # Restrict to classes belonging to the user-declared crop_type so
        # the field-context input actively participates in the pipeline
        # (true multimodal behavior, not "image model ignores context").
        valid_indices = [i for i, name in _label_map.items() if CLASS_TO_CROP[name] == crop_type]
        if valid_indices:
            masked_probs = np.zeros_like(probs)
            for i in valid_indices:
                masked_probs[i] = probs[i]
            if masked_probs.sum() > 0:
                masked_probs = masked_probs / masked_probs.sum()
                probs = masked_probs

        top_idx = int(np.argmax(probs))
        return {
            "predicted_disease": _label_map[top_idx],
            "confidence": float(round(probs[top_idx], 4)),
            "model_mode": "trained",
        }

    result = _fallback_heuristic_predict(image_path, crop_type)
    result["model_mode"] = "fallback_heuristic"
    return result
