"""
Feature-level (early) fusion — INFERENCE service.

EXPERIMENTAL — clearly distinguished from the production default
(weighted late fusion, app/services/fusion.py). See
app/ml/train_feature_fusion.py for the full honesty disclosure on how
this model is trained (synthetically paired field-context data, since
no real jointly-labeled dataset exists), and
docs/evaluation_plan.md Section 8 for measured comparison results.

This service loads the joint classifier (image embedding + field
features -> disease class) trained by train_feature_fusion.py and
exposes a single `predict_with_feature_fusion()` function. It is only
used by the /predict route when the caller explicitly opts in via
`fusion_method=feature_fusion` (default remains `late_fusion`) — see
app/routers/predict.py.

Like image_model.py and field_context.py, this module fails soft: if
either artifact (joint model or field preprocessor) is missing, it
reports itself as unavailable rather than raising, and the route falls
back to the production late-fusion pipeline automatically.
"""
import os
import numpy as np
import pandas as pd
import joblib

from app.config import settings
from app.ml.labels import CLASS_NAMES
from app.services import image_model

_fusion_model = None
_field_preprocessor = None
_load_attempted = False

NUMERIC_FEATURES = ["temperature_c", "humidity_pct", "rainfall_mm", "soil_moisture_pct"]
CATEGORICAL_FEATURES = ["crop_type", "growth_stage"]


def _try_load():
    global _fusion_model, _field_preprocessor, _load_attempted
    if _load_attempted:
        return
    _load_attempted = True

    model_path = settings.FEATURE_FUSION_MODEL_PATH
    preprocessor_path = settings.FEATURE_FUSION_PREPROCESSOR_PATH

    if os.path.exists(model_path) and os.path.exists(preprocessor_path):
        try:
            import tensorflow as tf
            _fusion_model = tf.keras.models.load_model(model_path)
            _field_preprocessor = joblib.load(preprocessor_path)
            print(f"[feature_fusion] Loaded experimental feature-fusion model from {model_path}")
        except Exception as e:
            print(f"[feature_fusion] Failed to load feature-fusion artifacts ({e}) — feature-level "
                  "fusion unavailable, /predict will fall back to late fusion.")
            _fusion_model, _field_preprocessor = None, None
    else:
        print("[feature_fusion] No trained feature-fusion model found — experimental feature-level "
              "fusion unavailable (this is expected unless you ran app.ml.train_feature_fusion). "
              "/predict will use late fusion.")


def is_feature_fusion_available() -> bool:
    _try_load()
    return _fusion_model is not None and _field_preprocessor is not None and image_model.is_model_loaded()


def predict_with_feature_fusion(image_path: str, crop_type: str, growth_stage: str,
                                 temperature_c: float, humidity_pct: float,
                                 rainfall_mm: float, soil_moisture_pct: float):
    """
    Returns {predicted_disease, confidence, method="feature_fusion"} or
    None if feature-level fusion is unavailable (missing artifacts, or
    no trained image model to extract embeddings from).
    """
    _try_load()
    if _fusion_model is None or _field_preprocessor is None:
        return None

    embedding = image_model.extract_image_embedding(image_path)
    if embedding is None:
        return None  # no trained image model loaded -> no embedding -> can't run feature fusion

    field_df = pd.DataFrame([{
        "crop_type": crop_type,
        "growth_stage": growth_stage,
        "temperature_c": temperature_c,
        "humidity_pct": humidity_pct,
        "rainfall_mm": rainfall_mm,
        "soil_moisture_pct": soil_moisture_pct,
    }])
    field_feat = np.asarray(_field_preprocessor.transform(field_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]))

    joint_input = np.concatenate([embedding.reshape(1, -1), field_feat], axis=1)
    probs = _fusion_model.predict(joint_input, verbose=0)[0]

    top_idx = int(np.argmax(probs))
    return {
        "predicted_disease": CLASS_NAMES[top_idx],
        "confidence": float(round(probs[top_idx], 4)),
        "method": "feature_fusion",
    }
