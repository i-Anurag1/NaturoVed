"""
Field-context processing service.

Loads the trained scikit-learn Pipeline (preprocessing + RandomForest,
produced by app/ml/train_field_model.py) and exposes a single
`assess_field_risk()` function used by the /predict route.

If the trained model file is missing (fresh checkout before training
has been run), falls back to a transparent rule-based risk score using
the SAME domain rules the synthetic training data was generated from
(see app/ml/train_field_model.py `_rule_based_label`), so the API
never breaks — it just runs a slightly less "learned" version of the
same logic until someone runs the training script.
"""
import os
import joblib
import numpy as np
import pandas as pd
from app.config import settings

_pipeline = None
_pipeline_loaded_from_disk = False

RISK_TO_SCORE = {"Low": 0.15, "Medium": 0.5, "High": 0.85}


def _try_load_pipeline():
    global _pipeline, _pipeline_loaded_from_disk
    if _pipeline is not None or _pipeline_loaded_from_disk:
        return
    if os.path.exists(settings.FIELD_MODEL_PATH):
        _pipeline = joblib.load(settings.FIELD_MODEL_PATH)
        print(f"[field_context] Loaded trained field-risk model from {settings.FIELD_MODEL_PATH}")
    else:
        print("[field_context] No trained field-risk model found — using rule-based fallback. "
              "Run `python -m app.ml.train_field_model` to train one.")
    _pipeline_loaded_from_disk = True


def is_field_model_loaded() -> bool:
    _try_load_pipeline()
    return _pipeline is not None


def _rule_based_fallback(crop_type, growth_stage, temperature_c, humidity_pct, rainfall_mm, soil_moisture_pct):
    score = 0.0
    if humidity_pct > 75 and 18 <= temperature_c <= 28:
        score += 1.0
    if rainfall_mm > 10:
        score += 0.6
    if soil_moisture_pct > 80:
        score += 0.4
    if temperature_c > 35:
        score -= 1.0
    if humidity_pct < 40 and rainfall_mm < 5:
        score -= 0.8

    if score >= 1.0:
        label = "High"
    elif score <= -0.5:
        label = "Low"
    else:
        label = "Medium"
    return label


def assess_field_risk(crop_type: str, growth_stage: str, temperature_c: float,
                       humidity_pct: float, rainfall_mm: float, soil_moisture_pct: float) -> dict:
    """
    Returns:
        {
          "risk_level": "Low" | "Medium" | "High",
          "risk_score": float in [0,1],
          "method": str
        }
    """
    _try_load_pipeline()

    if _pipeline is not None:
        X = pd.DataFrame([{
            "crop_type": crop_type,
            "growth_stage": growth_stage,
            "temperature_c": temperature_c,
            "humidity_pct": humidity_pct,
            "rainfall_mm": rainfall_mm,
            "soil_moisture_pct": soil_moisture_pct,
        }])
        proba = _pipeline.predict_proba(X)[0]
        classes = _pipeline.named_steps["clf"].classes_
        risk_level = classes[int(np.argmax(proba))]
        # risk_score = P(High) if available, else map predicted class to a fixed score
        if "High" in classes:
            high_idx = list(classes).index("High")
            risk_score = float(proba[high_idx])
        else:
            risk_score = RISK_TO_SCORE.get(risk_level, 0.5)
        return {"risk_level": str(risk_level), "risk_score": round(risk_score, 4), "method": "random_forest"}

    risk_level = _rule_based_fallback(crop_type, growth_stage, temperature_c, humidity_pct, rainfall_mm, soil_moisture_pct)
    return {
        "risk_level": risk_level,
        "risk_score": RISK_TO_SCORE[risk_level],
        "method": "rule_based_fallback",
    }


# --- v2: Field-context explainability -------------------------------------
#
# WHY GLOBAL FEATURE IMPORTANCE (not SHAP/LIME): scikit-learn's
# RandomForestClassifier exposes `feature_importances_` (mean decrease in
# impurity across all trees) essentially for free — no extra dependency,
# no extra inference-time cost, and it is a widely accepted, easily
# explained method for tree ensembles. SHAP would give per-prediction
# (local) explanations, which is a nice future upgrade (see
# docs/limitations_and_future_work.md), but adds a heavy dependency and
# noticeably slower per-request latency for a synchronous API endpoint.
# We combine the model's GLOBAL importances with the SPECIFIC input
# values for this request to produce a per-request, human-readable
# ranking ("this request's humidity was high, and humidity is generally
# the model's most influential feature") — a practical, honest middle
# ground for an undergraduate MVP.
_FEATURE_DISPLAY_NAMES = {
    "temperature_c": "Temperature",
    "humidity_pct": "Humidity",
    "rainfall_mm": "Rainfall (7-day)",
    "soil_moisture_pct": "Soil Moisture",
    "crop_type": "Crop Type",
    "growth_stage": "Growth Stage",
}


def get_field_feature_importance() -> list:
    """
    Returns the trained Random Forest's global feature importances as a
    sorted list of {feature, display_name, importance} dicts. Returns an
    empty list if no trained model is loaded (rule-based fallback mode
    has no learned importances to report).
    """
    _try_load_pipeline()
    if _pipeline is None:
        return []

    try:
        clf = _pipeline.named_steps["clf"]
        preprocessor = _pipeline.named_steps["preprocess"]
        feature_names = preprocessor.get_feature_names_out()
        importances = clf.feature_importances_
    except (AttributeError, KeyError):
        return []

    # Aggregate one-hot-encoded categorical columns (e.g. "cat__crop_type_tomato",
    # "cat__crop_type_potato", ...) back into a single importance per original
    # field so the explanation is readable ("Crop Type", not 3 separate rows).
    aggregated = {}
    for fname, importance in zip(feature_names, importances):
        # sklearn ColumnTransformer names look like "num__temperature_c" or
        # "cat__crop_type_tomato" — strip the transformer prefix, then strip
        # any one-hot category suffix by matching against known base names.
        stripped = fname.split("__", 1)[-1]
        base_name = stripped
        for known in _FEATURE_DISPLAY_NAMES:
            if stripped == known or stripped.startswith(known + "_"):
                base_name = known
                break
        aggregated[base_name] = aggregated.get(base_name, 0.0) + float(importance)

    total = sum(aggregated.values()) or 1.0
    ranked = sorted(aggregated.items(), key=lambda kv: kv[1], reverse=True)
    return [
        {
            "feature": name,
            "display_name": _FEATURE_DISPLAY_NAMES.get(name, name),
            "importance": round(value / total, 4),  # normalized to sum to 1.0
        }
        for name, value in ranked
    ]


def explain_field_risk(crop_type: str, growth_stage: str, temperature_c: float,
                        humidity_pct: float, rainfall_mm: float, soil_moisture_pct: float) -> dict:
    """
    Returns a per-request explanation combining the model's global
    feature importances with this request's specific values, e.g.:
        "Humidity (82%) was the most influential factor in this
         assessment (model-wide importance: 34%)."
    Falls back to a simple textual note (no ranking) in rule-based mode.
    """
    importances = get_field_feature_importance()
    if not importances:
        return {
            "top_features": [],
            "explanation": (
                "Field-risk explanation unavailable: no trained Random Forest model is "
                "loaded (running in rule-based fallback mode). Train one with "
                "`python -m app.ml.train_field_model` to enable feature-importance explanations."
            ),
            "method": "unavailable",
        }

    values = {
        "temperature_c": f"{temperature_c}°C",
        "humidity_pct": f"{humidity_pct}%",
        "rainfall_mm": f"{rainfall_mm}mm",
        "soil_moisture_pct": f"{soil_moisture_pct}%",
        "crop_type": crop_type,
        "growth_stage": growth_stage,
    }

    top3 = importances[:3]
    parts = []
    for rank, item in enumerate(top3, start=1):
        val_str = values.get(item["feature"], "")
        parts.append(
            f"{rank}. {item['display_name']} ({val_str}) — model-wide importance {item['importance'] * 100:.1f}%"
        )

    explanation = (
        "The field-risk model's most influential factors overall, evaluated against this "
        "submission's actual values: " + "; ".join(parts) + "."
    )

    return {"top_features": top3, "explanation": explanation, "method": "random_forest_feature_importance"}
