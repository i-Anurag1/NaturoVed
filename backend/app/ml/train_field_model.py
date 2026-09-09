"""
Train the structured field-context risk model.

HONESTY NOTE (also documented in docs/methodology.md and docs/limitations
section of the report): there is no public, labeled dataset mapping
(crop, growth stage, temperature, humidity, rainfall, soil moisture) to
a ground-truth "disease-favorable risk level". Real plant-pathology
epidemiology models (e.g. for late blight: Wallin/BLITECAST-style rules)
exist but require field-trial data we do not have access to for an
undergraduate MVP.

Our approach: we generate a SYNTHETIC, RULE-LABELED dataset using
well-established, textbook agronomy heuristics (documented below), then
train a Random Forest classifier on it. This lets us:
  1) Get a genuinely trained, non-trivial ML model (satisfies the
     "structured-data model" requirement) instead of a hardcoded if/else.
  2) Keep every generation rule fully transparent and defendable to
     evaluators (no fabricated "real-world accuracy" claims).
  3) Make it trivial to later replace this synthetic generator with a
     real field-trial dataset without touching the rest of the pipeline
     (train script + inference contract stay identical).

Domain rules used to synthesize labels (fungal/bacterial leaf disease
favorability, generalized across the 3 supported crops):
  - High humidity (>75%) + moderate temperature (18-28C) + recent
    rainfall (>10mm)  -> classic conditions for fungal spore germination
    (e.g. blight, rust) => HIGH risk.
  - Very low humidity (<40%) and low rainfall -> poor conditions for
    most foliar fungal pathogens => LOW risk.
  - Waterlogged soil (soil_moisture > 80%) independently raises risk
    (root stress + humidity microclimate).
  - Extreme heat (>35C) suppresses many common fungal pathogens
    (denatures spores) => lowers risk despite humidity.
  - Everything else => MEDIUM risk.
A controlled amount of label noise is added so the model does not
memorize a lookup table and must learn a genuine decision boundary.

Usage:
    python -m app.ml.train_field_model --n_samples 6000 --out ../../backend/storage/models/field_model.joblib
"""
import argparse
import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib

CROPS = ["tomato", "potato", "corn"]
GROWTH_STAGES = ["seedling", "vegetative", "flowering", "fruiting", "maturity"]
RISK_LABELS = ["Low", "Medium", "High"]

NUMERIC_FEATURES = ["temperature_c", "humidity_pct", "rainfall_mm", "soil_moisture_pct"]
CATEGORICAL_FEATURES = ["crop_type", "growth_stage"]


def _rule_based_label(row, rng):
    """Domain-rule label generator described in the module docstring."""
    score = 0.0
    if row["humidity_pct"] > 75 and 18 <= row["temperature_c"] <= 28:
        score += 1.0
    if row["rainfall_mm"] > 10:
        score += 0.6
    if row["soil_moisture_pct"] > 80:
        score += 0.4
    if row["temperature_c"] > 35:
        score -= 1.0
    if row["humidity_pct"] < 40 and row["rainfall_mm"] < 5:
        score -= 0.8

    # inject label noise ~8% of the time to avoid a trivially perfect
    # rule-memorization model
    if rng.random() < 0.08:
        score += rng.uniform(-1, 1)

    if score >= 1.0:
        return "High"
    elif score <= -0.5:
        return "Low"
    return "Medium"


def generate_synthetic_dataset(n_samples: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    py_rng = __import__("random")
    py_rng.seed(seed)

    rows = []
    for _ in range(n_samples):
        row = {
            "crop_type": rng.choice(CROPS),
            "growth_stage": rng.choice(GROWTH_STAGES),
            "temperature_c": float(np.clip(rng.normal(26, 6), -5, 45)),
            "humidity_pct": float(np.clip(rng.normal(65, 20), 0, 100)),
            "rainfall_mm": float(np.clip(rng.exponential(15), 0, 200)),
            "soil_moisture_pct": float(np.clip(rng.normal(50, 20), 0, 100)),
        }
        row["risk_level"] = _rule_based_label(row, py_rng)
        rows.append(row)

    return pd.DataFrame(rows)


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced",
    )
    return Pipeline([("preprocess", preprocessor), ("clf", clf)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=6000)
    parser.add_argument("--out", default="../../backend/storage/models/field_model.joblib")
    args = parser.parse_args()

    df = generate_synthetic_dataset(args.n_samples)
    print("Class balance:\n", df["risk_level"].value_counts())

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["risk_level"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"\nField-risk model — test accuracy={acc:.4f} weighted F1={f1:.4f}")
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(pipeline, args.out)

    metrics_path = args.out.replace(".joblib", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"accuracy": acc, "f1_weighted": f1}, f, indent=2)

    print(f"Saved field-context model to {args.out}")


if __name__ == "__main__":
    main()
