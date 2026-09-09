"""
Explanation builder.

Combines the outputs of every module (image model, severity, field
context, fusion) into a single human-readable explanation string
returned to the frontend. This module contains NO machine learning —
it is a deterministic template so the explanation is always
reproducible and auditable (a core "explainable AI" requirement from
the brief, deliberately implemented without any generative model).

Named `risk.py` because it also owns the mapping from raw field-risk
score to the display-friendly risk narrative used across the app.
"""
from app.ml.labels import CLASS_INFO

RISK_NARRATIVE = {
    "Low": "Current temperature, humidity, rainfall, and soil moisture levels are "
           "generally unfavorable for rapid disease spread.",
    "Medium": "Current environmental conditions could support disease development; "
              "monitoring is advised.",
    "High": "Current environmental conditions (humidity/temperature/moisture) closely "
            "match conditions known to favor rapid disease spread.",
}


def describe_field_risk(risk_level: str) -> str:
    return RISK_NARRATIVE.get(risk_level, "Field risk level could not be determined.")


def build_explanation(predicted_disease: str, image_confidence: float,
                       severity_class: str, affected_area_pct,
                       field_risk_level: str, fusion_rationale: str) -> str:
    disease_desc = CLASS_INFO.get(predicted_disease, "No description available for this class.")
    is_healthy = "Healthy" in predicted_disease

    parts = [
        f"Image analysis: {disease_desc} (model confidence: {image_confidence * 100:.1f}%).",
    ]

    if not is_healthy:
        area_str = f"{affected_area_pct:.1f}%" if affected_area_pct is not None else "unavailable"
        parts.append(
            f"Severity analysis: approximately {area_str} of the leaf surface shows "
            f"lesion-like discoloration, classified as {severity_class} severity."
        )

    parts.append(
        f"Field-context analysis: {describe_field_risk(field_risk_level)} "
        f"(overall field risk: {field_risk_level})."
    )
    parts.append(f"Fusion note: {fusion_rationale}")

    return " ".join(parts)
