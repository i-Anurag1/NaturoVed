"""
Deterministic, rule-based recommendation engine.

Per the project brief: "Do not rely on generative AI for core
predictions or recommendations ... Recommendations must be based on
disease, severity, and field conditions and should remain general and
safe unless reliable agricultural sources support more specific
guidance."

Implementation: a lookup table keyed by (disease_family, severity)
plus general field-condition-based advice, all sourced from widely
published, general agricultural extension guidance (e.g. university
extension fact sheets on early/late blight, rust, leaf mold). No
specific fungicide brand names, dosages, or region-specific legal
guidance are given — those require a licensed local agronomist and
vary by country/state regulation, which is explicitly out of scope
and called out in docs/limitations.
"""
from app.ml.labels import CLASS_TO_CROP

GENERAL_HEALTHY_TIPS = [
    "No disease symptoms detected — continue routine monitoring (2-3 times per week).",
    "Maintain balanced irrigation; avoid overhead watering late in the day to reduce leaf wetness duration.",
    "Continue standard crop nutrition and pest-scouting practices for this growth stage.",
]

DISEASE_FAMILY_TIPS = {
    "Early_Blight": [
        "Remove and destroy visibly infected lower leaves to reduce spore load.",
        "Improve airflow via proper plant spacing and pruning to lower humidity around foliage.",
        "Avoid overhead irrigation; water at the base of the plant.",
        "Consult a local agricultural extension office about approved fungicide options for early blight in your region.",
    ],
    "Late_Blight": [
        "Late blight can spread very rapidly in favorable weather — inspect the field daily.",
        "Remove and destroy infected plant material away from the field; do not compost.",
        "Avoid working in the field when foliage is wet to prevent spreading spores.",
        "Contact a local agricultural extension office promptly — late blight often requires timely, region-approved fungicide intervention.",
    ],
    "Leaf_Mold": [
        "Increase ventilation (greenhouse vents/plant spacing) to reduce humidity around leaves.",
        "Avoid leaf wetness by watering at soil level, ideally in the morning.",
        "Remove heavily infected leaves and dispose of them away from the crop.",
    ],
    "Common_Rust": [
        "Monitor pustule spread across the field; rust can spread quickly with wind and moisture.",
        "Avoid excess nitrogen fertilization, which can increase susceptibility.",
        "Consult local extension guidance on rust-resistant hybrid varieties for future planting.",
    ],
    "Gray_Leaf_Spot": [
        "Rotate crops and manage residue, since the pathogen survives in old corn debris.",
        "Improve field drainage and airflow to reduce prolonged leaf wetness.",
        "Consider resistant hybrids for future seasons if this disease recurs.",
    ],
}

SEVERITY_TIPS = {
    "Mild": [
        "Severity is currently mild — prioritize monitoring and cultural controls before considering chemical treatment.",
    ],
    "Moderate": [
        "Severity is moderate — consider intervention soon; consult a local expert about treatment thresholds for this crop.",
    ],
    "Severe": [
        "Severity is severe — prompt action is recommended; consult a local agronomist or extension service as soon as possible.",
        "Assess whether severely affected plants/sections should be isolated or removed to protect the rest of the field.",
    ],
}

RISK_TIPS = {
    "Low": [
        "Environmental risk is currently low; standard monitoring should be sufficient.",
    ],
    "Medium": [
        "Environmental conditions moderately favor disease spread — increase monitoring frequency this week.",
    ],
    "High": [
        "Environmental conditions strongly favor disease spread — inspect the field more frequently "
        "and consider preventive cultural measures (spacing, drainage, airflow) immediately.",
    ],
}


def _disease_family_key(predicted_disease: str) -> str:
    # "Tomato___Early_Blight" -> "Early_Blight"
    parts = predicted_disease.split("___")
    return parts[1] if len(parts) > 1 else predicted_disease


def generate_recommendations(predicted_disease: str, severity_class: str, field_risk_level: str) -> list:
    if "Healthy" in predicted_disease:
        recs = list(GENERAL_HEALTHY_TIPS)
        recs.extend(RISK_TIPS.get(field_risk_level, []))
        return recs

    family = _disease_family_key(predicted_disease)
    recs = []
    recs.extend(DISEASE_FAMILY_TIPS.get(family, [
        "Isolate/monitor affected plants and consult a local agricultural extension office for diagnosis confirmation."
    ]))
    recs.extend(SEVERITY_TIPS.get(severity_class, []))
    recs.extend(RISK_TIPS.get(field_risk_level, []))

    # de-duplicate while preserving order
    seen = set()
    unique_recs = []
    for r in recs:
        if r not in seen:
            unique_recs.append(r)
            seen.add(r)
    return unique_recs
