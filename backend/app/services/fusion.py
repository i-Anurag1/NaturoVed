"""
Multimodal fusion module.

The project brief asks us to compare three fusion strategies and pick
the most practical one for the MVP:

  1. Late fusion (decision-level): combine the two models' final
     outputs/scores with fixed or learned weights. Simple, robust,
     works even if one modality is weak/missing. No retraining needed
     when either sub-model changes.
  2. Feature-level fusion: concatenate the image model's penultimate
     embedding vector with the structured field features and train a
     single joint classifier on top. Usually the most accurate, but
     requires a paired, jointly-labeled dataset (image + field-context
     + outcome) which does not exist publicly for this problem, and
     requires retraining the whole fusion head whenever either input
     model changes.
  3. Rule-based fusion: hand-written if/else rules combining both
     outputs. Fully transparent but brittle and doesn't generalize.

DECISION FOR THE MVP: **Weighted late fusion**, because:
  - It works with the two models we CAN honestly train (image CNN/
    transfer model on PlantVillage-style images, RandomForest on
    synthetic-but-domain-grounded field data) without inventing a
    fake jointly-labeled dataset for feature-level fusion.
  - It is easy to explain and defend in a viva (explicit weights,
    explicit formula) — supports the "explainable" requirement.
  - It is simple for 4 undergraduates to implement, test, and tune
    within a semester.

How a more advanced multimodal model could be added later (documented
here and in docs/methodology.md "Future Work"): once real paired
image+field+outcome data is collected (e.g. via a field trial or
partnership with an agricultural extension office), replace this
module with a small joint neural network that takes the image
embedding (e.g. the 128-d Dense layer output before softmax) and the
structured features as a single concatenated input vector — this is
feature-level fusion, and the rest of the API contract (inputs/outputs
of `fuse_predictions`) would not need to change.

Fusion formula (weighted late fusion with a domain-consistency
adjustment):
    base_confidence = image_confidence
    if predicted_disease != Healthy AND field_risk_level in {Medium, High}:
        # field conditions are CONSISTENT with a disease being present
        # -> increase confidence in the image model's disease call
        final_confidence = base_confidence + (1 - base_confidence) * ALIGN_BONUS * risk_score
    elif predicted_disease != Healthy AND field_risk_level == Low:
        # field conditions do NOT typically favor disease -> the image
        # call is less certain from a field-epidemiology standpoint
        final_confidence = base_confidence * (1 - MISALIGN_PENALTY)
    else:
        # Healthy prediction: field risk does not change model confidence
        final_confidence = base_confidence
    final_confidence is clipped to [0.05, 0.99]
"""

ALIGN_BONUS = 0.25       # up to +25% of remaining headroom when field risk supports the disease call
MISALIGN_PENALTY = 0.15  # -15% relative confidence when field risk contradicts the disease call


def fuse_predictions(predicted_disease: str, image_confidence: float,
                      field_risk_level: str, field_risk_score: float) -> dict:
    is_healthy = "Healthy" in predicted_disease

    if is_healthy:
        final_confidence = image_confidence
        rationale = "Healthy prediction — field risk does not adjust confidence."
    elif field_risk_level in ("Medium", "High"):
        final_confidence = image_confidence + (1 - image_confidence) * ALIGN_BONUS * field_risk_score
        rationale = (f"Field conditions ({field_risk_level} risk) are consistent with disease "
                      "presence — confidence boosted.")
    else:  # disease predicted but field risk is Low
        final_confidence = image_confidence * (1 - MISALIGN_PENALTY)
        rationale = ("Field conditions (Low risk) are less typical for this disease — "
                      "confidence slightly reduced; recommend visual re-check.")

    final_confidence = float(min(max(final_confidence, 0.05), 0.99))

    return {
        "final_confidence": round(final_confidence, 4),
        "fusion_method": "weighted_late_fusion",
        "fusion_rationale": rationale,
    }
