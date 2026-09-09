"""
SQLAlchemy ORM models.

A single `Prediction` table stores every request end-to-end: the raw
inputs (image path + field context) and every intermediate/final
output of the pipeline (disease, confidence, severity, affected area,
risk, explanation, recommendations). This makes /prediction/{id} and
/history trivial to implement and gives full auditability for the
academic report (e.g. "show 5 example runs").
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean
from datetime import datetime
from app.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # --- raw inputs ---
    image_filename = Column(String, nullable=False)
    crop_type = Column(String, nullable=False)
    growth_stage = Column(String, nullable=False)
    location = Column(String, nullable=True)
    temperature_c = Column(Float, nullable=False)
    humidity_pct = Column(Float, nullable=False)
    rainfall_mm = Column(Float, nullable=False)
    soil_moisture_pct = Column(Float, nullable=False)

    # --- image model output ---
    predicted_disease = Column(String, nullable=False)
    image_confidence = Column(Float, nullable=False)

    # --- severity module output ---
    affected_area_pct = Column(Float, nullable=True)
    severity_class = Column(String, nullable=False)  # Mild / Moderate / Severe

    # --- field context module output ---
    field_risk_score = Column(Float, nullable=False)   # 0-1
    field_risk_level = Column(String, nullable=False)  # Low / Medium / High

    # --- fusion output ---
    final_confidence = Column(Float, nullable=False)

    # --- explanation & recommendations ---
    explanation = Column(Text, nullable=False)
    recommendations = Column(Text, nullable=False)  # JSON-encoded list[str]

    # --- v2 additions (all nullable/defaulted for backward compatibility
    # with rows created by v1 of this project) ---
    # Grad-CAM is NOT persisted as base64 in the DB (would bloat it);
    # instead we store whether it CAN be regenerated on demand from the
    # saved image file, and regenerate via GET /prediction/{id}/gradcam.
    gradcam_available = Column(Boolean, default=False, nullable=False)
    # weather source used for field context: "manual" (user-entered) or
    # "auto" (fetched from the weather API) — see app/services/weather.py
    weather_source = Column(String, default="manual", nullable=False)
    # which fusion strategy produced final_confidence: "late_fusion"
    # (default/baseline) or "feature_fusion" (experimental) — see
    # app/services/feature_fusion.py and docs/methodology.md
    fusion_method = Column(String, default="late_fusion", nullable=False)
