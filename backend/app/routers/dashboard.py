"""GET /dashboard/summary — aggregate analytics over all stored
predictions: totals, disease/severity/risk distributions, recent
predictions, and a model-performance/status summary. All aggregation
is done with plain SQL group-by queries (no extra dependency)."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database import get_db
from app.models.db_models import Prediction
from app.models.schemas import (
    DashboardSummary, DiseaseDistributionItem, SeverityDistributionItem,
    RiskDistributionItem, ModelPerformanceSummary, HistoryItem,
)
from app.services import image_model, field_context, feature_fusion

router = APIRouter(tags=["dashboard"])


@router.get("/dashboard/summary", response_model=DashboardSummary)
def get_dashboard_summary(db: Session = Depends(get_db), recent_limit: int = 10):
    total = db.query(func.count(Prediction.id)).scalar() or 0

    disease_rows = (
        db.query(Prediction.predicted_disease, func.count(Prediction.id))
        .group_by(Prediction.predicted_disease)
        .order_by(func.count(Prediction.id).desc())
        .all()
    )
    severity_rows = (
        db.query(Prediction.severity_class, func.count(Prediction.id))
        .group_by(Prediction.severity_class)
        .all()
    )
    risk_rows = (
        db.query(Prediction.field_risk_level, func.count(Prediction.id))
        .group_by(Prediction.field_risk_level)
        .all()
    )
    recent = (
        db.query(Prediction)
        .order_by(Prediction.created_at.desc())
        .limit(recent_limit)
        .all()
    )

    image_mode = image_model.get_model_mode()
    notes_parts = []
    if image_mode != "trained":
        notes_parts.append("Image model is running in fallback heuristic mode (no trained .keras model found).")
    if not field_context.is_field_model_loaded():
        notes_parts.append("Field-risk model is using the rule-based fallback (no trained Random Forest found).")
    if not feature_fusion.is_feature_fusion_available():
        notes_parts.append("Experimental feature-level fusion model is not available.")
    notes = " ".join(notes_parts) or "All models loaded from trained artifacts."

    return DashboardSummary(
        total_predictions=total,
        disease_distribution=[DiseaseDistributionItem(disease=d, count=c) for d, c in disease_rows],
        severity_distribution=[SeverityDistributionItem(severity=s, count=c) for s, c in severity_rows],
        risk_distribution=[RiskDistributionItem(risk_level=r, count=c) for r, c in risk_rows],
        recent_predictions=[HistoryItem.model_validate(r) for r in recent],
        model_performance=ModelPerformanceSummary(
            image_model_loaded=image_model.is_model_loaded(),
            field_model_loaded=field_context.is_field_model_loaded(),
            feature_fusion_available=feature_fusion.is_feature_fusion_available(),
            image_model_mode=image_mode,
            notes=notes,
        ),
    )
