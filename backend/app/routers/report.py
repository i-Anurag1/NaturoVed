"""GET /prediction/{id}/report — generates and streams a PDF diagnosis
report for a stored prediction (image, disease, confidence, severity,
affected area, field conditions, risk, explanation, recommendations,
and Grad-CAM overlay when available)."""
import os
import tempfile
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.db_models import Prediction
from app.services.report_generator import generate_pdf_report
from app.services import field_context

router = APIRouter(tags=["reports"])


@router.get("/prediction/{prediction_id}/report")
def download_report(prediction_id: int, db: Session = Depends(get_db)):
    row = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Prediction {prediction_id} not found")

    # Attach a transient field-explanation string (not persisted in the
    # DB schema) so the PDF can include it without a migration.
    try:
        field_explanation = field_context.explain_field_risk(
            crop_type=row.crop_type, growth_stage=row.growth_stage,
            temperature_c=row.temperature_c, humidity_pct=row.humidity_pct,
            rainfall_mm=row.rainfall_mm, soil_moisture_pct=row.soil_moisture_pct,
        )
        row._field_explanation_text = field_explanation["explanation"]
    except Exception:
        row._field_explanation_text = None

    output_path = os.path.join(tempfile.gettempdir(), f"crop_report_{prediction_id}.pdf")
    try:
        generate_pdf_report(row, output_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF report: {e}")

    return FileResponse(
        output_path,
        media_type="application/pdf",
        filename=f"crop_diagnosis_report_{prediction_id}.pdf",
    )
