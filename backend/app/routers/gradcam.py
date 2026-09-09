"""GET /prediction/{id}/gradcam — (re)generates the Grad-CAM heatmap
overlay for a stored prediction on demand. Grad-CAM images are NOT
stored as base64 in the database (would bloat it — see db_models.py);
instead we regenerate from the saved uploaded image file each time,
using the currently loaded trained model."""
import os
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models.db_models import Prediction
from app.models.schemas import GradCamResponse
from app.services import image_model

router = APIRouter(tags=["explainability"])


@router.get("/prediction/{prediction_id}/gradcam", response_model=GradCamResponse)
def get_gradcam(prediction_id: int, db: Session = Depends(get_db)):
    row = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Prediction {prediction_id} not found")

    if not image_model.is_model_loaded():
        return GradCamResponse(
            prediction_id=prediction_id, gradcam_image=None, available=False,
            message="Grad-CAM requires a trained image model; the system is currently running "
                    "in fallback heuristic mode. Train a model with `python -m app.ml.train_transfer`.",
        )

    image_path = os.path.join(settings.UPLOAD_DIR, row.image_filename)
    if not os.path.exists(image_path):
        return GradCamResponse(
            prediction_id=prediction_id, gradcam_image=None, available=False,
            message="Original uploaded image file is no longer available on disk.",
        )

    gradcam_data_uri = image_model.generate_gradcam_for_disease(image_path, row.predicted_disease)
    if gradcam_data_uri is None:
        return GradCamResponse(
            prediction_id=prediction_id, gradcam_image=None, available=False,
            message="Grad-CAM could not be computed for this model/image (see server logs for details).",
        )

    return GradCamResponse(prediction_id=prediction_id, gradcam_image=gradcam_data_uri, available=True)
