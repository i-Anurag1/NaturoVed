"""
POST /predict and GET /prediction/{id}

POST /predict is the heart of the system. It:
  1. Validates the uploaded image (type, size) and the field-context
     form fields (via Pydantic/Form validation).
  2. Saves the image to disk under a unique filename.
  3. Runs disease prediction using EITHER:
       - the production image model + weighted late fusion (default), or
       - the experimental feature-level fusion model, if the caller
         opts in via fusion_method="feature_fusion" AND it is available
         (falls back to late fusion automatically otherwise, with this
         noted in the response's explanation text — see v2 notes below).
  4. Runs the severity module (OpenCV) on the same image.
  5. Runs the field-context module -> field_risk_level, field_risk_score,
     plus a feature-importance-based explanation (v2).
  6. Builds the explanation string.
  7. Runs the recommendation engine.
  8. Generates a Grad-CAM heatmap overlay when a trained image model
     produced the prediction (v2; None in fallback-heuristic mode).
  9. Persists everything to the database (weather_source and
     fusion_method are recorded per-row for analytics/dashboard use).
 10. Returns the full PredictionResponse to the frontend.

BACKWARD COMPATIBILITY: every v2 request field (`fusion_method`,
`weather_source`) has a default value, so any v1 client that only sends
the original fields continues to work unmodified. Every v2 response
field likewise has a default, so v1 clients that ignore new JSON keys
are unaffected.
"""
import os
import uuid
import json
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models.db_models import Prediction
from app.models.schemas import PredictionResponse, GrowthStage, CropType, FieldFeatureImportance
from app.services import image_model, severity, field_context, fusion, recommendations, feature_fusion
from app.services.risk import build_explanation

router = APIRouter(tags=["prediction"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}
VALID_FUSION_METHODS = {"late_fusion", "feature_fusion"}
VALID_WEATHER_SOURCES = {"manual", "auto"}


def _validate_image(file: UploadFile, contents: bytes):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported image type: {file.content_type}. Use JPEG or PNG.")
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(status_code=400, detail=f"Image exceeds max size of {settings.MAX_UPLOAD_MB}MB.")
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")


def _row_to_response(row: Prediction) -> PredictionResponse:
    field_explanation_text, field_top_features = None, []
    try:
        exp = field_context.explain_field_risk(
            crop_type=row.crop_type, growth_stage=row.growth_stage,
            temperature_c=row.temperature_c, humidity_pct=row.humidity_pct,
            rainfall_mm=row.rainfall_mm, soil_moisture_pct=row.soil_moisture_pct,
        )
        field_explanation_text = exp["explanation"]
        field_top_features = [FieldFeatureImportance(**f) for f in exp["top_features"]]
    except Exception:
        pass

    gradcam_image = None
    if row.gradcam_available:
        try:
            image_path = os.path.join(settings.UPLOAD_DIR, row.image_filename)
            if os.path.exists(image_path):
                gradcam_image = image_model.generate_gradcam_for_disease(image_path, row.predicted_disease)
        except Exception:
            gradcam_image = None

    return PredictionResponse(
        id=row.id,
        created_at=row.created_at,
        image_filename=row.image_filename,
        crop_type=row.crop_type,
        growth_stage=row.growth_stage,
        location=row.location,
        temperature_c=row.temperature_c,
        humidity_pct=row.humidity_pct,
        rainfall_mm=row.rainfall_mm,
        soil_moisture_pct=row.soil_moisture_pct,
        predicted_disease=row.predicted_disease,
        image_confidence=row.image_confidence,
        affected_area_pct=row.affected_area_pct,
        severity_class=row.severity_class,
        field_risk_score=row.field_risk_score,
        field_risk_level=row.field_risk_level,
        final_confidence=row.final_confidence,
        explanation=row.explanation,
        recommendations=json.loads(row.recommendations),
        gradcam_image=gradcam_image,
        gradcam_available=row.gradcam_available,
        weather_source=row.weather_source,
        fusion_method=row.fusion_method,
        field_explanation=field_explanation_text,
        field_top_features=field_top_features,
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(..., description="Leaf/crop image (JPEG or PNG)"),
    crop_type: CropType = Form(...),
    growth_stage: GrowthStage = Form(...),
    temperature_c: float = Form(...),
    humidity_pct: float = Form(...),
    rainfall_mm: float = Form(...),
    soil_moisture_pct: float = Form(...),
    location: str = Form(default=None),
    fusion_method: str = Form(default="late_fusion", description="'late_fusion' (default) or 'feature_fusion' (experimental)"),
    weather_source: str = Form(default="manual", description="'manual' (default) or 'auto' (values came from GET /weather)"),
    db: Session = Depends(get_db),
):
    contents = await image.read()
    _validate_image(image, contents)

    if not (0 <= humidity_pct <= 100):
        raise HTTPException(status_code=422, detail="humidity_pct must be between 0 and 100")
    if not (0 <= soil_moisture_pct <= 100):
        raise HTTPException(status_code=422, detail="soil_moisture_pct must be between 0 and 100")
    if fusion_method not in VALID_FUSION_METHODS:
        raise HTTPException(status_code=422, detail=f"fusion_method must be one of {sorted(VALID_FUSION_METHODS)}")
    if weather_source not in VALID_WEATHER_SOURCES:
        raise HTTPException(status_code=422, detail=f"weather_source must be one of {sorted(VALID_WEATHER_SOURCES)}")

    # 1. Save image
    ext = os.path.splitext(image.filename or "upload.jpg")[1] or ".jpg"
    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(settings.UPLOAD_DIR, unique_name)
    with open(save_path, "wb") as f:
        f.write(contents)

    try:
        # 2. Field-context risk assessment (computed first — needed either
        #    way, and feature-fusion also needs the raw field values)
        field_result = field_context.assess_field_risk(
            crop_type=crop_type.value,
            growth_stage=growth_stage.value,
            temperature_c=temperature_c,
            humidity_pct=humidity_pct,
            rainfall_mm=rainfall_mm,
            soil_moisture_pct=soil_moisture_pct,
        )

        # 3. Disease prediction: feature-level fusion (if requested & available)
        #    or the production image model + weighted late fusion (default).
        actual_fusion_method = "late_fusion"
        fusion_rationale = None

        feature_fusion_result = None
        if fusion_method == "feature_fusion":
            feature_fusion_result = feature_fusion.predict_with_feature_fusion(
                image_path=save_path, crop_type=crop_type.value, growth_stage=growth_stage.value,
                temperature_c=temperature_c, humidity_pct=humidity_pct,
                rainfall_mm=rainfall_mm, soil_moisture_pct=soil_moisture_pct,
            )

        if feature_fusion_result is not None:
            predicted_disease = feature_fusion_result["predicted_disease"]
            image_confidence = feature_fusion_result["confidence"]
            final_confidence = feature_fusion_result["confidence"]
            actual_fusion_method = "feature_fusion"
            fusion_rationale = (
                "Prediction produced by the experimental feature-level fusion model, which jointly "
                "classifies from the image embedding and processed field-context features in a single "
                "trained network (see docs/methodology.md Section 6 and docs/evaluation_plan.md Section 8)."
            )
        else:
            if fusion_method == "feature_fusion":
                fallback_note = " (requested feature_fusion was unavailable; used late_fusion instead)"
            else:
                fallback_note = ""

            img_result = image_model.predict_disease(save_path, crop_type.value)
            predicted_disease = img_result["predicted_disease"]
            image_confidence = img_result["confidence"]

            fusion_result = fusion.fuse_predictions(
                predicted_disease=predicted_disease,
                image_confidence=image_confidence,
                field_risk_level=field_result["risk_level"],
                field_risk_score=field_result["risk_score"],
            )
            final_confidence = fusion_result["final_confidence"]
            fusion_rationale = fusion_result["fusion_rationale"] + fallback_note

        # 4. Severity estimation
        sev_result = severity.estimate_severity(save_path, predicted_disease)

        # 5. Explanation
        explanation = build_explanation(
            predicted_disease=predicted_disease,
            image_confidence=image_confidence,
            severity_class=sev_result["severity_class"],
            affected_area_pct=sev_result["affected_area_pct"],
            field_risk_level=field_result["risk_level"],
            fusion_rationale=fusion_rationale,
        )

        # 6. Recommendations
        recs = recommendations.generate_recommendations(
            predicted_disease=predicted_disease,
            severity_class=sev_result["severity_class"],
            field_risk_level=field_result["risk_level"],
        )

        # 7. Grad-CAM (only meaningful for a trained image model; None otherwise)
        gradcam_available = False
        if image_model.is_model_loaded():
            gradcam_check = image_model.generate_gradcam_for_disease(save_path, predicted_disease)
            gradcam_available = gradcam_check is not None

    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(status_code=500, detail=f"Prediction pipeline failed: {str(e)}")

    # 8. Persist to database
    row = Prediction(
        image_filename=unique_name,
        crop_type=crop_type.value,
        growth_stage=growth_stage.value,
        location=location,
        temperature_c=temperature_c,
        humidity_pct=humidity_pct,
        rainfall_mm=rainfall_mm,
        soil_moisture_pct=soil_moisture_pct,
        predicted_disease=predicted_disease,
        image_confidence=image_confidence,
        affected_area_pct=sev_result["affected_area_pct"],
        severity_class=sev_result["severity_class"],
        field_risk_score=field_result["risk_score"],
        field_risk_level=field_result["risk_level"],
        final_confidence=final_confidence,
        explanation=explanation,
        recommendations=json.dumps(recs),
        gradcam_available=gradcam_available,
        weather_source=weather_source,
        fusion_method=actual_fusion_method,
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    return _row_to_response(row)


@router.get("/prediction/{prediction_id}", response_model=PredictionResponse)
def get_prediction(prediction_id: int, db: Session = Depends(get_db)):
    row = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Prediction {prediction_id} not found")
    return _row_to_response(row)
