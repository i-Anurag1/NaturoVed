"""GET /health — simple liveness + model-status check used by the
frontend on load and by deployment health checks."""
from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.services import image_model, field_context

router = APIRouter(tags=["health"])

APP_VERSION = "2.0.0"


@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        image_model_loaded=image_model.is_model_loaded(),
        field_model_loaded=field_context.is_field_model_loaded(),
        version=APP_VERSION,
    )
