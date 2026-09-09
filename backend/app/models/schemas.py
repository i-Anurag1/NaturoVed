"""
Pydantic schemas for request validation and response serialization.
These define the exact API contract used by the React frontend.
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum


class GrowthStage(str, Enum):
    seedling = "seedling"
    vegetative = "vegetative"
    flowering = "flowering"
    fruiting = "fruiting"
    maturity = "maturity"


class CropType(str, Enum):
    tomato = "tomato"
    potato = "potato"
    corn = "corn"  # maize


class FieldContextIn(BaseModel):
    """Structured field-context data submitted alongside the image."""
    crop_type: CropType
    growth_stage: GrowthStage
    location: Optional[str] = Field(default=None, description="Free-text region/village/district name")
    temperature_c: float = Field(..., ge=-10, le=60, description="Ambient temperature in Celsius")
    humidity_pct: float = Field(..., ge=0, le=100, description="Relative humidity percentage")
    rainfall_mm: float = Field(..., ge=0, le=1000, description="Rainfall in the last 7 days (mm)")
    soil_moisture_pct: float = Field(..., ge=0, le=100, description="Soil moisture percentage")

    class Config:
        json_schema_extra = {
            "example": {
                "crop_type": "tomato",
                "growth_stage": "vegetative",
                "location": "Coimbatore, TN",
                "temperature_c": 29.5,
                "humidity_pct": 78,
                "rainfall_mm": 12.5,
                "soil_moisture_pct": 45,
            }
        }


class FieldFeatureImportance(BaseModel):
    feature: str
    display_name: str
    importance: float


class PredictionResponse(BaseModel):
    """Full response returned by POST /predict and GET /prediction/{id}."""
    id: int
    created_at: datetime
    image_filename: str = Field(description="Filename under /uploads/ — build the viewable URL as f'{API_BASE_URL}/uploads/{image_filename}'.")

    crop_type: str
    growth_stage: str
    location: Optional[str]
    temperature_c: float
    humidity_pct: float
    rainfall_mm: float
    soil_moisture_pct: float

    predicted_disease: str
    image_confidence: float

    affected_area_pct: Optional[float]
    severity_class: str

    field_risk_score: float
    field_risk_level: str

    final_confidence: float

    explanation: str
    recommendations: List[str]

    # --- v2 additions (all optional/defaulted so existing v1 API
    # consumers are unaffected if they ignore new fields) ---
    gradcam_image: Optional[str] = Field(
        default=None,
        description="Base64 data-URI PNG Grad-CAM overlay, only present when a trained image model produced this prediction.",
    )
    gradcam_available: bool = Field(
        default=False,
        description="Whether a Grad-CAM heatmap can be (re)generated for this prediction via GET /prediction/{id}/gradcam.",
    )
    weather_source: str = Field(default="manual", description="'manual' (user-entered) or 'auto' (fetched from weather API).")
    fusion_method: str = Field(default="late_fusion", description="'late_fusion' (baseline) or 'feature_fusion' (experimental).")
    field_explanation: Optional[str] = Field(
        default=None,
        description="Human-readable explanation of which field-context features most influenced the risk assessment.",
    )
    field_top_features: List[FieldFeatureImportance] = Field(default_factory=list)

    class Config:
        from_attributes = True


class GradCamResponse(BaseModel):
    prediction_id: int
    gradcam_image: Optional[str]
    available: bool
    message: Optional[str] = None


class WeatherResponse(BaseModel):
    """Response for GET /weather — used by the frontend's 'Use live
    weather' button to auto-fill the field-context form."""
    temperature_c: float
    humidity_pct: float
    rainfall_mm: float
    soil_moisture_pct: float
    source: str = "open-meteo"


class DiseaseDistributionItem(BaseModel):
    disease: str
    count: int


class SeverityDistributionItem(BaseModel):
    severity: str
    count: int


class RiskDistributionItem(BaseModel):
    risk_level: str
    count: int


class ModelPerformanceSummary(BaseModel):
    model_config = {"protected_namespaces": ()}

    image_model_loaded: bool
    field_model_loaded: bool
    feature_fusion_available: bool
    image_model_mode: str
    notes: str


class DashboardSummary(BaseModel):
    """Response for GET /dashboard/summary."""
    model_config = {"protected_namespaces": ()}

    total_predictions: int
    disease_distribution: List[DiseaseDistributionItem]
    severity_distribution: List[SeverityDistributionItem]
    risk_distribution: List[RiskDistributionItem]
    recent_predictions: List["HistoryItem"]
    model_performance: ModelPerformanceSummary


class HistoryItem(BaseModel):
    """Lightweight summary used in GET /history (avoids sending full payloads)."""
    id: int
    created_at: datetime
    crop_type: str
    predicted_disease: str
    severity_class: str
    field_risk_level: str
    final_confidence: float

    class Config:
        from_attributes = True


class HealthResponse(BaseModel):
    status: str
    image_model_loaded: bool
    field_model_loaded: bool
    version: str


# Resolve the forward reference to HistoryItem used in DashboardSummary
# (HistoryItem is defined below DashboardSummary in this file).
DashboardSummary.model_rebuild()
