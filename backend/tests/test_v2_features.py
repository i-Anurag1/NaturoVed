"""
v2 feature tests: Grad-CAM, feature-level fusion, field-context
explainability, analytics dashboard, PDF report generation, and
weather integration (unit-tested against a mocked response — see
module docstring in app/services/weather.py for why live API calls are
not exercised in this sandboxed environment).

These tests are designed to pass whether or not trained model
artifacts (disease_model.keras, feature_fusion_model.keras) are present
— they assert graceful, documented fallback behavior when artifacts
are missing, and assert real functional behavior when they exist. This
mirrors the project's "fail soft, disclose honestly" design used
throughout app/services/.

Run with:
    cd backend
    pytest tests/test_v2_features.py -v
"""
import io
import os
import sys
import json
import pytest
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app  # noqa: E402
from app.database import init_db  # noqa: E402
from app.config import settings  # noqa: E402

init_db()
client = TestClient(app)


def _make_test_image_bytes() -> bytes:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :] = [34, 139, 34]
    img[20:40, 20:40] = [101, 67, 33]
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


def _submit_prediction(fusion_method="late_fusion"):
    image_bytes = _make_test_image_bytes()
    resp = client.post(
        "/predict",
        files={"image": ("leaf.jpg", image_bytes, "image/jpeg")},
        data={
            "crop_type": "tomato",
            "growth_stage": "vegetative",
            "temperature_c": "27",
            "humidity_pct": "80",
            "rainfall_mm": "12",
            "soil_moisture_pct": "50",
            "location": "Test Field",
            "fusion_method": fusion_method,
        },
    )
    assert resp.status_code == 200
    return resp.json()


# ---------------------------------------------------------------------------
# Backward compatibility: v1-style request (no v2 fields) must still work
# ---------------------------------------------------------------------------
def test_v1_style_request_still_works():
    """A client that only sends the original v1 fields (no fusion_method,
    no weather_source) must get a 200 with sensible v2 defaults applied."""
    image_bytes = _make_test_image_bytes()
    resp = client.post(
        "/predict",
        files={"image": ("leaf.jpg", image_bytes, "image/jpeg")},
        data={
            "crop_type": "tomato",
            "growth_stage": "vegetative",
            "temperature_c": "27",
            "humidity_pct": "80",
            "rainfall_mm": "12",
            "soil_moisture_pct": "50",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["fusion_method"] == "late_fusion"
    assert data["weather_source"] == "manual"
    assert "gradcam_available" in data
    assert "field_explanation" in data


# ---------------------------------------------------------------------------
# Field-context explainability
# ---------------------------------------------------------------------------
def test_field_explanation_present_and_well_formed():
    data = _submit_prediction()
    assert data["field_explanation"] is not None
    assert isinstance(data["field_top_features"], list)
    if data["field_top_features"]:  # trained field model present in this environment
        top = data["field_top_features"][0]
        assert "feature" in top and "display_name" in top and "importance" in top
        assert 0.0 <= top["importance"] <= 1.0


def test_field_context_explain_function_directly():
    from app.services import field_context
    result = field_context.explain_field_risk(
        crop_type="tomato", growth_stage="vegetative",
        temperature_c=27, humidity_pct=82, rainfall_mm=15, soil_moisture_pct=55,
    )
    assert "explanation" in result
    assert "method" in result
    assert result["method"] in {"random_forest_feature_importance", "unavailable"}


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
def test_gradcam_field_reflects_model_availability():
    from app.services import image_model
    data = _submit_prediction()
    assert data["gradcam_available"] == image_model.is_model_loaded()
    if image_model.is_model_loaded():
        assert data["gradcam_image"] is not None
        assert data["gradcam_image"].startswith("data:image/png;base64,")
    else:
        assert data["gradcam_image"] is None


def test_gradcam_endpoint_for_existing_prediction():
    data = _submit_prediction()
    pred_id = data["id"]
    resp = client.get(f"/prediction/{pred_id}/gradcam")
    assert resp.status_code == 200
    gc = resp.json()
    assert gc["prediction_id"] == pred_id
    assert isinstance(gc["available"], bool)
    if gc["available"]:
        assert gc["gradcam_image"] is not None


def test_gradcam_endpoint_for_missing_prediction_returns_404():
    resp = client.get("/prediction/999999/gradcam")
    assert resp.status_code == 404


def test_gradcam_module_shape_on_synthetic_trained_model():
    """Directly exercises app.services.gradcam against whatever model is
    currently loaded (trained or absent). If a trained model is loaded,
    asserts the heatmap has the expected 2D shape and [0,1] value range."""
    from app.services import image_model, gradcam
    if not image_model.is_model_loaded():
        pytest.skip("No trained image model loaded in this environment — Grad-CAM has nothing to test against.")

    model = image_model.get_loaded_model()
    x = np.random.uniform(0, 255, size=(1, 224, 224, 3)).astype(np.float32)
    heatmap = gradcam.compute_gradcam_heatmap(model, x, class_idx=0)
    assert heatmap.ndim == 2
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Feature-level fusion
# ---------------------------------------------------------------------------
def test_feature_fusion_request_succeeds_or_falls_back():
    """Whether or not the experimental feature-fusion model is available,
    requesting fusion_method=feature_fusion must return 200 — either using
    real feature fusion, or transparently falling back to late fusion."""
    data = _submit_prediction(fusion_method="feature_fusion")
    assert data["fusion_method"] in {"feature_fusion", "late_fusion"}


def test_invalid_fusion_method_rejected():
    image_bytes = _make_test_image_bytes()
    resp = client.post(
        "/predict",
        files={"image": ("leaf.jpg", image_bytes, "image/jpeg")},
        data={
            "crop_type": "tomato", "growth_stage": "vegetative",
            "temperature_c": "27", "humidity_pct": "80", "rainfall_mm": "12", "soil_moisture_pct": "50",
            "fusion_method": "not_a_real_method",
        },
    )
    assert resp.status_code == 422


def test_feature_fusion_service_unavailable_returns_none_gracefully(tmp_path, monkeypatch):
    """Simulates missing feature-fusion artifacts and confirms the service
    reports unavailable rather than raising."""
    from app.services import feature_fusion
    monkeypatch.setattr(feature_fusion, "_fusion_model", None)
    monkeypatch.setattr(feature_fusion, "_field_preprocessor", None)
    monkeypatch.setattr(feature_fusion, "_load_attempted", True)  # skip real load attempt
    result = feature_fusion.predict_with_feature_fusion(
        image_path="/nonexistent/path.jpg", crop_type="tomato", growth_stage="vegetative",
        temperature_c=25, humidity_pct=60, rainfall_mm=5, soil_moisture_pct=40,
    )
    assert result is None


# ---------------------------------------------------------------------------
# Analytics dashboard
# ---------------------------------------------------------------------------
def test_dashboard_summary_structure():
    _submit_prediction()  # ensure at least one row exists
    resp = client.get("/dashboard/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_predictions"] >= 1
    assert isinstance(data["disease_distribution"], list)
    assert isinstance(data["severity_distribution"], list)
    assert isinstance(data["risk_distribution"], list)
    assert isinstance(data["recent_predictions"], list)
    assert "image_model_loaded" in data["model_performance"]
    assert "feature_fusion_available" in data["model_performance"]


def test_dashboard_recent_predictions_respect_limit():
    for _ in range(3):
        _submit_prediction()
    resp = client.get("/dashboard/summary?recent_limit=2")
    assert resp.status_code == 200
    assert len(resp.json()["recent_predictions"]) <= 2


# ---------------------------------------------------------------------------
# PDF report generation
# ---------------------------------------------------------------------------
def test_pdf_report_generation_for_existing_prediction():
    data = _submit_prediction()
    pred_id = data["id"]
    resp = client.get(f"/prediction/{pred_id}/report")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.content[:4] == b"%PDF"  # valid PDF magic bytes
    assert len(resp.content) > 1000  # not an empty/broken file


def test_pdf_report_missing_prediction_returns_404():
    resp = client.get("/prediction/999999/report")
    assert resp.status_code == 404


def test_report_generator_function_directly(tmp_path):
    from app.services.report_generator import generate_pdf_report
    from app.models.db_models import Prediction

    fake_row = Prediction(
        id=1, image_filename="nonexistent.jpg", crop_type="tomato", growth_stage="vegetative",
        location="Test", temperature_c=27, humidity_pct=80, rainfall_mm=10, soil_moisture_pct=50,
        predicted_disease="Tomato___Early_Blight", image_confidence=0.8,
        affected_area_pct=15.0, severity_class="Mild",
        field_risk_score=0.6, field_risk_level="Medium", final_confidence=0.75,
        explanation="Test explanation.", recommendations=json.dumps(["Tip 1", "Tip 2"]),
        gradcam_available=False, weather_source="manual", fusion_method="late_fusion",
    )
    out_path = str(tmp_path / "test_report.pdf")
    result_path = generate_pdf_report(fake_row, out_path)
    assert os.path.exists(result_path)
    with open(result_path, "rb") as f:
        assert f.read(4) == b"%PDF"


# ---------------------------------------------------------------------------
# Weather integration — mocked (see app/services/weather.py docstring for
# why live network calls to api.open-meteo.com are not exercised here)
# ---------------------------------------------------------------------------
def _mock_open_meteo_response():
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "current": {"temperature_2m": 28.5, "relative_humidity_2m": 75.0},
        "daily": {"precipitation_sum": [2.0, 0.0, 5.5, 0.0, 12.0, 0.0, 1.0, 0.0]},
        "hourly": {"soil_moisture_0_to_1cm": [0.25] * 24},
    }
    return mock_resp


def test_weather_service_parses_mocked_response():
    from app.services import weather
    with patch("app.services.weather.requests.get", return_value=_mock_open_meteo_response()):
        result = weather.fetch_live_weather(latitude=11.0, longitude=77.0)
    assert result["temperature_c"] == 28.5
    assert result["humidity_pct"] == 75.0
    assert result["rainfall_mm"] == pytest.approx(20.5)
    assert 0 <= result["soil_moisture_pct"] <= 100
    assert result["source"] == "open-meteo"


def test_weather_service_invalid_coordinates_raises():
    from app.services import weather
    with pytest.raises(weather.WeatherUnavailableError):
        weather.fetch_live_weather(latitude=999, longitude=0)


def test_weather_service_network_failure_raises_recoverable_error():
    from app.services import weather
    import requests as requests_lib
    with patch("app.services.weather.requests.get", side_effect=requests_lib.ConnectionError("no network")):
        with pytest.raises(weather.WeatherUnavailableError):
            weather.fetch_live_weather(latitude=11.0, longitude=77.0)


def test_weather_endpoint_with_mocked_service():
    with patch("app.routers.weather.fetch_live_weather", return_value={
        "temperature_c": 28.5, "humidity_pct": 75.0, "rainfall_mm": 20.5,
        "soil_moisture_pct": 25.0, "source": "open-meteo",
    }):
        resp = client.get("/weather", params={"latitude": 11.0, "longitude": 77.0})
    assert resp.status_code == 200
    data = resp.json()
    assert data["temperature_c"] == 28.5
    assert data["source"] == "open-meteo"


def test_weather_endpoint_service_unavailable_returns_503():
    from app.services.weather import WeatherUnavailableError
    with patch("app.routers.weather.fetch_live_weather", side_effect=WeatherUnavailableError("mocked failure")):
        resp = client.get("/weather", params={"latitude": 11.0, "longitude": 77.0})
    assert resp.status_code == 503


def test_weather_endpoint_rejects_invalid_coordinates():
    resp = client.get("/weather", params={"latitude": 999, "longitude": 0})
    assert resp.status_code == 422  # FastAPI query validation (ge=-90, le=90)


# ---------------------------------------------------------------------------
# Database persistence of v2 columns
# ---------------------------------------------------------------------------
def test_v2_columns_persist_correctly():
    data = _submit_prediction(fusion_method="late_fusion")
    resp = client.get(f"/prediction/{data['id']}")
    assert resp.status_code == 200
    refetched = resp.json()
    assert refetched["fusion_method"] == data["fusion_method"]
    assert refetched["weather_source"] == data["weather_source"]
    assert refetched["gradcam_available"] == data["gradcam_available"]
