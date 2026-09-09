"""
Basic API test suite using FastAPI's TestClient (httpx under the hood).

Run with:
    cd backend
    pytest -v

These tests exercise the full pipeline (image + field context ->
prediction) using a small synthetically generated test image, so they
work in CI/without any real dataset or trained model (fallback
heuristic mode is exercised when no trained model file is present).
"""
import io
import os
import sys
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app  # noqa: E402
from app.database import init_db  # noqa: E402

# Ensure tables exist before any test runs (mirrors what the FastAPI
# startup event does, but called directly so it's not dependent on
# TestClient lifespan behavior across httpx/FastAPI versions).
init_db()

client = TestClient(app)


def _make_test_image_bytes() -> bytes:
    """Create a small synthetic green-leaf-with-brown-spots JPEG in memory."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :] = [34, 139, 34]
    img[20:40, 20:40] = [101, 67, 33]  # brown lesion patch
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "image_model_loaded" in data
    assert "field_model_loaded" in data


def test_predict_success():
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
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "predicted_disease" in data
    assert data["crop_type"] == "tomato"
    assert 0.0 <= data["image_confidence"] <= 1.0
    assert data["severity_class"] in {"Mild", "Moderate", "Severe", "None"}
    assert data["field_risk_level"] in {"Low", "Medium", "High"}
    assert isinstance(data["recommendations"], list)
    assert len(data["recommendations"]) > 0


def _create_prediction_and_get_id() -> int:
    """Helper (not a test itself) that creates a prediction and returns its id,
    for reuse by tests that need an existing record to look up."""
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
        },
    )
    assert resp.status_code == 200
    return resp.json()["id"]


def test_predict_missing_field_returns_422():
    image_bytes = _make_test_image_bytes()
    resp = client.post(
        "/predict",
        files={"image": ("leaf.jpg", image_bytes, "image/jpeg")},
        data={
            "crop_type": "tomato",
            "growth_stage": "vegetative",
            # temperature_c intentionally missing
            "humidity_pct": "80",
            "rainfall_mm": "12",
            "soil_moisture_pct": "50",
        },
    )
    assert resp.status_code == 422


def test_predict_invalid_crop_type_returns_422():
    image_bytes = _make_test_image_bytes()
    resp = client.post(
        "/predict",
        files={"image": ("leaf.jpg", image_bytes, "image/jpeg")},
        data={
            "crop_type": "wheat",  # not a supported crop in this MVP
            "growth_stage": "vegetative",
            "temperature_c": "27",
            "humidity_pct": "80",
            "rainfall_mm": "12",
            "soil_moisture_pct": "50",
        },
    )
    assert resp.status_code == 422


def test_predict_rejects_bad_file_type():
    resp = client.post(
        "/predict",
        files={"image": ("notanimage.txt", b"hello world", "text/plain")},
        data={
            "crop_type": "tomato",
            "growth_stage": "vegetative",
            "temperature_c": "27",
            "humidity_pct": "80",
            "rainfall_mm": "12",
            "soil_moisture_pct": "50",
        },
    )
    assert resp.status_code == 400


def test_get_prediction_by_id():
    created_id = _create_prediction_and_get_id()
    resp = client.get(f"/prediction/{created_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == created_id


def test_get_prediction_not_found():
    resp = client.get("/prediction/999999")
    assert resp.status_code == 404


def test_history_returns_list():
    _create_prediction_and_get_id()
    resp = client.get("/history?limit=5")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert "predicted_disease" in data[0]
