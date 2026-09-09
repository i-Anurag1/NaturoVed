# API Documentation (v2)

Interactive Swagger UI is always available at `/docs` and ReDoc at
`/redoc` when the backend is running. This document is a static
reference covering the same endpoints, organized by feature area, with
notes on v2-specific behavior.

Base URL (local): `http://localhost:8000`

---

## Health

### `GET /health`
Liveness + model-status check.

**Response 200:**
```json
{
  "status": "ok",
  "image_model_loaded": true,
  "field_model_loaded": true,
  "version": "2.0.0"
}
```
`image_model_loaded`/`field_model_loaded` being `false` is NOT an
error — it means the system is running in the documented fallback
modes (see README.md "Honesty Notes"). The frontend and all endpoints
remain fully functional either way.

---

## Prediction (core pipeline)

### `POST /predict`
Multipart form. Runs the full multimodal pipeline and returns a
complete diagnosis.

**Form fields:**

| Field | Type | Required | v1/v2 | Notes |
|---|---|---|---|---|
| `image` | file (JPEG/PNG, ≤8MB) | Yes | v1 | Leaf/crop image |
| `crop_type` | enum: tomato, potato, corn | Yes | v1 | |
| `growth_stage` | enum: seedling, vegetative, flowering, fruiting, maturity | Yes | v1 | |
| `temperature_c` | float (-10 to 60) | Yes | v1 | |
| `humidity_pct` | float (0-100) | Yes | v1 | |
| `rainfall_mm` | float (0-1000) | Yes | v1 | 7-day total |
| `soil_moisture_pct` | float (0-100) | Yes | v1 | |
| `location` | string | No | v1 | Free text |
| `fusion_method` | enum: late_fusion, feature_fusion | No (default `late_fusion`) | v2 | See "Fusion Methods" below |
| `weather_source` | enum: manual, auto | No (default `manual`) | v2 | Set by frontend after a `/weather` call; purely informational for analytics |

**Response 200** (all v2 fields have defaults, so v1-only clients
reading a subset of these keys are unaffected):
```json
{
  "id": 1,
  "created_at": "2026-09-03T07:12:00",
  "image_filename": "ee5a8c27...jpg",
  "crop_type": "tomato",
  "growth_stage": "vegetative",
  "location": "Test Field",
  "temperature_c": 27.0,
  "humidity_pct": 82.0,
  "rainfall_mm": 15.0,
  "soil_moisture_pct": 55.0,
  "predicted_disease": "Tomato___Early_Blight",
  "image_confidence": 0.83,
  "affected_area_pct": 14.2,
  "severity_class": "Mild",
  "field_risk_score": 0.71,
  "field_risk_level": "High",
  "final_confidence": 0.88,
  "explanation": "Image analysis: ... Fusion note: ...",
  "recommendations": ["...", "..."],
  "gradcam_image": "data:image/png;base64,...",
  "gradcam_available": true,
  "weather_source": "manual",
  "fusion_method": "late_fusion",
  "field_explanation": "The field-risk model's most influential factors...",
  "field_top_features": [
    {"feature": "humidity_pct", "display_name": "Humidity", "importance": 0.34}
  ]
}
```

**Errors:** `400` (bad image type/size/empty), `422` (invalid
crop_type/growth_stage/fusion_method/weather_source, or out-of-range
numeric field), `500` (unexpected pipeline failure — the uploaded image
file is cleaned up automatically in this case).

**Fusion Methods:**
- `late_fusion` (default, production): image model runs independently;
  field-context Random Forest assesses risk; a weighted formula adjusts
  the DISPLAYED CONFIDENCE (never the predicted label) based on whether
  field conditions are consistent with the diagnosis. See
  `backend/app/services/fusion.py`.
- `feature_fusion` (experimental): a jointly-trained neural network
  classifies directly from [image embedding + processed field
  features] concatenated together — this CAN change the predicted
  label, not just confidence. Requires a trained feature-fusion model
  artifact; if unavailable, the API automatically and transparently
  falls back to `late_fusion` and reports the ACTUAL method used in the
  response's `fusion_method` field (which may differ from what you
  requested). See `docs/evaluation_plan.md` Section 8 for measured
  comparison results and important caveats about synthetic training data.

### `GET /prediction/{id}`
Fetch one full prediction by ID. Returns the same schema as `/predict`.
`404` if not found.

### `GET /prediction/{id}/gradcam` (v2)
Regenerates the Grad-CAM heatmap on demand from the saved uploaded
image (not cached in the database — see "Why Grad-CAM isn't stored" in
`backend/app/routers/gradcam.py`).

**Response 200:**
```json
{
  "prediction_id": 1,
  "gradcam_image": "data:image/png;base64,...",
  "available": true,
  "message": null
}
```
When `available` is `false`, `message` explains why (typically: no
trained image model loaded). `404` if the prediction ID doesn't exist.

### `GET /prediction/{id}/report` (v2)
Streams a generated PDF diagnosis report (`application/pdf`) containing
the image, Grad-CAM overlay (if available), disease, confidence,
severity, affected area, field conditions, risk, explanation, and
recommendations. `404` if the prediction doesn't exist.

---

## History & Analytics

### `GET /history?limit=&offset=&crop_type=`
Paginated list of past predictions (summary view). `limit` defaults to
20 (max 100), `offset` defaults to 0, `crop_type` is an optional filter.

### `GET /dashboard/summary?recent_limit=` (v2)
Aggregate analytics across all stored predictions.

**Response 200:**
```json
{
  "total_predictions": 42,
  "disease_distribution": [{"disease": "Tomato___Early_Blight", "count": 12}],
  "severity_distribution": [{"severity": "Mild", "count": 20}],
  "risk_distribution": [{"risk_level": "Medium", "count": 25}],
  "recent_predictions": [ "HistoryItem[]" ],
  "model_performance": {
    "image_model_loaded": true,
    "field_model_loaded": true,
    "feature_fusion_available": false,
    "image_model_mode": "trained",
    "notes": "..."
  }
}
```

---

## Weather (v2)

### `GET /weather?latitude=&longitude=`
Fetches current temperature/humidity and recent rainfall/soil-moisture
proxy values from Open-Meteo (no API key required) for the given
coordinates. Used by the frontend's "Use Live Weather" button to
pre-fill the field-context form — manual entry always remains
available as a fallback and the user can edit any auto-filled value.

**Response 200:**
```json
{
  "temperature_c": 28.5,
  "humidity_pct": 75.0,
  "rainfall_mm": 20.5,
  "soil_moisture_pct": 25.0,
  "source": "open-meteo"
}
```
**Errors:** `422` (latitude/longitude out of valid range), `503`
(upstream weather API unreachable or returned an error — see
`backend/app/services/weather.py` for the full disclosure on live
network dependency and testing limitations in this project's sandboxed
development environment).

---

## Static Files

### `GET /uploads/{filename}`
Serves the raw uploaded image file (used by the frontend to display
the original image next to its Grad-CAM overlay). Not a JSON endpoint.

---

## Backward Compatibility Summary
Every v2 request field has a default value, and every v2 response
field is additive. A v1 client (or a v1 copy of this frontend) sending
only the original 7 required form fields, and reading only the
original 13 response fields, continues to work unmodified against this
v2 backend. This is verified by
`backend/tests/test_v2_features.py::test_v1_style_request_still_works`.
