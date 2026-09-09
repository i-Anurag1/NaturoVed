# Proposed Methodology, System Architecture & Dataset Description (v2)

## 1. System Architecture (text-based diagram, updated for v2)

```
                        ┌───────────────────────────────────────┐
                        │        React Frontend (SPA)            │
                        │ Home | Result | History | Dashboard    │
                        │ i18n (EN/TA) · Live-weather auto-fill  │
                        │ Grad-CAM viewer · PDF download button  │
                        └───────────────────┬─────────────────────┘
                                            │ HTTPS (JSON / multipart)
                                            ▼
                        ┌───────────────────────────────────────┐
                        │      FastAPI Backend (REST API)         │
                        │  GET  /health          GET /weather      │
                        │  POST /predict         GET /dashboard/summary │
                        │  GET  /prediction/{id}                    │
                        │  GET  /prediction/{id}/gradcam  (v2)      │
                        │  GET  /prediction/{id}/report   (v2, PDF) │
                        │  GET  /history                             │
                        └───────────────────┬─────────────────────┘
                                            │
     ┌──────────────┬───────────────────────┼───────────────────────┬──────────────┐
     ▼              ▼                       ▼                       ▼              ▼
┌─────────┐  ┌──────────────┐   ┌───────────────────────┐  ┌──────────────┐  ┌───────────┐
│ Image    │  │ Severity      │   │ Field-Context Module   │  │ Live Weather  │  │ Feature-  │
│ Disease  │  │ Estimation    │   │ (Random Forest)         │  │ Service (v2)  │  │ Level     │
│ Model    │  │ (OpenCV)      │   │ - Encode crop/stage     │  │ Open-Meteo,   │  │ Fusion    │
│ (Mobile- │  │ - HSV leaf/   │   │ - Scale numerics        │  │ no API key    │  │ (v2, exp- │
│ NetV2)   │  │   lesion seg. │   │ -> risk_level/score      │  │ -> temp/      │  │ erimental)│
│ -> disease│  │ -> affected_  │   │ -> feature_importances_  │  │    humidity/  │  │ [image emb│
│ + conf.  │  │   area_pct,   │   │    (v2 explainability)   │  │    rainfall/  │  │  + field  │
│          │  │   severity    │   │                          │  │    soil moist.│  │  feats]   │
└────┬─────┘  └──────┬────────┘   └────────────┬─────────────┘  └───────┬───────┘  │ -> disease│
     │               │                         │                       │          └─────┬─────┘
     │        ┌──────┴─────────┐               │            (frontend pre-fill only,     │
     │        │ Grad-CAM (v2)   │               │             manual entry always         │
     │        │ tf.GradientTape │               │             remains the fallback)       │
     │        │ -> heatmap PNG  │               │                                          │
     │        └──────┬─────────┘               │                                          │
     │               │                         │                                          │
     └───────────────┴─────────────┬───────────┘◄─────────────────────────────────────────┘
                                    ▼
                     ┌─────────────────────────────────┐
                     │ Multimodal Fusion Module          │
                     │ DEFAULT: weighted late fusion      │
                     │ (adjusts confidence only)          │
                     │ OPT-IN: feature-level fusion (v2)  │
                     │ (can change predicted label)       │
                     │ -> final_confidence, fusion_method │
                     └─────────────────┬───────────────────┘
                                       ▼
                     ┌─────────────────────────────────┐
                     │ Explanation Module                │
                     │ -> explanation string              │
                     │ -> field_explanation (v2)          │
                     └─────────────────┬───────────────────┘
                                       ▼
                     ┌─────────────────────────────────┐
                     │ Recommendation Engine              │
                     │ (deterministic, rule-based)        │
                     └─────────────────┬───────────────────┘
                                       ▼
                     ┌─────────────────────────────────┐
                     │  SQLite Database (predictions)     │
                     └───────┬─────────────────────┬───────┘
                             │                     │
                             ▼                     ▼
                 Full PredictionResponse   GET /dashboard/summary (v2)
                    returned to React      aggregate analytics
                             │
                             ▼
                 GET /prediction/{id}/report (v2)
                 -> generated PDF (ReportLab)
```

**v2 additions are visually distinguished above with "(v2)" tags.**
Every v2 component is additive — none of the v1 data flow (image →
severity/field-context → late fusion → explanation → recommendations →
DB → response) was altered, only extended.

## 2. Module Inputs / Outputs / Integration Points


| Module | Inputs | Processing | Outputs | Integration point |
|---|---|---|---|---|
| Image Disease Model | Leaf image (224x224 RGB), crop_type | MobileNetV2 forward pass, softmax over classes filtered to crop_type | predicted_disease, image_confidence | `app/services/image_model.py::predict_disease` |
| Severity Module | Leaf image, predicted_disease | HSV color segmentation (leaf mask, lesion mask) | affected_area_pct, severity_class | `app/services/severity.py::estimate_severity` |
| Field-Context Module | crop_type, growth_stage, temperature_c, humidity_pct, rainfall_mm, soil_moisture_pct | One-hot encode categoricals, standard-scale numerics, RandomForest.predict_proba | risk_level, risk_score | `app/services/field_context.py::assess_field_risk` |
| Fusion Module | predicted_disease, image_confidence, risk_level, risk_score | Weighted late-fusion formula (see `fusion.py` docstring) | final_confidence, fusion_rationale | `app/services/fusion.py::fuse_predictions` |
| Explanation Module | All of the above | Deterministic string template | explanation (str) | `app/services/risk.py::build_explanation` |
| Recommendation Engine | predicted_disease, severity_class, risk_level | Rule lookup table | recommendations (list[str]) | `app/services/recommendations.py::generate_recommendations` |
| Grad-CAM (v2) | Trained model, preprocessed image, class index | `tf.GradientTape` gradient of class score w.r.t. last conv layer | Base64 PNG heatmap overlay | `app/services/gradcam.py`, `app/routers/gradcam.py` |
| Live Weather (v2) | latitude, longitude | HTTP GET to Open-Meteo, unit conversion | temperature_c, humidity_pct, rainfall_mm, soil_moisture_pct | `app/services/weather.py`, `app/routers/weather.py` |
| Feature-Level Fusion (v2, experimental) | Image embedding, processed field features | Concatenate + forward pass through trained joint classifier | predicted_disease, confidence | `app/services/feature_fusion.py` |
| Field Explainability (v2) | Trained RandomForest, submitted field values | `feature_importances_` + per-request value lookup | field_explanation (str), field_top_features (list) | `app/services/field_context.py::explain_field_risk` |
| Dashboard (v2) | All stored predictions | SQL `GROUP BY` aggregate queries | Distributions, recent predictions, model status | `app/routers/dashboard.py` |
| PDF Report (v2) | Stored prediction row, uploaded image, Grad-CAM | ReportLab document composition | PDF file stream | `app/services/report_generator.py`, `app/routers/report.py` |
| API Route | multipart form (image + fields) | Orchestrates all of the above, persists to DB | PredictionResponse JSON | `app/routers/predict.py::predict` |

## 3. Dataset Comparison & Selection

| Dataset | Size | Classes | Image conditions | Field/env metadata | Fit for MVP |
|---|---|---|---|---|---|
| **PlantVillage** | ~54,000 images | 38 classes, 14 crops | Lab/lightbox, uniform background | None | **Selected** — largest, cleanest, most widely used; ideal for a 4-person team to get a working classifier quickly |
| PlantDoc | ~2,600 images | 27 classes, 13 crops | Real field photos (cluttered background, varied lighting) | None | Good for testing generalization later; too small/noisy to train from scratch as the primary MVP dataset |
| Custom field-collected | N/A | N/A | Realistic | Could be paired with real environmental sensors | Best long-term but infeasible to collect + label in one semester |

**Decision:** Use **PlantVillage** as the primary training dataset for
the MVP (Tomato, Potato, Corn subsets only — see `app/ml/labels.py`),
and optionally use a small sample of **PlantDoc** images for the 3
selected crops as an additional, more realistic **test-only** set to
honestly report how much accuracy drops on field-like photos versus
lab photos. This generalization-gap discussion is valuable, honest,
and easy to include (see `docs/evaluation_plan.md`).

## 4. Dataset Download, Organize, Clean, Preprocess, Split — Step by Step
See `data/README.md` for exact commands. Summary:
1. Download PlantVillage (e.g. from Kaggle:
   `https://www.kaggle.com/datasets/emmarex/plantdisease` — verify the
   exact dataset link/license at download time) into `data/raw/`.
2. Confirm folder structure is `data/raw/<ClassName>/*.jpg`, with class
   names matching (or renamed to match) `app/ml/labels.py::CLASS_NAMES`.
3. Run `python -m app.ml.dataset_utils --raw_dir ../../data/raw --processed_dir ../../data/processed`
   to produce a 70/15/15 train/val/test split per class.
4. Basic cleaning: remove zero-byte/corrupted files (`PIL.Image.open`
   + `.verify()` in a small script) before splitting if you spot any
   during initial inspection.
5. Preprocessing at training time (handled inside `train_transfer.py`):
   resize to 224x224, MobileNetV2 `preprocess_input` scaling,
   augmentation (flip/rotate/zoom/contrast) applied only to the
   training split.

## 5. Model Selection Reasoning (image model)
Three options were considered and are compared experimentally
(see `docs/evaluation_plan.md`):

| Model | Params | Relative training speed | Expected accuracy on PlantVillage-style data | Deployment size |
|---|---|---|---|---|
| CNN Baseline (custom, from scratch) | ~0.5M | Fastest per epoch, but needs more epochs to converge | Lowest (no pretrained features) | Smallest |
| MobileNetV2 (transfer learning) | ~3.5M (head) + frozen/fine-tuned base | Fast, converges in fewer epochs | High | Small (~14MB) — **Selected** |
| ResNet50 / EfficientNetB0 | 11-25M | Slower per epoch on CPU | High, marginal gain over MobileNetV2 for this task | Larger |

**Decision:** MobileNetV2 transfer learning is the production model.
The CNN baseline is retained and reported purely as the required
comparison baseline, not deployed in the API.

## 6. Multimodal Fusion Strategy — Decision
See the full reasoning and formula documented directly in
`backend/app/services/fusion.py` (kept next to the code so
implementation and justification never drift apart). Summary: weighted
late fusion was chosen over feature-level fusion because feature-level
fusion requires a jointly-labeled (image + field-context + verified
outcome) dataset that does not exist publicly, and rule-based-only
fusion is too brittle/non-adaptive. The module docstring also explains
exactly how to upgrade to feature-level fusion later if such a dataset
is collected.

## 7. Field-Context Model — Honesty Note
The Random Forest field-risk model is trained on **synthetic,
domain-rule-labeled data** (see `backend/app/ml/train_field_model.py`
docstring for the exact rules used, based on textbook fungal-disease
favorability heuristics: humidity, temperature range, rainfall, soil
moisture). This is disclosed here, in the code, and in
`docs/limitations.md` — it must also appear in your submitted report's
methodology and limitations sections. Do not present this model's
accuracy figures as if they reflect real-world epidemiological
validity; they reflect how well the model learned the synthetic
labeling rules, which is still a legitimate ML exercise but a distinct
claim from "predicts real disease outbreaks."
