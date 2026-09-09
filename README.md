# 🌿 Multimodal Crop Disease Diagnosis with Severity & Field Context (v2)

A full-stack, end-to-end academic project (B.Tech CSE, 4-member team)
that diagnoses crop diseases from a leaf image **combined with**
structured field-context data, and returns disease + confidence,
severity + affected leaf area, field/environmental risk, an
explanation, and rule-based recommendations.

**v2 adds:** Grad-CAM explainability, live weather auto-fill,
experimental feature-level multimodal fusion (compared against the v1
late-fusion baseline), field-risk model explainability, an analytics
dashboard, PDF report generation, English/Tamil UI, Docker + CI. All
v1 functionality is preserved and unchanged — see `docs/api_documentation.md`
for the backward-compatibility guarantee.

**Before you present this project, read `docs/limitations_and_future_work.md`
Sections 2 and 3.** They disclose two important, specific caveats: (a)
Grad-CAM heatmaps are not yet diagnostically meaningful because no
ImageNet-pretrained weights could be downloaded in this project's
development sandbox, and (b) the feature-fusion-vs-late-fusion
comparison numbers are inflated by synthetic field-context data and
must not be quoted as real-world results. Both are explained with the
exact fix (train on real data with real weights) and are safe, honest
talking points for a viva — not something to hide.

## Repository Structure
```
crop-disease-diagnosis/
├── README.md
├── docker-compose.yml              <- v2: run both containers together
├── .github/workflows/ci.yml        <- v2: automated tests + build validation
├── docs/                           <- academic documentation (report material)
│   ├── problem_statement.md
│   ├── literature_review.md
│   ├── methodology.md              <- architecture diagram (updated for v2)
│   ├── evaluation_plan.md          <- real experiment results + v2 fusion comparison
│   ├── limitations_and_future_work.md  <- READ THIS (Grad-CAM/synthetic-data caveats)
│   ├── api_documentation.md        <- v2: full endpoint reference
│   ├── deployment.md               <- v2: Docker/cloud deployment guide
│   ├── viva_questions_and_answers.md  <- v2
│   ├── demo_and_presentation.md
│   └── team_and_workflow.md
├── data/                           <- dataset download/organize instructions
├── scripts/                        <- convenience setup scripts
├── backend/
│   ├── Dockerfile                  <- v2
│   ├── requirements.txt
│   ├── .env.example
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── models/                 <- SQLAlchemy models + Pydantic schemas
│   │   ├── routers/                <- health, predict, history, weather (v2),
│   │   │                              dashboard (v2), gradcam (v2), report (v2)
│   │   ├── services/                <- image_model, severity, field_context, fusion,
│   │   │                               recommendations, gradcam (v2), weather (v2),
│   │   │                               feature_fusion (v2), report_generator (v2)
│   │   └── ml/                      <- dataset utils, training scripts, evaluation,
│   │                                    synthetic_benchmark (v2), train_feature_fusion (v2),
│   │                                    compare_fusion_approaches (v2),
│   │                                    evaluate_severity_segmentation (v2)
│   ├── storage/                     <- trained models + uploaded images + SQLite DB
│   ├── sample_data/
│   └── tests/                       <- test_api.py (v1) + test_v2_features.py (v2, 20 tests)
└── frontend/
    ├── Dockerfile                   <- v2
    ├── nginx.conf                   <- v2
    ├── package.json
    ├── .env.example
    └── src/
        ├── api/                     <- axios API service layer
        ├── i18n/                    <- v2: English/Tamil translations + context
        ├── components/              <- + GradCamDisplay, WeatherFetchButton,
        │                               ReportDownloadButton, SimpleBarChart, LanguageSwitcher (v2)
        ├── pages/                   <- Home, Result, History, + Dashboard (v2)
        └── styles/
```

Full architecture diagram (with v2 components labeled): `docs/methodology.md`.

## Quick Start

### Option A: Docker Compose (recommended)
```bash
docker compose up --build
```
- Backend: http://localhost:8000/docs
- Frontend: http://localhost:3000

See `docs/deployment.md` for full details, including the disclosed
limitation that Docker itself was validated statically (not built and
run) in this project's own development sandbox, since Docker was not
installed there — run it yourself once before your final demo.

### Option B: Manual (local development, hot-reload)

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Works immediately with **no trained models required** — runs in the
documented fallback-heuristic / rule-based modes. Visit
`http://localhost:8000/docs` for interactive API docs.

**Frontend:**
```bash
cd frontend
npm install
cp .env.example .env
npm start
```
Visit `http://localhost:3000`.

### Run Tests
```bash
cd backend
pytest tests/ -v          # 30 tests: 10 v1 (test_api.py) + 20 v2 (test_v2_features.py)
```

## Training the Models

### a) Field-context Random Forest (fast, seconds, no dataset download)
```bash
cd backend
python -m app.ml.train_field_model --n_samples 6000 --out ./storage/models/field_model.joblib
```

### b) Image disease model — real training (requires PlantVillage + internet access)
```bash
cd backend
python -m app.ml.dataset_utils --raw_dir ../data/raw --processed_dir ../data/processed
python -m app.ml.train_cnn --data_dir ../data/processed --epochs 15 --out ./storage/models/cnn_baseline.keras
python -m app.ml.train_transfer --data_dir ../data/processed --epochs 10 --fine_tune_epochs 5 \
    --out ./storage/models/disease_model.keras
python -m app.ml.evaluate --model ./storage/models/disease_model.keras --data_dir ../data/processed
```
**This step is what makes Grad-CAM meaningful** — see
`docs/limitations_and_future_work.md` Section 2. `--weights imagenet`
is the default and requires network access to `storage.googleapis.com`
(not available in this project's own development sandbox, but normal
in most environments).

### c) Experimental feature-level fusion model (optional; requires (b) first)
```bash
python -m app.ml.train_feature_fusion --image_model ./storage/models/disease_model.keras \
    --data_dir ../data/processed --out ./storage/models/feature_fusion_model.keras
```

### d) Compare fusion approaches (image-only vs late-fusion vs feature-fusion)
```bash
python -m app.ml.compare_fusion_approaches \
    --image_model ./storage/models/disease_model.keras \
    --feature_fusion_model ./storage/models/feature_fusion_model.keras \
    --field_preprocessor ./storage/models/feature_fusion_model_field_preprocessor.joblib \
    --data_dir ../data/processed
```
See `docs/evaluation_plan.md` Section 8 for how this was run in this
project's sandbox (synthetic data, random-init backbone) and why those
specific numbers must not be quoted as real-world results.

After training (b), restart the backend — `GET /health` should report
`"image_model_loaded": true`.

## API Reference
Full reference with request/response schemas: `docs/api_documentation.md`,
or interactively at `/docs` once the backend is running. Summary:

| Method | Route | v1/v2 | Description |
|---|---|---|---|
| GET | `/health` | v1 | Liveness + model-load status |
| POST | `/predict` | v1 (+v2 optional fields) | Full diagnosis pipeline |
| GET | `/prediction/{id}` | v1 | Fetch a stored prediction |
| GET | `/history` | v1 | List past predictions |
| GET | `/weather` | v2 | Live weather lookup by coordinates |
| GET | `/dashboard/summary` | v2 | Aggregate analytics |
| GET | `/prediction/{id}/gradcam` | v2 | Regenerate Grad-CAM heatmap |
| GET | `/prediction/{id}/report` | v2 | Download PDF diagnosis report |

## Environment Variables
See `backend/.env.example` for the full list. New in v2:
`FEATURE_FUSION_MODEL_PATH`, `FEATURE_FUSION_PREPROCESSOR_PATH`,
`WEATHER_PROVIDER`, `ENABLE_LIVE_WEATHER`. All have sensible defaults —
no new variable is required to run the app.

## Supported Crops & Diseases (MVP scope, unchanged in v2)
Tomato, Potato, Corn — 10 classes total. See `backend/app/ml/labels.py`
and `docs/problem_statement.md` Section 3.

## Honesty Notes — Read Before Presenting This Project
- **Field-risk Random Forest**: trained on synthetic, domain-rule-labeled
  data (no public real-world dataset exists for this). Disclosed in
  `backend/app/ml/train_field_model.py` and `docs/methodology.md` Section 7.
- **Severity module**: an unsupervised OpenCV heuristic, not a trained
  segmentation model. A candidate improvement was tested in v2 and
  **rejected** after real experimentation — see `docs/evaluation_plan.md`
  Section 7.
- **Image model**: runs a labeled fallback heuristic mode until you train
  and save a real model (`model_mode` / `image_model_loaded` tell you
  which is active).
- **Grad-CAM (v2)**: implemented and tested, but not yet diagnostically
  meaningful in this environment (no ImageNet weights available) — see
  `docs/limitations_and_future_work.md` Section 2. Becomes meaningful
  automatically after real training.
- **Feature-level fusion comparison (v2)**: numbers are a sandboxed
  code-correctness smoke test on synthetic data, not a real-world
  result — see `docs/limitations_and_future_work.md` Section 3.
- **Live weather (v2)**: unit-tested against a mocked API response; a
  live call to Open-Meteo has not been exercised in this sandbox
  (network restricted) — see `docs/limitations_and_future_work.md` Section 4.
- **Docker (v2)**: validated statically (YAML/env-var/path checks);
  not actually built and run in this sandbox (Docker not installed
  here) — see `docs/deployment.md`.

## License / Academic Use
This is an academic MVP project. Recommendations given by the app are
general agricultural guidance only and are not a substitute for advice
from a licensed local agronomist or extension service.
