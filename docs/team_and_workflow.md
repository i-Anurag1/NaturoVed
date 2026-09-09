# Team Responsibilities, Development Plan & GitHub Workflow

## 1. Team Division (4 Members)

### Member 1 — Dataset & Image Disease Classification
- Download, clean, and organize the PlantVillage subset (`data/README.md`)
- Run `app/ml/dataset_utils.py` to produce train/val/test splits
- Train and tune the CNN baseline (`app/ml/train_cnn.py`)
- Train and tune the MobileNetV2 transfer model (`app/ml/train_transfer.py`)
- Run `app/ml/evaluate.py` for both models, produce confusion matrices
- Own: `app/ml/labels.py`, `app/ml/dataset_utils.py`, `app/ml/train_cnn.py`,
  `app/ml/train_transfer.py`, `app/ml/evaluate.py`

### Member 2 — Severity Estimation & Image Analysis
- Implement/tune the OpenCV severity module (`app/services/severity.py`)
- Run the small-scale severity pseudo-validation described in
  `docs/evaluation_plan.md` Section 4
- Own: `app/services/severity.py`, severity-related sections of the report

### Member 3 — Field-Context Modeling & Multimodal Fusion
- Build/tune the synthetic dataset generator and Random Forest
  (`app/ml/train_field_model.py`)
- Implement/tune `app/services/field_context.py`
- Implement/tune the fusion logic (`app/services/fusion.py`) and
  explanation builder (`app/services/risk.py`)
- Implement the recommendation engine (`app/services/recommendations.py`)
- Own: field-context and fusion sections of the report, methodology write-up

### Member 4 — Backend, Frontend, Database, Integration & Deployment
- Own the FastAPI backend (`app/main.py`, `app/routers/*`, `app/models/*`,
  `app/database.py`, `app/config.py`)
- Own the React frontend (all of `frontend/src/`)
- Wire the full pipeline together in `app/routers/predict.py`
- Handle deployment/README/setup instructions
- Own: `README.md`, `.env.example` files, deployment section of the report

### Shared Responsibilities
- Integration testing (does the full pipeline work end-to-end with each
  member's component swapped in?)
- Writing and reviewing `backend/tests/test_api.py`
- Report writing/review, presentation slide review
- Code review on every pull request (see workflow below) — every PR
  needs at least one other member's approval before merging

## 2. Phase-by-Phase Development Plan

| Phase | Deliverable | Suggested duration |
|---|---|---|
| 1. Project setup | Repo scaffolding, environments, `.env` files, everyone can run `uvicorn` and `npm start` locally | Week 1 |
| 2. Dataset preparation | PlantVillage subset downloaded, cleaned, split; stats documented | Week 1-2 |
| 3. Disease model (baseline + transfer) | Both models trained, evaluated, best one saved | Week 2-4 |
| 4. Severity module | OpenCV severity function working + pseudo-validated | Week 3-4 (parallel with Phase 3) |
| 5. Field-context model | Synthetic dataset + Random Forest trained and evaluated | Week 3-4 (parallel) |
| 6. Multimodal fusion | Fusion module implemented, explanation builder, recommendation engine | Week 5 |
| 7. Backend integration | All modules wired into `/predict`, DB persistence working | Week 5-6 |
| 8. Frontend | All pages/components built, connected to the API | Week 5-7 (parallel with Phase 7) |
| 9. Integration & testing | Full end-to-end manual + automated (`pytest`) testing | Week 7 |
| 10. Deployment | Local Docker/run instructions finalized, optional cloud deploy | Week 8 |
| 11. Documentation & report | Fill in all `docs/*.md` with real results | Week 8 |
| 12. Presentation & demo prep | Slides + rehearsed live demo | Week 9 |

## 3. GitHub Workflow

### Branch Strategy
- `main` — always deployable/demo-ready. Protected: no direct pushes.
- `develop` — integration branch where feature branches merge first.
- `feature/<short-description>` — one branch per task, e.g.
  `feature/mobilenet-training`, `feature/severity-opencv`,
  `feature/predict-route`, `feature/history-page`.

### Workflow
1. Create an **Issue** for each task (link to the phase/module table above).
2. Create a **feature branch** off `develop` for that issue.
3. Commit small, focused changes with clear messages
   (e.g. `feat(severity): add HSV lesion segmentation`).
4. Open a **Pull Request** into `develop`, link the Issue, request
   review from at least one teammate.
5. After approval + passing tests (`pytest backend/tests`), merge with
   **squash merge** to keep `develop` history clean.
6. Periodically (e.g. end of each phase), open a PR from `develop` into
   `main` once the integration is verified stable — tag a **Milestone**
   release (e.g. `v0.1-backend-mvp`, `v0.2-frontend-integrated`,
   `v1.0-demo-ready`).

### Reducing Merge Conflicts
- Keep the module boundaries in this document — each member primarily
  edits their own files (services vs. routers vs. frontend components),
  which naturally minimizes overlapping edits.
- Pull `develop` and rebase your feature branch frequently, especially
  before opening a PR.
- Agree on shared contracts early (e.g. the exact `PredictionResponse`
  schema in `app/models/schemas.py`) so backend and frontend work can
  proceed in parallel without breaking each other.

### Milestones (suggested)
- `M1: Environment & scaffolding ready`
- `M2: Disease model trained & evaluated`
- `M3: Severity + field-context modules working standalone`
- `M4: Backend /predict pipeline fully wired`
- `M5: Frontend fully connected to backend`
- `M6: Tests passing, docs complete`
- `M7: Demo-ready release`
