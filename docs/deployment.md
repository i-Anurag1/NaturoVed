# Deployment Guide

## 1. Local Deployment with Docker Compose (recommended for grading/demo)

**Prerequisites:** Docker Engine 24+ and Docker Compose v2 (bundled with
modern Docker Desktop / `docker compose` CLI plugin on Linux).

```bash
# From the repository root
docker compose up --build
```

- Backend: http://localhost:8000 (Swagger docs at `/docs`)
- Frontend: http://localhost:3000

Stop the stack:
```bash
docker compose down
```

Reset ALL data (database, uploaded images, trained models) and start
completely fresh:
```bash
docker compose down -v
```

**What works out of the box:** the app is fully usable immediately —
no dataset or training required. The image model runs in the
documented fallback-heuristic mode and the field-risk model in
rule-based fallback mode until trained artifacts are present (see
README.md "Honesty Notes"). This is intentional, not a bug.

**To use trained models inside Docker:** train on the host first (see
README.md "Training the Models"), then either:
- Restart `docker compose up` — the `backend_storage` named volume
  persists across restarts, so if you previously trained directly
  inside the container, artifacts remain; or
- Copy host-trained `.keras`/`.joblib` files into the running
  container's storage volume:
  ```bash
  docker cp backend/storage/models/. crop-disease-backend:/app/storage/models/
  docker compose restart backend
  ```

**Environment variables:** see `backend/.env.example` for the full
list (all consumed by `docker-compose.yml` directly — no separate
`.env` file is needed when running via Compose, though one can still
be used for local `uvicorn` development). `docs/environment_variables.md`
section further below documents each variable.

### Docker build note (important, disclosed honestly)
This project's own sandboxed development environment does not have
Docker installed, so the Dockerfiles and `docker-compose.yml` in this
repository were **validated statically** (YAML syntax validated,
environment variable names cross-checked one-by-one against
`app/config.py`'s `Settings` fields, file paths referenced in each
Dockerfile confirmed to exist, Nginx config brace-balance checked) but
**not actually built and run** in this environment. Before your final
submission/demo, run `docker compose up --build` yourself at least
once and confirm both containers report healthy — this is standard
practice for any handed-off project and is called out here rather than
silently assumed to work.

## 2. Manual (non-Docker) Local Deployment
See the root `README.md` "Quick Start" section — this remains the
simplest path for active development (hot-reload, direct log access,
no image rebuild needed after every code change).

## 3. Cloud Deployment Options (suitable for an academic demo)

These are **suggestions with concrete steps**, not a guarantee of a
specific provider's current free-tier terms (verify pricing/limits
yourself before deploying, as these change over time).

### Backend: Render / Railway / Fly.io (any Docker-friendly PaaS)
1. Push this repository to GitHub.
2. Create a new "Web Service" pointing at the `backend/` directory
   (or the repo root with `backend/Dockerfile` as the Docker context).
3. Set environment variables from `backend/.env.example` in the
   platform's dashboard. Set `CORS_ORIGINS` to your deployed frontend's
   URL (e.g. `https://your-app.vercel.app`).
4. Most platforms auto-detect `EXPOSE 8000` / respect a `PORT` env var
   — the backend Dockerfile's CMD already reads `$PORT` if set,
   falling back to 8000.
5. Attach a persistent volume/disk for `/app/storage` if the platform
   supports it (otherwise the SQLite DB and uploaded images reset on
   every redeploy — acceptable for a demo, not for real use).

### Frontend: Vercel / Netlify / GitHub Pages
1. Point the platform at the `frontend/` directory.
2. Build command: `npm run build`; output directory: `build`.
3. Set the environment variable `REACT_APP_API_BASE_URL` to your
   deployed backend's public URL (e.g. `https://your-api.onrender.com`)
   — remember this is baked in at BUILD time for Create React App, so
   changing it requires a rebuild/redeploy, not just a restart.

### Database note for cloud deployment
SQLite is appropriate for this MVP and for a graded demo, but is a
single file on local disk — most serverless/ephemeral-filesystem
platforms will lose it on redeploy or restart. For a longer-lived
public deployment, migrate `DATABASE_URL` to a managed Postgres
instance (SQLAlchemy makes this a connection-string change, not a code
change) — this is listed as future work in
`docs/limitations_and_future_work.md`, not implemented in this
project's scope.

## 4. GitHub Actions CI (already configured)
See `.github/workflows/ci.yml`. On every push/PR to `main` or
`develop`, it runs:
1. **Backend tests** — the full pytest suite (30 tests as of v2),
   installed from the pinned `requirements.txt`, with NO trained model
   artifacts present (deliberately exercises fallback-mode code paths
   — this is intentional so CI never depends on committing large
   binary model files to the repo).
2. **Frontend build validation** — `npm ci && npm run build`, confirms
   the production bundle compiles and `build/index.html` exists.
3. **ML experiment smoke test** — trains the field-context Random
   Forest, runs the leaf-segmentation comparison experiment, generates
   a tiny synthetic image benchmark, and trains a tiny (1-epoch,
   random-weights) image model to confirm the Grad-CAM code path loads
   a model without error. This step validates CODE CORRECTNESS only —
   it does not and cannot validate real-world model accuracy (see
   `docs/limitations_and_future_work.md`).

This CI workflow was **not run on actual GitHub infrastructure** during
this project's development (no GitHub repository was created in this
sandboxed session) — each step's commands were instead run manually,
locally, with the same working directory and arguments the workflow
file specifies, and confirmed to succeed (see the final verification
report for exact output). Push this repository to GitHub and confirm
the Actions tab shows green before relying on it for grading purposes.
