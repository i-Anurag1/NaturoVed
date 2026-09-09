#!/usr/bin/env bash
# Convenience script to set up both backend and frontend in one go.
# Usage: bash scripts/setup.sh
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== Setting up backend ==="
cd "$ROOT_DIR/backend"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp -n .env.example .env || true
echo "Training field-context model (fast, synthetic data)..."
python -m app.ml.train_field_model --n_samples 6000 --out ./storage/models/field_model.joblib
deactivate

echo
echo "=== Setting up frontend ==="
cd "$ROOT_DIR/frontend"
npm install
cp -n .env.example .env || true

echo
echo "=== Setup complete ==="
echo "Backend:  cd backend  && source venv/bin/activate && uvicorn app.main:app --reload"
echo "Frontend: cd frontend && npm start"
echo
echo "NOTE: the image disease model (MobileNetV2) still needs to be trained"
echo "separately once you have downloaded the PlantVillage dataset — see"
echo "data/README.md and the 'Training the Models' section of the root README."
