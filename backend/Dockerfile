# Backend Dockerfile — Multimodal Crop Disease Diagnosis API
#
# Build:  docker build -t crop-disease-backend ./backend
# Run:    docker run -p 8000:8000 -v crop_storage:/app/storage crop-disease-backend
# (Normally run via docker-compose — see docker-compose.yml at repo root)

FROM python:3.11-slim

# System dependencies required by opencv-python-headless and reportlab
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Storage directories must exist even on a fresh container (mirrors the
# .gitkeep pattern used in local development)
RUN mkdir -p storage/models storage/uploads

EXPOSE 8000

# Uses the PORT env var if set (common on managed platforms like Render/
# Railway), otherwise defaults to 8000 to match docker-compose.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
