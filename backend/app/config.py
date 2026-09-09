"""
Centralized application configuration.
Reads from environment variables / .env file using pydantic-settings.
"""
from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    APP_ENV: str = "development"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: str = "http://localhost:3000"

    DATABASE_URL: str = "sqlite:///./storage/app.db"

    IMAGE_MODEL_PATH: str = "./storage/models/disease_model.keras"
    FIELD_MODEL_PATH: str = "./storage/models/field_model.joblib"
    LABEL_MAP_PATH: str = "./storage/models/label_map.json"

    # --- v2: experimental feature-level fusion artifacts (optional) ---
    FEATURE_FUSION_MODEL_PATH: str = "./storage/models/feature_fusion_model.keras"
    FEATURE_FUSION_PREPROCESSOR_PATH: str = "./storage/models/feature_fusion_model_field_preprocessor.joblib"

    # --- v2: optional live weather integration ---
    WEATHER_PROVIDER: str = "open-meteo"  # informational; see app/services/weather.py
    ENABLE_LIVE_WEATHER: bool = True

    UPLOAD_DIR: str = "./storage/uploads"
    MAX_UPLOAD_MB: int = 8

    class Config:
        env_file = ".env"

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


settings = Settings()

# Ensure required directories exist at import time so the app never
# crashes on first run just because a folder is missing.
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.dirname(settings.DATABASE_URL.replace("sqlite:///", "")) or ".", exist_ok=True)
os.makedirs(os.path.dirname(settings.IMAGE_MODEL_PATH), exist_ok=True)
