"""GET /weather — optional live weather lookup used by the frontend's
'Use live weather' button. See app/services/weather.py for the full
honesty disclosure on data source, approximations, and testing status."""
from fastapi import APIRouter, HTTPException, Query
from app.models.schemas import WeatherResponse
from app.services.weather import fetch_live_weather, WeatherUnavailableError

router = APIRouter(tags=["weather"])


@router.get("/weather", response_model=WeatherResponse)
def get_weather(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
):
    try:
        data = fetch_live_weather(latitude, longitude)
    except WeatherUnavailableError as e:
        raise HTTPException(status_code=503, detail=f"Live weather unavailable: {e}")
    return WeatherResponse(**data)
