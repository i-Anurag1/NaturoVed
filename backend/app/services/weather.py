"""
Optional live weather integration.

WHY OPEN-METEO: it's a free, no-API-key-required, well-documented
weather API (https://open-meteo.com) that conveniently exposes not
just temperature/humidity/precipitation but also modeled soil-moisture
layers — covering every field-context input this project needs from a
single provider, with no signup friction for a student project. If your
team prefers a different provider (e.g. OpenWeatherMap, WeatherAPI),
swap the implementation of `fetch_live_weather()` below; the function
signature/return contract is what the rest of the app depends on.

WHAT THIS PROVIDES:
    fetch_live_weather(lat, lon) -> {
        "temperature_c": float,
        "humidity_pct": float,
        "rainfall_mm": float,      # summed over the past 7 days
        "soil_moisture_pct": float,  # approximate, see note below
        "source": "open-meteo",
    }

SOIL MOISTURE NOTE: Open-Meteo's `soil_moisture_0_to_1cm` variable is
volumetric water content (m³/m³, range ~0.0-0.5 for most soils), not a
"percent saturation" figure. We report `value * 100` as an approximate
percentage for consistency with this project's manual input field
(which also asks for a 0-100 percentage), but this is a modeled
estimate from a weather model's land-surface layer, not a ground
sensor reading — this approximation is disclosed to the user in the
UI (see frontend WeatherFetchButton component) and documented in
docs/limitations_and_future_work.md.

FALLBACK: if the API call fails for any reason (network issue, invalid
coordinates, provider outage), `fetch_live_weather` raises
`WeatherUnavailableError` with a human-readable message. The /predict
route (app/routers/predict.py) catches this and falls back to
requiring manual field-context input — live weather is always an
OPTIONAL convenience, never a hard dependency of the core pipeline.

TESTING NOTE (see backend/tests/test_weather.py): live outbound HTTP
calls to api.open-meteo.com are not exercised in this project's
sandboxed development/CI environment (network allowlist blocks it —
see docs/limitations_and_future_work.md). The parsing/aggregation logic
is unit-tested against a realistic mocked API response shape instead.
Before relying on this in a real demo, run a manual live check:
    curl "https://api.open-meteo.com/v1/forecast?latitude=11.0&longitude=77.0&current=temperature_2m,relative_humidity_2m&daily=precipitation_sum&past_days=7&hourly=soil_moisture_0_to_1cm"
"""
import requests

OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT_SECONDS = 8


class WeatherUnavailableError(Exception):
    """Raised when live weather cannot be fetched for any reason —
    network failure, invalid coordinates, malformed provider response,
    etc. Callers must treat this as recoverable and fall back to manual
    field-context input."""
    pass


def fetch_live_weather(latitude: float, longitude: float) -> dict:
    if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
        raise WeatherUnavailableError(f"Invalid coordinates: ({latitude}, {longitude})")

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m",
        "daily": "precipitation_sum",
        "hourly": "soil_moisture_0_to_1cm",
        "past_days": 7,
        "forecast_days": 1,
        "timezone": "auto",
    }

    try:
        response = requests.get(OPEN_METEO_BASE_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        raise WeatherUnavailableError(f"Could not reach weather service: {e}")
    except ValueError as e:
        raise WeatherUnavailableError(f"Weather service returned an unreadable response: {e}")

    try:
        current = data["current"]
        temperature_c = float(current["temperature_2m"])
        humidity_pct = float(current["relative_humidity_2m"])

        daily_precip = data.get("daily", {}).get("precipitation_sum", [])
        # Sum the last 7 days of daily precipitation (past_days=7 gives us
        # 7 historical days + today; we sum all available entries as a
        # reasonable "recent rainfall" proxy).
        rainfall_mm = float(sum(v for v in daily_precip if v is not None))

        hourly_soil = data.get("hourly", {}).get("soil_moisture_0_to_1cm", [])
        recent_soil_values = [v for v in hourly_soil[-24:] if v is not None] if hourly_soil else []
        if recent_soil_values:
            soil_moisture_fraction = sum(recent_soil_values) / len(recent_soil_values)
        else:
            soil_moisture_fraction = 0.0
        soil_moisture_pct = float(min(max(soil_moisture_fraction * 100, 0), 100))

    except (KeyError, TypeError, ValueError) as e:
        raise WeatherUnavailableError(f"Unexpected weather API response format: {e}")

    return {
        "temperature_c": round(temperature_c, 1),
        "humidity_pct": round(humidity_pct, 1),
        "rainfall_mm": round(rainfall_mm, 1),
        "soil_moisture_pct": round(soil_moisture_pct, 1),
        "source": "open-meteo",
    }
