import React, { useState } from "react";
import { getLiveWeather } from "../api/api";
import { useLanguage } from "../i18n/LanguageContext";

/**
 * "Use Live Weather" button: gets the browser's geolocation, calls
 * GET /weather, and hands the result back to the parent via onFetched.
 * Manual field entry always remains available as a fallback — this
 * button only ever pre-fills values the user can still edit.
 */
export default function WeatherFetchButton({ onFetched }) {
  const { t } = useLanguage();
  const [status, setStatus] = useState("idle"); // idle | loading | error

  const handleClick = () => {
    if (!navigator.geolocation) {
      setStatus("error");
      return;
    }
    setStatus("loading");
    navigator.geolocation.getCurrentPosition(
      async (position) => {
        try {
          const { latitude, longitude } = position.coords;
          const weather = await getLiveWeather(latitude, longitude);
          onFetched(weather);
          setStatus("idle");
        } catch (err) {
          setStatus("error");
        }
      },
      () => setStatus("error"),
      { timeout: 10000 }
    );
  };

  return (
    <div className="weather-fetch-wrap">
      <button
        type="button"
        className="btn-weather"
        onClick={handleClick}
        disabled={status === "loading"}
      >
        {status === "loading" ? t("fetching_weather") : `📍 ${t("use_live_weather")}`}
      </button>
      {status === "error" && <p className="field-error small">{t("weather_fetch_error")}</p>}
    </div>
  );
}
