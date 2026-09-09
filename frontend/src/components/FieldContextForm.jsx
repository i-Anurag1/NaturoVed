import React from "react";
import WeatherFetchButton from "./WeatherFetchButton";
import { useLanguage } from "../i18n/LanguageContext";

const CROP_VALUES = ["tomato", "potato", "corn"];
const GROWTH_STAGE_VALUES = ["seedling", "vegetative", "flowering", "fruiting", "maturity"];

/**
 * Controlled form for all structured field-context inputs, plus the
 * optional "Use Live Weather" auto-fill button. Manual entry always
 * remains the fallback: auto-fill only pre-populates the numeric
 * fields, which the user can still freely edit afterward.
 */
export default function FieldContextForm({ values, onChange, onWeatherFetched, weatherSource, errors = {} }) {
  const { t } = useLanguage();
  const handle = (field) => (e) => onChange(field, e.target.value);

  const handleWeatherFetched = (weather) => {
    onWeatherFetched(weather);
  };

  return (
    <div className="field-context-form">
      <h3>{t("field_context_heading")}</h3>

      <div className="form-row">
        <div className="form-group">
          <label className="field-label">{t("label_crop_type")}</label>
          <select value={values.crop_type} onChange={handle("crop_type")}>
            {CROP_VALUES.map((v) => (
              <option key={v} value={v}>{v.charAt(0).toUpperCase() + v.slice(1)}</option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label className="field-label">{t("label_growth_stage")}</label>
          <select value={values.growth_stage} onChange={handle("growth_stage")}>
            {GROWTH_STAGE_VALUES.map((v) => (
              <option key={v} value={v}>{v.charAt(0).toUpperCase() + v.slice(1)}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="form-group">
        <label className="field-label">{t("label_location")}</label>
        <input
          type="text"
          placeholder="e.g. Coimbatore, Tamil Nadu"
          value={values.location}
          onChange={handle("location")}
        />
      </div>

      <WeatherFetchButton onFetched={handleWeatherFetched} />
      {weatherSource === "auto" && <p className="info-note weather-note">{t("weather_autofilled_note")}</p>}

      <div className="form-row">
        <div className="form-group">
          <label className="field-label">{t("label_temperature")}</label>
          <input type="number" step="0.1" value={values.temperature_c} onChange={handle("temperature_c")} />
          {errors.temperature_c && <p className="field-error small">{errors.temperature_c}</p>}
        </div>
        <div className="form-group">
          <label className="field-label">{t("label_humidity")}</label>
          <input type="number" step="0.1" min="0" max="100" value={values.humidity_pct} onChange={handle("humidity_pct")} />
          {errors.humidity_pct && <p className="field-error small">{errors.humidity_pct}</p>}
        </div>
      </div>

      <div className="form-row">
        <div className="form-group">
          <label className="field-label">{t("label_rainfall")}</label>
          <input type="number" step="0.1" min="0" value={values.rainfall_mm} onChange={handle("rainfall_mm")} />
          {errors.rainfall_mm && <p className="field-error small">{errors.rainfall_mm}</p>}
        </div>
        <div className="form-group">
          <label className="field-label">{t("label_soil_moisture")}</label>
          <input type="number" step="0.1" min="0" max="100" value={values.soil_moisture_pct} onChange={handle("soil_moisture_pct")} />
          {errors.soil_moisture_pct && <p className="field-error small">{errors.soil_moisture_pct}</p>}
        </div>
      </div>
    </div>
  );
}
