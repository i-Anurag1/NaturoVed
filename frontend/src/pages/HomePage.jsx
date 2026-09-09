import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import ImageUpload from "../components/ImageUpload";
import FieldContextForm from "../components/FieldContextForm";
import LoadingState from "../components/LoadingState";
import { submitPrediction } from "../api/api";
import { useLanguage } from "../i18n/LanguageContext";

const DEFAULT_FIELD_CONTEXT = {
  crop_type: "tomato",
  growth_stage: "vegetative",
  location: "",
  temperature_c: 28,
  humidity_pct: 70,
  rainfall_mm: 5,
  soil_moisture_pct: 45,
};

function validateFieldContext(values, t) {
  const errors = {};
  const temp = parseFloat(values.temperature_c);
  const humidity = parseFloat(values.humidity_pct);
  const rainfall = parseFloat(values.rainfall_mm);
  const soilMoisture = parseFloat(values.soil_moisture_pct);

  if (Number.isNaN(temp) || temp < -10 || temp > 60) {
    errors.temperature_c = "Temperature must be between -10°C and 60°C.";
  }
  if (Number.isNaN(humidity) || humidity < 0 || humidity > 100) {
    errors.humidity_pct = "Humidity must be between 0% and 100%.";
  }
  if (Number.isNaN(rainfall) || rainfall < 0 || rainfall > 1000) {
    errors.rainfall_mm = "Rainfall must be between 0 and 1000mm.";
  }
  if (Number.isNaN(soilMoisture) || soilMoisture < 0 || soilMoisture > 100) {
    errors.soil_moisture_pct = "Soil moisture must be between 0% and 100%.";
  }
  return errors;
}

export default function HomePage() {
  const { t } = useLanguage();
  const [imageFile, setImageFile] = useState(null);
  const [imageError, setImageError] = useState(null);
  const [fieldContext, setFieldContext] = useState(DEFAULT_FIELD_CONTEXT);
  const [weatherSource, setWeatherSource] = useState("manual");
  const [fusionMethod, setFusionMethod] = useState("late_fusion");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [fieldErrors, setFieldErrors] = useState({});
  const [loading, setLoading] = useState(false);
  const [submitError, setSubmitError] = useState(null);
  const navigate = useNavigate();

  const handleFieldChange = (field, value) => {
    setFieldContext((prev) => ({ ...prev, [field]: value }));
    setWeatherSource("manual"); // any manual edit invalidates the "auto" tag
  };

  const handleWeatherFetched = (weather) => {
    setFieldContext((prev) => ({
      ...prev,
      temperature_c: weather.temperature_c,
      humidity_pct: weather.humidity_pct,
      rainfall_mm: weather.rainfall_mm,
      soil_moisture_pct: weather.soil_moisture_pct,
    }));
    setWeatherSource("auto");
  };

  const handleImageSelected = (file, error) => {
    setImageFile(file);
    setImageError(error);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSubmitError(null);

    if (!imageFile) {
      setSubmitError(t("error_no_image"));
      return;
    }

    const errors = validateFieldContext(fieldContext, t);
    setFieldErrors(errors);
    if (Object.keys(errors).length > 0) {
      setSubmitError("Please correct the highlighted fields before submitting.");
      return;
    }

    setLoading(true);
    try {
      const result = await submitPrediction(imageFile, {
        ...fieldContext,
        fusion_method: fusionMethod,
        weather_source: weatherSource,
      });
      navigate(`/result/${result.id}`, { state: { prediction: result, imageFile } });
    } catch (err) {
      const detail = err.response?.data?.detail || t("error_generic");
      setSubmitError(typeof detail === "string" ? detail : t("error_generic"));
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="page-container">
        <LoadingState message={t("btn_diagnosing")} />
      </div>
    );
  }

  return (
    <div className="page-container">
      <div className="hero">
        <h1>{t("hero_title")}</h1>
        <p className="subtitle">{t("hero_subtitle")}</p>
      </div>

      <form className="diagnosis-form" onSubmit={handleSubmit} noValidate>
        <ImageUpload onImageSelected={handleImageSelected} error={imageError} />
        <FieldContextForm
          values={fieldContext}
          onChange={handleFieldChange}
          onWeatherFetched={handleWeatherFetched}
          weatherSource={weatherSource}
          errors={fieldErrors}
        />

        <div className="advanced-options">
          <button
            type="button"
            className="advanced-toggle"
            onClick={() => setShowAdvanced((s) => !s)}
          >
            {showAdvanced ? "▾" : "▸"} {t("advanced_options")}
          </button>
          {showAdvanced && (
            <div className="advanced-panel">
              <label className="field-label">{t("fusion_method_label")}</label>
              <select value={fusionMethod} onChange={(e) => setFusionMethod(e.target.value)}>
                <option value="late_fusion">{t("fusion_late")}</option>
                <option value="feature_fusion">{t("fusion_feature")}</option>
              </select>
            </div>
          )}
        </div>

        {submitError && <p className="field-error" role="alert">{submitError}</p>}

        <button type="submit" className="btn-primary" disabled={loading}>
          {t("btn_diagnose")}
        </button>
      </form>
    </div>
  );
}
