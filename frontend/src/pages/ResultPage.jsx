import React, { useEffect, useState } from "react";
import { useParams, useLocation, Link } from "react-router-dom";
import PredictionCard from "../components/PredictionCard";
import SeverityDisplay from "../components/SeverityDisplay";
import RiskDisplay from "../components/RiskDisplay";
import RecommendationsList from "../components/RecommendationsList";
import GradCamDisplay from "../components/GradCamDisplay";
import ReportDownloadButton from "../components/ReportDownloadButton";
import LoadingState from "../components/LoadingState";
import { getPrediction } from "../api/api";
import { useLanguage } from "../i18n/LanguageContext";

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || "http://localhost:8000";

export default function ResultPage() {
  const { id } = useParams();
  const routerLocation = useLocation();
  const { t } = useLanguage();

  const [prediction, setPrediction] = useState(routerLocation.state?.prediction || null);
  const [loading, setLoading] = useState(!routerLocation.state?.prediction);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (prediction) return;
    let cancelled = false;
    setLoading(true);
    getPrediction(id)
      .then((data) => {
        if (!cancelled) setPrediction(data);
      })
      .catch(() => {
        if (!cancelled) setError("Could not load this prediction. It may not exist.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  if (loading) {
    return (
      <div className="page-container">
        <LoadingState message={t("loading_result")} />
      </div>
    );
  }

  if (error || !prediction) {
    return (
      <div className="page-container">
        <p className="field-error">{error || "Prediction not found."}</p>
        <Link to="/" className="btn-secondary">{t("btn_new_diagnosis")}</Link>
      </div>
    );
  }

  const originalImageUrl = prediction.image_filename
    ? `${API_BASE_URL}/uploads/${prediction.image_filename}`
    : null;

  return (
    <div className="page-container">
      <PredictionCard prediction={prediction} />

      <div className="result-grid">
        <SeverityDisplay
          severityClass={prediction.severity_class}
          affectedAreaPct={prediction.affected_area_pct}
        />
        <RiskDisplay riskLevel={prediction.field_risk_level} riskScore={prediction.field_risk_score} />
      </div>

      <GradCamDisplay
        originalImageUrl={originalImageUrl}
        gradcamImage={prediction.gradcam_image}
        available={prediction.gradcam_available}
      />

      <RecommendationsList
        recommendations={prediction.recommendations}
        explanation={prediction.explanation}
      />

      {prediction.field_explanation && (
        <div className="info-card wide">
          <h4>{t("field_explanation_heading")}</h4>
          <p className="explanation-text">{prediction.field_explanation}</p>
        </div>
      )}

      <div className="field-context-summary">
        <h4>{t("field_context_summary_heading")}</h4>
        <ul>
          <li>Crop: {prediction.crop_type}</li>
          <li>Growth stage: {prediction.growth_stage}</li>
          {prediction.location && <li>Location: {prediction.location}</li>}
          <li>Temperature: {prediction.temperature_c}°C</li>
          <li>Humidity: {prediction.humidity_pct}%</li>
          <li>Rainfall (7d): {prediction.rainfall_mm}mm</li>
          <li>Soil Moisture: {prediction.soil_moisture_pct}%</li>
          <li>Weather Source: {prediction.weather_source === "auto" ? "Live Weather API" : "Manual Entry"}</li>
          <li>Fusion Method: {prediction.fusion_method === "feature_fusion" ? "Feature-Level Fusion (experimental)" : "Weighted Late Fusion"}</li>
        </ul>
      </div>

      <div className="action-row">
        <Link to="/" className="btn-secondary">{t("btn_new_diagnosis")}</Link>
        <Link to="/history" className="btn-secondary">{t("btn_view_history")}</Link>
        <ReportDownloadButton predictionId={prediction.id} />
      </div>
    </div>
  );
}
