import React from "react";
import { useLanguage } from "../i18n/LanguageContext";

function formatDiseaseName(raw) {
  const parts = raw.split("___");
  const name = parts.length > 1 ? parts[1] : raw;
  return name.replace(/_/g, " ");
}

export default function PredictionCard({ prediction }) {
  const { t } = useLanguage();
  const isHealthy = prediction.predicted_disease.includes("Healthy");
  const cropName = prediction.predicted_disease.split("___")[0];

  return (
    <div className={`prediction-card ${isHealthy ? "healthy" : "diseased"}`}>
      <div className="prediction-card-header">
        <span className="crop-tag">{cropName}</span>
        <span className={`status-badge ${isHealthy ? "badge-healthy" : "badge-diseased"}`}>
          {isHealthy ? t("status_healthy") : t("status_diseased")}
        </span>
      </div>
      <h2 className="disease-name">{formatDiseaseName(prediction.predicted_disease)}</h2>

      <div className="confidence-row">
        <div className="confidence-block">
          <span className="confidence-label">{t("confidence_image")}</span>
          <ConfidenceBar value={prediction.image_confidence} />
        </div>
        <div className="confidence-block">
          <span className="confidence-label">{t("confidence_final")}</span>
          <ConfidenceBar value={prediction.final_confidence} emphasized />
        </div>
      </div>

      {prediction.fusion_method === "feature_fusion" && (
        <p className="experimental-tag">⚗️ Experimental: Feature-Level Fusion</p>
      )}
    </div>
  );
}

function ConfidenceBar({ value, emphasized }) {
  const pct = Math.round(value * 100);
  return (
    <div className="confidence-bar-wrap">
      <div className={`confidence-bar-fill ${emphasized ? "emphasized" : ""}`} style={{ width: `${pct}%` }} />
      <span className="confidence-value">{pct}%</span>
    </div>
  );
}
