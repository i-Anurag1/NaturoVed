import React from "react";
import { useLanguage } from "../i18n/LanguageContext";

const SEVERITY_COLORS = {
  None: "#4caf50",
  Mild: "#8bc34a",
  Moderate: "#ff9800",
  Severe: "#e53935",
};

export default function SeverityDisplay({ severityClass, affectedAreaPct }) {
  const { t } = useLanguage();
  const color = SEVERITY_COLORS[severityClass] || "#999";

  return (
    <div className="info-card">
      <h4>{t("severity_heading")}</h4>
      <div className="severity-pill" style={{ backgroundColor: color }}>
        {severityClass}
      </div>
      {affectedAreaPct !== null && affectedAreaPct !== undefined && (
        <p className="affected-area-text">
          {t("affected_area_text")}: <strong>{affectedAreaPct.toFixed(1)}%</strong>
        </p>
      )}
      <p className="info-note">
        Estimated via OpenCV color-segmentation heuristic (Mild &lt;20%, Moderate 20-50%, Severe &gt;50%).
      </p>
    </div>
  );
}
