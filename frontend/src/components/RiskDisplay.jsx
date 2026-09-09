import React from "react";
import { useLanguage } from "../i18n/LanguageContext";

const RISK_COLORS = { Low: "#4caf50", Medium: "#ff9800", High: "#e53935" };

export default function RiskDisplay({ riskLevel, riskScore }) {
  const { t } = useLanguage();
  const color = RISK_COLORS[riskLevel] || "#999";
  const pct = Math.round((riskScore ?? 0) * 100);

  return (
    <div className="info-card">
      <h4>{t("risk_heading")}</h4>
      <div className="severity-pill" style={{ backgroundColor: color }}>
        {riskLevel} Risk
      </div>
      <div className="confidence-bar-wrap" style={{ marginTop: "8px" }}>
        <div className="confidence-bar-fill" style={{ width: `${pct}%`, backgroundColor: color }} />
        <span className="confidence-value">{pct}%</span>
      </div>
      <p className="info-note">
        Based on temperature, humidity, rainfall, soil moisture, crop, and growth stage.
      </p>
    </div>
  );
}
