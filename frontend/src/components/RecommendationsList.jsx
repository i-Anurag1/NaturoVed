import React from "react";
import { useLanguage } from "../i18n/LanguageContext";

export default function RecommendationsList({ recommendations, explanation }) {
  const { t } = useLanguage();

  return (
    <div className="info-card wide">
      <h4>{t("explanation_heading")}</h4>
      <p className="explanation-text">{explanation}</p>

      <h4 style={{ marginTop: "16px" }}>{t("recommendations_heading")}</h4>
      <ul className="recommendations-list">
        {recommendations.map((rec, idx) => (
          <li key={idx}>{rec}</li>
        ))}
      </ul>
      <p className="disclaimer">{t("disclaimer_text")}</p>
    </div>
  );
}
