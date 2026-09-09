import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { getDashboardSummary } from "../api/api";
import SimpleBarChart from "../components/SimpleBarChart";
import LoadingState from "../components/LoadingState";
import { useLanguage } from "../i18n/LanguageContext";

const SEVERITY_COLORS = { None: "#4C8B5B", Mild: "#8bc34a", Moderate: "#C77D2E", Severe: "#B23A2E" };
const RISK_COLORS = { Low: "#4C8B5B", Medium: "#C77D2E", High: "#B23A2E" };

function formatDiseaseLabel(raw) {
  const parts = raw.split("___");
  const name = parts.length > 1 ? parts[1] : raw;
  return name.replace(/_/g, " ");
}

export default function DashboardPage() {
  const { t } = useLanguage();
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    getDashboardSummary(10)
      .then(setSummary)
      .catch(() => setError(t("error_generic")))
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (loading) {
    return (
      <div className="page-container">
        <LoadingState message={t("loading_default")} />
      </div>
    );
  }

  if (error || !summary) {
    return (
      <div className="page-container">
        <p className="field-error">{error}</p>
      </div>
    );
  }

  const diseaseData = summary.disease_distribution.map((d) => ({
    label: formatDiseaseLabel(d.disease),
    count: d.count,
  }));
  const severityData = summary.severity_distribution.map((d) => ({ label: d.severity, count: d.count }));
  const riskData = summary.risk_distribution.map((d) => ({ label: d.risk_level, count: d.count }));

  const mp = summary.model_performance;

  return (
    <div className="page-container">
      <h1>{t("dashboard_heading")}</h1>

      <div className="dashboard-total-card">
        <span className="dashboard-total-number">{summary.total_predictions}</span>
        <span className="dashboard-total-label">{t("dashboard_total")}</span>
      </div>

      <div className="dashboard-grid">
        <div className="info-card">
          <h4>{t("dashboard_disease_dist")}</h4>
          <SimpleBarChart data={diseaseData} />
        </div>
        <div className="info-card">
          <h4>{t("dashboard_severity_dist")}</h4>
          <SimpleBarChart data={severityData} colorMap={SEVERITY_COLORS} />
        </div>
        <div className="info-card">
          <h4>{t("dashboard_risk_dist")}</h4>
          <SimpleBarChart data={riskData} colorMap={RISK_COLORS} />
        </div>
      </div>

      <div className="info-card wide">
        <h4>{t("dashboard_model_status")}</h4>
        <ul className="model-status-list">
          <li>
            <span>{t("dashboard_image_model")}</span>
            <span className={mp.image_model_loaded ? "status-tag ok" : "status-tag warn"}>
              {mp.image_model_loaded ? t("status_loaded") : t("status_fallback")}
            </span>
          </li>
          <li>
            <span>{t("dashboard_field_model")}</span>
            <span className={mp.field_model_loaded ? "status-tag ok" : "status-tag warn"}>
              {mp.field_model_loaded ? t("status_loaded") : t("status_fallback")}
            </span>
          </li>
          <li>
            <span>{t("dashboard_feature_fusion")}</span>
            <span className={mp.feature_fusion_available ? "status-tag ok" : "status-tag warn"}>
              {mp.feature_fusion_available ? t("status_available") : t("status_unavailable")}
            </span>
          </li>
        </ul>
        {mp.notes && <p className="info-note">{mp.notes}</p>}
      </div>

      <div className="info-card wide">
        <h4>{t("dashboard_recent")}</h4>
        {summary.recent_predictions.length === 0 ? (
          <p className="empty-state">{t("history_empty")}</p>
        ) : (
          <table className="history-table">
            <thead>
              <tr>
                <th>{t("col_date")}</th>
                <th>{t("col_crop")}</th>
                <th>{t("col_disease")}</th>
                <th>{t("col_severity")}</th>
                <th>{t("col_risk")}</th>
                <th>{t("col_confidence")}</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {summary.recent_predictions.map((item) => (
                <tr key={item.id} className="history-row">
                  <td>{new Date(item.created_at).toLocaleString()}</td>
                  <td>{item.crop_type}</td>
                  <td>{formatDiseaseLabel(item.predicted_disease)}</td>
                  <td>{item.severity_class}</td>
                  <td>{item.field_risk_level}</td>
                  <td>{Math.round(item.final_confidence * 100)}%</td>
                  <td>
                    <Link to={`/result/${item.id}`} className="view-link">{t("col_view")}</Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
