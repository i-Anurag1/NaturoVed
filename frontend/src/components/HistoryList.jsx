import React from "react";
import { useNavigate } from "react-router-dom";
import { useLanguage } from "../i18n/LanguageContext";

export default function HistoryList({ items }) {
  const navigate = useNavigate();
  const { t } = useLanguage();

  if (!items || items.length === 0) {
    return <p className="empty-state">{t("history_empty")}</p>;
  }

  return (
    <div className="table-scroll-wrap">
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
          {items.map((item) => (
            <tr key={item.id} onClick={() => navigate(`/result/${item.id}`)} className="history-row">
              <td>{new Date(item.created_at).toLocaleString()}</td>
              <td>{item.crop_type}</td>
              <td>{item.predicted_disease.replace(/___/g, " - ").replace(/_/g, " ")}</td>
              <td>{item.severity_class}</td>
              <td>{item.field_risk_level}</td>
              <td>{Math.round(item.final_confidence * 100)}%</td>
              <td className="view-link">{t("col_view")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
