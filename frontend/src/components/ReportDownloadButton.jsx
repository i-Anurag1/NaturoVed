import React, { useState } from "react";
import { downloadReport } from "../api/api";
import { useLanguage } from "../i18n/LanguageContext";

export default function ReportDownloadButton({ predictionId }) {
  const { t } = useLanguage();
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState(null);

  const handleDownload = async () => {
    setDownloading(true);
    setError(null);
    try {
      await downloadReport(predictionId);
    } catch (err) {
      setError(t("error_generic"));
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div>
      <button className="btn-secondary" onClick={handleDownload} disabled={downloading}>
        {downloading ? t("generating_report") : `📄 ${t("btn_download_report")}`}
      </button>
      {error && <p className="field-error small">{error}</p>}
    </div>
  );
}
