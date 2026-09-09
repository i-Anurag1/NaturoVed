import React, { useEffect, useState } from "react";
import HistoryList from "../components/HistoryList";
import LoadingState from "../components/LoadingState";
import { getHistory } from "../api/api";
import { useLanguage } from "../i18n/LanguageContext";

export default function HistoryPage() {
  const { t } = useLanguage();
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    getHistory({ limit: 50 })
      .then(setItems)
      .catch(() => setError(t("error_generic")))
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="page-container">
      <h1>{t("history_heading")}</h1>
      {loading && <LoadingState message={t("loading_default")} />}
      {error && <p className="field-error">{error}</p>}
      {!loading && !error && <HistoryList items={items} />}
    </div>
  );
}
