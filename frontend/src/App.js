import React from "react";
import { Routes, Route, Link, useLocation } from "react-router-dom";
import HomePage from "./pages/HomePage";
import ResultPage from "./pages/ResultPage";
import HistoryPage from "./pages/HistoryPage";
import DashboardPage from "./pages/DashboardPage";
import LanguageSwitcher from "./components/LanguageSwitcher";
import { useLanguage } from "./i18n/LanguageContext";

export default function App() {
  const location = useLocation();
  const { t } = useLanguage();

  return (
    <div className="app-shell">
      <nav className="navbar">
        <Link to="/" className="brand">🌿 CropDoc</Link>
        <div className="nav-links">
          <Link to="/" className={location.pathname === "/" ? "active" : ""}>{t("nav_home")}</Link>
          <Link to="/history" className={location.pathname === "/history" ? "active" : ""}>{t("nav_history")}</Link>
          <Link to="/dashboard" className={location.pathname === "/dashboard" ? "active" : ""}>{t("nav_dashboard")}</Link>
          <LanguageSwitcher />
        </div>
      </nav>

      <main>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/result/:id" element={<ResultPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
        </Routes>
      </main>

      <footer className="app-footer">
        <p>{t("footer_text")}</p>
      </footer>
    </div>
  );
}
