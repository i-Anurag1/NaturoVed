import React from "react";
import { useLanguage } from "../i18n/LanguageContext";

export default function LanguageSwitcher() {
  const { language, setLanguage } = useLanguage();

  return (
    <div className="language-switcher">
      <button
        className={language === "en" ? "lang-btn active" : "lang-btn"}
        onClick={() => setLanguage("en")}
        aria-label="Switch to English"
      >
        EN
      </button>
      <button
        className={language === "ta" ? "lang-btn active" : "lang-btn"}
        onClick={() => setLanguage("ta")}
        aria-label="தமிழுக்கு மாறவும்"
      >
        தமிழ்
      </button>
    </div>
  );
}
