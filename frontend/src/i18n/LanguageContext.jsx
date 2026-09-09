import React, { createContext, useContext, useState, useCallback } from "react";
import translations from "./translations";

const LanguageContext = createContext(null);

const STORAGE_KEY = "cropdoc_language";
const DEFAULT_LANGUAGE = "en";

export function LanguageProvider({ children }) {
  const [language, setLanguageState] = useState(() => {
    try {
      return localStorage.getItem(STORAGE_KEY) || DEFAULT_LANGUAGE;
    } catch {
      return DEFAULT_LANGUAGE; // localStorage may be unavailable (private browsing, etc.)
    }
  });

  const setLanguage = useCallback((lang) => {
    setLanguageState(lang);
    try {
      localStorage.setItem(STORAGE_KEY, lang);
    } catch {
      // ignore — language just won't persist across reloads
    }
  }, []);

  const t = useCallback(
    (key) => {
      const dict = translations[language] || translations[DEFAULT_LANGUAGE];
      return dict[key] ?? translations[DEFAULT_LANGUAGE][key] ?? key;
    },
    [language]
  );

  return (
    <LanguageContext.Provider value={{ language, setLanguage, t }}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  const ctx = useContext(LanguageContext);
  if (!ctx) {
    throw new Error("useLanguage must be used within a LanguageProvider");
  }
  return ctx;
}
