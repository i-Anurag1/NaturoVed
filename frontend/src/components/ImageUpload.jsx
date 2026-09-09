import React, { useRef, useState } from "react";
import { useLanguage } from "../i18n/LanguageContext";

export default function ImageUpload({ onImageSelected, error }) {
  const { t } = useLanguage();
  const [preview, setPreview] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef(null);

  const handleFile = (file) => {
    if (!file) return;
    if (!["image/jpeg", "image/png", "image/jpg"].includes(file.type)) {
      onImageSelected(null, t("error_invalid_image_type"));
      return;
    }
    const reader = new FileReader();
    reader.onload = () => setPreview(reader.result);
    reader.readAsDataURL(file);
    onImageSelected(file, null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    handleFile(e.dataTransfer.files?.[0]);
  };

  return (
    <div className="upload-section">
      <label className="field-label">{t("upload_label")}</label>
      <div
        className={`upload-dropzone ${dragActive ? "active" : ""}`}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") inputRef.current?.click(); }}
        role="button"
        tabIndex={0}
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
      >
        {preview ? (
          <img src={preview} alt="Leaf preview" className="upload-preview" />
        ) : (
          <div className="upload-placeholder">
            <p>{t("upload_placeholder")}</p>
            <p className="upload-hint">{t("upload_hint")}</p>
          </div>
        )}
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png"
        style={{ display: "none" }}
        onChange={(e) => handleFile(e.target.files?.[0])}
      />
      {error && <p className="field-error">{error}</p>}
    </div>
  );
}
