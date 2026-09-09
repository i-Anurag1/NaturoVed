import React from "react";
import { useLanguage } from "../i18n/LanguageContext";

/**
 * Shows the original leaf image side-by-side with the Grad-CAM heatmap
 * overlay, when available. Gracefully explains why it's unavailable
 * (fallback-heuristic mode / no trained model) instead of hiding the
 * section silently, so users understand this is a known limitation,
 * not a bug.
 */
export default function GradCamDisplay({ originalImageUrl, gradcamImage, available }) {
  const { t } = useLanguage();

  return (
    <div className="info-card wide gradcam-card">
      <h4>{t("gradcam_heading")}</h4>
      {available && gradcamImage ? (
        <div className="gradcam-image-row">
          <div className="gradcam-image-col">
            <p className="image-caption">{t("original_image")}</p>
            {originalImageUrl && <img src={originalImageUrl} alt="Original leaf" className="gradcam-image" />}
          </div>
          <div className="gradcam-image-col">
            <p className="image-caption">{t("gradcam_heading")}</p>
            <img src={gradcamImage} alt="Grad-CAM heatmap overlay" className="gradcam-image" />
          </div>
        </div>
      ) : (
        <p className="info-note">{t("gradcam_unavailable")}</p>
      )}
    </div>
  );
}
