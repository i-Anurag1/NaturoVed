import axios from "axios";

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || "http://localhost:8000";

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

/** GET /health */
export async function checkHealth() {
  const res = await apiClient.get("/health");
  return res.data;
}

/**
 * POST /predict
 * @param {File} imageFile - the leaf/crop image file
 * @param {object} fieldContext - { crop_type, growth_stage, location,
 *   temperature_c, humidity_pct, rainfall_mm, soil_moisture_pct }
 */
export async function submitPrediction(imageFile, fieldContext) {
  const formData = new FormData();
  formData.append("image", imageFile);
  Object.entries(fieldContext).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      formData.append(key, value);
    }
  });

  const res = await apiClient.post("/predict", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
}

/** GET /prediction/{id} */
export async function getPrediction(id) {
  const res = await apiClient.get(`/prediction/${id}`);
  return res.data;
}

/** GET /history?limit=&offset=&crop_type= */
export async function getHistory(params = {}) {
  const res = await apiClient.get("/history", { params });
  return res.data;
}

/** GET /weather?latitude=&longitude= — live weather auto-fill (v2) */
export async function getLiveWeather(latitude, longitude) {
  const res = await apiClient.get("/weather", { params: { latitude, longitude } });
  return res.data;
}

/** GET /dashboard/summary?recent_limit= — analytics dashboard (v2) */
export async function getDashboardSummary(recentLimit = 10) {
  const res = await apiClient.get("/dashboard/summary", { params: { recent_limit: recentLimit } });
  return res.data;
}

/** GET /prediction/{id}/gradcam — (re)generate Grad-CAM heatmap (v2) */
export async function getGradCam(id) {
  const res = await apiClient.get(`/prediction/${id}/gradcam`);
  return res.data;
}

/**
 * GET /prediction/{id}/report — downloads the PDF diagnosis report (v2).
 * Triggers a browser file download rather than returning JSON.
 */
export async function downloadReport(id) {
  const res = await apiClient.get(`/prediction/${id}/report`, { responseType: "blob" });
  const url = window.URL.createObjectURL(new Blob([res.data], { type: "application/pdf" }));
  const link = document.createElement("a");
  link.href = url;
  link.setAttribute("download", `crop_diagnosis_report_${id}.pdf`);
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(url);
}

export default apiClient;
