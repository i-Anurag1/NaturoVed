"""
PDF diagnosis report generator.

WHY REPORTLAB: pure-Python, no system-level binary dependency (unlike
wkhtmltopdf/weasyprint which need external libraries), MIT-style
license, and simple enough for a 4-person student team to read and
extend. It renders directly from Python objects (no HTML/CSS templating
layer needed), which keeps this module self-contained.

Produces a single-page (or slightly longer, if recommendations are
long) PDF containing: the uploaded leaf image, predicted disease,
confidence, severity + affected-area %, submitted field conditions,
field risk assessment, the explanation text, recommendations, and the
Grad-CAM heatmap overlay when available.
"""
import base64
import io
import json
import os
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, ListFlowable, ListItem,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

from app.config import settings

PINE = colors.HexColor("#1F3D2B")
OCHRE = colors.HexColor("#C77D2E")
BRICK = colors.HexColor("#B23A2E")
LIGHT_BG = colors.HexColor("#F3F6EE")


def _severity_color(severity_class: str):
    return {
        "None": colors.HexColor("#4C8B5B"),
        "Mild": colors.HexColor("#8bc34a"),
        "Moderate": OCHRE,
        "Severe": BRICK,
    }.get(severity_class, colors.grey)


def _risk_color(risk_level: str):
    return {"Low": colors.HexColor("#4C8B5B"), "Medium": OCHRE, "High": BRICK}.get(risk_level, colors.grey)


def _decode_data_uri_to_image_reader(data_uri: str):
    """Converts a 'data:image/png;base64,...' string into an in-memory
    file-like object reportlab's Image flowable can read directly."""
    header, encoded = data_uri.split(",", 1)
    raw = base64.b64decode(encoded)
    return io.BytesIO(raw)


def generate_pdf_report(prediction_row, output_path: str):
    """
    prediction_row: an app.models.db_models.Prediction ORM instance (or
    any object exposing the same attributes).
    output_path: where to write the generated PDF file.
    """
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleGreen", parent=styles["Title"], textColor=PINE, alignment=TA_CENTER)
    heading_style = ParagraphStyle("HeadingGreen", parent=styles["Heading2"], textColor=PINE, spaceBefore=12)
    body_style = ParagraphStyle("Body", parent=styles["BodyText"], leading=15)
    disclaimer_style = ParagraphStyle("Disclaimer", parent=styles["BodyText"], fontSize=8, textColor=colors.grey)

    doc = SimpleDocTemplate(output_path, pagesize=A4,
                             topMargin=18 * mm, bottomMargin=18 * mm,
                             leftMargin=18 * mm, rightMargin=18 * mm)
    story = []

    story.append(Paragraph("Crop Disease Diagnosis Report", title_style))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} &nbsp;|&nbsp; "
        f"Prediction ID: {prediction_row.id}",
        ParagraphStyle("Meta", parent=styles["Normal"], alignment=TA_CENTER, textColor=colors.grey),
    ))
    story.append(Spacer(1, 6 * mm))

    # --- Images: original + Grad-CAM side by side (if available) ---
    image_path = os.path.join(settings.UPLOAD_DIR, prediction_row.image_filename)
    image_cells = []
    if os.path.exists(image_path):
        image_cells.append([Paragraph("Uploaded Image", heading_style), ""])
        img = RLImage(image_path, width=75 * mm, height=75 * mm, kind="proportional")
        gradcam_flowable = ""
        if getattr(prediction_row, "gradcam_available", False):
            try:
                from app.services import image_model
                gradcam_data_uri = image_model.generate_gradcam_for_disease(image_path, prediction_row.predicted_disease)
                if gradcam_data_uri:
                    gradcam_flowable = RLImage(
                        _decode_data_uri_to_image_reader(gradcam_data_uri),
                        width=75 * mm, height=75 * mm, kind="proportional",
                    )
            except Exception:
                gradcam_flowable = ""
        header_row = [Paragraph("Original Image", body_style),
                      Paragraph("Grad-CAM Explanation" if gradcam_flowable else "Grad-CAM (unavailable)", body_style)]
        img_row = [img, gradcam_flowable if gradcam_flowable else Paragraph("Not available for this prediction.", body_style)]
        img_table = Table([header_row, img_row], colWidths=[85 * mm, 85 * mm])
        img_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
        ]))
        story.append(img_table)
        story.append(Spacer(1, 6 * mm))

    # --- Disease + confidence summary table ---
    is_healthy = "Healthy" in prediction_row.predicted_disease
    disease_display = prediction_row.predicted_disease.replace("___", " - ").replace("_", " ")
    summary_data = [
        ["Predicted Disease", disease_display],
        ["Status", "Healthy" if is_healthy else "Disease Detected"],
        ["Image-Model Confidence", f"{prediction_row.image_confidence * 100:.1f}%"],
        ["Final (Fused) Confidence", f"{prediction_row.final_confidence * 100:.1f}%"],
        ["Fusion Method", prediction_row.fusion_method.replace("_", " ").title()],
    ]
    summary_table = Table(summary_data, colWidths=[60 * mm, 110 * mm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHT_BG),
        ("TEXTCOLOR", (0, 0), (0, -1), PINE),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DCE3D5")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 6 * mm))

    # --- Severity + Risk side by side ---
    severity_data = [
        ["Severity", prediction_row.severity_class],
        ["Affected Leaf Area", f"{prediction_row.affected_area_pct:.1f}%" if prediction_row.affected_area_pct is not None else "N/A"],
    ]
    risk_data = [
        ["Field Risk Level", prediction_row.field_risk_level],
        ["Field Risk Score", f"{prediction_row.field_risk_score * 100:.1f}%"],
    ]
    sev_table = Table(severity_data, colWidths=[40 * mm, 40 * mm])
    sev_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), _severity_color(prediction_row.severity_class)),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DCE3D5")),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    risk_table = Table(risk_data, colWidths=[40 * mm, 40 * mm])
    risk_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), _risk_color(prediction_row.field_risk_level)),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DCE3D5")),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    combined = Table([[sev_table, risk_table]], colWidths=[85 * mm, 85 * mm])
    story.append(combined)
    story.append(Spacer(1, 6 * mm))

    # --- Field conditions submitted ---
    story.append(Paragraph("Submitted Field Conditions", heading_style))
    field_data = [
        ["Crop", prediction_row.crop_type.title()],
        ["Growth Stage", prediction_row.growth_stage.title()],
        ["Location", prediction_row.location or "Not provided"],
        ["Temperature", f"{prediction_row.temperature_c}°C"],
        ["Humidity", f"{prediction_row.humidity_pct}%"],
        ["Rainfall (7-day)", f"{prediction_row.rainfall_mm}mm"],
        ["Soil Moisture", f"{prediction_row.soil_moisture_pct}%"],
        ["Weather Source", "Live Weather API" if prediction_row.weather_source == "auto" else "Manually Entered"],
    ]
    field_table = Table(field_data, colWidths=[50 * mm, 120 * mm])
    field_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DCE3D5")),
        ("BACKGROUND", (0, 0), (0, -1), LIGHT_BG),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(field_table)
    story.append(Spacer(1, 6 * mm))

    # --- Explanation ---
    story.append(Paragraph("Explanation", heading_style))
    story.append(Paragraph(prediction_row.explanation, body_style))

    field_explanation = getattr(prediction_row, "_field_explanation_text", None)
    if field_explanation:
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph(field_explanation, body_style))

    story.append(Spacer(1, 4 * mm))

    # --- Recommendations ---
    story.append(Paragraph("Recommended Actions", heading_style))
    try:
        recs = json.loads(prediction_row.recommendations)
    except (TypeError, ValueError):
        recs = []
    if recs:
        story.append(ListFlowable(
            [ListItem(Paragraph(r, body_style)) for r in recs],
            bulletType="bullet",
        ))

    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph(
        "This report was generated by an automated decision-support system and is general, "
        "rule-based guidance only. It is not a substitute for advice from a licensed local "
        "agronomist or agricultural extension service.",
        disclaimer_style,
    ))

    doc.build(story)
    return output_path
