"""
Dental OPG X-ray Analysis — Gradio Web Application
Deployable to HuggingFace Spaces
"""
import io
import os
import sys
import json
import logging
import tempfile
import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import gradio as gr
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  MODEL LOADING                                                       #
# ------------------------------------------------------------------ #

MODEL_PATH = Path("models/best/best.pt")
FALLBACK_MODEL = "yolov8s.pt"
LOAD_ERROR = None


def load_pipeline():
    global LOAD_ERROR
    try:
        from ultralytics import YOLO  # noqa: F401
    except ImportError:
        LOAD_ERROR = "ultralytics not installed. Run: pip install ultralytics"
        logger.error(LOAD_ERROR)
        return None
    try:
        from dental_opg.pipeline.prediction_pipeline import PredictionPipeline
        if MODEL_PATH.exists():
            return PredictionPipeline(model_path=MODEL_PATH)
        else:
            logger.warning(f"Trained model not found at {MODEL_PATH}. Using pretrained fallback.")
            return PredictionPipeline(model_path=FALLBACK_MODEL)
    except Exception as e:
        LOAD_ERROR = str(e)
        logger.error(f"Failed to load pipeline: {e}")
        return None


pipeline = load_pipeline()
logger.info(f"Pipeline loaded: {pipeline is not None}")

# ------------------------------------------------------------------ #
#  PDF REPORT GENERATOR                                                #
# ------------------------------------------------------------------ #

def generate_pdf_report(result: dict, image: np.ndarray) -> Optional[str]:
    """
    Generate a downloadable PDF clinical report from prediction results.
    Returns the path to the saved PDF file, or None if reportlab is unavailable.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable, Image as RLImage,
        )
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    except ImportError:
        logger.warning("reportlab not installed. PDF generation skipped.")
        return None

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()
    pdf_path = tmp.name

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    W = A4[0] - 4 * cm  # usable width

    # Custom styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontSize=20,
        textColor=colors.HexColor("#0f2d52"),
        spaceAfter=4,
        alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#64748b"),
        alignment=TA_CENTER,
        spaceAfter=12,
    )
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontSize=12,
        textColor=colors.HexColor("#1a56db"),
        spaceBefore=14,
        spaceAfter=4,
        borderPad=2,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=9.5,
        textColor=colors.HexColor("#1e293b"),
        leading=15,
        spaceAfter=6,
    )
    small_style = ParagraphStyle(
        "Small",
        parent=styles["Normal"],
        fontSize=8.5,
        textColor=colors.HexColor("#64748b"),
        leading=13,
    )

    SEVERITY_COLORS = {
        "urgent":   colors.HexColor("#fee2e2"),
        "moderate": colors.HexColor("#fef3c7"),
        "monitor":  colors.HexColor("#e0f2fe"),
        "normal":   colors.HexColor("#d1fae5"),
    }
    SEVERITY_TEXT = {
        "urgent":   colors.HexColor("#991b1b"),
        "moderate": colors.HexColor("#92400e"),
        "monitor":  colors.HexColor("#075985"),
        "normal":   colors.HexColor("#065f46"),
    }

    story = []
    now = datetime.datetime.now().strftime("%d %B %Y at %H:%M")

    # Header
    story.append(Paragraph("Dental OPG X-ray Analysis Report", title_style))
    story.append(Paragraph(f"Generated on {now} — AI-assisted screening tool", subtitle_style))
    story.append(HRFlowable(width=W, color=colors.HexColor("#1a56db"), thickness=2))
    story.append(Spacer(1, 0.4 * cm))

    # Scan image
    if image is not None:
        try:
            img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            img_bytes = io.BytesIO()
            img_pil.save(img_bytes, format="JPEG", quality=85)
            img_bytes.seek(0)
            max_w = W
            orig_w, orig_h = img_pil.size
            scale = min(max_w / orig_w, 7 * cm / orig_h)
            disp_w = orig_w * scale
            disp_h = orig_h * scale
            rl_img = RLImage(img_bytes, width=disp_w, height=disp_h)
            story.append(rl_img)
            story.append(Spacer(1, 0.3 * cm))
        except Exception as e:
            logger.warning(f"Could not embed image in PDF: {e}")

    # Validation status
    valid = result.get("valid", True)
    val_msg = result.get("validation_msg", "")
    story.append(Paragraph("Image Validation", section_style))
    val_color = colors.HexColor("#d1fae5") if valid else colors.HexColor("#fee2e2")
    val_text_color = colors.HexColor("#065f46") if valid else colors.HexColor("#991b1b")
    val_status = "PASSED — Image accepted as a dental OPG X-ray" if valid else f"FAILED — {val_msg}"
    val_table = Table([[Paragraph(val_status, ParagraphStyle("vt", fontSize=9, textColor=val_text_color))]], colWidths=[W])
    val_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), val_color),
        ("ROUNDEDCORNERS", [6]),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(val_table)
    story.append(Spacer(1, 0.3 * cm))

    if not valid:
        story.append(Paragraph(
            "The image did not pass validation. Results below are not clinically meaningful.",
            body_style,
        ))
        story.append(Spacer(1, 0.5 * cm))

    # Overall summary
    clinical = result.get("clinical_report", {})
    severity = result.get("severity", "")
    story.append(Paragraph("Overall Assessment", section_style))
    story.append(Paragraph(clinical.get("summary", "No findings."), body_style))
    story.append(Paragraph(f"<b>Severity:</b> {severity}", body_style))
    story.append(Paragraph(
        f"<b>Total findings:</b> {result.get('cavity_count', 0)}   "
        f"<b>Average confidence:</b> {result.get('confidence_avg', 0):.1%}",
        body_style,
    ))
    story.append(Spacer(1, 0.2 * cm))

    # Priority action
    priority = clinical.get("priority_action", "")
    if priority:
        p_sev = clinical.get("overall_severity", "normal")
        p_bg = SEVERITY_COLORS.get(p_sev, colors.HexColor("#f0f9ff"))
        p_tc = SEVERITY_TEXT.get(p_sev, colors.HexColor("#1e293b"))
        pt = Table(
            [[Paragraph(f"<b>Priority Action:</b> {priority}", ParagraphStyle("pa", fontSize=9.5, textColor=p_tc))]],
            colWidths=[W],
        )
        pt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), p_bg),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(pt)
        story.append(Spacer(1, 0.3 * cm))

    # Per-condition findings table
    findings = clinical.get("findings", [])
    if findings:
        story.append(Paragraph("Detected Conditions", section_style))
        table_data = [[
            Paragraph("<b>Condition</b>", body_style),
            Paragraph("<b>Count</b>", body_style),
            Paragraph("<b>Severity</b>", body_style),
            Paragraph("<b>Recommended Action</b>", body_style),
        ]]
        col_widths = [W * 0.22, W * 0.08, W * 0.13, W * 0.57]
        ts = [
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f2d52")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]
        for i, f in enumerate(findings):
            sev = f.get("severity", "normal")
            row_bg = SEVERITY_COLORS.get(sev, colors.white)
            sev_tc = SEVERITY_TEXT.get(sev, colors.black)
            row = [
                Paragraph(f.get("full_name", f.get("condition", "")), body_style),
                Paragraph(str(f.get("count", 0)), body_style),
                Paragraph(
                    f"<b>{f.get('severity_label', '')}</b>",
                    ParagraphStyle("sl", fontSize=9, textColor=sev_tc),
                ),
                Paragraph(f.get("recommended_action", ""), body_style),
            ]
            table_data.append(row)
            ts.append(("BACKGROUND", (0, i + 1), (-1, i + 1), row_bg))

        findings_table = Table(table_data, colWidths=col_widths)
        findings_table.setStyle(TableStyle(ts))
        story.append(findings_table)
        story.append(Spacer(1, 0.3 * cm))

    # Individual detections
    detections = result.get("detections", [])
    if detections:
        story.append(Paragraph("Individual Detection Details", section_style))
        det_data = [[
            Paragraph("<b>#</b>", small_style),
            Paragraph("<b>Condition</b>", small_style),
            Paragraph("<b>Confidence</b>", small_style),
            Paragraph("<b>Bounding Box (x1, y1, x2, y2)</b>", small_style),
        ]]
        for i, det in enumerate(detections):
            bb = det["bbox"]
            det_data.append([
                Paragraph(str(i + 1), small_style),
                Paragraph(det["class_name"], small_style),
                Paragraph(f"{det['confidence']:.1%}", small_style),
                Paragraph(f"({bb[0]}, {bb[1]}, {bb[2]}, {bb[3]})", small_style),
            ])
        det_table = Table(det_data, colWidths=[W * 0.06, W * 0.3, W * 0.18, W * 0.46])
        det_table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a56db")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ]))
        story.append(det_table)
        story.append(Spacer(1, 0.3 * cm))

    # Disclaimer
    story.append(HRFlowable(width=W, color=colors.HexColor("#e2e8f0"), thickness=1))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        "<b>Medical Disclaimer:</b> This report is generated by an AI screening tool and is "
        "intended to assist qualified dental professionals. It does not constitute a clinical "
        "diagnosis and must not replace examination by a licensed dentist. All findings should "
        "be verified through direct clinical assessment before any treatment decision is made.",
        small_style,
    ))
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph(
        "Model: YOLOv8s trained on annotated dental OPG X-ray dataset (6 classes)  |  "
        "Author: Paul Sentongo  |  "
        "GitHub: github.com/sentongo-web/Dental-OPG-XRAY-Analysis-MLOPS",
        small_style,
    ))

    doc.build(story)
    return pdf_path


# ------------------------------------------------------------------ #
#  PREDICTION + REPORT FUNCTION                                        #
# ------------------------------------------------------------------ #

def predict_and_report(
    image: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
) -> Tuple[np.ndarray, str, str]:
    """
    Returns: annotated image, markdown summary, path to PDF file
    """
    if image is None:
        return None, "Please upload an OPG X-ray image.", None

    if pipeline is None:
        error_detail = LOAD_ERROR or "Unknown error during model loading."
        msg = f"**Model failed to load.**\n\nReason: `{error_detail}`"
        return image, msg, None

    try:
        pipeline.conf = conf_threshold
        pipeline.iou = iou_threshold

        result = pipeline.predict(image, return_visualization=True, validate=True)
        vis_image = result.get("visualization")

        # Convert to RGB for Gradio
        if vis_image is not None:
            vis_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        else:
            vis_rgb = image

        summary = _build_markdown_summary(result)

        # Generate PDF using the annotated BGR image
        pdf_path = generate_pdf_report(result, vis_image if vis_image is not None else image)

        return vis_rgb, summary, pdf_path

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        return image, f"Analysis failed: {str(e)}", None


def _build_markdown_summary(result: dict) -> str:
    valid = result.get("valid", True)
    val_msg = result.get("validation_msg", "")

    if not valid:
        return (
            "## Image Rejected\n\n"
            f"{val_msg}\n\n"
            "Please upload a panoramic dental OPG X-ray image in JPG, PNG, or TIFF format."
        )

    count = result.get("cavity_count", 0)
    severity = result.get("severity", "")
    avg_conf = result.get("confidence_avg", 0)
    class_counts = result.get("class_counts", {})
    detections = result.get("detections", [])
    clinical = result.get("clinical_report", {})

    # Severity icon
    sev_lower = severity.lower()
    if "urgent" in sev_lower:
        icon = "🔴"
    elif "moderate" in sev_lower:
        icon = "🟠"
    elif "mild" in sev_lower:
        icon = "🟡"
    elif "monitor" in sev_lower:
        icon = "🔵"
    elif "no pathological" in sev_lower:
        icon = "✅"
    else:
        icon = "⚪"

    lines = [
        f"## {icon} Scan Classification",
        "",
        "**Image type:** Panoramic OPG Dental X-ray",
        f"**Total findings:** {count}",
        f"**Average confidence:** {avg_conf:.1%}",
        "",
        f"**Overall severity:** {severity}",
        "",
    ]

    # Summary sentence
    summary_text = clinical.get("summary", "")
    if summary_text:
        lines.append(f"> {summary_text}")
        lines.append("")

    # Priority action
    priority = clinical.get("priority_action", "")
    if priority:
        lines.append(f"**Priority action:** {priority}")
        lines.append("")

    # Per-condition breakdown
    findings = clinical.get("findings", [])
    if findings:
        lines.append("### Defect Predictions")
        lines.append("")
        lines.append("| Condition | Count | Severity | What This Means |")
        lines.append("|-----------|-------|----------|-----------------|")
        for f in findings:
            lines.append(
                f"| **{f['full_name']}** | {f['count']} | "
                f"{f['severity_label']} | {f['description'][:80]}... |"
            )
        lines.append("")

    # Detailed detection list
    if detections:
        lines.append("### All Detections")
        lines.append("")
        lines.append("| # | Condition | Confidence |")
        lines.append("|---|-----------|-----------|")
        for i, det in enumerate(detections):
            lines.append(f"| {i+1} | {det['class_name']} | {det['confidence']:.1%} |")
        lines.append("")

    if count == 0:
        lines.append(
            "No pathological findings were detected in this scan. "
            "This does not guarantee the absence of dental disease. "
            "Regular check-ups every 6 months are still recommended."
        )
        lines.append("")

    lines.extend([
        "---",
        "_This AI system supports dental professionals. "
        "Always confirm findings with a clinical examination._",
    ])

    return "\n".join(lines)


# ------------------------------------------------------------------ #
#  CSS                                                                 #
# ------------------------------------------------------------------ #

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@700&display=swap');

* { font-family: 'Inter', sans-serif !important; }

body, .gradio-container { background: #f0f4f8 !important; }

.hero-banner {
    background: linear-gradient(135deg, #0f2d52 0%, #1a56db 60%, #0e9f6e 100%);
    border-radius: 20px; padding: 52px 48px; margin-bottom: 8px;
    color: white; position: relative; overflow: hidden;
}
.hero-banner::before {
    content:''; position:absolute; top:-60px; right:-60px;
    width:300px; height:300px;
    background:rgba(255,255,255,0.05); border-radius:50%;
}
.hero-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.5rem !important; font-weight: 700 !important;
    line-height: 1.2 !important; margin: 0 0 14px 0 !important;
    color: #ffffff !important;
}
.hero-subtitle {
    font-size: 1.1rem !important; font-weight: 300 !important;
    color: rgba(255,255,255,0.88) !important;
    max-width: 680px; line-height: 1.7 !important;
    margin: 0 0 26px 0 !important;
}
.hero-badges { display:flex; gap:10px; flex-wrap:wrap; }
.badge {
    background:rgba(255,255,255,0.15); border:1px solid rgba(255,255,255,0.25);
    color:white !important; padding:6px 14px; border-radius:100px;
    font-size:0.8rem !important; font-weight:500 !important;
}
.info-card {
    background:white; border-radius:16px; padding:30px;
    box-shadow:0 1px 3px rgba(0,0,0,0.06),0 4px 24px rgba(0,0,0,0.04);
    margin-bottom:8px;
}
.info-card h2 {
    font-size:1.25rem !important; font-weight:600 !important;
    color:#0f2d52 !important; margin:0 0 10px 0 !important;
}
.info-card p, .info-card li {
    font-size:0.93rem !important; line-height:1.75 !important; color:#374151 !important;
}
.info-card ul { padding-left:20px; margin:10px 0; }
.info-card li { margin-bottom:5px; }
.accent-blue { color:#1a56db !important; font-weight:600 !important; }
.accent-green { color:#0e9f6e !important; font-weight:600 !important; }
.legend-grid {
    display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
    gap:10px; margin-top:14px;
}
.legend-item {
    display:flex; align-items:center; gap:10px;
    padding:10px 14px; background:#f8fafc; border-radius:10px;
    border:1px solid #e2e8f0;
}
.legend-dot {
    width:14px; height:14px; border-radius:50%; flex-shrink:0;
}
.legend-label { font-size:0.83rem !important; color:#1e293b !important; font-weight:500 !important; }
.steps-grid {
    display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
    gap:14px; margin-top:14px;
}
.step-item {
    text-align:center; padding:18px 10px;
    background:#f8fafc; border-radius:12px; border:1px solid #e2e8f0;
}
.step-num {
    display:inline-flex; align-items:center; justify-content:center;
    width:34px; height:34px; background:#1a56db; color:white;
    border-radius:50%; font-size:0.82rem; font-weight:700; margin-bottom:8px;
}
.step-label { font-size:0.83rem !important; font-weight:500 !important; color:#1e293b !important; }
.disclaimer-box {
    background:#fffbeb; border:1px solid #fde68a;
    border-left:4px solid #f59e0b; border-radius:10px; padding:14px 18px; margin-top:8px;
}
.disclaimer-box p {
    font-size:0.87rem !important; color:#78350f !important; margin:0 !important; line-height:1.6 !important;
}
.stats-row { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin:18px 0; }
.stat-box { background:#f0f7ff; border-left:4px solid #1a56db; border-radius:8px; padding:14px; }
.stat-number { font-size:1.4rem !important; font-weight:700 !important; color:#0f2d52 !important; display:block; }
.stat-label { font-size:0.77rem !important; color:#64748b !important; margin-top:2px; }
.footer-section {
    background:#0f2d52; border-radius:16px; padding:30px;
    text-align:center; margin-top:8px;
}
.footer-section p { color:rgba(255,255,255,0.7) !important; font-size:0.87rem !important; margin:4px 0 !important; }
.footer-section a { color:#60a5fa !important; text-decoration:none; font-weight:500; }
.section-tag {
    display:inline-block; background:#eff6ff; color:#1a56db;
    font-size:0.73rem; font-weight:700; text-transform:uppercase;
    letter-spacing:1px; padding:3px 10px; border-radius:100px; margin-bottom:8px;
}
footer { display: none !important; }
"""

# ------------------------------------------------------------------ #
#  HTML BLOCKS                                                         #
# ------------------------------------------------------------------ #

HERO_HTML = """
<div class="hero-banner">
  <p style="font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:2px;color:rgba(255,255,255,0.6);margin:0 0 10px 0;">
    AI for Oral Health
  </p>
  <h1 class="hero-title">Detecting Dental Conditions Before They Cause Harm</h1>
  <p class="hero-subtitle">
    Upload a panoramic OPG X-ray and this system will automatically identify cavities,
    infections, fractures, impacted teeth, and more — then generate a downloadable clinical report
    for your review. Built for dentists, dental officers, and health workers across Uganda and beyond.
  </p>
  <div class="hero-badges">
    <span class="badge">YOLOv8s Vision AI</span>
    <span class="badge">6-Class Detection</span>
    <span class="badge">OPG Validation</span>
    <span class="badge">PDF Report</span>
    <span class="badge">Open Source</span>
  </div>
</div>
"""

HOW_TO_USE_HTML = """
<div class="info-card">
  <span class="section-tag">Quick Start</span>
  <h2>How to Use This Tool</h2>
  <div class="steps-grid">
    <div class="step-item">
      <div class="step-num">1</div>
      <p class="step-label">Upload an OPG X-ray image in JPG, PNG, or TIFF format</p>
    </div>
    <div class="step-item">
      <div class="step-num">2</div>
      <p class="step-label">The system checks whether the image is a valid dental X-ray</p>
    </div>
    <div class="step-item">
      <div class="step-num">3</div>
      <p class="step-label">Click Analyse Scan and wait a moment for results</p>
    </div>
    <div class="step-item">
      <div class="step-num">4</div>
      <p class="step-label">Review findings and download the full PDF clinical report</p>
    </div>
  </div>
</div>
"""

LEGEND_HTML = """
<div class="info-card">
  <span class="section-tag">Detection Classes</span>
  <h2>What the Model Looks For</h2>
  <p>The AI detects six conditions in a single scan. Each is colour-coded on the annotated image.</p>
  <div class="legend-grid">
    <div class="legend-item">
      <div class="legend-dot" style="background:#dc2020;"></div>
      <span class="legend-label">BDC-BDR — Badly Decayed Crown / Root</span>
    </div>
    <div class="legend-item">
      <div class="legend-dot" style="background:#f97316;"></div>
      <span class="legend-label">Caries — Dental Cavity</span>
    </div>
    <div class="legend-item">
      <div class="legend-dot" style="background:#eab308;"></div>
      <span class="legend-label">Fractured Teeth</span>
    </div>
    <div class="legend-item">
      <div class="legend-dot" style="background:#16a34a;"></div>
      <span class="legend-label">Healthy Teeth</span>
    </div>
    <div class="legend-item">
      <div class="legend-dot" style="background:#a21caf;"></div>
      <span class="legend-label">Impacted Teeth</span>
    </div>
    <div class="legend-item">
      <div class="legend-dot" style="background:#e03030;"></div>
      <span class="legend-label">Infection — Periapical Abscess</span>
    </div>
  </div>
</div>
"""

ABOUT_MODEL_HTML = """
<div class="info-card">
  <span class="section-tag">The Technology</span>
  <h2>How the AI Works</h2>
  <p>
    The system uses <span class="accent-blue">YOLOv8s</span>, which stands for You Only Look Once,
    version 8 (small variant). When you upload an X-ray, the model scans the entire image in a
    single pass and draws precise boxes around every area that matches one of the six dental
    conditions it was trained to recognise. It does not examine one region at a time — it sees
    everything at once, the way an experienced radiologist might glance at a scan and immediately
    notice something in the lower right molar.
  </p>
  <p style="margin-top:12px;">
    The model was trained on annotated OPG X-ray images using an end-to-end MLOps pipeline
    with DVC for data versioning and MLflow for experiment tracking. Training used medical-safe
    image augmentations only — no horizontal flips (to preserve anatomical orientation) and no
    colour distortion (X-rays are grayscale).
  </p>
  <h2 style="margin-top:24px;">Why YOLOv8s Was Chosen</h2>
  <ul>
    <li><span class="accent-blue">Speed without sacrifice.</span> Results appear in seconds, even on a standard laptop.</li>
    <li><span class="accent-blue">Proven in medical imaging.</span> The YOLO family handles the low-contrast nature of X-rays well.</li>
    <li><span class="accent-blue">Localization built in.</span> The model draws a box around exactly where the finding is, not just a label.</li>
    <li><span class="accent-blue">Runs on modest hardware.</span> No expensive GPU server required — deployable in district hospitals.</li>
    <li><span class="accent-blue">Open and reproducible.</span> Anyone can retrain, verify, or improve the model from the public GitHub repo.</li>
  </ul>
</div>
"""

UGANDA_HTML = """
<div class="info-card">
  <span class="section-tag" style="background:#d1fae5;color:#065f46;">Uganda Context</span>
  <h2>Why This Matters in Uganda</h2>
  <p>
    Uganda has fewer than <span class="accent-green">200 registered dentists</span> serving a
    population of over 47 million people. In rural areas, a patient may travel three hours to
    reach a clinic, only to learn that a detailed X-ray analysis requires a trip to Kampala.
    By the time treatment begins, a small cavity can become an extraction or a spreading infection.
  </p>
  <div class="stats-row">
    <div class="stat-box">
      <span class="stat-number">1 : 235,000</span>
      <span class="stat-label">Dentist to population ratio in Uganda</span>
    </div>
    <div class="stat-box">
      <span class="stat-number">&gt;80%</span>
      <span class="stat-label">Of Ugandans live outside major cities</span>
    </div>
    <div class="stat-box">
      <span class="stat-number">Early</span>
      <span class="stat-label">Detection saves teeth that late detection cannot</span>
    </div>
  </div>
  <h2 style="margin-top:24px;">Who Benefits</h2>
  <ul>
    <li><span class="accent-green">Dental officers and nurses</span> at district hospitals reading X-rays without a specialist present.</li>
    <li><span class="accent-green">Dental schools</span> at Makerere University and KIU, where students can build diagnostic intuition alongside real training.</li>
    <li><span class="accent-green">Mobile health outreach teams</span> conducting community screenings in schools and markets.</li>
    <li><span class="accent-green">Private dental clinics</span> looking to review X-rays faster and see more patients each day.</li>
    <li><span class="accent-green">Public health researchers</span> mapping dental caries prevalence across regions and age groups.</li>
  </ul>
  <p style="margin-top:18px;padding:14px;background:#f0fdf4;border-radius:10px;border-left:4px solid #10b981;">
    The vision is not to replace dentists. Uganda needs more dentists. The vision is to make
    every dentist already working in the country dramatically more effective, and to give every
    health worker the ability to catch dental disease earlier.
  </p>
</div>
"""

DISCLAIMER_HTML = """
<div class="disclaimer-box">
  <p>
    <strong>Medical Disclaimer:</strong> This tool is designed to assist qualified dental professionals
    and trained health workers. It is not a substitute for clinical examination, professional diagnosis,
    or treatment planning. All findings must be confirmed by a licensed dentist before any clinical
    decision is made.
  </p>
</div>
"""

FOOTER_HTML = """
<div class="footer-section">
  <p style="font-size:0.98rem !important;font-weight:600 !important;color:white !important;margin-bottom:6px !important;">
    Dental OPG X-ray Analysis System
  </p>
  <p>Built by <strong style="color:white;">Paul Sentongo</strong> &mdash;
    <a href="https://github.com/sentongo-web/Dental-OPG-XRAY-Analysis-MLOPS" target="_blank">
      View on GitHub
    </a>
  </p>
  <p style="margin-top:8px !important;">
    Model: YOLOv8s &nbsp;|&nbsp; 6 Classes &nbsp;|&nbsp; Framework: Gradio &nbsp;|&nbsp;
    Tracking: MLflow + DVC
  </p>
  <p style="margin-top:5px !important;font-size:0.77rem !important;">
    For research, educational, and clinical support purposes. Not a certified medical device.
  </p>
</div>
"""

# ------------------------------------------------------------------ #
#  EXAMPLE IMAGES                                                      #
# ------------------------------------------------------------------ #

EXAMPLES = []
example_dir = Path("data/examples")
if example_dir.exists():
    EXAMPLES = [[str(f), 0.25, 0.45] for f in example_dir.glob("*.jpg")]


# ------------------------------------------------------------------ #
#  GRADIO INTERFACE                                                    #
# ------------------------------------------------------------------ #

def create_interface():
    with gr.Blocks(
        title="Dental OPG Analysis | AI Screening for Uganda",
        theme=gr.themes.Base(
            primary_hue="blue",
            secondary_hue="emerald",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
        ),
        css=CUSTOM_CSS,
    ) as demo:

        gr.HTML(HERO_HTML)
        gr.HTML(HOW_TO_USE_HTML)
        gr.HTML(LEGEND_HTML)

        # Main analysis panel
        with gr.Group():
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    gr.Markdown("### Upload X-ray")
                    input_image = gr.Image(
                        type="numpy",
                        label="OPG X-ray Image",
                        sources=["upload", "clipboard"],
                        height=370,
                    )
                    with gr.Accordion("Detection Settings", open=False):
                        conf_slider = gr.Slider(
                            minimum=0.10, maximum=0.90, value=0.25, step=0.05,
                            label="Confidence Threshold",
                            info="How certain the model must be before reporting a finding. "
                                 "Lower values catch more but may include false positives.",
                        )
                        iou_slider = gr.Slider(
                            minimum=0.20, maximum=0.80, value=0.45, step=0.05,
                            label="Overlap Threshold",
                            info="Controls how much detected boxes can overlap. "
                                 "Leave at default unless results look duplicated.",
                        )
                    with gr.Row():
                        analyse_btn = gr.Button("Analyse Scan", variant="primary", size="lg")
                        clear_btn = gr.ClearButton([input_image], value="Clear", variant="secondary")

                with gr.Column(scale=1):
                    gr.Markdown("### Annotated Result")
                    output_image = gr.Image(
                        type="numpy",
                        label="Annotated X-ray",
                        height=370,
                        interactive=False,
                    )

            with gr.Row():
                with gr.Column(scale=2):
                    summary_output = gr.Markdown(
                        value="Upload an OPG image above and click Analyse Scan to see the full report here.",
                    )
                with gr.Column(scale=1):
                    gr.Markdown("### Download Report")
                    pdf_output = gr.File(
                        label="Clinical PDF Report",
                        file_types=[".pdf"],
                        interactive=False,
                    )
                    gr.Markdown(
                        "_The PDF includes the annotated scan, all detected conditions, "
                        "per-condition clinical descriptions, recommended actions, and a "
                        "medical disclaimer. Ready to print or share._",
                        elem_id="pdf-info",
                    )

        if EXAMPLES:
            with gr.Group():
                gr.Markdown("### Example Scans")
                gr.Examples(
                    examples=EXAMPLES,
                    inputs=[input_image, conf_slider, iou_slider],
                    outputs=[output_image, summary_output, pdf_output],
                    fn=predict_and_report,
                    cache_examples=True,
                )

        gr.HTML(DISCLAIMER_HTML)
        gr.HTML(ABOUT_MODEL_HTML)
        gr.HTML(UGANDA_HTML)
        gr.HTML(FOOTER_HTML)

        # Events
        analyse_btn.click(
            fn=predict_and_report,
            inputs=[input_image, conf_slider, iou_slider],
            outputs=[output_image, summary_output, pdf_output],
            show_progress=True,
        )
        input_image.upload(
            fn=predict_and_report,
            inputs=[input_image, conf_slider, iou_slider],
            outputs=[output_image, summary_output, pdf_output],
        )

    return demo


# ------------------------------------------------------------------ #
#  ENTRY POINT                                                         #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    demo = create_interface()
    demo.launch(share=args.share, server_port=args.port, server_name=args.host, show_error=True)
