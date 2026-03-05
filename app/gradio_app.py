"""
Dental OPG Cavity Detection — Gradio Web Application
Deployable to HuggingFace Spaces
"""
import os
import sys
import json
import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  MODEL LOADING                                                       #
# ------------------------------------------------------------------ #

MODEL_PATH = Path("models/best/best.pt")
FALLBACK_MODEL = "yolov8n.pt"

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
            logger.info(f"Loading trained model: {MODEL_PATH}")
            return PredictionPipeline(model_path=MODEL_PATH)
        else:
            logger.warning(f"Trained model not found at {MODEL_PATH}")
            logger.info("Downloading pretrained YOLOv8n as demo fallback...")
            return PredictionPipeline(model_path=FALLBACK_MODEL)
    except Exception as e:
        LOAD_ERROR = str(e)
        logger.error(f"Failed to load pipeline: {e}")
        return None


pipeline = load_pipeline()
logger.info(f"Pipeline loaded: {pipeline is not None}")

# ------------------------------------------------------------------ #
#  PREDICTION FUNCTION                                                 #
# ------------------------------------------------------------------ #

def predict_cavities(
    image: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
) -> Tuple[np.ndarray, str, str]:
    if image is None:
        return None, "Please upload an OPG X-ray image.", "{}"

    if pipeline is None:
        error_detail = LOAD_ERROR or "Unknown error during model loading."
        msg = f"**Model failed to load.**\n\nReason: `{error_detail}`"
        return image, msg, f'{{"error": "{error_detail}"}}'

    try:
        pipeline.conf = conf_threshold
        pipeline.iou = iou_threshold

        result = pipeline.predict(image, return_visualization=True)
        vis_image = result["visualization"]

        if vis_image is not None:
            vis_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        else:
            vis_rgb = image

        summary = _build_markdown_summary(result)

        report = {k: v for k, v in result.items() if k != "visualization"}
        json_report = json.dumps(report, indent=2)

        return vis_rgb, summary, json_report

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        error_msg = f"Prediction failed: {str(e)}"
        return image, error_msg, f'{{"error": "{str(e)}"}}'


def _build_markdown_summary(result: dict) -> str:
    count = result.get("cavity_count", 0)
    severity = result.get("severity", "Unknown")
    avg_conf = result.get("confidence_avg", 0)
    detections = result.get("detections", [])
    class_counts = result.get("class_counts", {})

    URGENT_CLASSES = {"Infection", "BDC-BDR"}
    classes_found = {d["class_name"] for d in detections}

    if count == 0:
        status_icon = "✅"
    elif classes_found & URGENT_CLASSES:
        status_icon = "🔴"
    else:
        status_icon = "⚠️"

    lines = [
        f"## {status_icon} Scan Results",
        "",
        f"**Total Findings:** {count}",
        f"**Average Confidence:** {avg_conf:.1%}",
        f"**Assessment:** {severity}",
        "",
    ]

    if class_counts:
        lines.append("### Findings by Type")
        lines.append("")
        for cls_name, cls_count in sorted(class_counts.items()):
            lines.append(f"- **{cls_name}:** {cls_count}")
        lines.append("")

    if detections:
        lines.append("### All Detections")
        lines.append("")
        lines.append("| # | Condition | Confidence |")
        lines.append("|---|-----------|-----------|")
        for i, det in enumerate(detections):
            lines.append(
                f"| {i+1} | {det['class_name']} | {det['confidence']:.1%} |"
            )
        lines.append("")

    lines.extend([
        "---",
        "**Important Notice**",
        "",
        "This AI system is built to support dental professionals, not replace them. "
        "The results shown here are a screening aid and should always be reviewed alongside "
        "a full clinical examination by a qualified dentist.",
        "",
        "Model: YOLOv8s trained on annotated dental OPG X-ray images (6 classes)",
    ])

    return "\n".join(lines)


# ------------------------------------------------------------------ #
#  CUSTOM CSS                                                          #
# ------------------------------------------------------------------ #

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@700&display=swap');

* {
    font-family: 'Inter', sans-serif !important;
}

body, .gradio-container {
    background: #f0f4f8 !important;
}

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #0f2d52 0%, #1a56db 60%, #0e9f6e 100%);
    border-radius: 20px;
    padding: 56px 48px;
    margin-bottom: 8px;
    color: white;
    position: relative;
    overflow: hidden;
}

.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: rgba(255,255,255,0.05);
    border-radius: 50%;
}

.hero-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.6rem !important;
    font-weight: 700 !important;
    line-height: 1.2 !important;
    margin: 0 0 16px 0 !important;
    color: #ffffff !important;
}

.hero-subtitle {
    font-size: 1.15rem !important;
    font-weight: 300 !important;
    color: rgba(255,255,255,0.88) !important;
    max-width: 680px;
    line-height: 1.7 !important;
    margin: 0 0 28px 0 !important;
}

.hero-badges {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}

.badge {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    color: white !important;
    padding: 6px 14px;
    border-radius: 100px;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    backdrop-filter: blur(4px);
}

/* Info cards */
.info-card {
    background: white;
    border-radius: 16px;
    padding: 32px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 24px rgba(0,0,0,0.04);
    margin-bottom: 8px;
}

.info-card h2 {
    font-size: 1.35rem !important;
    font-weight: 600 !important;
    color: #0f2d52 !important;
    margin: 0 0 12px 0 !important;
}

.info-card p, .info-card li {
    font-size: 0.95rem !important;
    line-height: 1.75 !important;
    color: #374151 !important;
}

.info-card ul {
    padding-left: 20px;
    margin: 12px 0;
}

.info-card li {
    margin-bottom: 6px;
}

.accent-blue { color: #1a56db !important; font-weight: 600 !important; }
.accent-green { color: #0e9f6e !important; font-weight: 600 !important; }

/* Steps */
.steps-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 16px;
    margin-top: 16px;
}

.step-item {
    text-align: center;
    padding: 20px 12px;
    background: #f8fafc;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
}

.step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 36px; height: 36px;
    background: #1a56db;
    color: white;
    border-radius: 50%;
    font-size: 0.85rem;
    font-weight: 700;
    margin-bottom: 10px;
}

.step-label {
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: #1e293b !important;
}

/* Uganda context tag */
.ug-flag {
    display: inline-block;
    background: #d4edda;
    color: #155724;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 12px;
}

/* Stat highlights */
.stats-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin: 20px 0;
}

.stat-box {
    background: #f0f7ff;
    border-left: 4px solid #1a56db;
    border-radius: 8px;
    padding: 16px;
}

.stat-number {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: #0f2d52 !important;
    display: block;
}

.stat-label {
    font-size: 0.78rem !important;
    color: #64748b !important;
    margin-top: 2px;
}

/* Disclaimer box */
.disclaimer-box {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 4px solid #f59e0b;
    border-radius: 10px;
    padding: 16px 20px;
    margin-top: 8px;
}

.disclaimer-box p {
    font-size: 0.88rem !important;
    color: #78350f !important;
    margin: 0 !important;
    line-height: 1.6 !important;
}

/* Footer */
.footer-section {
    background: #0f2d52;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    margin-top: 8px;
}

.footer-section p {
    color: rgba(255,255,255,0.7) !important;
    font-size: 0.88rem !important;
    margin: 4px 0 !important;
}

.footer-section a {
    color: #60a5fa !important;
    text-decoration: none;
    font-weight: 500;
}

.footer-section a:hover {
    color: #93c5fd !important;
    text-decoration: underline;
}

/* Gradio component polish */
.gr-button-primary {
    background: linear-gradient(135deg, #1a56db, #0e9f6e) !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    transition: opacity 0.2s !important;
}

.gr-button-primary:hover {
    opacity: 0.9 !important;
}

.gr-button-secondary {
    border-radius: 10px !important;
    font-weight: 500 !important;
}

.gr-panel, .gr-box {
    border-radius: 14px !important;
    border: 1px solid #e2e8f0 !important;
}

label.svelte-1hnfib2 span {
    font-weight: 600 !important;
    color: #1e293b !important;
}

footer { display: none !important; }

/* Section divider */
.section-tag {
    display: inline-block;
    background: #eff6ff;
    color: #1a56db;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 4px 12px;
    border-radius: 100px;
    margin-bottom: 10px;
}
"""

# ------------------------------------------------------------------ #
#  HTML CONTENT BLOCKS                                                 #
# ------------------------------------------------------------------ #

HERO_HTML = """
<div class="hero-banner">
  <p style="font-size:0.8rem; font-weight:600; text-transform:uppercase; letter-spacing:2px; color:rgba(255,255,255,0.6); margin:0 0 10px 0;">
    AI for Oral Health
  </p>
  <h1 class="hero-title">Detecting Cavities Before They Cause Harm</h1>
  <p class="hero-subtitle">
    An intelligent screening system that reads dental X-rays and flags cavities automatically.
    Built for clinics, dental schools, and health workers who need fast, reliable support
    wherever they are in Uganda and beyond.
  </p>
  <div class="hero-badges">
    <span class="badge">YOLOv8 Vision AI</span>
    <span class="badge">OPG X-ray Analysis</span>
    <span class="badge">Real-time Detection</span>
    <span class="badge">MLflow Tracked</span>
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
      <p class="step-label">Adjust the confidence threshold if needed (default is fine)</p>
    </div>
    <div class="step-item">
      <div class="step-num">3</div>
      <p class="step-label">Click Scan for Cavities and wait a moment</p>
    </div>
    <div class="step-item">
      <div class="step-num">4</div>
      <p class="step-label">Review the annotated image and the detailed report</p>
    </div>
  </div>
</div>
"""

ABOUT_MODEL_HTML = """
<div class="info-card">
  <span class="section-tag">The Technology</span>
  <h2>Meet YOLOv8: The Brain Behind the Scans</h2>
  <p>
    At the heart of this system is <span class="accent-blue">YOLOv8</span>, which stands for
    "You Only Look Once, version 8." Despite the technical name, the idea is simple and powerful.
    When you upload an X-ray, the model scans the entire image in a single pass and draws
    precise boxes around every area that looks like a cavity. It does not need to examine one
    region at a time. It sees everything at once.
  </p>
  <p style="margin-top:14px;">
    Think of it the way an experienced radiologist might glance at a scan and immediately
    notice something off in the upper right molar. YOLOv8 has been trained to develop that
    same kind of pattern recognition, only it does it in milliseconds.
  </p>

  <h2 style="margin-top:28px;">Why We Chose YOLOv8</h2>
  <ul>
    <li>
      <span class="accent-blue">Speed without sacrifice.</span> YOLOv8 is one of the fastest
      object detection models available, yet it maintains high accuracy. This matters enormously
      in a busy clinic where patients cannot wait long.
    </li>
    <li>
      <span class="accent-blue">Proven in medical imaging.</span> The YOLO family of models
      has been successfully applied across radiology, pathology, and dental research. It handles
      the low-contrast, grayscale nature of X-rays well.
    </li>
    <li>
      <span class="accent-blue">Runs on modest hardware.</span> Unlike large hospital-grade
      AI systems that require expensive servers, YOLOv8 can run on a standard laptop. This
      makes it deployable in resource-limited settings such as district hospitals and rural clinics.
    </li>
    <li>
      <span class="accent-blue">Built-in localization.</span> The model does not just say
      "there is a cavity." It draws a box around exactly where it is, giving the dentist a
      visual reference to guide the examination.
    </li>
    <li>
      <span class="accent-blue">Open and reproducible.</span> This entire pipeline is open
      source, tracked with DVC and MLflow, so any researcher or clinic can retrain, verify,
      or improve it.
    </li>
  </ul>
</div>
"""

UGANDA_CONTEXT_HTML = """
<div class="info-card">
  <span class="section-tag" style="background:#d1fae5; color:#065f46;">Uganda Context</span>
  <h2>Why This Matters in Uganda</h2>

  <p>
    Uganda has fewer than <span class="accent-green">200 registered dentists</span> serving
    a population of over 47 million people. In rural areas, a person might travel three hours
    to reach the nearest dental clinic, only to be told they need a more detailed scan that
    requires a trip to Kampala. By the time treatment begins, what started as a small cavity
    can become a full extraction or an infection that spreads.
  </p>

  <div class="stats-row">
    <div class="stat-box">
      <span class="stat-number">1 : 235,000</span>
      <span class="stat-label">Dentist to population ratio in Uganda</span>
    </div>
    <div class="stat-box">
      <span class="stat-number">&gt; 80%</span>
      <span class="stat-label">Of Ugandans live outside major cities</span>
    </div>
    <div class="stat-box">
      <span class="stat-number">Early</span>
      <span class="stat-label">Detection can save a tooth that late detection cannot</span>
    </div>
  </div>

  <p>
    This is where AI screening changes the story. A trained dental officer at a health centre
    level III or IV can upload an OPG scan and get an immediate second opinion. The model
    highlights suspicious areas, allowing the clinician to prioritize which patients need
    urgent referral and which can be managed locally.
  </p>

  <h2 style="margin-top:28px;">Who Benefits</h2>
  <ul>
    <li>
      <span class="accent-green">Dental officers and nurses</span> in district hospitals who
      read X-rays without a specialist present. The AI acts as a quiet second pair of eyes.
    </li>
    <li>
      <span class="accent-green">Dental schools</span> like those at Makerere University and
      Kampala International University, where students can use this tool to build their diagnostic
      intuition alongside real clinical training.
    </li>
    <li>
      <span class="accent-green">Mobile health outreach teams</span> that conduct community
      screenings in schools and markets. A laptop and a portable X-ray unit is now enough to
      offer cavity screening at scale.
    </li>
    <li>
      <span class="accent-green">Private dental clinics</span> looking to reduce the time it
      takes to review X-rays and see more patients in a day without compromising thoroughness.
    </li>
    <li>
      <span class="accent-green">Researchers and public health teams</span> mapping the
      prevalence of dental caries across different regions or age groups in Uganda.
    </li>
  </ul>

  <p style="margin-top:20px; padding:16px; background:#f0fdf4; border-radius:10px; border-left:4px solid #10b981;">
    The vision is not to replace dentists. Uganda needs more dentists, full stop. The vision
    is to make every dentist already working in the country dramatically more effective, and
    to give every health worker who sees patients the ability to catch dental disease earlier.
  </p>
</div>
"""

DISCLAIMER_HTML = """
<div class="disclaimer-box">
  <p>
    <strong>Medical Disclaimer:</strong> This tool is designed to assist qualified dental
    professionals and trained health workers. It is not a substitute for clinical examination,
    professional diagnosis, or treatment planning. All findings should be reviewed and
    confirmed by a licensed dentist before any clinical decision is made.
  </p>
</div>
"""

FOOTER_HTML = """
<div class="footer-section">
  <p style="font-size:1rem !important; font-weight:600 !important; color:white !important; margin-bottom:8px !important;">
    Dental OPG Cavity Detection System
  </p>
  <p>Built by <strong style="color:white;">Paul Sentongo</strong> &mdash;
    <a href="https://github.com/sentongo-web/Dental-OPG-XRAY-Analysis-MLOPS" target="_blank">
      View on GitHub
    </a>
  </p>
  <p style="margin-top:10px !important;">
    Model: YOLOv8 &nbsp;|&nbsp; Framework: Gradio &nbsp;|&nbsp;
    Tracking: MLflow + DVC &nbsp;|&nbsp; Task: Dental Caries Detection
  </p>
  <p style="margin-top:6px !important; font-size:0.78rem !important;">
    For research, educational, and clinical support purposes.
    Not a certified medical device.
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
        title="Dental OPG Cavity Detection | AI Screening for Uganda",
        theme=gr.themes.Base(
            primary_hue="blue",
            secondary_hue="emerald",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
            font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace"],
        ),
        css=CUSTOM_CSS,
    ) as demo:

        # Hero
        gr.HTML(HERO_HTML)

        # How to use
        gr.HTML(HOW_TO_USE_HTML)

        # Main detection area
        with gr.Group():
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    gr.Markdown("### Upload X-ray")
                    input_image = gr.Image(
                        type="numpy",
                        label="OPG X-ray Image",
                        sources=["upload", "clipboard"],
                        height=380,
                    )

                    with gr.Accordion("Detection Settings", open=False):
                        conf_slider = gr.Slider(
                            minimum=0.10,
                            maximum=0.90,
                            value=0.25,
                            step=0.05,
                            label="Confidence Threshold",
                            info="How certain the model must be before flagging a cavity. Lower values catch more but may include false positives.",
                        )
                        iou_slider = gr.Slider(
                            minimum=0.20,
                            maximum=0.80,
                            value=0.45,
                            step=0.05,
                            label="Overlap Threshold (IoU)",
                            info="Controls how much detected boxes can overlap before one is removed. Keep the default unless you know what to change.",
                        )

                    with gr.Row():
                        detect_btn = gr.Button(
                            "Scan for Cavities",
                            variant="primary",
                            size="lg",
                        )
                        clear_btn = gr.ClearButton(
                            [input_image],
                            value="Clear",
                            variant="secondary",
                        )

                with gr.Column(scale=1):
                    gr.Markdown("### Annotated Result")
                    output_image = gr.Image(
                        type="numpy",
                        label="Annotated X-ray",
                        height=380,
                        interactive=False,
                    )

            with gr.Row():
                with gr.Column(scale=2):
                    summary_output = gr.Markdown(
                        label="Clinical Summary",
                        value="Upload an OPG image above and click Scan for Cavities to see results here.",
                    )

                with gr.Column(scale=1):
                    json_output = gr.Code(
                        language="json",
                        label="Raw Detection Data (JSON)",
                        lines=14,
                    )

        if EXAMPLES:
            with gr.Group():
                gr.Markdown("### Example Scans")
                gr.Examples(
                    examples=EXAMPLES,
                    inputs=[input_image, conf_slider, iou_slider],
                    outputs=[output_image, summary_output, json_output],
                    fn=predict_cavities,
                    cache_examples=True,
                )

        # Disclaimer
        gr.HTML(DISCLAIMER_HTML)

        # About the model
        gr.HTML(ABOUT_MODEL_HTML)

        # Uganda context
        gr.HTML(UGANDA_CONTEXT_HTML)

        # Footer
        gr.HTML(FOOTER_HTML)

        # Event handlers
        detect_btn.click(
            fn=predict_cavities,
            inputs=[input_image, conf_slider, iou_slider],
            outputs=[output_image, summary_output, json_output],
            show_progress=True,
        )

        input_image.upload(
            fn=predict_cavities,
            inputs=[input_image, conf_slider, iou_slider],
            outputs=[output_image, summary_output, json_output],
        )

    return demo


# ------------------------------------------------------------------ #
#  ENTRY POINT                                                         #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--host", default="0.0.0.0", help="Host")
    args = parser.parse_args()

    demo = create_interface()
    demo.launch(
        share=args.share,
        server_port=args.port,
        server_name=args.host,
        show_error=True,
    )
