"""
Dental OPG Cavity Detection — Gradio Web Application
Deployable to HuggingFace Spaces

Features:
- Single image cavity detection with annotated output
- Confidence score and cavity count
- Severity assessment
- Dental report generation
- Example OPG images
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

# Ensure src is on path when running directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  MODEL LOADING                                                       #
# ------------------------------------------------------------------ #

MODEL_PATH = Path("models/best/best.pt")
FALLBACK_MODEL = "yolov8n.pt"  # Use nano YOLOv8 as fallback for demo


def load_pipeline():
    """Load prediction pipeline with fallback."""
    try:
        from dental_opg.pipeline.prediction_pipeline import PredictionPipeline
        if MODEL_PATH.exists():
            return PredictionPipeline(model_path=MODEL_PATH)
        else:
            logger.warning(f"Trained model not found at {MODEL_PATH}, using pretrained YOLOv8n")
            return PredictionPipeline(model_path=FALLBACK_MODEL)
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        return None


pipeline = load_pipeline()

# ------------------------------------------------------------------ #
#  PREDICTION FUNCTION                                                 #
# ------------------------------------------------------------------ #

def predict_cavities(
    image: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
) -> Tuple[np.ndarray, str, str]:
    """
    Main prediction function for Gradio interface.

    Returns:
        - Annotated image
        - Detection summary (markdown)
        - JSON report
    """
    if image is None:
        return None, "Please upload an OPG X-ray image.", "{}"

    if pipeline is None:
        return image, "Model not loaded. Please check deployment.", "{}"

    try:
        # Update thresholds
        pipeline.conf = conf_threshold
        pipeline.iou = iou_threshold

        # Run prediction
        result = pipeline.predict(image, return_visualization=True)
        vis_image = result["visualization"]

        # Convert BGR to RGB for Gradio
        if vis_image is not None:
            vis_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        else:
            vis_rgb = image

        # Build summary
        summary = _build_markdown_summary(result)

        # Build JSON report
        report = {k: v for k, v in result.items() if k != "visualization"}
        json_report = json.dumps(report, indent=2)

        return vis_rgb, summary, json_report

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        error_msg = f"Prediction failed: {str(e)}"
        return image, error_msg, f'{{"error": "{str(e)}"}}'


def _build_markdown_summary(result: dict) -> str:
    """Build a nicely formatted markdown summary."""
    count = result.get("cavity_count", 0)
    severity = result.get("severity", "Unknown")
    avg_conf = result.get("confidence_avg", 0)
    detections = result.get("detections", [])

    # Severity indicator
    if count == 0:
        status_icon = "✅"
        status_color = "green"
    elif count <= 2:
        status_icon = "⚠️"
    else:
        status_icon = "🚨"

    lines = [
        f"## {status_icon} Detection Results",
        "",
        f"**Cavities Detected:** `{count}`",
        f"**Average Confidence:** `{avg_conf:.1%}`",
        f"**Assessment:** {severity}",
        "",
    ]

    if detections:
        lines.append("### Detected Cavities")
        lines.append("")
        lines.append("| # | Class | Confidence | Location (x1,y1,x2,y2) |")
        lines.append("|---|-------|-----------|------------------------|")
        for i, det in enumerate(detections):
            bbox = det["bbox"]
            lines.append(
                f"| {i+1} | {det['class_name']} | {det['confidence']:.1%} | "
                f"({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}) |"
            )
        lines.append("")

    lines.extend([
        "---",
        "### Clinical Notes",
        "",
        "> **⚕️ Important:** This AI tool is designed to assist dental professionals.",
        "> It should NOT replace clinical examination and professional diagnosis.",
        "> Always consult a qualified dentist for proper evaluation and treatment.",
        "",
        "**Model:** YOLOv8 trained on dental OPG X-ray dataset",
        "**Task:** Dental cavity (caries) detection",
    ])

    return "\n".join(lines)


# ------------------------------------------------------------------ #
#  GRADIO UI                                                           #
# ------------------------------------------------------------------ #

DESCRIPTION = """
# 🦷 Dental OPG Cavity Detection

**AI-powered cavity detection in Orthopantomogram (OPG) X-ray images using YOLOv8.**

Upload an OPG X-ray image and the model will automatically detect and localize dental cavities.

### How to use:
1. Upload an OPG X-ray image (JPG, PNG, TIFF)
2. Adjust confidence/IoU thresholds if needed
3. Click **Detect Cavities**
4. View annotated results and clinical report

> ⚠️ **Disclaimer:** This tool is for research and educational purposes. Always consult a qualified dental professional.
"""

EXAMPLES = [
    # Will be populated if example images exist
]

# Check for example images
example_dir = Path("data/examples")
if example_dir.exists():
    EXAMPLES = [[str(f), 0.25, 0.45] for f in example_dir.glob("*.jpg")]


def create_interface():
    """Create and return the Gradio interface."""

    with gr.Blocks(
        title="Dental OPG Cavity Detection",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
        css="""
        .main-header { text-align: center; margin-bottom: 20px; }
        .result-box { border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; }
        footer { display: none !important; }
        """,
    ) as demo:
        gr.Markdown(DESCRIPTION, elem_classes=["main-header"])

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Input")
                input_image = gr.Image(
                    type="numpy",
                    label="OPG X-ray Image",
                    sources=["upload", "clipboard"],
                    height=400,
                )

                with gr.Accordion("⚙️ Detection Settings", open=False):
                    conf_slider = gr.Slider(
                        minimum=0.10,
                        maximum=0.90,
                        value=0.25,
                        step=0.05,
                        label="Confidence Threshold",
                        info="Minimum confidence score to report a detection",
                    )
                    iou_slider = gr.Slider(
                        minimum=0.20,
                        maximum=0.80,
                        value=0.45,
                        step=0.05,
                        label="IoU Threshold (NMS)",
                        info="IoU threshold for non-maximum suppression",
                    )

                detect_btn = gr.Button(
                    "🔍 Detect Cavities",
                    variant="primary",
                    size="lg",
                )
                clear_btn = gr.ClearButton(
                    [input_image],
                    value="🗑️ Clear",
                    variant="secondary",
                )

            with gr.Column(scale=1):
                gr.Markdown("### 📊 Detection Results")
                output_image = gr.Image(
                    type="numpy",
                    label="Annotated X-ray",
                    height=400,
                    interactive=False,
                )

        with gr.Row():
            with gr.Column(scale=2):
                summary_output = gr.Markdown(
                    label="Clinical Summary",
                    value="*Upload an OPG image to see detection results.*",
                    elem_classes=["result-box"],
                )

            with gr.Column(scale=1):
                json_output = gr.Code(
                    language="json",
                    label="JSON Report",
                    lines=15,
                )

        if EXAMPLES:
            gr.Markdown("### 📁 Example Images")
            gr.Examples(
                examples=EXAMPLES,
                inputs=[input_image, conf_slider, iou_slider],
                outputs=[output_image, summary_output, json_output],
                fn=predict_cavities,
                cache_examples=True,
            )

        # Model info
        with gr.Accordion("ℹ️ Model Information", open=False):
            gr.Markdown("""
            **Architecture:** YOLOv8 (You Only Look Once v8)
            **Task:** Object Detection — Dental Cavity Localization
            **Input Size:** 640×640 pixels
            **Dataset:** Dental OPG X-ray images with cavity annotations
            **Training:** 100 epochs, AdamW optimizer, medical-safe augmentation

            **Pipeline:**
            - Data Ingestion → Validation → Transformation (YOLO format)
            - YOLOv8 Training with MLflow tracking
            - Model Evaluation (mAP, Precision, Recall, F1)
            - DVC for data/model versioning

            **Author:** Paul Sentongo
            **GitHub:** [Dental-OPG-XRAY-Analysis-MLOPS](https://github.com/paulsentongo/Dental-OPG-XRAY-Analysis-MLOPS)
            """)

        # Event handlers
        detect_btn.click(
            fn=predict_cavities,
            inputs=[input_image, conf_slider, iou_slider],
            outputs=[output_image, summary_output, json_output],
            show_progress=True,
        )

        # Also trigger on image upload
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
