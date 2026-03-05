# Dental OPG X-ray Analysis System

**AI-powered detection of dental conditions in panoramic OPG X-ray images.**
Built for dentists, dental officers, and health workers — with Uganda's healthcare context in mind.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8s-orange?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![Gradio](https://img.shields.io/badge/UI-Gradio-pink?style=flat-square)](https://gradio.app)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue?style=flat-square)](https://mlflow.org)
[![DVC](https://img.shields.io/badge/Data-DVC-green?style=flat-square)](https://dvc.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

---

## What This System Does

Upload a panoramic OPG dental X-ray and the system will:

- **Validate** that the image is actually a dental X-ray before processing it
- **Detect and classify** dental conditions into 6 categories with colour-coded bounding boxes
- **Generate a clinical report** in plain language with per-condition descriptions and recommended actions
- **Download a PDF** containing the annotated scan, all findings, and clinical recommendations — ready to print or share

---

## Detected Conditions

| Class | Condition | Severity |
| ----- | --------- | -------- |
| 0 | **BDC-BDR** — Badly Decayed Crown / Root | Urgent |
| 1 | **Caries** — Dental Cavities | Moderate |
| 2 | **Fractured Teeth** | Moderate |
| 3 | **Healthy Teeth** | Normal |
| 4 | **Impacted Teeth** | Monitor |
| 5 | **Infection** — Periapical Abscess | Urgent |

---

## Why Uganda

Uganda has fewer than **200 registered dentists** serving over 47 million people — a ratio of roughly 1 to 235,000. In rural areas, patients may travel hours to reach a clinic, and detailed X-ray analysis often requires a trip to Kampala. By the time treatment begins, a small cavity can become an extraction or a spreading infection.

This system is designed to help:

- **Dental officers and nurses** at district hospitals read X-rays without a specialist present
- **Dental schools** (Makerere University, KIU) to build diagnostic intuition alongside clinical training
- **Mobile outreach teams** conducting community screenings in schools and markets
- **Private clinics** reviewing X-rays faster and seeing more patients each day
- **Researchers** mapping dental disease prevalence across regions and age groups

---

## The Model

The system uses **YOLOv8s** (You Only Look Once, version 8, small variant). When an X-ray is uploaded, the model scans the entire image in a single pass and draws precise boxes around every detected condition. Results appear in seconds, even on a standard laptop.

### Why YOLOv8s

- Fast enough for real-time clinical use
- Handles the low-contrast, grayscale nature of X-rays well
- Produces localised bounding boxes, not just image-level labels
- Runs on modest hardware without an expensive GPU server
- Open source and fully reproducible

### Training Configuration

- Image size: 640×640 px
- Optimizer: AdamW, lr=0.001
- Epochs: 100 with early stopping (patience 20)
- Augmentations: CLAHE, brightness ±20%, slight rotation ±5°
- No horizontal/vertical flips (OPG anatomical orientation matters)
- No colour augmentation (X-rays are grayscale)

---

## Project Structure

```text
Dental-OPG-XRAY-Analysis-MLOPS/
├── app/
│   └── gradio_app.py             # Gradio web application
├── app.py                        # HuggingFace Spaces entry point
├── src/dental_opg/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── model_evaluation.py
│   └── pipeline/
│       └── prediction_pipeline.py
├── config/config.yaml            # Artifact paths
├── params.yaml                   # All YOLOv8 hyperparameters
├── dvc.yaml                      # DVC pipeline (5 stages)
├── main.py                       # Pipeline orchestrator CLI
├── requirements.txt
└── tests/
```

---

## Pipeline Stages

```text
Stage 1  Data Ingestion        Download OPG dataset from Google Drive
Stage 2  Data Validation       Verify image and annotation integrity
Stage 3  Data Transformation   Convert to YOLO format, split 75/15/10, augment
Stage 4  Model Training        YOLOv8s training with MLflow experiment tracking
Stage 5  Model Evaluation      mAP50, Precision, Recall, F1 with visual report
```

---

## Running Locally (Anaconda Prompt)

### 1. Clone and set up

```bat
git clone https://github.com/sentongo-web/Dental-OPG-XRAY-Analysis-MLOPS.git
cd Dental-OPG-XRAY-Analysis-MLOPS
pip install -e .
pip install -r requirements.txt
```

### 2. Run the full training pipeline

```bat
set PYTHONPATH=src && python main.py
```

### 3. Launch the web app

```bat
set PYTHONPATH=src && python app/gradio_app.py
```

Open [http://localhost:7860](http://localhost:7860) in your browser.

---

## Deploy to HuggingFace Spaces

Get a Write token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), then run:

```bat
python scripts/deploy_to_huggingface.py --token YOUR_HF_TOKEN --author sentongo-web
```

---

## Running Tests

```bat
set PYTHONPATH=src && pytest tests/unit/ -v
```

---

## Dataset

- **Source:** Dental OPG X-ray dataset via Google Drive
- **Detection subset:** ~260 panoramic OPG images with YOLO bounding box annotations
- **Classes:** 6 dental conditions (indices 0–5)
- **Split:** 75% train / 15% val / 10% test
- **Augmentation:** 2× per training image (CLAHE, brightness variation, slight rotation)

---

## OPG Image Validation

Before running detection, the system checks that the uploaded image is a valid dental X-ray:

- **Aspect ratio:** OPG images are panoramic — significantly wider than tall
- **Grayscale check:** X-rays have near-identical RGB channels; colourful images are rejected
- **Brightness range:** Too-dark or too-bright images are flagged
- **Minimum size:** Very small or thumbnail images are rejected

Non-dental images receive a clear rejection message instead of a meaningless detection result.

---

## Medical Disclaimer

This system is designed to **assist** qualified dental professionals. It is not a substitute for clinical examination, professional diagnosis, or treatment planning. All findings must be confirmed by a licensed dentist before any clinical decision is made.

---

## Author

**Paul Sentongo**
[GitHub](https://github.com/sentongo-web) · [Dental-OPG-XRAY-Analysis-MLOPS](https://github.com/sentongo-web/Dental-OPG-XRAY-Analysis-MLOPS)
