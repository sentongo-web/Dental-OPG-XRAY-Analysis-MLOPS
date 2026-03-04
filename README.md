# 🦷 Dental OPG Cavity Detection — MLOps Pipeline

[![CI](https://github.com/Sentoz/Dental-OPG-XRAY-Analysis-MLOPS/actions/workflows/ci.yaml/badge.svg)](https://github.com/Sentoz/Dental-OPG-XRAY-Analysis-MLOPS/actions/workflows/ci.yaml)
[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-blue)](https://huggingface.co/spaces/Sentoz/dental-opg-cavity-detection)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-orange)](https://github.com/ultralytics/ultralytics)
[![DVC](https://img.shields.io/badge/DVC-Enabled-purple)](https://dvc.org)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)](https://mlflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Production-grade MLOps pipeline for automated dental cavity detection in Orthopantomogram (OPG) X-ray images using YOLOv8.**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Pipeline Stages](#pipeline-stages)
- [Model Performance](#model-performance)
- [Configuration](#configuration)
- [MLOps Features](#mlops-features)
- [HuggingFace Deployment](#huggingface-deployment)
- [API Reference](#api-reference)
- [Contributing](#contributing)

---

## 🔬 Overview

This project implements a complete **end-to-end MLOps pipeline** for detecting dental cavities (caries) in OPG X-ray images. It covers:

- **Automated data ingestion** from Google Drive
- **Multi-format support**: YOLO, COCO, Pascal VOC annotation formats
- **Medical-safe augmentation** (no flips, no color jitter — X-ray appropriate)
- **YOLOv8 fine-tuning** with full hyperparameter control
- **MLflow experiment tracking** for all runs
- **DVC data versioning** for reproducible pipelines
- **Gradio web app** deployed to HuggingFace Spaces
- **GitHub Actions CI/CD** for automated testing and deployment

---

## 🏗️ Architecture

```
Raw OPG X-rays
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│                    MLOps Pipeline                        │
│                                                          │
│  Stage 1      Stage 2         Stage 3                   │
│  Data    →   Validate    →   Transform                  │
│  Ingest      (format,        (YOLO splits,              │
│  (GDrive)    integrity)      augmentation)              │
│                                    │                    │
│  Stage 4                   Stage 5 │                   │
│  Train YOLOv8  ←───────────────────┘                   │
│  (MLflow tracking)                                      │
│       │                                                  │
│       ▼                                                  │
│  Stage 5: Evaluate (mAP, Precision, Recall, F1)         │
│       │                                                  │
│       ▼                                                  │
│  Best Model  →  HuggingFace Hub  →  Gradio Space        │
└─────────────────────────────────────────────────────────┘
         │              │
    DVC Cache     MLflow Runs
```

**Model:** YOLOv8 (nano → large variants) | **Framework:** Ultralytics | **Tracking:** MLflow | **Versioning:** DVC

---

## 📁 Project Structure

```
Dental-OPG-XRAY-Analysis-MLOPS/
├── .github/
│   └── workflows/
│       ├── ci.yaml              # Lint, test, Docker build
│       ├── cd.yaml              # Deploy to HuggingFace
│       └── train.yaml           # Manual/scheduled training
├── app/
│   └── gradio_app.py            # Gradio web interface
├── config/
│   └── config.yaml              # Path & artifact configuration
├── notebooks/
│   └── EDA.ipynb                # Exploratory data analysis
├── reports/
│   └── figures/                 # Generated plots
├── scripts/
│   ├── deploy_to_huggingface.py # HF deployment script
│   └── download_data.py         # Standalone data downloader
├── src/
│   └── dental_opg/
│       ├── components/
│       │   ├── data_ingestion.py      # GDrive download + extract
│       │   ├── data_validation.py     # Format detection + validation
│       │   ├── data_transformation.py # YOLO/COCO/VOC → splits
│       │   ├── model_trainer.py       # YOLOv8 + MLflow
│       │   └── model_evaluation.py    # mAP, P, R, F1 + reports
│       ├── config/configuration.py    # ConfigurationManager
│       ├── constants/__init__.py      # Path constants
│       ├── entity/config_entity.py    # Dataclass configs
│       ├── pipeline/
│       │   ├── stage_01-05_*.py       # Pipeline stage runners
│       │   └── prediction_pipeline.py # Inference
│       └── utils/common.py            # Shared utilities
├── tests/
│   ├── unit/                    # Unit tests (pytest)
│   └── integration/             # Integration tests
├── app.py                       # HuggingFace entry point
├── dvc.yaml                     # DVC pipeline definition
├── main.py                      # Pipeline orchestrator CLI
├── params.yaml                  # All hyperparameters
├── requirements.txt
├── setup.py
├── Dockerfile
└── README.md
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.11+
- Git
- CUDA (recommended) or CPU

### 1. Clone and Install

```bash
git clone https://github.com/Sentoz/Dental-OPG-XRAY-Analysis-MLOPS.git
cd Dental-OPG-XRAY-Analysis-MLOPS

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

### 2. Initialize DVC

```bash
dvc init
dvc remote add -d gdrive gdrive://1mm0L2jRPKqCpyxXRsSJKVoj63-SXncv1
```

### 3. Run the Full Pipeline

```bash
# Run all 5 stages
python main.py

# Or run stages individually
python main.py --stage 1    # Download data
python main.py --stage 2    # Validate data
python main.py --stage 3    # Transform to YOLO format
python main.py --stage 4    # Train YOLOv8
python main.py --stage 5    # Evaluate model
```

### 4. View Experiment Tracking

```bash
mlflow ui
# Open http://localhost:5000
```

### 5. Launch Web App

```bash
python app/gradio_app.py
# Open http://localhost:7860
```

---

## 🔄 Pipeline Stages

### Stage 1: Data Ingestion
- Downloads dataset from Google Drive using `gdown`
- Extracts ZIP archive
- Logs dataset structure and file counts

```bash
python -m dental_opg.pipeline.stage_01_data_ingestion
```

### Stage 2: Data Validation
- Auto-detects annotation format (YOLO/COCO/VOC)
- Validates image/label counts
- Generates validation report JSON

```bash
python -m dental_opg.pipeline.stage_02_data_validation
```

### Stage 3: Data Transformation
- Converts any format → YOLO format
- Creates 75/15/10 train/val/test split
- Applies medical-safe augmentations:
  - CLAHE contrast enhancement
  - Brightness variation (±20%)
  - Slight rotation (±5°)
  - Gaussian blur
- Writes `data.yaml` for YOLOv8

```bash
python -m dental_opg.pipeline.stage_03_data_transformation
```

### Stage 4: Model Training
- Loads YOLOv8 (nano/small/medium/large/xlarge)
- Full hyperparameter control via `params.yaml`
- MLflow logging: params, metrics, model artifacts, plots
- Saves best model to `models/best/best.pt`

```bash
python -m dental_opg.pipeline.stage_04_model_training
```

### Stage 5: Model Evaluation
- Evaluates on test split
- Computes mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1
- Generates evaluation report (PNG)
- Quality check against thresholds
- MLflow logging of all evaluation metrics

```bash
python -m dental_opg.pipeline.stage_05_model_evaluation
```

---

## 📊 Model Performance

After training (metrics updated automatically):

| Metric | Score |
|--------|-------|
| mAP@0.5 | TBD after training |
| mAP@0.5:0.95 | TBD after training |
| Precision | TBD after training |
| Recall | TBD after training |
| F1 Score | TBD after training |

### Why YOLOv8?
- **Speed:** Real-time inference (30+ FPS on GPU)
- **Accuracy:** State-of-the-art detection on medical images
- **Flexible:** nano → xlarge variants for different hardware
- **Production-ready:** ONNX, TorchScript, TensorRT export

---

## ⚙️ Configuration

### Model Hyperparameters (`params.yaml`)

```yaml
training:
  model: yolov8n.pt      # Change to yolov8s/m/l/x for better accuracy
  epochs: 100
  batch: 16
  imgsz: 640
  lr0: 0.01
  optimizer: AdamW
  # Medical imaging specific - NO flips, NO mosaic
  fliplr: 0.0
  flipud: 0.0
  mosaic: 0.0
  hsv_h: 0.0             # X-rays are grayscale
  hsv_s: 0.0
  hsv_v: 0.4             # Brightness augmentation OK
```

### Switching Model Size

```yaml
# For better accuracy (requires more GPU memory):
model: yolov8s.pt   # Small (11.1M params)
model: yolov8m.pt   # Medium (25.9M params)
model: yolov8l.pt   # Large (43.7M params)
model: yolov8x.pt   # XLarge (68.2M params)
```

---

## 🚀 MLOps Features

| Feature | Tool | Details |
|---------|------|---------|
| Data Versioning | DVC | Track raw data, processed splits, model artifacts |
| Experiment Tracking | MLflow | Parameters, metrics, artifacts, model registry |
| Pipeline Orchestration | DVC + Python | Stage dependencies, caching, reproducibility |
| CI/CD | GitHub Actions | Lint, test, Docker build, HF deployment |
| Containerization | Docker | Multi-stage build, production-ready |
| Configuration | YAML + Python | Config + Params separation, type-safe entities |
| Testing | pytest | Unit + integration tests, coverage reporting |
| Code Quality | black + isort + flake8 | Pre-commit hooks |

### DVC Commands

```bash
# Run full pipeline (DVC-managed)
dvc repro

# Run specific stage
dvc repro model_training

# Check what would run (dry run)
dvc repro --dry

# Push data/model to remote
dvc push

# Pull latest data/model
dvc pull

# Compare parameters across runs
dvc params diff

# Compare metrics across runs
dvc metrics diff
```

### MLflow Commands

```bash
# Start UI
mlflow ui

# List experiments
mlflow experiments list

# Get best run
mlflow runs list --experiment-name "Dental-OPG-Cavity-Detection" --order-by "metrics.mAP50 DESC"
```

---

## 🤗 HuggingFace Deployment

### Quick Deploy

```bash
# After training
python scripts/deploy_to_huggingface.py \
    --token YOUR_HF_TOKEN \
    --author Sentoz \
    --space-name dental-opg-cavity-detection
```

### CI/CD Deployment
Deployment triggers automatically on push to `main` when model changes are detected.

Required GitHub Secrets:
- `HF_TOKEN`: HuggingFace API token (Settings → Access Tokens)

### Space URL
```
https://huggingface.co/spaces/Sentoz/dental-opg-cavity-detection
```

---

## 📡 API Reference

### Python API

```python
from dental_opg.pipeline.prediction_pipeline import PredictionPipeline
import cv2

# Initialize
pipeline = PredictionPipeline(
    model_path="models/best/best.pt",
    conf_threshold=0.25,
    iou_threshold=0.45,
)

# Predict
image = cv2.imread("opg_xray.jpg")
result = pipeline.predict(image)

print(f"Cavities: {result['cavity_count']}")
print(f"Severity: {result['severity']}")
for det in result['detections']:
    print(f"  {det['class_name']}: {det['confidence']:.2%} at {det['bbox']}")

# Show annotated image
cv2.imshow("Result", result['visualization'])
```

### CLI

```bash
python -m dental_opg.pipeline.prediction_pipeline \
    --image path/to/opg.jpg \
    --model models/best/best.pt \
    --conf 0.25 \
    --output annotated.jpg
```

---

## 🐳 Docker

```bash
# Build
docker build -t dental-opg-cavity-detection .

# Run web app
docker run -p 7860:7860 dental-opg-cavity-detection

# Run training
docker run -v $(pwd)/data:/app/data dental-opg-cavity-detection python main.py
```

---

## 🧪 Testing

```bash
# Run unit tests
pytest tests/unit/ -v

# Run with coverage
pytest tests/unit/ --cov=src/dental_opg --cov-report=html

# Run integration tests (requires data + model)
pytest tests/integration/ -v -m integration
```

---

## ⚕️ Medical Disclaimer

This tool is designed to **assist** dental professionals and is **NOT** a substitute for:
- Professional clinical examination
- Radiologist review
- Dentist diagnosis and treatment planning

Always consult a qualified dental professional for proper evaluation and treatment.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

## 👤 Author

**Paul Sentongo**

---

*Built with ❤️ for better dental healthcare through AI*
