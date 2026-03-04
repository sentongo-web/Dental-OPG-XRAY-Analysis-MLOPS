"""
Deploy Dental OPG Cavity Detection model and app to HuggingFace.

Usage:
    python scripts/deploy_to_huggingface.py --token YOUR_HF_TOKEN
    python scripts/deploy_to_huggingface.py --token YOUR_HF_TOKEN --space-name my-dental-opg-app
"""
import os
import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_hf_readme(space_name: str, author: str) -> str:
    """Generate HuggingFace Space README (YAML front matter + description)."""
    return f"""---
title: Dental OPG Cavity Detection
emoji: 🦷
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.19.0"
app_file: app.py
pinned: true
license: mit
tags:
  - medical
  - dental
  - object-detection
  - yolov8
  - xray
  - cavity-detection
  - computer-vision
  - healthcare
short_description: AI-powered dental cavity detection in OPG X-ray images
---

# 🦷 Dental OPG Cavity Detection

**AI-powered dental cavity (caries) detection in Orthopantomogram (OPG) X-ray images.**

Built with [YOLOv8](https://github.com/ultralytics/ultralytics) and deployed with [MLOps best practices](https://github.com/{author}/Dental-OPG-XRAY-Analysis-MLOPS).

## 📋 How to Use

1. Upload an OPG X-ray image (JPG, PNG, TIFF)
2. Adjust the confidence threshold if desired (default: 0.25)
3. Click **Detect Cavities**
4. View annotated results with cavity locations and confidence scores

## 🏗️ Architecture

- **Model:** YOLOv8 (nano) fine-tuned on dental OPG dataset
- **Input:** 640×640 px OPG X-ray images
- **Output:** Bounding boxes around detected cavities + confidence scores
- **MLOps:** DVC (data versioning) + MLflow (experiment tracking)

## ⚠️ Disclaimer

This tool is designed to **assist** dental professionals and is **not** a substitute for professional clinical examination. Always consult a qualified dentist for proper diagnosis and treatment.

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| mAP@0.5 | Coming after training |
| Precision | Coming after training |
| Recall | Coming after training |
| F1 Score | Coming after training |

## 🔬 Dataset

Trained on dental OPG X-ray images with annotated cavities.

## 👤 Author

**{author}** — [GitHub](https://github.com/{author}) | [LinkedIn](https://linkedin.com/in/{author.lower()})
"""


def deploy_space(token: str, space_name: str, author: str = "paulsentongo"):
    """Deploy Gradio app to HuggingFace Spaces."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    repo_id = f"{author}/{space_name}"

    logger.info(f"Deploying to HuggingFace Space: {repo_id}")

    # Create Space
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            exist_ok=True,
            private=False,
        )
        logger.info(f"Space created/found: {repo_id}")
    except Exception as e:
        logger.error(f"Failed to create Space: {e}")
        raise

    # Upload README
    readme_content = create_hf_readme(space_name, author)
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="space",
    )
    logger.info("README.md uploaded")

    # Files to upload
    upload_files = [
        ("app.py", "app.py"),
        ("requirements.txt", "requirements.txt"),
        ("setup.py", "setup.py"),
        ("params.yaml", "params.yaml"),
        ("config/config.yaml", "config/config.yaml"),
    ]

    for local_path, remote_path in upload_files:
        if Path(local_path).exists():
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=remote_path,
                repo_id=repo_id,
                repo_type="space",
            )
            logger.info(f"Uploaded: {local_path} → {remote_path}")

    # Upload app directory
    api.upload_folder(
        folder_path="app",
        path_in_repo="app",
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=["__pycache__", "*.pyc"],
    )
    logger.info("Uploaded: app/")

    # Upload src directory
    api.upload_folder(
        folder_path="src",
        path_in_repo="src",
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=["__pycache__", "*.pyc", "*.egg-info"],
    )
    logger.info("Uploaded: src/")

    # Upload model if exists
    model_path = Path("models/best/best.pt")
    if model_path.exists():
        logger.info(f"Uploading model ({model_path.stat().st_size / 1e6:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo="models/best/best.pt",
            repo_id=repo_id,
            repo_type="space",
        )
        logger.info("Model uploaded!")
    else:
        logger.warning(f"Model not found at {model_path}. App will use pretrained YOLOv8.")

    space_url = f"https://huggingface.co/spaces/{repo_id}"
    logger.info("=" * 60)
    logger.info(f"✅ Deployment complete!")
    logger.info(f"🌐 Space URL: {space_url}")
    logger.info("=" * 60)
    return space_url


def upload_model_to_hub(token: str, model_name: str, author: str = "paulsentongo"):
    """Upload trained model to HuggingFace Model Hub."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    repo_id = f"{author}/{model_name}"

    logger.info(f"Uploading model to HuggingFace Hub: {repo_id}")

    # Create model repo
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        exist_ok=True,
        private=False,
    )

    # Generate model card
    model_card = f"""---
language: en
license: mit
tags:
  - dental
  - medical
  - object-detection
  - yolov8
  - cavity-detection
  - xray
datasets:
  - custom-dental-opg
metrics:
  - mAP
model-index:
  - name: YOLOv8 Dental Cavity Detector
    results:
      - task:
          type: object-detection
        metrics:
          - type: mAP
            value: 0.0  # Update after training
            name: mAP@0.5
---

# YOLOv8 Dental Cavity Detection Model

Fine-tuned YOLOv8 for detecting dental cavities in OPG X-ray images.

## Usage

```python
from ultralytics import YOLO

model = YOLO("paulsentongo/dental-opg-cavity-detection-model")
results = model("opg_xray.jpg", conf=0.25)
results[0].show()
```

## Training Details

- Base model: YOLOv8n
- Dataset: Dental OPG X-ray images
- Task: Object detection (cavity localization)
- Input size: 640×640
"""

    api.upload_file(
        path_or_fileobj=model_card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )

    # Upload model weights
    model_path = Path("models/best/best.pt")
    if model_path.exists():
        api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo="best.pt",
            repo_id=repo_id,
            repo_type="model",
        )
        logger.info(f"Model uploaded: https://huggingface.co/{repo_id}")
    else:
        logger.warning("Model file not found. Train first with: python main.py")


def main():
    parser = argparse.ArgumentParser(description="Deploy to HuggingFace")
    parser.add_argument("--token", required=True, help="HuggingFace API token")
    parser.add_argument("--author", default="paulsentongo", help="HuggingFace username")
    parser.add_argument("--space-name", default="dental-opg-cavity-detection",
                       help="Space repository name")
    parser.add_argument("--model-name", default="dental-opg-cavity-detection-model",
                       help="Model repository name")
    parser.add_argument("--skip-space", action="store_true", help="Skip Space deployment")
    parser.add_argument("--skip-model", action="store_true", help="Skip model upload")
    args = parser.parse_args()

    if not args.skip_space:
        deploy_space(args.token, args.space_name, args.author)

    if not args.skip_model:
        upload_model_to_hub(args.token, args.model_name, args.author)


if __name__ == "__main__":
    main()
