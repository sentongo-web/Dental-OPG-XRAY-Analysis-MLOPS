"""
HuggingFace Spaces entry point for Dental OPG Cavity Detection.
This file is the main entry point for HuggingFace Spaces.
"""
import os
import sys
import logging
from pathlib import Path

# Ensure src is on Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up logging
logging.basicConfig(level=logging.INFO)

# Download model from HuggingFace Hub if not present
def download_model_if_needed():
    """Download trained model from HuggingFace Hub."""
    model_path = Path("models/best/best.pt")
    if model_path.exists():
        return

    try:
        from huggingface_hub import hf_hub_download
        os.makedirs("models/best", exist_ok=True)
        # Update with your HuggingFace repo ID after uploading
        hf_hub_download(
            repo_id="Sentoz/dental-opg-cavity-detection",
            filename="best.pt",
            local_dir="models/best",
        )
        print("Model downloaded from HuggingFace Hub.")
    except Exception as e:
        print(f"Could not download model: {e}. Using pretrained YOLOv8n.")


download_model_if_needed()

# Import and launch the Gradio app
from app.gradio_app import create_interface

demo = create_interface()

if __name__ == "__main__":
    demo.launch()
