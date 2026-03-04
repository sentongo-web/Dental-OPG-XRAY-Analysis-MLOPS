import os
import yaml
import json
import shutil
import logging
from pathlib import Path
from typing import Any, Union
from box import ConfigBox
from ensure import ensure_annotations

logger = logging.getLogger(__name__)


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Read YAML file and return as ConfigBox for dot-notation access."""
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML loaded: {path_to_yaml}")
            return ConfigBox(content)
    except Exception as e:
        raise ValueError(f"Error reading YAML at {path_to_yaml}: {e}")


@ensure_annotations
def create_directories(path_to_directories: list, verbose: bool = True):
    """Create a list of directories."""
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Directory created: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """Save a dictionary to JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON saved: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Load JSON file as ConfigBox."""
    with open(path) as f:
        content = json.load(f)
    logger.info(f"JSON loaded: {path}")
    return ConfigBox(content)


def get_size(path: Path) -> str:
    """Get file size in KB."""
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~{size_in_kb} KB"


def copy_file(src: Path, dest: Path):
    """Copy a file from src to dest."""
    os.makedirs(dest.parent, exist_ok=True)
    shutil.copy2(src, dest)


def get_image_files(directory: Path, extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")) -> list:
    """Recursively get all image files from a directory."""
    image_files = []
    for ext in extensions:
        image_files.extend(directory.rglob(f"*{ext}"))
        image_files.extend(directory.rglob(f"*{ext.upper()}"))
    return sorted(image_files)


def count_files(directory: Path, extension: str = None) -> int:
    """Count files in a directory, optionally filtered by extension."""
    if extension:
        return len(list(directory.rglob(f"*{extension}")))
    return len([f for f in directory.rglob("*") if f.is_file()])
