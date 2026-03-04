from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    required_files: List[str]
    required_annotation_formats: List[str]


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    output_dir: Path
    train_ratio: float
    val_ratio: float
    test_ratio: float
    image_size: int
    augmentation_factor: int


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    data_yaml: Path
    model_name: str
    results_dir: Path


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    data_yaml: Path
    model_path: Path
    eval_results_dir: Path
