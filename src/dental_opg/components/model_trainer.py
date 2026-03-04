import os
import shutil
import logging
from pathlib import Path
from typing import Optional
import yaml
import mlflow
import mlflow.artifacts
from dental_opg.entity.config_entity import ModelTrainerConfig
from dental_opg.utils.common import read_yaml, save_json
from dental_opg.constants import PARAMS_FILE_PATH

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    YOLOv8-based model trainer for dental cavity detection.
    Integrates MLflow experiment tracking.
    Supports multiple model sizes (n/s/m/l/x).
    """

    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.params = read_yaml(PARAMS_FILE_PATH)

    def _get_training_args(self) -> dict:
        """Build YOLO training arguments from params.yaml."""
        p = self.params.training
        return {
            "data": str(self.config.data_yaml),
            "epochs": p.epochs,
            "imgsz": p.imgsz,
            "batch": p.batch,
            "patience": p.patience,
            "lr0": p.lr0,
            "lrf": p.lrf,
            "momentum": p.momentum,
            "weight_decay": p.weight_decay,
            "warmup_epochs": p.warmup_epochs,
            "warmup_momentum": p.warmup_momentum,
            "warmup_bias_lr": p.warmup_bias_lr,
            "box": p.box,
            "cls": p.cls,
            "dfl": p.dfl,
            "degrees": p.degrees,
            "translate": p.translate,
            "scale": p.scale,
            "fliplr": p.fliplr,
            "flipud": p.flipud,
            "mosaic": p.mosaic,
            "mixup": p.mixup,
            "hsv_h": p.hsv_h,
            "hsv_s": p.hsv_s,
            "hsv_v": p.hsv_v,
            "optimizer": p.optimizer,
            "seed": p.seed,
            "cos_lr": p.cos_lr,
            "amp": p.amp,
            "val": p.val,
            "save": p.save,
            "save_period": p.save_period,
            "plots": p.plots,
            "verbose": p.verbose,
            "project": str(self.config.results_dir),
            "name": "cavity_detection",
            "exist_ok": True,
            "device": p.device if p.device else "",
            "workers": p.workers,
            "close_mosaic": p.close_mosaic,
            "fraction": p.fraction,
            "cache": p.cache,
        }

    def train(self):
        """Train YOLOv8 model with MLflow tracking."""
        from ultralytics import YOLO

        os.makedirs(self.config.root_dir, exist_ok=True)
        os.makedirs(self.config.results_dir, exist_ok=True)

        # Validate data.yaml exists
        if not self.config.data_yaml.exists():
            raise FileNotFoundError(f"data.yaml not found: {self.config.data_yaml}")

        logger.info(f"Loading model: {self.config.model_name}")
        model = YOLO(self.config.model_name)

        train_args = self._get_training_args()
        logger.info(f"Training args: {train_args}")

        # Set up MLflow
        mlflow.set_experiment("Dental-OPG-Cavity-Detection")

        with mlflow.start_run(run_name=f"yolov8_{self.config.model_name.replace('.pt', '')}") as run:
            logger.info(f"MLflow Run ID: {run.info.run_id}")

            # Log parameters
            mlflow.log_params({
                "model": self.config.model_name,
                "epochs": train_args["epochs"],
                "batch": train_args["batch"],
                "imgsz": train_args["imgsz"],
                "optimizer": train_args["optimizer"],
                "lr0": train_args["lr0"],
                "augmentation": "CLAHE+brightness+rotation",
            })

            # Load data config for logging
            with open(self.config.data_yaml) as f:
                data_cfg = yaml.safe_load(f)
            mlflow.log_params({
                "num_classes": data_cfg.get("nc", 1),
                "class_names": str(data_cfg.get("names", ["cavity"])),
            })

            # Train
            logger.info("Starting YOLOv8 training...")
            results = model.train(**train_args)

            # Log metrics
            metrics = results.results_dict if hasattr(results, "results_dict") else {}
            mlflow.log_metrics({
                "mAP50": float(metrics.get("metrics/mAP50(B)", 0)),
                "mAP50_95": float(metrics.get("metrics/mAP50-95(B)", 0)),
                "precision": float(metrics.get("metrics/precision(B)", 0)),
                "recall": float(metrics.get("metrics/recall(B)", 0)),
                "box_loss": float(metrics.get("train/box_loss", 0)),
                "cls_loss": float(metrics.get("train/cls_loss", 0)),
                "dfl_loss": float(metrics.get("train/dfl_loss", 0)),
            })

            # Save best model path
            best_model_path = (
                self.config.results_dir / "cavity_detection" / "weights" / "best.pt"
            )

            if best_model_path.exists():
                # Log model artifact to MLflow
                mlflow.log_artifact(str(best_model_path), "model")

                # Copy best model to models/best/
                best_dir = Path("models/best")
                best_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(best_model_path, best_dir / "best.pt")
                logger.info(f"Best model saved to: {best_dir / 'best.pt'}")

                # Save model info
                model_info = {
                    "model_name": self.config.model_name,
                    "best_model_path": str(best_model_path),
                    "mlflow_run_id": run.info.run_id,
                    "metrics": {k: float(v) for k, v in metrics.items()},
                }
                save_json(Path(self.config.root_dir) / "model_info.json", model_info)

            # Log training plots
            results_plot_dir = self.config.results_dir / "cavity_detection"
            if results_plot_dir.exists():
                for plot_file in results_plot_dir.glob("*.png"):
                    mlflow.log_artifact(str(plot_file), "plots")

            logger.info(f"Training complete. Run ID: {run.info.run_id}")
            logger.info(f"mAP@0.5: {metrics.get('metrics/mAP50(B)', 'N/A')}")
            logger.info(f"mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")

        return results

    def run(self):
        """Execute model training stage."""
        logger.info("=" * 60)
        logger.info("STAGE: Model Training")
        logger.info("=" * 60)
        results = self.train()
        logger.info("Model Training complete.")
        return results
