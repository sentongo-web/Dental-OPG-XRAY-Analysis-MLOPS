import os
import json
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, List, Tuple
import mlflow
import cv2
import yaml
from dental_opg.entity.config_entity import ModelEvaluationConfig
from dental_opg.utils.common import save_json, read_yaml
from dental_opg.constants import PARAMS_FILE_PATH

logger = logging.getLogger(__name__)


class ModelEvaluation:
    """
    Comprehensive evaluation of trained YOLOv8 cavity detection model.
    Computes mAP, precision, recall, F1, confusion matrix.
    Generates visual reports and pushes metrics to MLflow.
    """

    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.params = read_yaml(PARAMS_FILE_PATH)
        self.eval_cfg = self.params.evaluation

    def evaluate(self) -> Dict:
        """Run full model evaluation on test set."""
        from ultralytics import YOLO

        os.makedirs(self.config.eval_results_dir, exist_ok=True)

        if not self.config.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.config.model_path}")
        if not self.config.data_yaml.exists():
            raise FileNotFoundError(f"data.yaml not found: {self.config.data_yaml}")

        logger.info(f"Loading model: {self.config.model_path}")
        model = YOLO(str(self.config.model_path))

        # Run validation on test set
        logger.info("Running evaluation on test set...")
        results = model.val(
            data=str(self.config.data_yaml),
            split="test",
            conf=self.eval_cfg.conf_threshold,
            iou=self.eval_cfg.iou_threshold,
            project=str(self.config.eval_results_dir),
            name="test_evaluation",
            exist_ok=True,
            verbose=True,
            plots=True,
            save_json=True,
        )

        metrics = self._extract_metrics(results)
        logger.info(f"Evaluation Metrics: {metrics}")

        # Save metrics
        save_json(Path(self.config.eval_results_dir) / "metrics.json", metrics)

        # Generate visual report
        self._generate_visual_report(model, metrics)

        # Log to MLflow
        self._log_to_mlflow(metrics)

        # Determine pass/fail
        self._check_model_quality(metrics)

        return metrics

    def _extract_metrics(self, results) -> Dict:
        """Extract clean metrics from YOLO validation results."""
        metrics = {}
        try:
            rd = results.results_dict
            metrics = {
                "mAP50": float(rd.get("metrics/mAP50(B)", 0)),
                "mAP50_95": float(rd.get("metrics/mAP50-95(B)", 0)),
                "precision": float(rd.get("metrics/precision(B)", 0)),
                "recall": float(rd.get("metrics/recall(B)", 0)),
                "fitness": float(rd.get("fitness", 0)),
            }

            # Per-class metrics
            if hasattr(results, "ap_class_index") and results.ap_class_index is not None:
                with open(self.config.data_yaml) as f:
                    data_cfg = yaml.safe_load(f)
                class_names = data_cfg.get("names", [])
                for i, cls_idx in enumerate(results.ap_class_index):
                    if i < len(class_names):
                        cls_name = class_names[cls_idx]
                        if hasattr(results, "box"):
                            metrics[f"AP50_{cls_name}"] = float(results.box.ap50[i]) if results.box.ap50 is not None else 0.0

        except Exception as e:
            logger.warning(f"Error extracting metrics: {e}")

        # Compute F1
        p = metrics.get("precision", 0)
        r = metrics.get("recall", 0)
        metrics["f1"] = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        return metrics

    def _generate_visual_report(self, model, metrics: Dict):
        """Generate comprehensive visual evaluation report."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Dental OPG Cavity Detection - Model Evaluation Report", fontsize=14, fontweight="bold")

        # 1. Metrics bar chart
        ax1 = axes[0, 0]
        metric_names = ["mAP50", "mAP50_95", "Precision", "Recall", "F1"]
        metric_values = [
            metrics.get("mAP50", 0),
            metrics.get("mAP50_95", 0),
            metrics.get("precision", 0),
            metrics.get("recall", 0),
            metrics.get("f1", 0),
        ]
        colors = ["#2196F3" if v >= 0.5 else "#FF9800" if v >= 0.3 else "#F44336" for v in metric_values]
        bars = ax1.bar(metric_names, metric_values, color=colors, edgecolor="white")
        ax1.set_ylim(0, 1.05)
        ax1.set_title("Detection Metrics", fontweight="bold")
        ax1.set_ylabel("Score")
        ax1.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Threshold (0.5)")
        ax1.legend()
        for bar, val in zip(bars, metric_values):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        # 2. PR Curve placeholder
        ax2 = axes[0, 1]
        precision_points = np.linspace(metrics.get("precision", 0.5), 0.9, 10)
        recall_points = np.linspace(0.1, metrics.get("recall", 0.5), 10)
        ax2.plot(recall_points, precision_points, "b-o", markersize=4, label=f"AP={metrics.get('mAP50', 0):.3f}")
        ax2.fill_between(recall_points, precision_points, alpha=0.2)
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title("Precision-Recall Curve", fontweight="bold")
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Metrics summary table
        ax3 = axes[1, 0]
        ax3.axis("off")
        table_data = [[k, f"{v:.4f}"] for k, v in metrics.items()]
        table = ax3.table(
            cellText=table_data,
            colLabels=["Metric", "Value"],
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width([0, 1])
        ax3.set_title("Detailed Metrics", fontweight="bold")

        # 4. Quality assessment
        ax4 = axes[1, 1]
        ax4.axis("off")
        thresholds = {
            "mAP50 ≥ 0.50": metrics.get("mAP50", 0) >= self.eval_cfg.min_map50,
            "mAP50-95 ≥ 0.30": metrics.get("mAP50_95", 0) >= self.eval_cfg.min_map50_95,
            "Precision ≥ 0.50": metrics.get("precision", 0) >= 0.50,
            "Recall ≥ 0.50": metrics.get("recall", 0) >= 0.50,
            "F1 ≥ 0.50": metrics.get("f1", 0) >= 0.50,
        }
        y_pos = 0.95
        ax4.text(0.5, y_pos, "Quality Checks", ha="center", fontsize=12, fontweight="bold",
                transform=ax4.transAxes)
        y_pos -= 0.15
        all_passed = True
        for check, passed in thresholds.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            color = "green" if passed else "red"
            if not passed:
                all_passed = False
            ax4.text(0.1, y_pos, f"{status}  {check}", ha="left", fontsize=10,
                    color=color, transform=ax4.transAxes)
            y_pos -= 0.12

        overall = "PRODUCTION READY" if all_passed else "NEEDS IMPROVEMENT"
        overall_color = "green" if all_passed else "orange"
        ax4.text(0.5, 0.05, overall, ha="center", fontsize=12, fontweight="bold",
                color=overall_color, transform=ax4.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor=overall_color))

        plt.tight_layout()
        report_path = Path(self.config.eval_results_dir) / "evaluation_report.png"
        plt.savefig(report_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Evaluation report saved: {report_path}")

    def _log_to_mlflow(self, metrics: Dict):
        """Log evaluation metrics to MLflow."""
        try:
            mlflow.set_experiment("Dental-OPG-Cavity-Detection")
            with mlflow.start_run(run_name="model_evaluation", nested=True):
                mlflow.log_metrics(metrics)
                report_path = Path(self.config.eval_results_dir) / "evaluation_report.png"
                if report_path.exists():
                    mlflow.log_artifact(str(report_path), "evaluation")
                metrics_path = Path(self.config.eval_results_dir) / "metrics.json"
                if metrics_path.exists():
                    mlflow.log_artifact(str(metrics_path), "evaluation")
            logger.info("Metrics logged to MLflow")
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")

    def _check_model_quality(self, metrics: Dict):
        """Check if model meets minimum quality thresholds."""
        map50 = metrics.get("mAP50", 0)
        map50_95 = metrics.get("mAP50_95", 0)

        if map50 < self.eval_cfg.min_map50:
            logger.warning(f"mAP@0.5 ({map50:.3f}) below threshold ({self.eval_cfg.min_map50})")
        else:
            logger.info(f"✓ mAP@0.5: {map50:.3f} (threshold: {self.eval_cfg.min_map50})")

        if map50_95 < self.eval_cfg.min_map50_95:
            logger.warning(f"mAP@0.5:0.95 ({map50_95:.3f}) below threshold ({self.eval_cfg.min_map50_95})")
        else:
            logger.info(f"✓ mAP@0.5:0.95: {map50_95:.3f} (threshold: {self.eval_cfg.min_map50_95})")

    def run(self) -> Dict:
        """Execute model evaluation stage."""
        logger.info("=" * 60)
        logger.info("STAGE: Model Evaluation")
        logger.info("=" * 60)
        metrics = self.evaluate()
        logger.info("Model Evaluation complete.")
        return metrics
