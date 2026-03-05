"""
Prediction pipeline for dental OPG cavity detection.
Used by the Gradio app and API endpoints.
"""
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional
import yaml

logger = logging.getLogger(__name__)


class PredictionPipeline:
    """
    Inference pipeline for cavity detection in OPG X-ray images.
    Handles single images and batches.
    """

    # Color map for visualization per class (BGR), 6 classes
    CLASS_COLORS = [
        (0, 0, 255),     # class 0 BDC-BDR: Red
        (0, 165, 255),   # class 1 Caries: Orange
        (0, 255, 255),   # class 2 Fractured Teeth: Yellow
        (0, 200, 0),     # class 3 Healthy Teeth: Green
        (255, 0, 255),   # class 4 Impacted Teeth: Magenta
        (255, 50, 50),   # class 5 Infection: Blue-red
    ]
    TEXT_COLOR = (255, 255, 255)
    OVERLAY_ALPHA = 0.25

    def __init__(
        self,
        model_path: Union[str, Path] = "models/best/best.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "",
    ):
        self.model_path = Path(model_path)
        self.conf = conf_threshold
        self.iou = iou_threshold
        self.device = device
        self.model = None
        self.class_names = []
        self._load_model()

    def _load_model(self):
        """Load YOLOv8 model. Accepts local .pt path or pretrained name (e.g. 'yolov8n.pt')."""
        from ultralytics import YOLO

        # Only check existence for local paths (not pretrained names like 'yolov8n.pt')
        is_pretrained_name = not str(self.model_path).startswith(("/", ".", "\\")) and \
                             not Path(self.model_path).is_absolute() and \
                             str(self.model_path) == self.model_path.name

        if not is_pretrained_name and not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        logger.info(f"Loading model: {self.model_path} (pretrained={is_pretrained_name})")
        self.model = YOLO(str(self.model_path))
        logger.info(f"Model loaded: {self.model_path}")

        # Try to load class names
        data_yaml = Path("artifacts/data_transformation/dataset/data.yaml")
        if data_yaml.exists():
            with open(data_yaml) as f:
                cfg = yaml.safe_load(f)
            self.class_names = cfg.get("names", ["cavity"])
        else:
            self.class_names = ["cavity"]

    def predict(
        self,
        image: Union[np.ndarray, str, Path],
        return_visualization: bool = True,
    ) -> Dict:
        """
        Run cavity detection on a single image.

        Args:
            image: numpy array (BGR/RGB), file path, or Path object
            return_visualization: whether to return annotated image

        Returns:
            dict with keys: detections, count, confidence_avg, visualization
        """
        # Load image
        if isinstance(image, (str, Path)):
            img_bgr = cv2.imread(str(image))
            if img_bgr is None:
                raise ValueError(f"Cannot load image: {image}")
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:  # grayscale
                img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:  # RGBA
                img_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            elif image.shape[2] == 3:
                img_bgr = image.copy()
            else:
                img_bgr = image.copy()
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Run inference
        results = self.model(
            img_bgr,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        # Parse detections
        detections = []
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else "cavity"
                    detections.append({
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": round(conf, 4),
                        "bbox": [round(x1), round(y1), round(x2), round(y2)],
                        "bbox_normalized": [
                            round(x1 / img_bgr.shape[1], 4),
                            round(y1 / img_bgr.shape[0], 4),
                            round(x2 / img_bgr.shape[1], 4),
                            round(y2 / img_bgr.shape[0], 4),
                        ],
                    })

        avg_conf = np.mean([d["confidence"] for d in detections]) if detections else 0.0

        # Count per class
        class_counts = {}
        for d in detections:
            name = d["class_name"]
            class_counts[name] = class_counts.get(name, 0) + 1

        result = {
            "cavity_count": len(detections),
            "class_counts": class_counts,
            "detections": detections,
            "confidence_avg": round(float(avg_conf), 4),
            "severity": self._assess_severity(detections),
        }

        if return_visualization:
            result["visualization"] = self._visualize(img_bgr, detections)

        return result

    def _assess_severity(self, detections: List[Dict]) -> str:
        """Assess overall severity based on detected conditions."""
        if not detections:
            return "No pathological findings detected"

        URGENT = {"Infection", "BDC-BDR"}
        MODERATE = {"Caries", "Fractured Teeth"}

        classes_found = {d["class_name"] for d in detections}
        count = len(detections)

        if classes_found & URGENT:
            return "Urgent — infection or severe decay detected. Immediate dental referral recommended."
        elif classes_found & MODERATE and count > 3:
            return "Moderate — multiple cavities or fractures found. Dental treatment needed soon."
        elif classes_found & MODERATE:
            return "Mild — early cavities or minor fractures. Schedule dental appointment."
        elif "Impacted Teeth" in classes_found:
            return "Impacted teeth detected. Orthodontic or surgical evaluation may be needed."
        else:
            return "Minor findings detected. Follow-up with a dentist recommended."

    def _visualize(self, img_bgr: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels on image."""
        img_vis = img_bgr.copy()
        h, w = img_vis.shape[:2]

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            cls_id = det["class_id"]
            cls_name = det["class_name"]
            color = self.CLASS_COLORS[cls_id % len(self.CLASS_COLORS)]

            # Draw rectangle
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)

            # Semi-transparent fill
            overlay = img_vis.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, self.OVERLAY_ALPHA, img_vis, 1 - self.OVERLAY_ALPHA, 0, img_vis)

            # Label
            label = f"{cls_name} {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(y1 - 5, lh + 5)
            cv2.rectangle(img_vis, (x1, label_y - lh - 4), (x1 + lw + 4, label_y + 2), color, -1)
            cv2.putText(img_vis, label, (x1 + 2, label_y - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_COLOR, 1, cv2.LINE_AA)

        # Summary header
        summary = f"Findings: {len(detections)}"
        if detections:
            summary += f" | Avg Conf: {np.mean([d['confidence'] for d in detections]):.2f}"
        cv2.rectangle(img_vis, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(img_vis, summary, (10, 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        return img_vis

    def predict_batch(self, images: List[Union[np.ndarray, str, Path]]) -> List[Dict]:
        """Run prediction on a batch of images."""
        return [self.predict(img) for img in images]


def main():
    """CLI entry point for batch prediction."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Dental OPG Cavity Detection")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", default="models/best/best.pt", help="Path to model")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--output", help="Output image path")
    args = parser.parse_args()

    pipeline = PredictionPipeline(model_path=args.model, conf_threshold=args.conf)
    result = pipeline.predict(args.image)

    print(json.dumps({k: v for k, v in result.items() if k != "visualization"}, indent=2))

    if args.output and "visualization" in result:
        cv2.imwrite(args.output, result["visualization"])
        print(f"Annotated image saved: {args.output}")


if __name__ == "__main__":
    main()
