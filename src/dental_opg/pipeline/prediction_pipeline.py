"""
Prediction pipeline for dental OPG X-ray analysis.
Handles image validation, multi-class detection, and clinical reporting.
"""
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional
import yaml

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  CLINICAL KNOWLEDGE BASE                                             #
# ------------------------------------------------------------------ #

CLASS_INFO = {
    "BDC-BDR": {
        "full_name": "Badly Decayed Crown / Root",
        "description": "Advanced tooth decay affecting the crown or root structure. "
                       "This typically represents end-stage caries where the tooth is "
                       "extensively damaged.",
        "severity": "urgent",
        "action": "Urgent extraction or root canal therapy required. "
                  "Delay risks infection spreading to surrounding bone.",
        "color": (0, 0, 220),      # BGR: deep red
    },
    "Caries": {
        "full_name": "Dental Caries (Cavity)",
        "description": "Bacterial decay causing a cavity in the tooth enamel or dentine. "
                       "Early intervention prevents deeper damage.",
        "severity": "moderate",
        "action": "Restorative treatment (dental filling) recommended. "
                  "Schedule appointment within 2 to 4 weeks.",
        "color": (0, 140, 255),    # BGR: orange
    },
    "Fractured Teeth": {
        "full_name": "Tooth Fracture",
        "description": "A crack or break in the tooth structure, which may affect the "
                       "enamel, dentine, or root depending on depth.",
        "severity": "moderate",
        "action": "Dental evaluation needed to assess fracture depth. "
                  "Treatment ranges from bonding to extraction depending on severity.",
        "color": (0, 220, 220),    # BGR: yellow
    },
    "Healthy Teeth": {
        "full_name": "Healthy Tooth Structure",
        "description": "Normal, intact tooth with no visible pathology detected.",
        "severity": "normal",
        "action": "Continue regular brushing, flossing, and dental check-ups every 6 months.",
        "color": (0, 190, 0),      # BGR: green
    },
    "Impacted Teeth": {
        "full_name": "Impacted Tooth",
        "description": "A tooth that has failed to emerge fully into its expected position, "
                       "often a wisdom tooth pressing against adjacent structures.",
        "severity": "monitor",
        "action": "Orthodontic or oral surgery evaluation recommended. "
                  "Impacted teeth can cause crowding and infection if left unmanaged.",
        "color": (200, 0, 200),    # BGR: magenta
    },
    "Infection": {
        "full_name": "Periapical Infection / Dental Abscess",
        "description": "Bacterial infection at the root tip or surrounding bone, "
                       "often presenting as a periapical radiolucency on X-ray.",
        "severity": "urgent",
        "action": "Immediate treatment required. Antibiotic therapy plus root canal "
                  "treatment or extraction. Untreated infection can spread to the jaw, "
                  "neck, or become systemic.",
        "color": (50, 50, 230),    # BGR: red-blue
    },
}

SEVERITY_ORDER = ["urgent", "moderate", "monitor", "normal"]

SEVERITY_LABELS = {
    "urgent":   "URGENT",
    "moderate": "MODERATE",
    "monitor":  "MONITOR",
    "normal":   "NORMAL",
}


class PredictionPipeline:
    """
    Full dental OPG analysis pipeline:
    - OPG image validation (rejects non-dental or non-X-ray images)
    - Multi-class detection (6 dental conditions)
    - Clinical reporting with per-class descriptions and recommendations
    """

    TEXT_COLOR = (255, 255, 255)
    OVERLAY_ALPHA = 0.22

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

    # ------------------------------------------------------------------ #
    #  MODEL LOADING                                                       #
    # ------------------------------------------------------------------ #

    def _load_model(self):
        from ultralytics import YOLO

        is_pretrained_name = (
            not str(self.model_path).startswith(("/", ".", "\\"))
            and not Path(self.model_path).is_absolute()
            and str(self.model_path) == self.model_path.name
        )

        if not is_pretrained_name and not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        logger.info(f"Loading model: {self.model_path}")
        self.model = YOLO(str(self.model_path))

        data_yaml = Path("artifacts/data_transformation/dataset/data.yaml")
        if data_yaml.exists():
            with open(data_yaml) as f:
                cfg = yaml.safe_load(f)
            self.class_names = cfg.get("names", list(CLASS_INFO.keys()))
        else:
            self.class_names = list(CLASS_INFO.keys())

        logger.info(f"Classes: {self.class_names}")

    # ------------------------------------------------------------------ #
    #  OPG IMAGE VALIDATION                                                #
    # ------------------------------------------------------------------ #

    def validate_opg_image(self, image: np.ndarray) -> Tuple[bool, str]:
        """
        Validate whether an image is likely a dental OPG X-ray.

        Checks:
        - Image must be large enough to be a real scan
        - Must be panoramic (wider than tall)
        - Must look like an X-ray (near-grayscale, appropriate brightness range)
        - Must not be a solid color, blank, or overly colorful photo

        Returns:
            (is_valid, message)
        """
        if image is None or image.size == 0:
            return False, "No image data received."

        h, w = image.shape[:2]

        # Minimum size check
        if w < 300 or h < 150:
            return False, (
                "This image is too small to be an OPG X-ray. "
                "Please upload a full-resolution panoramic X-ray image."
            )

        # Aspect ratio check — OPG X-rays are panoramic and significantly wider than tall
        ratio = w / h
        if ratio < 1.1:
            return False, (
                "This image appears to be portrait or square-shaped. "
                "An OPG (orthopantomogram) is a panoramic X-ray and should be "
                "significantly wider than it is tall. Please check that you have "
                "uploaded the correct image."
            )

        # Convert to work with
        if len(image.shape) == 2:
            gray = image
            is_already_gray = True
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            is_already_gray = False

        # Blank / near-blank check
        std_val = float(np.std(gray))
        if std_val < 8.0:
            return False, (
                "This image appears to be blank or nearly uniform. "
                "Please upload a valid dental OPG X-ray."
            )

        # Color saturation check — X-rays are grayscale
        if not is_already_gray and len(image.shape) == 3:
            r = image[:, :, 0].astype(np.float32)
            g = image[:, :, 1].astype(np.float32)
            b = image[:, :, 2].astype(np.float32)
            # Mean channel-to-channel difference (grayscale images have ~0 difference)
            color_divergence = float(
                np.mean(np.abs(r - g)) + np.mean(np.abs(g - b)) + np.mean(np.abs(r - b))
            ) / 3.0
            if color_divergence > 25.0:
                return False, (
                    "This does not appear to be an X-ray image. It looks like a "
                    "color photograph or graphic. Please upload a grayscale dental "
                    "OPG X-ray for analysis."
                )

        # Brightness range check — X-rays should not be mostly white or mostly black
        mean_brightness = float(np.mean(gray))
        if mean_brightness < 10:
            return False, (
                "This image is almost completely black. "
                "Please upload a properly exposed OPG X-ray."
            )
        if mean_brightness > 240:
            return False, (
                "This image is almost completely white and does not contain "
                "visible dental structures. Please upload a valid OPG X-ray."
            )

        return True, "Image validated as a dental X-ray."

    # ------------------------------------------------------------------ #
    #  PREDICTION                                                          #
    # ------------------------------------------------------------------ #

    def predict(
        self,
        image: Union[np.ndarray, str, Path],
        return_visualization: bool = True,
        validate: bool = True,
    ) -> Dict:
        """
        Run full dental analysis on a single image.

        Returns dict with:
            valid           — whether image passed OPG validation
            validation_msg  — human-readable validation result
            cavity_count    — total number of detections
            class_counts    — per-class detection counts
            detections      — list of individual detection dicts
            confidence_avg  — average confidence across all detections
            severity        — overall severity string
            clinical_report — structured clinical findings dict
            visualization   — annotated image (numpy array, BGR)
        """
        # Load image to numpy
        if isinstance(image, (str, Path)):
            img_raw = cv2.imread(str(image))
            if img_raw is None:
                raise ValueError(f"Cannot load image: {image}")
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                img_raw = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                img_raw = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            else:
                img_raw = image.copy()
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Validate image
        if validate:
            is_valid, val_msg = self.validate_opg_image(img_raw)
            if not is_valid:
                return {
                    "valid": False,
                    "validation_msg": val_msg,
                    "cavity_count": 0,
                    "class_counts": {},
                    "detections": [],
                    "confidence_avg": 0.0,
                    "severity": "Invalid image",
                    "clinical_report": {},
                    "visualization": img_raw if return_visualization else None,
                }

        # Run inference
        results = self.model(
            img_raw,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        # Parse detections
        detections = []
        h_img, w_img = img_raw.shape[:2]
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = (
                        self.class_names[cls_id]
                        if cls_id < len(self.class_names)
                        else f"class_{cls_id}"
                    )
                    detections.append({
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": round(conf, 4),
                        "bbox": [round(x1), round(y1), round(x2), round(y2)],
                        "bbox_normalized": [
                            round(x1 / w_img, 4),
                            round(y1 / h_img, 4),
                            round(x2 / w_img, 4),
                            round(y2 / h_img, 4),
                        ],
                    })

        avg_conf = float(np.mean([d["confidence"] for d in detections])) if detections else 0.0

        class_counts: Dict[str, int] = {}
        for d in detections:
            class_counts[d["class_name"]] = class_counts.get(d["class_name"], 0) + 1

        severity = self._assess_severity(detections)
        clinical = self._build_clinical_report(detections, class_counts)

        result = {
            "valid": True,
            "validation_msg": "Image validated as a dental X-ray.",
            "cavity_count": len(detections),
            "class_counts": class_counts,
            "detections": detections,
            "confidence_avg": round(avg_conf, 4),
            "severity": severity,
            "clinical_report": clinical,
        }

        if return_visualization:
            result["visualization"] = self._visualize(img_raw, detections)

        return result

    # ------------------------------------------------------------------ #
    #  CLINICAL ASSESSMENT                                                 #
    # ------------------------------------------------------------------ #

    def _assess_severity(self, detections: List[Dict]) -> str:
        if not detections:
            return "No pathological findings detected"

        classes_found = {d["class_name"] for d in detections}
        count = len(detections)

        if classes_found & {"Infection", "BDC-BDR"}:
            return "URGENT — serious conditions detected. Immediate dental referral required."
        elif classes_found & {"Caries", "Fractured Teeth"} and count > 3:
            return "MODERATE — multiple cavities or fractures found. Dental treatment needed soon."
        elif classes_found & {"Caries", "Fractured Teeth"}:
            return "MILD — early cavities or minor fractures detected. Schedule a dental appointment."
        elif "Impacted Teeth" in classes_found:
            return "MONITOR — impacted teeth detected. Orthodontic or surgical evaluation recommended."
        else:
            return "LOW — minor findings detected. Routine dental follow-up recommended."

    def _build_clinical_report(
        self, detections: List[Dict], class_counts: Dict[str, int]
    ) -> Dict:
        """Build a structured clinical findings report."""
        if not detections:
            return {
                "summary": "No dental pathology detected in this scan.",
                "findings": [],
                "priority_action": "No immediate action required. Continue regular dental check-ups.",
                "overall_severity": "normal",
            }

        # Sort found classes by severity priority
        found_classes = list(class_counts.keys())
        found_classes.sort(
            key=lambda c: SEVERITY_ORDER.index(
                CLASS_INFO.get(c, {}).get("severity", "normal")
            )
        )

        findings = []
        for cls_name in found_classes:
            info = CLASS_INFO.get(cls_name, {})
            findings.append({
                "condition": cls_name,
                "full_name": info.get("full_name", cls_name),
                "count": class_counts[cls_name],
                "description": info.get("description", ""),
                "severity": info.get("severity", "normal"),
                "severity_label": SEVERITY_LABELS.get(info.get("severity", "normal"), ""),
                "recommended_action": info.get("action", "Consult a dentist."),
            })

        # Overall severity from highest priority class found
        overall_severity = findings[0]["severity"] if findings else "normal"

        # Priority action from highest severity condition
        priority_action = findings[0]["recommended_action"] if findings else ""

        # Summary sentence
        conditions_str = ", ".join(
            f"{v} {k}" if v > 1 else f"1 {k}"
            for k, v in class_counts.items()
        )
        summary = f"This scan shows {conditions_str}."

        return {
            "summary": summary,
            "findings": findings,
            "priority_action": priority_action,
            "overall_severity": overall_severity,
        }

    # ------------------------------------------------------------------ #
    #  VISUALIZATION                                                       #
    # ------------------------------------------------------------------ #

    def _get_class_color(self, cls_name: str) -> Tuple[int, int, int]:
        return CLASS_INFO.get(cls_name, {}).get("color", (0, 128, 255))

    def _visualize(self, img_bgr: np.ndarray, detections: List[Dict]) -> np.ndarray:
        img_vis = img_bgr.copy()
        h, w = img_vis.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            cls_name = det["class_name"]
            color = self._get_class_color(cls_name)

            # Box + fill
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
            overlay = img_vis.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, self.OVERLAY_ALPHA, img_vis, 1 - self.OVERLAY_ALPHA, 0, img_vis)

            # Label
            label = f"{cls_name} {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(y1 - 5, lh + 5)
            cv2.rectangle(img_vis, (x1, label_y - lh - 4), (x1 + lw + 4, label_y + 2), color, -1)
            cv2.putText(
                img_vis, label, (x1 + 2, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_COLOR, 1, cv2.LINE_AA,
            )

        # Header bar
        header = f"Findings: {len(detections)}"
        if detections:
            header += f"  |  Avg Confidence: {np.mean([d['confidence'] for d in detections]):.0%}"
        cv2.rectangle(img_vis, (0, 0), (w, 32), (15, 15, 15), -1)
        cv2.putText(
            img_vis, header, (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA,
        )

        return img_vis

    # ------------------------------------------------------------------ #
    #  BATCH                                                               #
    # ------------------------------------------------------------------ #

    def predict_batch(self, images: List[Union[np.ndarray, str, Path]]) -> List[Dict]:
        return [self.predict(img) for img in images]


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Dental OPG Analysis")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", default="models/best/best.pt", help="Path to model")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--output", help="Output annotated image path")
    args = parser.parse_args()

    pipe = PredictionPipeline(model_path=args.model, conf_threshold=args.conf)
    result = pipe.predict(args.image)
    print(json.dumps({k: v for k, v in result.items() if k != "visualization"}, indent=2))

    if args.output and result.get("visualization") is not None:
        cv2.imwrite(args.output, result["visualization"])
        print(f"Annotated image saved: {args.output}")


if __name__ == "__main__":
    main()
