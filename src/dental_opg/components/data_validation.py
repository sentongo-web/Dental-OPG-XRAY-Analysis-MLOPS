import os
import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple, Dict, List
from dental_opg.entity.config_entity import DataValidationConfig

logger = logging.getLogger(__name__)


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        self.unzip_dir = Path("artifacts/data_ingestion/unzipped")

    def _find_dataset_root(self) -> Path:
        """Locate the actual dataset root (handles nested zip structures)."""
        if not self.unzip_dir.exists():
            raise FileNotFoundError(f"Unzip dir not found: {self.unzip_dir}")

        # Check for one-level-deep directories first
        contents = list(self.unzip_dir.iterdir())
        if len(contents) == 1 and contents[0].is_dir():
            return contents[0]
        return self.unzip_dir

    def _detect_annotation_format(self, dataset_root: Path) -> str:
        """Auto-detect annotation format: yolo, coco, or voc."""
        # Check for COCO JSON
        json_files = list(dataset_root.rglob("*.json"))
        if json_files:
            for jf in json_files:
                try:
                    with open(jf) as f:
                        data = json.load(f)
                    if "annotations" in data and "categories" in data:
                        logger.info(f"Detected COCO format: {jf}")
                        return "coco"
                except Exception:
                    continue

        # Check for VOC XML
        xml_files = list(dataset_root.rglob("*.xml"))
        if xml_files:
            try:
                tree = ET.parse(xml_files[0])
                root = tree.getroot()
                if root.tag == "annotation":
                    logger.info(f"Detected VOC format: {xml_files[0]}")
                    return "voc"
            except Exception:
                pass

        # Check for YOLO txt labels
        txt_files = list(dataset_root.rglob("*.txt"))
        non_yaml_txt = [f for f in txt_files if f.suffix == ".txt" and "yaml" not in f.name]
        if non_yaml_txt:
            try:
                with open(non_yaml_txt[0]) as f:
                    first_line = f.readline().strip()
                parts = first_line.split()
                if len(parts) == 5 and all(self._is_float(p) for p in parts):
                    logger.info(f"Detected YOLO format: {non_yaml_txt[0]}")
                    return "yolo"
            except Exception:
                pass

        # Check for existing data.yaml (already YOLO format)
        yaml_files = list(dataset_root.rglob("data.yaml")) + list(dataset_root.rglob("dataset.yaml"))
        if yaml_files:
            logger.info(f"Detected YOLO format via yaml: {yaml_files[0]}")
            return "yolo"

        return "unknown"

    @staticmethod
    def _is_float(value: str) -> bool:
        try:
            float(value)
            return True
        except ValueError:
            return False

    def _get_dataset_stats(self, dataset_root: Path) -> Dict:
        """Collect dataset statistics."""
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        images = [f for f in dataset_root.rglob("*") if f.suffix.lower() in image_exts]
        txt_labels = list(dataset_root.rglob("*.txt"))
        xml_labels = list(dataset_root.rglob("*.xml"))
        json_files = list(dataset_root.rglob("*.json"))

        return {
            "total_images": len(images),
            "txt_labels": len([f for f in txt_labels if "yaml" not in f.name]),
            "xml_labels": len(xml_labels),
            "json_labels": len(json_files),
            "image_formats": list({f.suffix.lower() for f in images}),
            "dataset_root": str(dataset_root),
        }

    def validate_all(self) -> bool:
        """Run all validation checks and write status."""
        os.makedirs(self.config.root_dir, exist_ok=True)
        validation_status = True
        issues = []

        logger.info("Starting data validation...")

        # 1. Check unzip directory exists
        if not self.unzip_dir.exists():
            issues.append("ERROR: Unzipped data directory not found")
            validation_status = False
        else:
            dataset_root = self._find_dataset_root()
            logger.info(f"Dataset root: {dataset_root}")

            # 2. Get stats
            stats = self._get_dataset_stats(dataset_root)
            logger.info(f"Dataset stats: {stats}")

            # 3. Check images exist
            if stats["total_images"] == 0:
                issues.append("ERROR: No images found in dataset")
                validation_status = False
            else:
                logger.info(f"✓ Images found: {stats['total_images']}")

            # 4. Check annotations exist
            total_labels = stats["txt_labels"] + stats["xml_labels"] + stats["json_labels"]
            if total_labels == 0:
                issues.append("WARNING: No annotation files found - check dataset format")
            else:
                logger.info(f"✓ Annotation files found: {total_labels}")

            # 5. Detect format
            fmt = self._detect_annotation_format(dataset_root)
            if fmt == "unknown":
                issues.append("WARNING: Could not detect annotation format")
            else:
                logger.info(f"✓ Annotation format detected: {fmt}")

            # 6. Write stats to validation dir
            self._save_validation_report(stats, fmt, issues)

        # Write status file
        with open(self.config.STATUS_FILE, "w") as f:
            if validation_status:
                f.write("Validation status: True\n")
                f.write("All required checks passed.\n")
            else:
                f.write("Validation status: False\n")
                f.write("\n".join(issues) + "\n")

        if issues:
            for issue in issues:
                logger.warning(issue)

        logger.info(f"Validation status: {validation_status}")
        return validation_status

    def _save_validation_report(self, stats: Dict, fmt: str, issues: List):
        """Save detailed validation report."""
        report = {
            "dataset_stats": stats,
            "detected_format": fmt,
            "issues": issues,
            "validation_passed": len([i for i in issues if i.startswith("ERROR")]) == 0,
        }
        import json
        report_path = Path(self.config.root_dir) / "validation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Validation report saved: {report_path}")

    def run(self):
        """Execute data validation."""
        logger.info("=" * 60)
        logger.info("STAGE: Data Validation")
        logger.info("=" * 60)
        status = self.validate_all()
        if not status:
            raise ValueError("Data validation failed. Check the status file.")
        logger.info("Data Validation complete.")
        return status
