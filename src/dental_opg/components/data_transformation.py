import os
import cv2
import json
import yaml
import shutil
import random
import logging
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
from dental_opg.entity.config_entity import DataTransformationConfig

logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)


class DataTransformation:
    """
    Transforms raw dental OPG dataset into YOLO-format train/val/test splits.
    Handles YOLO, COCO, and VOC input formats.
    Applies medical-imaging-safe augmentations.
    """

    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.data_root = None
        self.annotation_format = None
        self.class_names = []

    # ------------------------------------------------------------------ #
    #  FORMAT DETECTION & ROUTING                                          #
    # ------------------------------------------------------------------ #

    def _find_dataset_root(self) -> Path:
        contents = list(self.config.data_path.iterdir())
        if len(contents) == 1 and contents[0].is_dir():
            return contents[0]
        return self.config.data_path

    def _detect_format(self, root: Path) -> str:
        json_files = list(root.rglob("*.json"))
        for jf in json_files:
            try:
                with open(jf) as f:
                    d = json.load(f)
                if "annotations" in d and "categories" in d:
                    return "coco"
            except Exception:
                pass

        xml_files = list(root.rglob("*.xml"))
        if xml_files:
            try:
                tree = ET.parse(xml_files[0])
                if tree.getroot().tag == "annotation":
                    return "voc"
            except Exception:
                pass

        # Check for pre-split YOLO dataset (train/valid/test dirs)
        yaml_files = list(root.rglob("data.yaml")) + list(root.rglob("dataset.yaml"))
        if yaml_files:
            return "yolo_presplit"

        txt_files = [f for f in root.rglob("*.txt") if f.suffix == ".txt"]
        if txt_files:
            try:
                with open(txt_files[0]) as f:
                    line = f.readline().strip().split()
                if len(line) == 5:
                    return "yolo"
            except Exception:
                pass

        return "unknown"

    # ------------------------------------------------------------------ #
    #  COCO → YOLO CONVERSION                                              #
    # ------------------------------------------------------------------ #

    def _coco_to_yolo(self, root: Path) -> Tuple[List, List, List]:
        """Convert COCO JSON annotations to YOLO format."""
        json_files = list(root.rglob("*.json"))
        coco_json = None
        for jf in json_files:
            with open(jf) as f:
                d = json.load(f)
            if "annotations" in d:
                coco_json = d
                break

        self.class_names = [c["name"] for c in sorted(coco_json["categories"], key=lambda x: x["id"])]
        logger.info(f"COCO classes: {self.class_names}")

        id_to_file = {img["id"]: img["file_name"] for img in coco_json["images"]}
        id_to_shape = {img["id"]: (img["height"], img["width"]) for img in coco_json["images"]}
        cat_id_to_idx = {c["id"]: i for i, c in enumerate(sorted(coco_json["categories"], key=lambda x: x["id"]))}

        # Group annotations by image
        ann_by_image = {}
        for ann in coco_json["annotations"]:
            iid = ann["image_id"]
            if iid not in ann_by_image:
                ann_by_image[iid] = []
            ann_by_image[iid].append(ann)

        # Convert each image
        img_label_pairs = []
        image_dir = root / "images" if (root / "images").exists() else root
        all_images = list(image_dir.rglob("*.jpg")) + list(image_dir.rglob("*.png")) + \
                     list(image_dir.rglob("*.jpeg"))
        img_by_name = {f.name: f for f in all_images}

        for img_id, file_name in id_to_file.items():
            fname = Path(file_name).name
            img_path = img_by_name.get(fname)
            if img_path is None:
                continue

            h, w = id_to_shape[img_id]
            yolo_lines = []
            for ann in ann_by_image.get(img_id, []):
                bx, by, bw, bh = ann["bbox"]
                cx = (bx + bw / 2) / w
                cy = (by + bh / 2) / h
                nw = bw / w
                nh = bh / h
                cls = cat_id_to_idx[ann["category_id"]]
                yolo_lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            img_label_pairs.append((img_path, yolo_lines))

        return self._split_pairs(img_label_pairs)

    # ------------------------------------------------------------------ #
    #  VOC → YOLO CONVERSION                                               #
    # ------------------------------------------------------------------ #

    def _voc_to_yolo(self, root: Path) -> Tuple[List, List, List]:
        """Convert Pascal VOC XML annotations to YOLO format."""
        xml_files = list(root.rglob("*.xml"))
        classes_set = set()
        for xml_f in xml_files:
            tree = ET.parse(xml_f)
            for obj in tree.getroot().findall("object"):
                classes_set.add(obj.find("name").text)
        self.class_names = sorted(list(classes_set))
        logger.info(f"VOC classes: {self.class_names}")

        img_label_pairs = []
        image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

        for xml_f in xml_files:
            tree = ET.parse(xml_f)
            root_elem = tree.getroot()
            size = root_elem.find("size")
            w = int(size.find("width").text)
            h = int(size.find("height").text)

            # Find corresponding image
            img_name = root_elem.find("filename").text if root_elem.find("filename") is not None else xml_f.stem
            img_path = None
            for ext in image_exts:
                candidate = xml_f.parent / (Path(img_name).stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break
                # Also search parent directories
                for search_dir in [xml_f.parent.parent, xml_f.parent.parent / "images"]:
                    candidate = search_dir / (Path(img_name).stem + ext)
                    if candidate.exists():
                        img_path = candidate
                        break
                if img_path:
                    break

            if img_path is None:
                continue

            yolo_lines = []
            for obj in root_elem.findall("object"):
                cls_name = obj.find("name").text
                if cls_name not in self.class_names:
                    continue
                cls_idx = self.class_names.index(cls_name)
                bnd = obj.find("bndbox")
                xmin = float(bnd.find("xmin").text)
                ymin = float(bnd.find("ymin").text)
                xmax = float(bnd.find("xmax").text)
                ymax = float(bnd.find("ymax").text)
                cx = (xmin + xmax) / 2 / w
                cy = (ymin + ymax) / 2 / h
                bw = (xmax - xmin) / w
                bh = (ymax - ymin) / h
                yolo_lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            img_label_pairs.append((img_path, yolo_lines))

        return self._split_pairs(img_label_pairs)

    # ------------------------------------------------------------------ #
    #  YOLO FORMAT (already labeled)                                        #
    # ------------------------------------------------------------------ #

    def _process_yolo(self, root: Path) -> Tuple[List, List, List]:
        """Process existing YOLO-format dataset."""
        img_dirs = []
        for d in ["images", "JPEGImages", "imgs", "."]:
            candidate = root / d
            if candidate.exists():
                imgs = list(candidate.rglob("*.jpg")) + list(candidate.rglob("*.png")) + \
                       list(candidate.rglob("*.jpeg"))
                if imgs:
                    img_dirs.extend(imgs)

        label_dirs = list(root.rglob("*.txt"))
        label_dirs = [f for f in label_dirs if f.suffix == ".txt" and "classes" not in f.name.lower()]

        # Extract class names
        classes_file = root / "classes.txt"
        if not classes_file.exists():
            classes_file = root / "obj.names"
        if classes_file.exists():
            with open(classes_file) as f:
                self.class_names = [line.strip() for line in f if line.strip()]
        else:
            self.class_names = ["cavity"]

        # Match images to labels
        label_map = {f.stem: f for f in label_dirs}
        img_label_pairs = []
        for img_path in img_dirs:
            label_path = label_map.get(img_path.stem)
            if label_path:
                with open(label_path) as f:
                    lines = [l.strip() for l in f if l.strip()]
                img_label_pairs.append((img_path, lines))

        return self._split_pairs(img_label_pairs)

    def _process_yolo_presplit(self, root: Path) -> Tuple[List, List, List]:
        """Handle pre-split YOLO dataset (train/valid/test folders)."""
        yaml_files = list(root.rglob("data.yaml")) + list(root.rglob("dataset.yaml"))
        if yaml_files:
            with open(yaml_files[0]) as f:
                data_cfg = yaml.safe_load(f)
            nc = data_cfg.get("nc", 1)
            self.class_names = data_cfg.get("names", [f"class_{i}" for i in range(nc)])

        train_imgs, val_imgs, test_imgs = [], [], []
        for split, container in [("train", train_imgs), ("valid", val_imgs), ("val", val_imgs), ("test", test_imgs)]:
            split_dir = root / split / "images"
            if not split_dir.exists():
                split_dir = root / split
            if split_dir.exists():
                imgs = list(split_dir.rglob("*.jpg")) + list(split_dir.rglob("*.png")) + \
                       list(split_dir.rglob("*.jpeg"))
                for img in imgs:
                    lbl_candidates = [
                        img.parent.parent / "labels" / (img.stem + ".txt"),
                        img.parent / (img.stem + ".txt"),
                    ]
                    lbl = next((l for l in lbl_candidates if l.exists()), None)
                    if lbl:
                        with open(lbl) as f:
                            lines = [l.strip() for l in f if l.strip()]
                        container.append((img, lines))
                    else:
                        container.append((img, []))

        if not val_imgs and not test_imgs:
            # Re-split
            all_pairs = train_imgs
            return self._split_pairs(all_pairs)

        return train_imgs, val_imgs, test_imgs

    # ------------------------------------------------------------------ #
    #  SPLIT                                                               #
    # ------------------------------------------------------------------ #

    def _split_pairs(self, pairs: List) -> Tuple[List, List, List]:
        random.shuffle(pairs)
        n = len(pairs)
        n_train = int(n * self.config.train_ratio)
        n_val = int(n * self.config.val_ratio)
        return pairs[:n_train], pairs[n_train:n_train + n_val], pairs[n_train + n_val:]

    # ------------------------------------------------------------------ #
    #  AUGMENTATION (Medical-Imaging Safe)                                  #
    # ------------------------------------------------------------------ #

    def _augment_image(self, image: np.ndarray, labels: List[str]) -> List[Tuple[np.ndarray, List[str]]]:
        """
        Apply medical-imaging-safe augmentations to OPG X-rays.
        NO flips (anatomical orientation), NO color jitter (grayscale X-rays).
        Safe: brightness/contrast, CLAHE, slight rotation, slight zoom.
        """
        augmented = []
        h, w = image.shape[:2]

        # 1. CLAHE contrast enhancement
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        if len(image.shape) == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        augmented.append((enhanced, labels))

        # 2. Brightness variation (±20%)
        alpha = random.uniform(0.80, 1.20)
        bright = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        augmented.append((bright, labels))

        # 3. Gaussian blur (slight)
        sigma = random.uniform(0.5, 1.5)
        blurred = cv2.GaussianBlur(image, (3, 3), sigma)
        augmented.append((blurred, labels))

        # 4. Slight rotation (±5 degrees max for OPG)
        angle = random.uniform(-5, 5)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        # Adjust labels for rotation
        rotated_labels = self._rotate_labels(labels, angle, w, h)
        augmented.append((rotated, rotated_labels))

        return augmented[:self.config.augmentation_factor]

    def _rotate_labels(self, labels: List[str], angle: float, w: int, h: int) -> List[str]:
        """Rotate YOLO bounding boxes."""
        if abs(angle) < 0.5 or not labels:
            return labels
        theta = np.radians(angle)
        cos_a, sin_a = np.cos(theta), np.sin(theta)
        new_labels = []
        for lbl in labels:
            parts = lbl.split()
            if len(parts) != 5:
                new_labels.append(lbl)
                continue
            cls, cx, cy, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            # Rotate center point
            cx_px = cx * w - w / 2
            cy_px = cy * h - h / 2
            new_cx_px = cx_px * cos_a - cy_px * sin_a
            new_cy_px = cx_px * sin_a + cy_px * cos_a
            new_cx = (new_cx_px + w / 2) / w
            new_cy = (new_cy_px + h / 2) / h
            new_cx = max(0, min(1, new_cx))
            new_cy = max(0, min(1, new_cy))
            new_labels.append(f"{cls} {new_cx:.6f} {new_cy:.6f} {bw:.6f} {bh:.6f}")
        return new_labels

    # ------------------------------------------------------------------ #
    #  WRITE OUTPUT                                                         #
    # ------------------------------------------------------------------ #

    def _write_split(self, pairs: List, split: str):
        """Write image-label pairs to YOLO output structure."""
        img_dir = self.config.output_dir / split / "images"
        lbl_dir = self.config.output_dir / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        written = 0
        for idx, (img_path, labels) in enumerate(tqdm(pairs, desc=f"Writing {split}")):
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Cannot read image: {img_path}")
                continue

            # Resize to configured size
            img_resized = cv2.resize(img, (self.config.image_size, self.config.image_size))
            out_name = f"{split}_{idx:05d}.jpg"
            cv2.imwrite(str(img_dir / out_name), img_resized)

            # Write labels
            lbl_out = lbl_dir / f"{split}_{idx:05d}.txt"
            with open(lbl_out, "w") as f:
                f.write("\n".join(labels) + ("\n" if labels else ""))

            written += 1

            # Augment training data
            if split == "train" and labels and self.config.augmentation_factor > 0:
                augmented = self._augment_image(img_resized, labels)
                for aug_idx, (aug_img, aug_labels) in enumerate(augmented):
                    aug_name = f"train_{idx:05d}_aug{aug_idx}.jpg"
                    cv2.imwrite(str(img_dir / aug_name), aug_img)
                    aug_lbl = lbl_dir / f"train_{idx:05d}_aug{aug_idx}.txt"
                    with open(aug_lbl, "w") as f:
                        f.write("\n".join(aug_labels) + ("\n" if aug_labels else ""))

        logger.info(f"  {split}: {written} original images written")

    def _write_data_yaml(self):
        """Write YOLO data.yaml config file."""
        yaml_content = {
            "path": str(self.config.output_dir.resolve()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": len(self.class_names),
            "names": self.class_names,
        }
        yaml_path = self.config.output_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"data.yaml written: {yaml_path}")
        logger.info(f"Classes ({len(self.class_names)}): {self.class_names}")

    # ------------------------------------------------------------------ #
    #  MAIN                                                                #
    # ------------------------------------------------------------------ #

    def run(self):
        """Execute full data transformation pipeline."""
        logger.info("=" * 60)
        logger.info("STAGE: Data Transformation")
        logger.info("=" * 60)

        os.makedirs(self.config.output_dir, exist_ok=True)

        # Already transformed
        if (self.config.output_dir / "data.yaml").exists():
            logger.info("Transformation already done, skipping.")
            return

        self.data_root = self._find_dataset_root()
        self.annotation_format = self._detect_format(self.data_root)
        logger.info(f"Dataset root: {self.data_root}")
        logger.info(f"Detected format: {self.annotation_format}")

        if self.annotation_format == "coco":
            train, val, test = self._coco_to_yolo(self.data_root)
        elif self.annotation_format == "voc":
            train, val, test = self._voc_to_yolo(self.data_root)
        elif self.annotation_format == "yolo_presplit":
            train, val, test = self._process_yolo_presplit(self.data_root)
        elif self.annotation_format == "yolo":
            train, val, test = self._process_yolo(self.data_root)
        else:
            logger.warning("Unknown format, attempting YOLO processing")
            train, val, test = self._process_yolo(self.data_root)

        if not self.class_names:
            self.class_names = ["cavity"]

        logger.info(f"Split sizes — Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

        self._write_split(train, "train")
        self._write_split(val, "val")
        self._write_split(test, "test")
        self._write_data_yaml()

        logger.info("Data Transformation complete.")
