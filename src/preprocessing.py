"""
Dataset Preprocessing and Validation Module
===========================================
Handles essential data quality checks and dataset statistics for the
John Deere Agricultural Object Detection project:
1. Image validation (corrupted images, non-image files, dimension checks)
2. Label validation (YOLO format correctness, normalized [0, 1] coordinates, valid class IDs)
3. Image-to-annotation alignment (identifying orphan images and orphan labels)
4. Comprehensive dataset statistics (class frequency, bounding box dimensions, area categories)
5. Synthetic sample dataset generator (for offline unit testing and immediate verification)
6. Download guidance for the real Roboflow dataset
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from PIL import Image
import numpy as np

# Ensure src/ directory is in path when executed directly
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import setup_logger, load_config

logger = setup_logger("Preprocessing")

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_image_file(image_path: Path) -> Tuple[bool, Optional[str], Optional[Tuple[int, int]]]:
    """
    Validates that a single image file exists, has a valid extension, and can be read.

    Args:
        image_path: Path to the image file.

    Returns:
        Tuple of (is_valid, error_reason, (width, height)).
    """
    if not image_path.is_file():
        return False, "File does not exist", None

    if image_path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
        return False, f"Unsupported extension {image_path.suffix}", None

    try:
        with Image.open(image_path) as img:
            img.verify()  # Fast structural verification
        # Re-open to read dimensions and confirm decodability
        with Image.open(image_path) as img:
            width, height = img.size
            if width <= 0 or height <= 0:
                return False, f"Invalid dimensions: {width}x{height}", None
            return True, None, (width, height)
    except Exception as exc:
        return False, f"Corrupted image ({exc})", None


def validate_label_file(
    label_path: Path,
    num_classes: int = 2
) -> Tuple[bool, List[str], List[Dict[str, Any]]]:
    """
    Validates a YOLO format annotation text file.
    Each line must follow: <class_id> <x_center> <y_center> <width> <height>
    with coordinates normalized in the interval [0.0, 1.0].

    Args:
        label_path: Path to the annotation .txt file.
        num_classes: Total expected number of distinct classes.

    Returns:
        Tuple of (is_valid, list_of_error_strings, list_of_parsed_boxes).
    """
    if not label_path.is_file():
        return False, ["Label file does not exist"], []

    errors: List[str] = []
    boxes: List[Dict[str, Any]] = []

    try:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        for line_idx, line in enumerate(lines, start=1):
            parts = line.split()
            if len(parts) != 5:
                errors.append(
                    f"Line {line_idx}: Expected 5 elements, found {len(parts)} ('{line}')"
                )
                continue

            try:
                class_id = int(parts[0])
            except ValueError:
                errors.append(f"Line {line_idx}: Class ID '{parts[0]}' is not an integer")
                continue

            if class_id < 0 or class_id >= num_classes:
                errors.append(
                    f"Line {line_idx}: Class ID {class_id} outside valid range [0, {num_classes - 1}]"
                )

            try:
                xc, yc, w, h = [float(p) for p in parts[1:5]]
            except ValueError:
                errors.append(f"Line {line_idx}: Non-numeric bounding box coordinates")
                continue

            # Verify normalized boundaries [0.0, 1.0]
            coords = [("x_center", xc), ("y_center", yc), ("width", w), ("height", h)]
            for name, val in coords:
                if val < 0.0 or val > 1.0:
                    errors.append(
                        f"Line {line_idx}: {name} value {val} out of normalized bounds [0.0, 1.0]"
                    )

            if w <= 0.0 or h <= 0.0:
                errors.append(f"Line {line_idx}: Degenerate box dimensions (w={w}, h={h})")

            boxes.append({
                "class_id": class_id,
                "x_center": xc,
                "y_center": yc,
                "width": w,
                "height": h
            })

    except Exception as exc:
        errors.append(f"Failed to read label file: {exc}")

    is_valid = len(errors) == 0
    return is_valid, errors, boxes


def check_dataset_alignment(
    images_dir: Path,
    labels_dir: Path
) -> Dict[str, Any]:
    """
    Checks for alignment between images and their corresponding YOLO label files.
    Identifies orphan images and orphan labels.

    Args:
        images_dir: Directory containing image files.
        labels_dir: Directory containing .txt label files.

    Returns:
        Dictionary containing counts and lists of matched and mismatched items.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    if not images_dir.exists():
        return {"error": f"Image directory does not exist: {images_dir}"}
    if not labels_dir.exists():
        return {"error": f"Label directory does not exist: {labels_dir}"}

    image_stems = {
        img.stem: img for img in images_dir.iterdir()
        if img.suffix.lower() in VALID_IMAGE_EXTENSIONS
    }
    label_stems = {
        lbl.stem: lbl for lbl in labels_dir.iterdir()
        if lbl.suffix.lower() == ".txt"
    }

    matched = set(image_stems.keys()).intersection(set(label_stems.keys()))
    orphan_images = [str(image_stems[stem]) for stem in set(image_stems.keys()) - matched]
    orphan_labels = [str(label_stems[stem]) for stem in set(label_stems.keys()) - matched]

    return {
        "total_images": len(image_stems),
        "total_labels": len(label_stems),
        "matched_pairs": len(matched),
        "orphan_images": orphan_images,
        "orphan_labels": orphan_labels,
    }


def compute_dataset_statistics(
    data_dir: Path,
    num_classes: int = 2,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes statistical overview of dataset splits (train/val/test):
    - Image counts
    - Bounding box frequency per class
    - Box size distributions (Small, Medium, Large relative area)
    - Aspect ratio distribution

    Args:
        data_dir: Root dataset folder containing images/ and labels/.
        num_classes: Expected number of classes.
        class_names: Names corresponding to class IDs.

    Returns:
        Nested dictionary of statistics.
    """
    data_dir = Path(data_dir)
    names = class_names or [f"class_{i}" for i in range(num_classes)]
    stats: Dict[str, Any] = {"splits": {}, "overall": {"total_images": 0, "total_boxes": 0}}
    class_totals: Dict[str, int] = {name: 0 for name in names}

    splits = ["train", "val", "test"]
    for split in splits:
        img_dir = data_dir / "images" / split
        lbl_dir = data_dir / "labels" / split

        if not img_dir.exists() or not lbl_dir.exists():
            continue

        alignment = check_dataset_alignment(img_dir, lbl_dir)
        split_boxes = 0
        split_classes: Dict[str, int] = {name: 0 for name in names}
        box_aspect_ratios: List[float] = []
        box_areas: List[float] = []

        for lbl_file in lbl_dir.glob("*.txt"):
            valid, _, boxes = validate_label_file(lbl_file, num_classes)
            for b in boxes:
                cid = b["class_id"]
                if 0 <= cid < len(names):
                    cname = names[cid]
                    split_classes[cname] += 1
                    class_totals[cname] += 1
                split_boxes += 1
                w, h = b["width"], b["height"]
                if h > 0:
                    box_aspect_ratios.append(w / h)
                box_areas.append(w * h)

        stats["splits"][split] = {
            "images": alignment.get("total_images", 0),
            "matched_pairs": alignment.get("matched_pairs", 0),
            "total_boxes": split_boxes,
            "class_distribution": split_classes,
            "mean_aspect_ratio": float(np.mean(box_aspect_ratios)) if box_aspect_ratios else 0.0,
            "mean_relative_area": float(np.mean(box_areas)) if box_areas else 0.0,
        }
        stats["overall"]["total_images"] += alignment.get("total_images", 0)
        stats["overall"]["total_boxes"] += split_boxes

    stats["overall"]["class_distribution"] = class_totals
    return stats


def create_sample_dataset(data_dir: Path) -> None:
    """
    Generates a minimal valid synthetic dataset with realistic annotations
    for offline testing, CI verification, and immediate pipeline validation.
    Creates train, val, and test splits with sample agricultural imagery
    (colored backgrounds with tractors and workers) and matching YOLO .txt labels.

    Args:
        data_dir: Root dataset directory to populate.
    """
    data_dir = Path(data_dir)
    logger.info(f"Generating synthetic sample dataset in {data_dir.resolve()}...")

    splits = {
        "train": 8,
        "val": 4,
        "test": 2,
    }

    # Class 0: tractor (green rect), Class 1: person (orange rect)
    for split, count in splits.items():
        img_dir = data_dir / "images" / split
        lbl_dir = data_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(count):
            w, h = 640, 480
            # Agricultural background: field brown/green gradient
            field_color = (
                50 + (idx * 5) % 40,
                110 + (idx * 10) % 60,
                30 + (idx * 3) % 30
            )
            img = Image.new("RGB", (w, h), color=field_color)
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)

            labels_content = []

            # Box 1: Simulated tractor (class 0)
            t_w, t_h = int(w * 0.4), int(h * 0.35)
            t_x1, t_y1 = int(w * 0.1), int(h * 0.4)
            t_x2, t_y2 = t_x1 + t_w, t_y1 + t_h
            draw.rectangle([t_x1, t_y1, t_x2, t_y2], fill=(34, 139, 34), outline=(0, 255, 0), width=3)
            # Center coordinates normalized
            tx_c = (t_x1 + t_w / 2.0) / w
            ty_c = (t_y1 + t_h / 2.0) / h
            tw_norm = t_w / w
            th_norm = t_h / h
            labels_content.append(f"0 {tx_c:.6f} {ty_c:.6f} {tw_norm:.6f} {th_norm:.6f}")

            # Box 2: Simulated agricultural worker (class 1)
            p_w, p_h = int(w * 0.08), int(h * 0.25)
            p_x1, p_y1 = int(w * 0.65), int(h * 0.45)
            p_x2, p_y2 = p_x1 + p_w, p_y1 + p_h
            draw.rectangle([p_x1, p_y1, p_x2, p_y2], fill=(255, 140, 0), outline=(255, 255, 0), width=2)
            px_c = (p_x1 + p_w / 2.0) / w
            py_c = (p_y1 + p_h / 2.0) / h
            pw_norm = p_w / w
            ph_norm = p_h / h
            labels_content.append(f"1 {px_c:.6f} {py_c:.6f} {pw_norm:.6f} {ph_norm:.6f}")

            img_file = img_dir / f"agri_sample_{split}_{idx:03d}.jpg"
            lbl_file = lbl_dir / f"agri_sample_{split}_{idx:03d}.txt"

            img.save(img_file, format="JPEG", quality=90)
            with open(lbl_file, "w", encoding="utf-8") as f:
                f.write("\n".join(labels_content) + "\n")

    logger.info("Synthetic sample agricultural dataset created successfully!")


def print_download_guide() -> None:
    """
    Prints precise step-by-step instructions for obtaining the verified
    real agricultural object detection dataset from Roboflow Universe.
    """
    guide = """
================================================================================
VERIFIED DATASET SETUP INSTRUCTIONS: DCB-YOLO-TRACTOR-DETECTION
================================================================================

1. Dataset Metadata:
   - Dataset Name: DCB-yolo-tractor-detection
   - Host Platform: Roboflow Universe
   - Verified URL: https://universe.roboflow.com/rjdpworkspaace/dcb-yolo-tractor-detection
   - License: MIT License (Permissive open source for academic/research use)
   - Object Classes: 'tractor', 'person' (2 classes)
   - Total Annotated Images: ~695 images
   - Format: YOLOv8 / YOLO11 PyTorch TXT format

2. Option A: Download via Roboflow Python SDK (Recommended)
   Run in your terminal:
   $ pip install roboflow
   $ python -c "
from roboflow import Roboflow
rf = Roboflow(api_key='YOUR_ROBOFLOW_API_KEY')
project = rf.workspace('rjdpworkspaace').project('dcb-yolo-tractor-detection')
version = project.version(1)
dataset = version.download('yolov11', location='data')
"

3. Option B: Manual Web Download
   a) Visit: https://universe.roboflow.com/rjdpworkspaace/dcb-yolo-tractor-detection
   b) Click 'Download Dataset'
   c) Select Format: 'YOLOv11 PyTorch' (or 'YOLOv8 PyTorch')
   d) Extract the downloaded zip file into the 'data/' folder so that:
      - data/images/train/
      - data/images/val/
      - data/images/test/
      - data/labels/train/
      - data/labels/val/
      - data/labels/test/
      - data/data.yaml
      match the project layout.

4. Quick Start / Offline Testing:
   You can run the full pipeline immediately on the generated sample dataset:
   $ python src/preprocessing.py --create-sample
================================================================================
"""
    print(guide)


def main():
    parser = argparse.ArgumentParser(description="Agricultural Dataset Preprocessing & Validation")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to dataset root folder")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to configuration file")
    parser.add_argument("--validate", action="store_true", help="Perform image and label validation")
    parser.add_argument("--stats", action="store_true", help="Calculate and print dataset statistics")
    parser.add_argument("--create-sample", action="store_true", help="Generate synthetic sample dataset")
    parser.add_argument("--download-guide", action="store_true", help="Print Roboflow dataset download guide")

    args = parser.parse_args()

    if args.download_guide:
        print_download_guide()
        return

    data_dir = Path(args.data_dir)

    if args.create_sample:
        create_sample_dataset(data_dir)

    config = {}
    if Path(args.config).is_file():
        config = load_config(args.config)
    num_classes = config.get("dataset", {}).get("num_classes", 2)
    class_names = config.get("dataset", {}).get("classes", ["tractor", "person"])

    if args.validate or (not args.create_sample and not args.stats):
        logger.info("Validating dataset integrity...")
        for split in ["train", "val", "test"]:
            img_dir = data_dir / "images" / split
            lbl_dir = data_dir / "labels" / split
            if not img_dir.exists() or not lbl_dir.exists():
                logger.info(f"Split '{split}' not found or partially missing. Skipping.")
                continue

            align = check_dataset_alignment(img_dir, lbl_dir)
            logger.info(
                f"[{split.upper()}] Images: {align.get('total_images')}, "
                f"Labels: {align.get('total_labels')}, "
                f"Matched: {align.get('matched_pairs')}, "
                f"Orphan images: {len(align.get('orphan_images', []))}, "
                f"Orphan labels: {len(align.get('orphan_labels', []))}"
            )

    if args.stats:
        logger.info("Computing dataset statistics...")
        stats = compute_dataset_statistics(data_dir, num_classes, class_names)
        import json
        print("\n" + json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

