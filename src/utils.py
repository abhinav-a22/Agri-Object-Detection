"""
Utils Module - John Deere Agricultural Object Detection
======================================================
Provides reusable, modular utility functions for:
- Configuration loading and validation (PyYAML)
- Reproducible random seed setting (Python, NumPy, PyTorch)
- Hardware device resolution (CUDA GPU vs. CPU)
- Logging configuration
- Bounding box coordinate transformations (YOLO normalized xywh <-> pixel xyxy)
- Intersection over Union (IoU) calculation
- Visualizing detections with bounding boxes and confidence labels
"""

import os
import random
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

import yaml
import numpy as np


def setup_logger(name: str = "AgriDetection", log_file: Optional[Union[str, Path]] = None) -> logging.Logger:
    """
    Configures and returns a structured console and optional file logger.

    Args:
        name: Name identifier for the logger instance.
        log_file: Optional path to an output log file.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console stream handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = setup_logger("Utils")


def load_config(config_path: Union[str, Path] = "configs/config.yaml") -> Dict[str, Any]:
    """
    Loads and parses the project YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing parsed configuration parameters.

    Raises:
        FileNotFoundError: If the specified config file does not exist.
        ValueError: If the file is not valid YAML.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise ValueError(f"Expected YAML dictionary in {path}, got {type(config)}")
        return config
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML configuration: {exc}") from exc


def set_seed(seed: int = 42) -> None:
    """
    Enforces deterministic reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value (default: 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device(preferred: str = "auto") -> str:
    """
    Resolves the execution device ('cuda:0' vs 'cpu') based on system capability
    and user preference.

    Args:
        preferred: Device preference ('auto', 'cpu', '0', 'cuda:0', etc.)

    Returns:
        Resolved device string suitable for Ultralytics YOLO.
    """
    if preferred == "cpu":
        return "cpu"

    try:
        import torch
        if torch.cuda.is_available():
            if preferred in ("auto", "cuda", "gpu"):
                return "0"
            return str(preferred)
        else:
            return "cpu"
    except ImportError:
        return "cpu"


def xywh_to_xyxy(
    box: Union[List[float], Tuple[float, float, float, float]],
    img_width: int,
    img_height: int
) -> List[float]:
    """
    Converts normalized YOLO bounding box [x_center, y_center, width, height]
    into absolute pixel coordinates [x1, y1, x2, y2].

    Args:
        box: Normalized coordinates [x_center, y_center, width, height] in [0, 1].
        img_width: Width of image in pixels.
        img_height: Height of image in pixels.

    Returns:
        Pixel coordinates [x1, y1, x2, y2].
    """
    xc, yc, w, h = box
    x1 = (xc - w / 2.0) * img_width
    y1 = (yc - h / 2.0) * img_height
    x2 = (xc + w / 2.0) * img_width
    y2 = (yc + h / 2.0) * img_height

    # Clip to image boundaries
    x1 = max(0.0, min(float(img_width), x1))
    y1 = max(0.0, min(float(img_height), y1))
    x2 = max(0.0, min(float(img_width), x2))
    y2 = max(0.0, min(float(img_height), y2))

    return [x1, y1, x2, y2]


def xyxy_to_xywh(
    box: Union[List[float], Tuple[float, float, float, float]],
    img_width: int,
    img_height: int
) -> List[float]:
    """
    Converts absolute pixel coordinates [x1, y1, x2, y2]
    into normalized YOLO bounding box [x_center, y_center, width, height].

    Args:
        box: Pixel coordinates [x1, y1, x2, y2].
        img_width: Width of image in pixels.
        img_height: Height of image in pixels.

    Returns:
        Normalized coordinates [x_center, y_center, width, height] in range [0, 1].
    """
    x1, y1, x2, y2 = box
    w = (x2 - x1) / float(img_width)
    h = (y2 - y1) / float(img_height)
    xc = (x1 + x2) / (2.0 * float(img_width))
    yc = (y1 + y2) / (2.0 * float(img_height))
    return [round(xc, 6), round(yc, 6), round(w, 6), round(h, 6)]


def compute_iou(
    box1: Union[List[float], Tuple[float, float, float, float]],
    box2: Union[List[float], Tuple[float, float, float, float]]
) -> float:
    """
    Computes Intersection over Union (IoU) between two bounding boxes
    in pixel coordinate format [x1, y1, x2, y2].

    Args:
        box1: First box [x1, y1, x2, y2].
        box2: Second box [x1, y1, x2, y2].

    Returns:
        Float IoU value in range [0.0, 1.0].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_w = max(0.0, x2 - x1)
    intersection_h = max(0.0, y2 - y1)
    intersection_area = intersection_w * intersection_h

    box1_area = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    box2_area = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area
    if union_area <= 0.0:
        return 0.0

    return float(intersection_area / union_area)


def draw_detections(
    image: np.ndarray,
    boxes: List[List[float]],
    scores: List[float],
    class_ids: List[int],
    class_names: List[str],
    palette: Optional[Dict[int, Tuple[int, int, int]]] = None
) -> np.ndarray:
    """
    Draws clear bounding boxes with class label pills and confidence percentages
    on a BGR image using OpenCV (or pure NumPy fallback if cv2 is not present).

    Args:
        image: Input image array in BGR format (H, W, C).
        boxes: List of bounding boxes [x1, y1, x2, y2] in pixel coords.
        scores: Detection confidence scores in [0.0, 1.0].
        class_ids: Integer class index for each detection.
        class_names: List of class name strings mapped by index.
        palette: Optional mapping of class_id -> BGR color tuple.

    Returns:
        Annotated image copy.
    """
    annotated = image.copy()
    try:
        import cv2
    except ImportError:
        # If OpenCV is not installed, return image unmodified with warning
        logger.warning("OpenCV (cv2) is not installed; skipping bounding box drawing.")
        return annotated

    # Default color palette:
    # Class 0 (tractor): Bright John Deere Yellow-Green (0, 200, 0)
    # Class 1 (person): High-visibility Safety Amber/Orange (0, 140, 255)
    default_colors = {
        0: (36, 179, 36),     # Agricultural green (tractor)
        1: (0, 140, 255),     # Warning Orange (person)
        2: (255, 105, 180),   # Hot pink (extra class)
        3: (255, 215, 0),     # Gold (extra class)
    }
    colors = palette or default_colors

    for box, score, cid in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = [int(round(coord)) for coord in box]
        color = colors.get(cid, (0, 255, 255))
        cname = class_names[cid] if 0 <= cid < len(class_names) else f"class_{cid}"
        label_text = f"{cname} {score:.2f}"

        # Draw primary bounding rectangle
        thickness = max(2, int(round(min(image.shape[:2]) / 300)))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Draw label background pill
        font_scale = 0.5
        font_thickness = 1
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        pill_y1 = max(0, y1 - th - baseline - 4)
        pill_y2 = y1
        pill_x2 = min(image.shape[1], x1 + tw + 6)
        
        cv2.rectangle(annotated, (x1, pill_y1), (pill_x2, pill_y2), color, -1)
        # Draw contrast text
        text_color = (0, 0, 0) if (color[0] + color[1] + color[2]) / 3.0 > 128 else (255, 255, 255)
        cv2.putText(
            annotated,
            label_text,
            (x1 + 3, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            font_thickness,
            lineType=cv2.LINE_AA
        )

    return annotated


if __name__ == "__main__":
    print("Testing utils.py helper functions...")
    cfg = load_config()
    print(f"Successfully loaded config.yaml. Model architecture: {cfg['model']['architecture']}")
    dev = get_device(cfg['training']['device'])
    print(f"Resolved execution device: {dev}")

    # IoU unit check
    b1 = [10.0, 10.0, 50.0, 50.0]
    b2 = [20.0, 20.0, 60.0, 60.0]
    iou = compute_iou(b1, b2)
    print(f"Computed IoU between sample boxes: {iou:.4f}")
    assert 0.0 < iou < 1.0, "IoU math sanity check failed!"
    print("utils.py self-test passed successfully.")

