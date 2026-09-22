"""
Unit Tests for Inference and Evaluation Logic
=============================================
Tests:
- Configuration parsing and integrity
- Intersection over Union (IoU) calculation logic
- Coordinate space transformations (normalized xywh <-> pixel xyxy)
- Error analysis logic (TP, FP, FN, scale categorization)
- Exception handling for missing weights and invalid inputs
"""

import os
import sys
from pathlib import Path
import pytest

# Ensure project root is in system path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    load_config,
    compute_iou,
    xywh_to_xyxy,
    xyxy_to_xywh,
    get_device
)
from src.evaluate import (
    check_model_availability,
    run_error_analysis
)
from src.inference import (
    load_detection_model
)


def test_load_config():
    """Verifies that configs/config.yaml parses into expected structure."""
    cfg = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    assert "dataset" in cfg
    assert "model" in cfg
    assert "training" in cfg
    assert "inference" in cfg
    assert cfg["model"]["architecture"] in ("yolo11s", "yolo11n")
    assert cfg["dataset"]["num_classes"] == 2
    assert "tractor" in cfg["dataset"]["classes"]
    assert "person" in cfg["dataset"]["classes"]


def test_load_config_missing_file():
    """Verifies that missing configuration files raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config(PROJECT_ROOT / "configs" / "non_existent_config.yaml")


def test_compute_iou_identical_boxes():
    """IoU between identical boxes must be exactly 1.0."""
    box = [10.0, 10.0, 50.0, 50.0]
    assert pytest.approx(compute_iou(box, box), 1e-5) == 1.0


def test_compute_iou_disjoint_boxes():
    """IoU between completely non-overlapping boxes must be 0.0."""
    b1 = [0.0, 0.0, 10.0, 10.0]
    b2 = [20.0, 20.0, 30.0, 30.0]
    assert compute_iou(b1, b2) == 0.0


def test_compute_iou_partial_overlap():
    """IoU for known geometry: two 10x10 boxes overlapping by 5x10."""
    b1 = [0.0, 0.0, 10.0, 10.0]    # area = 100
    b2 = [5.0, 0.0, 15.0, 10.0]    # area = 100
    # intersection: x in [5, 10] -> w=5, h=10 -> area = 50
    # union: 100 + 100 - 50 = 150
    # IoU: 50 / 150 = 1/3 ~ 0.33333
    assert pytest.approx(compute_iou(b1, b2), 1e-4) == (1.0 / 3.0)


def test_coordinate_transforms_roundtrip():
    """Verifies round-trip accuracy between xywh and xyxy."""
    img_w, img_h = 640, 480
    orig_xywh = [0.5, 0.5, 0.2, 0.4]  # xc, yc, w, h
    xyxy = xywh_to_xyxy(orig_xywh, img_w, img_h)
    recovered_xywh = xyxy_to_xywh(xyxy, img_w, img_h)

    for orig, recovered in zip(orig_xywh, recovered_xywh):
        assert pytest.approx(orig, 1e-4) == recovered


def test_coordinate_transforms_clipping():
    """Verifies that coordinates are clipped properly within [0, W] and [0, H]."""
    img_w, img_h = 640, 480
    # Box extending past left/top edges
    box_overflow = [0.05, 0.05, 0.2, 0.2]
    xyxy = xywh_to_xyxy(box_overflow, img_w, img_h)
    assert xyxy[0] >= 0.0
    assert xyxy[1] >= 0.0
    assert xyxy[2] <= img_w
    assert xyxy[3] <= img_h


def test_get_device_cpu_fallback():
    """Verifies get_device returns 'cpu' when explicit 'cpu' requested."""
    assert get_device("cpu") == "cpu"


def test_error_analysis_calculation():
    """Verifies computation of TP, FP, FN, precision, recall, and scale breakdown."""
    predictions = [
        # True Positive (matches gt_1)
        {"image_id": "img_1", "box": [10.0, 10.0, 50.0, 50.0], "class_id": 0, "score": 0.90},
        # False Positive (no matching ground truth)
        {"image_id": "img_1", "box": [100.0, 100.0, 150.0, 150.0], "class_id": 1, "score": 0.75}
    ]
    ground_truths = [
        # Matches prediction
        {"image_id": "img_1", "box": [12.0, 12.0, 52.0, 52.0], "class_id": 0, "area": 0.10},
        # Missed ground truth (False Negative)
        {"image_id": "img_1", "box": [200.0, 200.0, 220.0, 220.0], "class_id": 1, "area": 0.01}
    ]

    analysis = run_error_analysis(predictions, ground_truths, iou_threshold=0.5)

    assert analysis["true_positives"] == 1
    assert analysis["false_positives"] == 1
    assert analysis["false_negatives"] == 1
    assert pytest.approx(analysis["precision_at_iou50"], 1e-4) == 0.5
    assert pytest.approx(analysis["recall_at_iou50"], 1e-4) == 0.5
    assert analysis["missed_scale_breakdown"]["small_objects_missed"] == 1


def test_missing_model_handling(tmp_path):
    """Verifies check_model_availability returns False for non-existent file."""
    fake_weights = tmp_path / "fake_best.pt"
    assert check_model_availability(fake_weights) is False


def test_load_detection_model_missing_raises():
    """Verifies load_detection_model raises FileNotFoundError on missing weights."""
    fake_weights = Path("models/definitely_does_not_exist.pt")
    with pytest.raises(FileNotFoundError):
        load_detection_model(fake_weights)

