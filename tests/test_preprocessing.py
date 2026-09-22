"""
Unit Tests for Dataset Preprocessing and Validation
===================================================
Tests core validation logic:
- Image file integrity verification
- YOLO label format verification (bounds, class limits, element count)
- Image-to-annotation alignment (orphan image and label detection)
- Dataset statistics calculation
"""

import os
import sys
from pathlib import Path
import pytest
from PIL import Image

# Ensure project root is in system path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    validate_image_file,
    validate_label_file,
    check_dataset_alignment,
    compute_dataset_statistics,
    create_sample_dataset
)


@pytest.fixture
def temp_dataset_dir(tmp_path):
    """Creates a temporary isolated directory for dataset testing."""
    img_dir = tmp_path / "images" / "train"
    lbl_dir = tmp_path / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    return tmp_path


def test_validate_image_file_valid(tmp_path):
    """Verifies that a valid RGB image passes verification."""
    img_path = tmp_path / "valid.jpg"
    img = Image.new("RGB", (640, 480), color=(100, 150, 200))
    img.save(img_path, format="JPEG")

    is_valid, reason, dims = validate_image_file(img_path)
    assert is_valid is True
    assert reason is None
    assert dims == (640, 480)


def test_validate_image_file_corrupted(tmp_path):
    """Verifies that a corrupted non-image file is caught and rejected."""
    corrupted_path = tmp_path / "corrupted.jpg"
    with open(corrupted_path, "wb") as f:
        f.write(b"NOT_A_VALID_JPEG_HEADER_CONTENT")

    is_valid, reason, dims = validate_image_file(corrupted_path)
    assert is_valid is False
    assert reason is not None
    assert dims is None


def test_validate_image_file_nonexistent(tmp_path):
    """Verifies that a non-existent file path returns False."""
    fake_path = tmp_path / "does_not_exist.png"
    is_valid, reason, dims = validate_image_file(fake_path)
    assert is_valid is False
    assert "File does not exist" in reason


def test_validate_label_file_valid(tmp_path):
    """Verifies that a well-formed YOLO label file passes validation."""
    lbl_path = tmp_path / "valid.txt"
    # class_id xc yc w h
    with open(lbl_path, "w") as f:
        f.write("0 0.500000 0.500000 0.200000 0.300000\n")
        f.write("1 0.250000 0.300000 0.100000 0.150000\n")

    is_valid, errors, boxes = validate_label_file(lbl_path, num_classes=2)
    assert is_valid is True
    assert len(errors) == 0
    assert len(boxes) == 2
    assert boxes[0]["class_id"] == 0
    assert boxes[1]["class_id"] == 1


def test_validate_label_file_out_of_bounds(tmp_path):
    """Verifies that bounding boxes exceeding normalized [0, 1] range are flagged."""
    lbl_path = tmp_path / "invalid_bounds.txt"
    with open(lbl_path, "w") as f:
        # xc = 1.25 (exceeds 1.0)
        f.write("0 1.250000 0.500000 0.200000 0.300000\n")

    is_valid, errors, boxes = validate_label_file(lbl_path, num_classes=2)
    assert is_valid is False
    assert any("out of normalized bounds" in err for err in errors)


def test_validate_label_file_invalid_class_id(tmp_path):
    """Verifies that out-of-range class IDs are flagged."""
    lbl_path = tmp_path / "invalid_class.txt"
    with open(lbl_path, "w") as f:
        # Class 5 when num_classes=2
        f.write("5 0.500000 0.500000 0.200000 0.300000\n")

    is_valid, errors, boxes = validate_label_file(lbl_path, num_classes=2)
    assert is_valid is False
    assert any("outside valid range" in err for err in errors)


def test_check_dataset_alignment(temp_dataset_dir):
    """Tests identification of matched pairs and orphan files."""
    img_dir = temp_dataset_dir / "images" / "train"
    lbl_dir = temp_dataset_dir / "labels" / "train"

    # Matched pair
    Image.new("RGB", (100, 100)).save(img_dir / "frame_001.jpg")
    (lbl_dir / "frame_001.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    # Orphan image (no label)
    Image.new("RGB", (100, 100)).save(img_dir / "frame_orphan_img.jpg")

    # Orphan label (no image)
    (lbl_dir / "frame_orphan_lbl.txt").write_text("1 0.5 0.5 0.2 0.2\n")

    alignment = check_dataset_alignment(img_dir, lbl_dir)
    assert alignment["total_images"] == 2
    assert alignment["total_labels"] == 2
    assert alignment["matched_pairs"] == 1
    assert len(alignment["orphan_images"]) == 1
    assert len(alignment["orphan_labels"]) == 1


def test_sample_dataset_generation(tmp_path):
    """Tests synthetic dataset creation and statistics computation."""
    sample_dir = tmp_path / "agri_test_data"
    create_sample_dataset(sample_dir)

    stats = compute_dataset_statistics(sample_dir, num_classes=2, class_names=["tractor", "person"])
    assert stats["overall"]["total_images"] == 14
    assert stats["overall"]["total_boxes"] == 28
    assert stats["overall"]["class_distribution"]["tractor"] == 14
    assert stats["overall"]["class_distribution"]["person"] == 14

