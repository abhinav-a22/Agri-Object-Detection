"""
Image and Video Inference Module - John Deere Agricultural Object Detection
===========================================================================
Executes object detection inference on agricultural images, directories of images,
and video streams using trained YOLO11 weights.

Features:
- Single image inference: generates bounding boxes, class labels, and confidence scores
- Batch directory inference: processes multiple agricultural field images sequentially
- Video inference: frame-by-frame detection with real-time FPS overlay and MP4/AVI export
- Configurable confidence and NMS Intersection-over-Union (IoU) thresholds
- Seamless integration with configs/config.yaml
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import numpy as np

# Ensure project root is in system path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import setup_logger, load_config, get_device, draw_detections

logger = setup_logger("Inference")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def load_detection_model(
    weights_path: Path,
    device: str = "cpu"
) -> Any:
    """
    Loads YOLO detection model with safety and availability validation.

    Args:
        weights_path: Path to model checkpoint file (.pt).
        device: Device to load model onto ('cpu', '0').

    Returns:
        Loaded Ultralytics YOLO model instance.
    """
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Model weights not found at: {weights_path.resolve()}\n"
            f"Please train the model first by running:\n"
            f"    python src/train.py --epochs 25\n"
            f"Alternatively, provide an existing checkpoint via: --weights <path>"
        )

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics is required for inference. Run: pip install ultralytics")
        raise SystemExit(1)

    logger.info(f"Loading YOLO weights from: {weights_path.resolve()} on {device}...")
    model = YOLO(str(weights_path))
    return model


def process_single_image(
    model: Any,
    image_path: Path,
    output_dir: Path,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    device: str = "cpu",
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Executes inference on a single agricultural image and saves the annotated result.

    Args:
        model: Ultralytics YOLO model instance.
        image_path: Path to input image file.
        output_dir: Folder to save annotated output image.
        conf_thresh: Minimum confidence score threshold.
        iou_thresh: Non-maximum suppression IoU threshold.
        device: Execution device ('cpu' or '0').
        class_names: Optional list of class names.

    Returns:
        Dictionary containing detected bounding boxes, classes, and confidence scores.
    """
    try:
        import cv2
    except ImportError:
        logger.error("OpenCV (cv2) is required for inference image operations.")
        raise SystemExit(1)

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image at: {image_path}")

    # Run YOLO inference
    t0 = time.perf_counter()
    results = model.predict(
        source=img,
        conf=conf_thresh,
        iou=iou_thresh,
        device=device,
        verbose=False
    )
    inference_time_ms = (time.perf_counter() - t0) * 1000.0

    boxes_out: List[List[float]] = []
    scores_out: List[float] = []
    classes_out: List[int] = []

    res = results[0]
    names = class_names or (list(model.names.values()) if hasattr(model, "names") else ["tractor", "person"])

    if res.boxes is not None and len(res.boxes) > 0:
        boxes_xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cid in zip(boxes_xyxy, confs, cls_ids):
            boxes_out.append([float(c) for c in box])
            scores_out.append(float(conf))
            classes_out.append(int(cid))

    annotated = draw_detections(
        image=img,
        boxes=boxes_out,
        scores=scores_out,
        class_ids=classes_out,
        class_names=names
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{image_path.stem}_detected{image_path.suffix}"
    cv2.imwrite(str(out_file), annotated)

    logger.info(
        f"Processed '{image_path.name}' -> {len(boxes_out)} object(s) detected "
        f"in {inference_time_ms:.1f} ms | Saved to: {out_file.name}"
    )

    return {
        "source": str(image_path),
        "output_path": str(out_file),
        "detections_count": len(boxes_out),
        "boxes": boxes_out,
        "scores": scores_out,
        "classes": classes_out,
        "latency_ms": round(inference_time_ms, 2)
    }


def process_video_stream(
    model: Any,
    video_path: Path,
    output_dir: Path,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    device: str = "cpu",
    class_names: Optional[List[str]] = None
) -> Path:
    """
    Executes frame-by-frame inference on a video file with real-time FPS overlay
    and saves the annotated video.

    Args:
        model: Ultralytics YOLO model instance.
        video_path: Path to input video file (.mp4, .avi, etc.).
        output_dir: Directory where annotated video will be written.
        conf_thresh: Confidence threshold.
        iou_thresh: NMS IoU threshold.
        device: Execution device.
        class_names: Optional class name mappings.

    Returns:
        Path to output video file.
    """
    try:
        import cv2
    except ImportError:
        logger.error("OpenCV (cv2) is required for video inference.")
        raise SystemExit(1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video source: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{video_path.stem}_detected.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_file), fourcc, fps_in, (width, height))

    logger.info(f"Processing video '{video_path.name}' ({total_frames} frames @ {fps_in:.1f} FPS)...")

    names = class_names or (list(model.names.values()) if hasattr(model, "names") else ["tractor", "person"])
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        t0 = time.perf_counter()

        # Run inference
        results = model.predict(
            source=frame,
            conf=conf_thresh,
            iou=iou_thresh,
            device=device,
            verbose=False
        )
        res = results[0]

        boxes_out: List[List[float]] = []
        scores_out: List[float] = []
        classes_out: List[int] = []

        if res.boxes is not None and len(res.boxes) > 0:
            for box, conf, cid in zip(
                res.boxes.xyxy.cpu().numpy(),
                res.boxes.conf.cpu().numpy(),
                res.boxes.cls.cpu().numpy().astype(int)
            ):
                boxes_out.append([float(c) for c in box])
                scores_out.append(float(conf))
                classes_out.append(int(cid))

        annotated = draw_detections(
            image=frame,
            boxes=boxes_out,
            scores=scores_out,
            class_ids=classes_out,
            class_names=names
        )

        dt = time.perf_counter() - t0
        fps_current = 1.0 / dt if dt > 0 else 0.0

        # Overlay FPS badge
        cv2.putText(
            annotated,
            f"FPS: {fps_current:.1f} | Frame: {frame_idx}/{total_frames}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            lineType=cv2.LINE_AA
        )

        writer.write(annotated)

        if frame_idx % 30 == 0:
            logger.info(f"Processed {frame_idx}/{total_frames} frames ({fps_current:.1f} FPS)")

    cap.release()
    writer.release()
    logger.info(f"Video inference completed. Saved output to: {out_file}")
    return out_file


def run_inference(
    source: str,
    weights_path: Optional[str] = None,
    config_path: str = "configs/config.yaml",
    conf_override: Optional[float] = None,
    iou_override: Optional[float] = None,
    device_override: Optional[str] = None,
    output_dir_override: Optional[str] = None
) -> None:
    """
    Top-level inference runner supporting images, directories, and videos.

    Args:
        source: Path to image, directory, or video.
        weights_path: Explicit weights path.
        config_path: Path to configuration YAML.
        conf_override: Optional confidence threshold.
        iou_override: Optional IoU threshold.
        device_override: Optional device.
        output_dir_override: Optional output directory.
    """
    config = load_config(config_path)

    # 1. Resolve weights
    if weights_path:
        w_path = Path(weights_path)
    else:
        w_path = (PROJECT_ROOT / config["model"]["save_dir"] / config["model"]["best_model_name"]).resolve()

    # 2. Resolve thresholds and device
    conf = conf_override if conf_override is not None else config["inference"]["confidence_threshold"]
    iou = iou_override if iou_override is not None else config["inference"]["iou_threshold"]
    device = get_device(device_override or config["training"]["device"])
    out_dir = Path(output_dir_override or config["inference"]["output_detections_dir"])
    class_names = config["dataset"].get("classes", ["tractor", "person"])

    # 3. Load model
    model = load_detection_model(w_path, device=device)

    src_path = Path(source)
    if not src_path.exists():
        raise FileNotFoundError(f"Specified inference source not found: {src_path.resolve()}")

    # 4. Route according to source type
    if src_path.is_file():
        suffix = src_path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            process_single_image(
                model=model,
                image_path=src_path,
                output_dir=out_dir,
                conf_thresh=conf,
                iou_thresh=iou,
                device=device,
                class_names=class_names
            )
        elif suffix in VIDEO_EXTENSIONS:
            process_video_stream(
                model=model,
                video_path=src_path,
                output_dir=out_dir,
                conf_thresh=conf,
                iou_thresh=iou,
                device=device,
                class_names=class_names
            )
        else:
            raise ValueError(f"Unsupported file format '{suffix}'. Supported: {IMAGE_EXTENSIONS | VIDEO_EXTENSIONS}")

    elif src_path.is_dir():
        image_files = [f for f in src_path.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
        logger.info(f"Processing directory '{src_path.name}' ({len(image_files)} images found)...")
        for img_file in image_files:
            process_single_image(
                model=model,
                image_path=img_file,
                output_dir=out_dir,
                conf_thresh=conf,
                iou_thresh=iou,
                device=device,
                class_names=class_names
            )


def main():
    parser = argparse.ArgumentParser(description="Agricultural Object Detection Inference (Image/Video)")
    parser.add_argument("--source", type=str, required=True, help="Path to input image, directory, or video")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights (.pt)")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold (e.g. 0.25)")
    parser.add_argument("--iou", type=float, default=None, help="NMS IoU threshold (e.g. 0.45)")
    parser.add_argument("--device", type=str, default=None, help="Execution device ('cpu', '0')")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save detections")

    args = parser.parse_args()

    run_inference(
        source=args.source,
        weights_path=args.weights,
        config_path=args.config,
        conf_override=args.conf,
        iou_override=args.iou,
        device_override=args.device,
        output_dir_override=args.output_dir,
    )


if __name__ == "__main__":
    main()

