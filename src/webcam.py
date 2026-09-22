"""
Real-Time Webcam Object Detection Module
========================================
Executes live computer vision object detection on webcam video feeds
simulating edge-mounted perception sensors on agricultural equipment.

Responsibilities:
1. Initialises webcam video capture device with configurable index.
2. Streams frames continuously into YOLO11 inference pipeline.
3. Renders bounding boxes, class labels, and confidence scores.
4. Overlays real-time framerate (FPS) indicator.
5. Handles exceptions gracefully:
   - Missing model weights
   - Disconnected / invalid camera hardware
   - Camera stream interruptions
6. Exits cleanly upon pressing 'q' key, releasing hardware locks.
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Optional, List

# Ensure project root is in system path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import setup_logger, load_config, get_device, draw_detections

logger = setup_logger("Webcam")


def run_webcam_detection(
    camera_id: int = 0,
    weights_path: Optional[str] = None,
    config_path: str = "configs/config.yaml",
    conf_thresh: Optional[float] = None,
    iou_thresh: Optional[float] = None,
    device_override: Optional[str] = None
) -> None:
    """
    Launches live webcam detection loop.

    Args:
        camera_id: System index for camera hardware (default: 0).
        weights_path: Path to YOLO model checkpoint (.pt).
        config_path: Path to configuration YAML.
        conf_thresh: Optional confidence threshold.
        iou_thresh: Optional IoU threshold.
        device_override: Optional device string ('cpu', '0').
    """
    config = load_config(config_path)

    # 1. Resolve model weights
    if weights_path:
        w_path = Path(weights_path)
    else:
        w_path = (PROJECT_ROOT / config["model"]["save_dir"] / config["model"]["best_model_name"]).resolve()

    if not w_path.is_file():
        logger.error("=" * 70)
        logger.error(f"Trained model weights not found at: {w_path.resolve()}")
        logger.error("Please train the model first by running:")
        logger.error("    python src/train.py --epochs 25")
        logger.error("Or pass an existing weights file using --weights <path>")
        logger.error("=" * 70)
        return

    # 2. Check OpenCV and Ultralytics
    try:
        import cv2
    except ImportError:
        logger.error("OpenCV (opencv-python) is required for webcam operation. Run: pip install opencv-python")
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics library is required for YOLO inference. Run: pip install ultralytics")
        return

    # 3. Resolve parameters
    conf = conf_thresh if conf_thresh is not None else config["inference"]["confidence_threshold"]
    iou = iou_thresh if iou_thresh is not None else config["inference"]["iou_threshold"]
    device = get_device(device_override or config["training"]["device"])
    class_names = config["dataset"].get("classes", ["tractor", "person"])

    # 4. Load Model
    logger.info(f"Loading YOLO model from: {w_path}...")
    try:
        model = YOLO(str(w_path))
    except Exception as exc:
        logger.error(f"Failed to initialize YOLO model: {exc}")
        return

    # 5. Open Video Capture Device
    logger.info(f"Attempting to open webcam device index [{camera_id}]...")
    cap = cv2.VideoCapture(camera_id)

    # Configure hardware resolution if supported
    webcam_cfg = config["inference"].get("webcam", {})
    width = webcam_cfg.get("frame_width", 1280)
    height = webcam_cfg.get("frame_height", 720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        logger.error(
            f"Unable to access camera index {camera_id}.\n"
            f"Possible causes:\n"
            f"  1. No webcam is connected to the machine.\n"
            f"  2. Another application (e.g. Zoom, Teams) is currently using the camera.\n"
            f"  3. The camera index is different (try running with --camera-id 1 or 2)."
        )
        return

    window_title = "John Deere Agricultural Object Detection - Live Stream"
    logger.info("=" * 60)
    logger.info("WEBCAM INFERENCE ACTIVE")
    logger.info(f"Resolution : {width}x{height}")
    logger.info(f"Device     : {device}")
    logger.info("Controls   : Press 'q' in the video window to exit cleanly")
    logger.info("=" * 60)

    prev_time = time.perf_counter()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Failed to grab frame from camera stream. Exiting loop.")
                break

            # Calculate running FPS
            current_time = time.perf_counter()
            dt = current_time - prev_time
            prev_time = current_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)  # Exponential moving average

            # Run inference
            try:
                results = model.predict(
                    source=frame,
                    conf=conf,
                    iou=iou,
                    device=device,
                    verbose=False
                )
            except Exception as inf_err:
                logger.error(f"Inference error on frame: {inf_err}")
                break

            res = results[0]
            boxes_out: List[List[float]] = []
            scores_out: List[float] = []
            classes_out: List[int] = []

            if res.boxes is not None and len(res.boxes) > 0:
                for box, score, cid in zip(
                    res.boxes.xyxy.cpu().numpy(),
                    res.boxes.conf.cpu().numpy(),
                    res.boxes.cls.cpu().numpy().astype(int)
                ):
                    boxes_out.append([float(c) for c in box])
                    scores_out.append(float(score))
                    classes_out.append(int(cid))

            # Render detections
            annotated = draw_detections(
                image=frame,
                boxes=boxes_out,
                scores=scores_out,
                class_ids=classes_out,
                class_names=class_names
            )

            # Draw HUD Status Bar
            hud_text = f"John Deere Agri Perception | FPS: {fps:.1f} | Detections: {len(boxes_out)} | 'q' to quit"
            cv2.rectangle(annotated, (10, 10), (700, 45), (0, 0, 0), -1)
            cv2.putText(
                annotated,
                hud_text,
                (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                lineType=cv2.LINE_AA
            )

            cv2.imshow(window_title, annotated)

            # Check for 'q' key to quit cleanly
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("User requested exit ('q' pressed).")
                break

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt caught. Shutting down webcam...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera resources released successfully.")


def main():
    parser = argparse.ArgumentParser(description="Real-Time Webcam Agricultural Object Detection")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--weights", type=str, default=None, help="Path to trained model weights (.pt)")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold (e.g. 0.25)")
    parser.add_argument("--iou", type=float, default=None, help="IoU threshold (e.g. 0.45)")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu', '0')")

    args = parser.parse_args()

    run_webcam_detection(
        camera_id=args.camera_id,
        weights_path=args.weights,
        config_path=args.config,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        device_override=args.device,
    )


if __name__ == "__main__":
    main()

