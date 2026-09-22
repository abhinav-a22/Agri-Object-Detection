"""
Model Training Module - John Deere Agricultural Object Detection
================================================================
Fine-tunes modern YOLO11 vision models (YOLO11s primary, YOLO11n baseline)
using Transfer Learning on agricultural object detection datasets.

Why Transfer Learning?
----------------------
Training deep convolutional vision architectures from scratch requires hundreds
of thousands of annotated images and weeks of high-end compute. In agricultural
applications where domain-specific datasets are typically hundreds or thousands of
images, transfer learning allows us to initialize our model with weights pretrained
on the diverse COCO dataset.

Early convolutional layers contain universal visual representations (edges, textures,
gradients, localized shapes), while deeper layers learn domain-specific object semantics.
By fine-tuning on agricultural imagery, the network adapts its higher-level representations
to field machinery, varied crop backgrounds, dust, and outdoor lighting conditions with
high sample efficiency and rapid convergence.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure project root is in system path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import setup_logger, load_config, set_seed, get_device

logger = setup_logger("Train")


def train_agricultural_model(
    config_path: str = "configs/config.yaml",
    model_override: Optional[str] = None,
    epochs_override: Optional[int] = None,
    batch_override: Optional[int] = None,
    device_override: Optional[str] = None,
    data_override: Optional[str] = None,
    imgsz_override: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Executes fine-tuning of YOLO11 for agricultural object detection.

    Args:
        config_path: Path to configuration YAML.
        model_override: Optional override for model weights (e.g. 'yolo11s.pt' or 'yolo11n.pt').
        epochs_override: Optional override for number of epochs.
        batch_override: Optional override for batch size.
        device_override: Optional override for compute device ('cpu', '0', etc.).
        data_override: Optional override for data.yaml path.
        imgsz_override: Optional override for input image resolution.

    Returns:
        Dictionary with training outcomes, saved weight paths, and metrics.
    """
    config = load_config(config_path)

    # 1. Resolve parameters (CLI overrides take precedence over config.yaml)
    data_yaml = data_override or config["dataset"]["data_yaml"]
    model_weights = model_override or config["model"]["pretrained_weights"]
    epochs = epochs_override or config["training"]["epochs"]
    batch_size = batch_override or config["training"]["batch_size"]
    imgsz = imgsz_override or config["dataset"]["img_size"]
    device = get_device(device_override or config["training"]["device"])
    seed = config["training"].get("seed", 42)
    workers = config["training"].get("workers", 0)

    # Resolve paths relative to project root
    data_yaml_path = (PROJECT_ROOT / data_yaml).resolve()
    if not data_yaml_path.is_file():
        raise FileNotFoundError(
            f"Dataset specification file not found at: {data_yaml_path}.\n"
            f"Run 'python src/preprocessing.py --create-sample' to generate test data "
            f"or follow instructions in README.md to download the full dataset."
        )

    # Enforce deterministic reproducibility
    set_seed(seed)

    logger.info("=" * 70)
    logger.info("JOHN DEERE AGRICULTURAL OBJECT DETECTION - MODEL TRAINING")
    logger.info("=" * 70)
    logger.info(f"Base Pretrained Model : {model_weights}")
    logger.info(f"Dataset Specification : {data_yaml_path}")
    logger.info(f"Target Epochs         : {epochs}")
    logger.info(f"Batch Size            : {batch_size}")
    logger.info(f"Image Resolution      : {imgsz}x{imgsz}")
    logger.info(f"Execution Device      : {device}")
    logger.info(f"Random Seed           : {seed}")
    logger.info("=" * 70)

    # 2. Verify Ultralytics availability
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error(
            "Ultralytics library is not installed in the active environment.\n"
            "Please install the required dependencies using:\n"
            "    pip install -r requirements.txt\n"
            "or:\n"
            "    pip install ultralytics torch torchvision"
        )
        raise SystemExit(1)

    # 3. Load model with pretrained weights for Transfer Learning
    logger.info(f"Loading pretrained weights '{model_weights}' for transfer learning...")
    model = YOLO(model_weights)

    # Extract agricultural augmentation hyperparameters
    aug = config.get("augmentation", {})
    runs_dir = (PROJECT_ROOT / "runs" / "train").resolve()

    # 4. Initiate training run
    logger.info("Starting model fine-tuning...")
    results = model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        workers=workers,
        seed=seed,
        project=str(runs_dir),
        name="agri_run",
        exist_ok=True,
        # Agricultural data augmentations
        hsv_h=aug.get("hsv_h", 0.015),
        hsv_s=aug.get("hsv_s", 0.7),
        hsv_v=aug.get("hsv_v", 0.4),
        degrees=aug.get("degrees", 10.0),
        translate=aug.get("translate", 0.1),
        scale=aug.get("scale", 0.5),
        fliplr=aug.get("fliplr", 0.5),
        flipud=aug.get("flipud", 0.0),
        mosaic=aug.get("mosaic", 1.0),
        mixup=aug.get("mixup", 0.0),
        # Training optimizations
        patience=config["training"].get("patience", 15),
        optimizer=config["training"].get("optimizer", "SGD"),
        lr0=config["training"].get("learning_rate", 0.01),
        lrf=config["training"].get("lr_final", 0.01),
        verbose=True,
    )

    # 5. Archive trained weights into models/
    models_dir = (PROJECT_ROOT / config["model"]["save_dir"]).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    exp_dir = runs_dir / "agri_run"
    best_weight_src = exp_dir / "weights" / "best.pt"
    last_weight_src = exp_dir / "weights" / "last.pt"

    best_target = models_dir / config["model"]["best_model_name"]
    last_target = models_dir / config["model"]["last_model_name"]

    if best_weight_src.is_file():
        shutil.copy2(best_weight_src, best_target)
        logger.info(f"Saved best model weights to: {best_target}")

    if last_weight_src.is_file():
        shutil.copy2(last_weight_src, last_target)
        logger.info(f"Saved final checkpoint weights to: {last_target}")

    # 6. Copy training curves and metric plots into outputs/plots/
    plots_dir = (PROJECT_ROOT / config["inference"]["output_plots_dir"]).resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)
    for plot_file in ["results.png", "confusion_matrix.png", "F1_curve.png", "PR_curve.png"]:
        src_plot = exp_dir / plot_file
        if src_plot.is_file():
            shutil.copy2(src_plot, plots_dir / plot_file)
            logger.info(f"Exported training curve: {plots_dir / plot_file}")

    logger.info("Model training pipeline completed successfully!")
    return {
        "best_weights": str(best_target) if best_target.is_file() else None,
        "last_weights": str(last_target) if last_target.is_file() else None,
        "run_directory": str(exp_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="Train YOLO11 on Agricultural Object Detection")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, default=None, help="YOLO weights (e.g., yolo11s.pt, yolo11n.pt)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cpu', '0', 'cuda:0')")
    parser.add_argument("--data", type=str, default=None, help="Path to data.yaml")
    parser.add_argument("--imgsz", type=int, default=None, help="Image size (e.g. 640)")

    args = parser.parse_args()

    train_agricultural_model(
        config_path=args.config,
        model_override=args.model,
        epochs_override=args.epochs,
        batch_override=args.batch_size,
        device_override=args.device,
        data_override=args.data,
        imgsz_override=args.imgsz,
    )


if __name__ == "__main__":
    main()

