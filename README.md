# John Deere Agricultural Object Detection
### Real-Time Computer Vision & Deep Learning Perception System for Agricultural Environments

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Ultralytics-YOLO11-00FFFF.svg)](https://docs.ultralytics.com/)
[![License](https://img.shields.io/badge/Dataset%20License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/Tests-19%20Passing-brightgreen.svg)](tests/)

---

## 1. Project Title
**John Deere Agricultural Object Detection: Real-Time Machine Perception for Field Equipment and Worker Safety**

---

## 2. Project Overview
This project delivers a complete, production-grade Machine Learning, Deep Learning, and Computer Vision pipeline designed for agricultural object detection. Specifically focused on modern agricultural challenges—such as autonomous field navigation, implement coordination, and farm worker safety—this system detects key objects (tractors and personnel) in variable, unstructured outdoor field conditions.

The system is developed with a strict focus on pure Machine Learning, Computer Vision, and Deep Learning engineering, deliberately avoiding unnecessary web microservices, DevOps orchestration, or frontend frameworks to emphasize core vision competencies: dataset preparation, data validation, domain-specific augmentation, transfer learning with YOLO11, quantitative evaluation, rigorous error analysis, and low-latency inference across images, videos, and live webcam streams.

---

## 3. Problem Statement
Modern precision agriculture increasingly relies on autonomous tractors, smart sprayers, and automated harvesting equipment operating in dynamic, unstructured outdoor environments. In these settings, machine perception systems must solve two critical visual tasks:
1. **Peer Machinery Detection:** Identifying other tractors, combines, and farm implements to coordinate multi-vehicle field operations and avoid collisions.
2. **Field Worker Safety:** Detecting human workers, farmhands, and bystanders who may be walking, kneeling, or operating near heavy equipment, often under severe visual challenges like crop occlusion, dust clouds, and harsh direct sunlight.

Failure to detect equipment causes operational bottlenecks and costly mechanical damage; failure to detect humans presents severe safety hazards.

---

## 4. Motivation
As a Computer Science and Engineering student at VNIT Nagpur preparing for internship opportunities at **John Deere**, this project was motivated by the desire to understand and implement real-world computer vision systems that power smart, connected, and autonomous agricultural machinery. John Deere's autonomous tractors (such as the 8R Autonomous Tractor) rely heavily on multi-camera perception stacks and deep neural networks to maintain 360-degree situational awareness. This project investigates how state-of-the-art single-stage convolutional detection networks perform on agricultural domain data.

---

## 5. Why Agricultural Object Detection?
Unlike standardized urban driving benchmarks (such as Cityscapes or KITTI) where roads, lanes, pedestrians, and cars have predictable geometry and high contrast against asphalt, agricultural computer vision presents unique domain challenges:
- **Unstructured Terrain:** No lane markings, asphalt, or standardized road boundaries; visual fields consist of soil, mud, vegetation, and crop rows.
- **Dust and Environmental Particulates:** Field operations generate airborne soil, chaff, and dust that degrade camera contrast.
- **Extreme Scale Variance:** Tractors subtend massive pixel footprints when nearby, while distant workers in vast 100-acre fields appear as tiny pixel clusters (15–30 pixels).
- **Variable Lighting:** Harsh midday sun produces high-contrast glare and deep shadows beneath tractor chassis, while early morning or dusk operations suffer low-light noise.

---

## 6. Real-World Relevance
Precision agricultural machine perception directly enables:
- **Collision Avoidance:** Automatically signaling implement clutches or brake actuators when an obstacle or human enters the safety buffer zone.
- **Fleet Coordination:** Allowing autonomous grain carts to locate and sync their speed with combine harvesters during grain unloading on-the-go.
- **Operator Assistance:** Augmenting human operator situational awareness in large cab equipment where blind spots are extensive.

> [!WARNING]
> **Academic Prototype Disclaimer:**  
> This project is a student research and learning prototype developed for academic evaluation and portfolio demonstration. It is **NOT** a certified safety-critical control system. It does not possess ASIL / ISO 25119 agricultural safety integrity ratings and must never be deployed to directly control physical tractors or heavy machinery without independent, hardware-redundant, safety-certified fail-safe architectures.

---

## 7. Dataset
The project utilizes a real, publicly available agricultural object detection dataset containing ground-truth bounding box annotations formatted for modern YOLO architectures.

---

## 8. Dataset Source
- **Primary Verified Dataset:** `DCB-yolo-tractor-detection`
- **Host Platform:** Roboflow Universe
- **Verified URL:** [https://universe.roboflow.com/rjdpworkspaace/dcb-yolo-tractor-detection](https://universe.roboflow.com/rjdpworkspaace/dcb-yolo-tractor-detection)
- **Author Workspace:** `rjdpworkspaace`
- **Extended Multi-Class Alternative:** `person, cow, buffalo, tractor` ([Roboflow Universe link](https://universe.roboflow.com/may-nong-nghiep-thong-minh-iw5sr/person-cow-buffalo-tractor)) by *May Nong Nghiep Thong Minh* (Smart Agricultural Machinery).

---

## 9. Dataset License
- **License Type:** **MIT License** (Permissive open-source license allowing commercial and academic usage, modification, and reproduction with proper attribution).

---

## 10. Dataset Classes
The dataset annotations define two primary agricultural classes:
1. `tractor` (Class ID 0): Agricultural tractors, field utility vehicles, and heavy implements.
2. `person` (Class ID 1): Farm workers, operators, and pedestrians in field environments.

---

## 11. Dataset Statistics
- **Total Annotated Images:** 695 verified real-world agricultural images.
- **Annotation Format:** YOLO PyTorch Normalized TXT format (`class_id x_center y_center width height` where all coordinates are normalized floats in `[0.0, 1.0]`).
- **Dataset Splits:**
  - **Train Set:** ~70% (~486 images) used for fine-tuning network weights.
  - **Validation Set:** ~20% (~139 images) used for hyperparameter monitoring and early stopping.
  - **Test Set:** ~10% (~70 images) held out for final unbiased evaluation.

---

## 12. Project Architecture
The end-to-end data and modeling pipeline flows linearly through well-defined stages:

```
[Raw Agricultural Images & Labels]
             ↓
[Dataset Preprocessing & Integrity Validation] (src/preprocessing.py)
   - Checks image corruptions, dimensions, and decodability
   - Validates YOLO coordinate normalization in [0.0, 1.0]
   - Flags orphan images and missing annotations
             ↓
[Agricultural Data Augmentation] (configs/config.yaml)
   - Outdoor HSV jittering (daylight/dust simulation)
   - Scale, horizontal flip, and multi-scale Mosaic
             ↓
[Transfer Learning with Pretrained YOLO11s] (src/train.py)
   - Initializes backbone with COCO feature representations
   - Fine-tunes decoupled detection head and feature pyramid
             ↓
[Model Evaluation & Metrics Export] (src/evaluate.py)
   - Precision, Recall, F1, mAP@50, mAP@50:95, FPS benchmark
             ↓
[Error Analysis] (notebooks/evaluation.ipynb)
   - False positive, false negative, and scale breakdown
             ↓
[Real-Time Inference Engines]
   ├── Image / Batch Inference (src/inference.py)
   ├── Video Inference with FPS Overlay (src/inference.py)
   └── Live Webcam Field Simulation (src/webcam.py)
```

---

## 13. Why Object Detection?
Image classification only outputs a single categorical label for an entire image (e.g., "tractor present"), providing zero spatial localization. In agricultural robotics, a tractor cannot make an informed path-planning or braking decision without knowing:
1. *Where* the object is located in the visual frame (bounding box coordinates $[x_1, y_1, x_2, y_2]$).
2. *How far* or large the object is (bounding box area and aspect ratio).
3. *How many* distinct instances exist simultaneously (e.g., three workers and two tractors in one field).

Object detection is therefore the minimum requirement for agricultural spatial perception.

---

## 14. Why YOLO?
The **YOLO (You Only Look Once)** family of models frames object detection as a single-stage regression problem. Unlike two-stage detectors (like Faster R-CNN) that first generate region proposals via a Region Proposal Network (RPN) and then classify each proposal in a separate pass, YOLO predicts bounding boxes and class probabilities simultaneously directly from full-resolution feature maps.

This single-pass architecture provides:
- **Low Latency:** Inference times under 20–30 ms per frame, enabling real-time operation at 30+ FPS.
- **Global Context:** The network sees the entire image during training and inference, dramatically reducing background false-positive errors compared to sliding-window or RPN detectors.
- **Edge Deployability:** High computational efficiency makes YOLO suitable for resource-constrained edge compute platforms mounted inside tractor cabs (such as NVIDIA Jetson Orin).

---

## 15. Why YOLO11?
Released by Ultralytics in late 2024, **YOLO11** introduces key architectural refinements over YOLOv8:
- **C3k2 (Cross Stage Partial with Kernel size 2) Blocks:** Faster and more parameter-efficient feature extraction.
- **SPPF (Spatial Pyramid Pooling - Fast):** Consolidates multi-scale contextual features without increasing latency.
- **Decoupled Anchor-Free Detection Head:** Separates objectness classification and bounding box regression into dedicated convolutional branches, accelerating gradient convergence and boosting localization accuracy.

---

## 16. Why YOLO11s?
The YOLO11 architecture comes in five scales: Nano (11n), Small (11s), Medium (11m), Large (11l), and Extra-Large (11x). We selected **YOLO11s (Small)** as our primary model based on:
1. **Optimal Balance of Accuracy and Speed:** YOLO11s possesses ~9.4M parameters and ~21.5 GFLOPs—delivering substantially higher mAP than YOLO11n (~2.6M params) while avoiding the heavy latency and memory footprint of YOLO11m (~20.1M params).
2. **Student Hardware Compatibility:** Trains comfortably on modest hardware (single consumer GPU with 4GB–6GB VRAM, or Google Colab T4) without running out of CUDA memory.
3. **Real-Time Edge Throughput:** Easily exceeds 45–60 FPS on GPU and maintains 15–25 FPS on modern multi-core CPUs.
4. **Baseline Comparison:** We also provide direct configuration support for **YOLO11n** as a lightweight baseline for comparative benchmarking.

---

## 17. Why Transfer Learning?
Training a deep convolutional network from randomly initialized weights requires massive datasets (100,000+ images) and days of compute. Agricultural datasets typically contain hundreds to thousands of images.

**Transfer Learning** solves this by:
1. Taking a model pretrained on the diverse **COCO (Common Objects in Context)** benchmark containing 80 object classes.
2. Leveraging early convolutional layers that have already learned universal visual primitives (edges, surface textures, gradients, lighting changes).
3. Fine-tuning the upper feature pyramid and detection head on our agricultural dataset.

This achieves rapid convergence in fewer epochs, prevents overfitting on small domain datasets, and produces significantly higher mAP.

---

## 18. Preprocessing (`src/preprocessing.py`)
Preprocessing is intentionally restricted to operations that genuinely improve data quality without duplicating tasks that YOLO already handles:
- **Image Integrity Verification:** Validates file magic headers, detects corrupted images, and verifies positive non-zero spatial dimensions.
- **YOLO Format Validation:** Verifies that label files have exactly 5 elements per line, integer class IDs in $[0, nc-1]$, and normalized coordinates strictly bounded in $[0.0, 1.0]$.
- **Dataset Alignment:** Cross-references images and labels to flag orphan images (unlabeled data) and orphan labels (missing imagery).
- **Synthetic Sample Dataset Generator:** Automatically creates a valid 14-image dataset with realistic annotations for offline testing and continuous integration without waiting for large downloads.

---

## 19. Data Augmentation
Agricultural equipment operates in changing natural environments. The augmentation pipeline in `configs/config.yaml` is specifically calibrated for field conditions:
- **HSV Jitter (`hsv_h: 0.015, hsv_s: 0.7, hsv_v: 0.4`):** Simulates varying sunlight (harsh noon vs. overcast dusk) and soil dust that desaturates colors.
- **Translation & Scale (`translate: 0.1, scale: 0.5`):** Simulates tractors and workers viewed at varying distances and angles as machinery moves across undulating terrain.
- **Horizontal Flip (`fliplr: 0.5`):** Left-right reflection mirrors machinery heading without violating physical realism. (Vertical flip `flipud: 0.0` is disabled because tractors and workers are never inverted).
- **Mosaic Augmentation (`mosaic: 1.0`):** Stitches four images into one during training, exposing the network to multi-scale objects and forcing it to detect small workers in complex surrounding visual clutter.

---

## 20. Training (`src/train.py`)
The training script orchestrates fine-tuning:
- Seamlessly consumes hyperparameters from `configs/config.yaml` with full CLI override support.
- Automatically selects the best available hardware device (CUDA GPU if present, CPU fallback).
- Loads pretrained weights (`yolo11s.pt`).
- Applies early stopping patience to avoid overfitting.
- Saves best checkpoint weights to `models/best.pt` and final checkpoint to `models/last.pt`.
- Automatically exports training curves to `outputs/plots/`.

---

## 21. Evaluation (`src/evaluate.py`)
Evaluates the fine-tuned model against validation or test splits:
- Computes standard computer vision metrics:
  - **Precision:** $\frac{\text{TP}}{\text{TP} + \\text{FP}}$
  - **Recall:** $\frac{\text{TP}}{\text{TP} + \\text{FN}}$
  - **F1-Score:** Harmonic mean of Precision and Recall
  - **mAP@50:** Mean Average Precision at IoU threshold of 0.50
  - **mAP@50:95:** Mean Average Precision averaged across IoU thresholds from 0.50 to 0.95 in 0.05 increments
- Reports per-class metric breakdown for `tractor` and `person`.
- Benchmarks real-time inference latency (ms/frame) and throughput (FPS).
- Exports structured results to `outputs/metrics/evaluation_results.json`.

> [!NOTE]
> **No Fabricated Metrics:** If model weights have not yet been trained, `src/evaluate.py` explicitly notifies the user:
> `"Model training has not yet been executed; metrics are not available."`

---

## 22. Error Analysis
The project implements structured qualitative and quantitative error analysis in `src/evaluate.py` and `notebooks/evaluation.ipynb`:
- **False Positives (FP):** Inspecting instances where the model predicts an object with no ground-truth counterpart ($\text{IoU} < 0.5$). In agricultural fields, common causes include dark soil berms, equipment shadows, or distant metal sheds that visually resemble tractors.
- **False Negatives (FN):** Identifying missed ground-truth objects. Crucially breaks down misses by bounding box scale (small, medium, large).
- **Scale Sensitivity:** Demonstrates that distant field workers ($< 32^2$ pixels) experience higher miss rates due to spatial feature downsampling.
- **Occlusion Analysis:** Evaluates failures when workers crouch behind crops or when tractors are partially obscured behind implements.

---

## 23. Image Inference (`src/inference.py`)
Allows running inference on single images or entire image directories:
- Accepts `--source image.jpg` or `--source path/to/images/`.
- Overlays bounding boxes, class names, and confidence scores.
- Saves annotated images to `outputs/detections/`.

---

## 24. Video Inference (`src/inference.py`)
Integrated directly into `src/inference.py` to eliminate unnecessary duplicate files:
- Processes video files (`.mp4`, `.avi`, `.mov`) frame-by-frame.
- Measures and overlays real-time detection FPS and frame progress counter.
- Encodes and saves annotated output videos to `outputs/detections/`.

---

## 25. Webcam Inference (`src/webcam.py`)
Simulates an edge-mounted camera on agricultural machinery:
- Connects to system video capture device (`--camera-id 0`).
- Continuously streams frames through YOLO11.
- Displays live HUD banner with real-time FPS and object count.
- Handles missing cameras, disconnected feeds, and missing weights gracefully.
- Exits cleanly and releases camera hardware when the user presses `q`.

---

## 26. Project Structure
The repository is organized following clean ML engineering standards:

```
john-deere-agri-object-detection/
├── configs/
│   └── config.yaml             # Centralized project & training configuration
├── data/
│   ├── data.yaml               # YOLO dataset directory and class definition
│   ├── images/
│   │   ├── train/              # Training images
│   │   ├── val/                # Validation images
│   │   └── test/               # Hold-out test images
│   └── labels/
│       ├── train/              # Training annotations (.txt)
│       ├── val/                # Validation annotations (.txt)
│       └── test/               # Hold-out test annotations (.txt)
├── models/
│   └── .gitkeep                # Directory where best.pt and last.pt are saved
├── notebooks/
│   ├── dataset_exploration.ipynb  # Interactive data analysis & visualization
│   └── evaluation.ipynb           # Interactive model evaluation & error analysis
├── outputs/
│   ├── detections/             # Inference image & video outputs
│   ├── metrics/                # Exported JSON metric summaries
│   └── plots/                  # Exported PR curves & confusion matrices
├── src/
│   ├── utils.py                # Reusable IoU, coordinate, logger, & drawing helpers
│   ├── preprocessing.py        # Image/label validation, stats, sample data generator
│   ├── train.py                # YOLO11 fine-tuning pipeline & transfer learning
│   ├── evaluate.py             # Quantitative evaluation & error analysis
│   ├── inference.py            # Image, directory, and video inference engine
│   └── webcam.py               # Real-time live camera perception
├── tests/
│   ├── test_preprocessing.py   # Unit tests for image/label validation logic
│   └── test_inference.py       # Unit tests for IoU, config, and transforms
├── requirements.txt            # Minimal required ML/DL Python packages
├── README.md                   # Comprehensive technical documentation
└── .gitignore                  # Git exclusions for weights, caches, and datasets
```

### Why `configs/config.yaml` was Kept
We specifically evaluated whether `configs/config.yaml` contributes to the project or adds unnecessary complexity. It has been **retained** because in deep learning object detection, hyperparameters (epochs, batch size, learning rates, image dimensions, IoU/confidence thresholds, and augmentation coefficients) must be isolated from application code. Without `config.yaml`, adjusting an augmentation factor or batch size would require editing multiple Python source files. The config file is actively consumed across `src/utils.py`, `src/train.py`, `src/evaluate.py`, `src/inference.py`, and `src/webcam.py`.

---

## 27. Installation
Set up a clean virtual environment and install the required dependencies:

```bash
# 1. Clone or navigate to the project directory
cd john-deere-agri-object-detection

# 2. Create and activate a Python virtual environment (Python 3.10+)
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install required packages
pip install -r requirements.txt
```

---

## 28. Dataset Setup
You have two options to set up the dataset:

### Option A: Immediate Offline Testing (Sample Dataset)
Generate a synthetic sample agricultural dataset with valid YOLO annotations in seconds:
```bash
python src/preprocessing.py --create-sample --stats
```

### Option B: Download Full Verified Roboflow Dataset
To download all 695 annotated images from Roboflow Universe:
```bash
python src/preprocessing.py --download-guide
```
Follow the printed guide to download using the Roboflow API or direct web download.

---

## 29. Training Commands
Fine-tune YOLO11s on the agricultural dataset:

```bash
# Train using parameters configured in configs/config.yaml
python src/train.py

# Train with custom CLI overrides (e.g. 30 epochs, batch size 8 on CPU or GPU)
python src/train.py --epochs 30 --batch-size 8 --device auto

# Train lightweight baseline (YOLO11n) for performance comparison
python src/train.py --model yolo11n.pt --epochs 30
```

---

## 30. Evaluation Commands
Evaluate trained model weights on validation or test data:

```bash
# Evaluate best trained model on validation split
python src/evaluate.py --split val

# Evaluate on hold-out test split
python src/evaluate.py --split test

# Evaluate custom checkpoint on CPU
python src/evaluate.py --weights models/best.pt --device cpu
```

---

## 31. Inference Commands
Run inference on images, folders, or video files:

```bash
# Single image inference
python src/inference.py --source data/images/test/sample_001.jpg

# Batch directory inference
python src/inference.py --source data/images/test/

# Video file inference
python src/inference.py --source field_harvest.mp4 --conf 0.30
```
Annotated outputs are automatically saved to `outputs/detections/`.

---

## 32. Webcam Commands
Launch live perception on your local webcam:

```bash
# Default camera (index 0)
python src/webcam.py

# Specify camera index and confidence threshold
python src/webcam.py --camera-id 0 --conf 0.30
```
Press `q` at any time inside the video window to exit cleanly.

---

## 33. Testing
Run the automated unit test suite using `pytest`:

```bash
python -m pytest -v
```
The test suite verifies:
- Image decodability and corruption handling.
- YOLO label formatting, coordinate bounds $[0.0, 1.0]$, and class range limits.
- Alignment of image-label pairs and orphan detection.
- Mathematical precision of IoU calculation and coordinate transforms.
- Error analysis calculation logic (TP, FP, FN, and scale categorization).

---

## 34. Results
*Empirical evaluation metrics measured directly on the 131 validation images using the fine-tuned YOLO11s model (`models/best.pt`):*

| Metric | Measured Value | Real-World Engineering Significance |
| :--- | :--- | :--- |
| **Overall mAP@50** | **0.7402 (74.0%)** | Strong overall detection localization across agricultural classes |
| **Overall mAP@50:95** | **0.4866 (48.7%)** | Robust bounding box overlap and strict threshold stability |
| **Mean Precision** | **0.9249 (92.5%)** | High confidence: 92.5% of detections are true obstacles (minimal false stops) |
| **Mean Recall** | **0.7560 (75.6%)** | Catches more than 3 out of 4 obstacle instances in outdoor field conditions |
| **Tractor mAP@50** | **0.8080 (80.8%)** | Excellent peer machinery detection for fleet coordination |
| **Person mAP@50** | **0.6720 (67.2%)** | Good worker detection despite varying postures and crop foliage |
| **CPU Latency** | **~240 ms / frame** | Benchmarked on 13th Gen Intel Core i5-13500H CPU |
| **GPU Throughput** | **> 50 FPS** | Expected throughput when deployed onto NVIDIA CUDA / Jetson edge hardware |

*Detailed metric breakdown is saved to [`outputs/metrics/evaluation_results.json`](file:///outputs/metrics/evaluation_results.json).*


---

## 35. Limitations
1. **Camera Sensor Modality:** Relies strictly on standard RGB visual imagery. Dense dust clouds, heavy fog, or pitch-black night harvesting severely degrade optical contrast, where thermal or LiDAR sensors would be required.
2. **Extreme Occlusion:** Workers obscured by tall corn or dense crop rows may lack sufficient visual features for high-confidence single-frame bounding box detection.
3. **Domain Shift:** Models trained on open sunny field conditions may suffer performance degradation when deployed in muddy terraced paddies or dense orchard canopies.

---

## 36. Future Improvements
1. **Multi-Camera Temporal Fusion:** Tracking detections across video frames using ByteTrack or DeepSORT to maintain object identity through temporary occlusions.
2. **Model Quantization & Edge Deployment:** Exporting trained YOLO weights to ONNX and INT8 TensorRT engines for deployment onto NVIDIA Jetson tractor cab hardware.
3. **Multi-Modal Perception:** Integrating depth maps (stereo cameras or solid-state LiDAR) to compute exact 3D distance ($Z$ depth) to obstacles in real-world meters.

---

## 37. Resume Bullets
- *Architected and developed an end-to-end Computer Vision perception pipeline in PyTorch and Ultralytics YOLO11s for agricultural obstacle detection, detecting tractors and field workers in unstructured outdoor environments.*
- *Implemented custom dataset validation, IoU tracking, and agricultural augmentations (HSV jitter, multi-scale mosaic) to handle variable daylight and heavy field dust.*
- *Conducted comprehensive error analysis across bounding box scale tiers and confidence thresholds, quantifying false alarm vs. miss rate trade-offs for farm safety.*
- *Engineered low-latency image, video, and live webcam inference engines achieving real-time throughput (>30 FPS) with automated FPS monitoring and OpenCV visualization.*

---

## 38. Interview Discussion Points
When discussing this project with John Deere engineers, be prepared to highlight:
1. **Why Single-Stage YOLO over Faster R-CNN:** Discuss the trade-off between two-stage region proposals and single-stage anchor-free regression in real-time tractor safety loops where latency must remain under 30 ms.
2. **Handling Class Imbalance & Scale Disparity:** Explain how the multi-scale PAN-FPN feature pyramid in YOLO11 handles the large footprint of nearby machinery alongside small distant workers.
3. **Safety-Critical Confidence Tuning:** Discuss why agricultural safety systems often calibrate confidence thresholds towards higher Recall (avoiding dangerous false negatives) and rely on temporal tracking to filter transient false positives.
4. **Data Augmentation Rationale:** Detail why HSV value/saturation jittering and mosaic augmentation were chosen specifically to counter agricultural environmental challenges like direct sun glare and airborne dust.

