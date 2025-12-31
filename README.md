# Traffic Analysis using Computer Vision

## Overview

Traffic monitoring is essential for urban planning, road safety, and congestion management. This project applies computer vision and deep learning to automatically detect, classify, count, track, and analyze vehicles from images and video streams.

Computer vision models process video frames to locate vehicles with bounding boxes, classify vehicle types, and feed detections into tracking and analytics modules to compute counts, density, and (optionally) speed estimates.

## Key Features

- **Vehicle detection and classification:** Localize vehicles and identify categories (car, truck, bus, motorcycle, etc.)
- **Vehicle counting and tracking:** Maintain identities across frames to count vehicles and generate trajectories
- **Traffic density / congestion analysis:** Compute per-frame and per-zone vehicle densities and congestion metrics
- **Speed estimation (optional):** Estimate vehicle speeds using trajectory displacement and camera calibration
- **Real-time or offline processing:** Supports batch processing of recorded video and streaming input for near-real-time monitoring

## Tech Stack

- **Languages:** Python
- **Computer vision:** OpenCV
- **Deep learning frameworks:** PyTorch (primary)
- **Pretrained models:** YOLO family (YOLOv8/YOLO11 variants)
- **Utilities:** NumPy, pandas, matplotlib, seaborn

## Model & Approach

- **Detection approach:** One-stage detectors (YOLO) are used for fast, accurate vehicle bounding box predictions.
- **Tracking approach:** Simple Online and Realtime Tracking (SORT) / DeepSORT (optional) or custom Kalman filter + Hungarian assignment for data association.
- **Model architecture:** YOLO variant. Exact architecture: YOLOv8n.
- **Training / fine-tuning:** Models can be fine-tuned on domain-specific traffic datasets. See `train.py` and `train_model.py` for training scripts and arguments. Training hyperparameters: epochs=50, batch_size=16, learning_rate=0.001.

## Dataset

- **Source:** Traffic camera footage and public datasets (e.g., UA-DETRAC, KITTI, COCO-vehicle subsets). Replace with your dataset in `data/my_dataset/`.
- **Format:** Images or videos. Labels follow YOLO-style text files per image (`label.txt` with `class x_center y_center width height`).
- **Annotations:** Bounding boxes with class ids. Example files are under `data/my_dataset/test/labels/`.

## Installation & Setup

### Prerequisites

- Python 3.8+ (recommend 3.10+)
- GPU with CUDA for training / fast inference (optional)

### Environment setup

Create and activate a virtual environment (example using venv):

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### Dependency installation

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

If you plan to use GPU acceleration for PyTorch, install the appropriate CUDA-enabled PyTorch build as documented at https://pytorch.org/.

## Usage

Run detection on an image:

```bash
python main.py --source data/input/images/example.jpg --weights trained_models/yolo_traffic/weights/best.pt --task detect
```

Run detection on a video:

```bash
python main.py --source data/input/videos/traffic.mp4 --weights trained_models/yolo_traffic/weights/best.pt --task detect --output output/videos/
```

Run tracking + counting (example):

```bash
python main.py --source data/input/videos/traffic.mp4 --weights trained_models/yolo_traffic/weights/best.pt --task track --tracker deep-sort --output output/json/
```

Train / fine-tune a model:

```bash
python train.py --data data/my_dataset/data.yaml --cfg config/config.yaml --weights yolov8n.pt --epochs 50
```

Replace CLI flags and script names with actual project options. See `config/` and the top of `main.py`, `train.py` for available arguments.

## Project Structure

- `main.py` — entrypoint for inference and pipeline orchestration
- `train.py` — training and fine-tuning scripts
- `src/` — core modules
  - `detector.py` — model loading and detection utilities
  - `tracker.py` — tracking logic (SORT / DeepSORT)
  - `analyzer.py` — analytics (counting, density, speed estimation)
  - `visualizer.py` — drawing bounding boxes, trajectories and summary plots
  - `exporter.py` — save outputs (CSV, JSON, annotated video)
  - `config_loader.py` — load config and hyperparameters
  - `utils.py` — helper utilities
- `data/` — input datasets and annotation folders
- `models/` & `trained_models/` — pretrained weights and experiment artifacts
- `output/` — inference outputs (videos, JSON, CSV, plots)

## Results & Visualizations

- Annotated frames with bounding boxes and labels are stored in `output/[trained_model]/videos/`.
- Detection outputs are exported as `output/[trained_model]/json/detections.json` and `output/[trained_model]/csv/statistics.csv`.
- Visual analytics (counts, density over time) can be plotted with `src/visualizer.py`.
- Example metrics:
  - FPS: 30
  - mAP: 0.7
  - Precision / Recall: 0.86 / 0.64 

## Limitations

- Occlusions and heavy traffic can reduce detection and tracking accuracy.
- Low-light or extreme weather degrades performance.
- Fixed camera angle assumptions may be required for accurate speed estimation and zone-based counting.
- Model bias: pretrained models may underperform on city-specific vehicles or rare classes.

## Future Enhancements

- Real-time deployment with optimized inference (TorchScript, ONNX, TensorRT)
- Edge / IoT integration for on-camera or gateway inference
- Smart traffic signal control integration using aggregated metrics
- Multi-camera fusion and cross-camera tracking
- Automated retraining pipeline with active learning from new annotations

## License & Acknowledgements

- License: MIT
- Acknowledgements: pretrained model authors (YOLO), public datasets (UA-DETRAC, KITTI), and open-source libraries (PyTorch, OpenCV).

---
