# How to Train a Model in AI-Forge

This guide walks you through preparing your dataset, configuring the stack, and training a YOLO object detection model entirely from the AI-Forge web UI.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Dataset Preparation](#2-dataset-preparation)
3. [Quick Start with Pascal VOC (Public Dataset)](#3-quick-start-with-pascal-voc-public-dataset)
4. [Making Your Dataset Accessible to the Trainer](#4-making-your-dataset-accessible-to-the-trainer)
5. [Starting the Stack](#5-starting-the-stack)
6. [Opening the Training Page](#6-opening-the-training-page)
7. [Filling in the Training Form](#7-filling-in-the-training-form)
8. [Monitoring Live Log Output](#8-monitoring-live-log-output)
9. [Viewing Results in MLflow](#9-viewing-results-in-mlflow)
10. [Using the Trained Model for Inference](#10-using-the-trained-model-for-inference)
11. [Model Export Options](#11-model-export-options)
12. [GPU Training](#12-gpu-training)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Prerequisites

Before training, make sure the following are in place:

| Requirement | Check |
|---|---|
| Docker + Docker Compose installed | `docker compose version` |
| Full AI-Forge stack built and running | `docker compose ps` — all services `healthy` |
| Dataset in Ultralytics YAML format | See [Section 2](#2-dataset-preparation) |
| Dataset files accessible inside the trainer container | See [Section 3](#3-making-your-dataset-accessible-to-the-trainer) |

The services you will interact with:

| Service | URL | Purpose |
|---|---|---|
| **AI-Forge UI** | `http://localhost:3000` | Training form and log viewer |
| **MLflow UI** | `http://localhost:5000` | Experiment and metric tracking |

---

## 2. Dataset Preparation

The training pipeline uses the **Ultralytics dataset YAML format**.

### Directory structure

Organize your dataset on the host like this:

```
data/
├── dataset.yaml          ← the config file you will reference in the UI
├── images/
│   ├── train/            ← training images (.jpg / .png)
│   └── val/              ← validation images
└── labels/
    ├── train/            ← YOLO .txt label files (one per image)
    └── val/
```

### dataset.yaml format

```yaml
# dataset.yaml — Ultralytics format
path: /data            # root directory INSIDE the trainer container
train: images/train    # relative to 'path'
val:   images/val      # relative to 'path'

nc: 3                  # number of classes
names:
  0: cat
  1: dog
  2: person
```

> **Key rule:** The `path` field must be an **absolute path as seen from inside the trainer container** — not the host path.
> If you mount your host `./data` folder to `/data` inside the container (see Section 3), then `path: /data` is correct.

### Label format (per image, one .txt file)

Each line = one object:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are **normalized** (0.0 – 1.0) relative to image width/height. Example:

```
0 0.512 0.374 0.210 0.305
1 0.128 0.680 0.095 0.140
```

---

## 3. Quick Start with Pascal VOC (Public Dataset)

If you don't have your own dataset yet, the repository ships a ready-made YAML for **Pascal VOC** — a classic 20-class object detection benchmark (~2.8 GB).

> **Reference:** [Ultralytics VOC Dataset Docs](https://docs.ultralytics.com/datasets/detect/voc/)

### What is Pascal VOC?

| Attribute | Details |
|---|---|
| Classes | 20 (aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor) |
| Train images | 16,551 (VOC2007 + VOC2012 train/val splits) |
| Validation images | 4,952 (VOC2007 test split) |
| Annotation format | Pascal VOC XML → converted to YOLO .txt automatically |
| Download size | ~2.8 GB (3 ZIP archives from Ultralytics GitHub assets) |

### File location

```
data/VOC.yaml          ← in the repo (host)
/data/VOC.yaml         ← inside the trainer container (after volume mount)
```

### Steps

**1. The data volume mount is pre-configured** in `docker-compose.yaml`:

```yaml
trainer:
  volumes:
    - "./train_service:/app"
    - "./data:/data"          # already present — exposes data/VOC.yaml at /data/VOC.yaml
```

**2. Rebuild the trainer** (needed once after any compose change):

```bash
docker compose up -d --build trainer
```

**3. Verify the YAML is visible:**

```bash
docker exec trainer cat /data/VOC.yaml | head -5
```

**4. Open the Training page** (`http://localhost:3000/training`) and enter:

| Field | Value |
|---|---|
| **Dataset YAML** | `/data/VOC.yaml` |
| **Model** | `yolov8n.pt` (fastest for a first run) |
| **Epochs** | `10` (enough to see the pipeline work) |
| **Experiment Name** | `voc_quickstart` |

**5. Click Start Training.**

Ultralytics detects that `/data/VOC` does not exist and automatically runs the embedded download + conversion script before starting training. Download progress appears in the live log viewer.

> The first run will show lines like:
> ```
> Downloading VOCtrainval_06-Nov-2007.zip to /data/VOC/tmp ...
> VOC2012/train: 100%|████| 8218/8218
> VOC2007/train: 100%|████| 2501/2501
> ```
> This is normal — subsequent runs skip the download.

---

## 4. Making Your Dataset Accessible to the Trainer

The trainer container reads datasets from `/data` inside the container. This directory is already bind-mounted from the host's `./data/` folder in `docker-compose.yaml`:

```yaml
trainer:
  volumes:
    - "./train_service:/app"
    - "./data:/data"        # ← pre-configured
```

**To add your own dataset**, place it (or symlink it) inside `./data/` on the host:

```
AI-Forge/
└── data/
    ├── VOC.yaml              ← shipped with repo
    ├── my_dataset.yaml       ← add your own YAML here
    └── my_dataset/
        ├── images/
        └── labels/
```

The container path `/data` must match the `path:` field in your `dataset.yaml`.
Rebuild once after adding new datasets to pick up changes:

```bash
docker compose up -d --build trainer
```

### Option B — Copy data into the running container (quick test)

If you don't want to edit `docker-compose.yaml`, copy files directly:

```bash
docker cp ./data/. trainer:/data/
```

> This is temporary — data disappears when the container is recreated. Use Option A for repeated training.

### Verify the data is visible

```bash
docker exec trainer ls /data
# Expected output: dataset.yaml  images/  labels/
```

---

## 5. Starting the Stack

If the stack is not already running:

```bash
docker compose up -d --build
```

Wait for all services to become healthy:

```bash
docker compose ps
```

All services should show `(healthy)` or `Up`. The trainer service typically takes 15–20 seconds on first start as it initialises uvicorn.

---

## 6. Opening the Training Page

1. Open your browser and go to `http://localhost:3000`
2. Click **Training** in the left sidebar (or the **Training** card on the Dashboard)
3. You will see the training form on the left and an empty job panel on the right

---

## 7. Filling in the Training Form

### Field reference

#### Dataset YAML * (required)

```
/data/dataset.yaml
```

Enter the **absolute path to your dataset config file as it appears inside the trainer container**.
This is the most important field — training will fail immediately if this path is wrong.

> Tip: Run `docker exec trainer ls /data` to confirm the path before submitting.

---

#### Model

Select a YOLO preset from the dropdown. Ultralytics will automatically download the weights on first use if they are not already cached.

| Size | Preset | Description |
|------|--------|-------------|
| Nano | `yolov8n.pt` | Fastest, least accurate — good for quick experiments |
| Small | `yolov8s.pt` | Good balance for edge devices |
| Medium | `yolov8m.pt` | Recommended for most use cases |
| Large | `yolov8l.pt` | Higher accuracy, more VRAM |
| XLarge | `yolov8x.pt` | Maximum accuracy, slowest |

YOLOv10 (`yolov10n/s/m/b/l/x.pt`) and YOLO11 (`yolo11n/s/m/l/x.pt`) variants are also available.

> For a first run, use **`yolov8n.pt`** (nano) to verify your dataset loads correctly before committing to a long training job.

---

#### Epochs

Number of full passes through the training dataset.

| Use case | Recommended |
|---|---|
| Quick sanity check | 3–5 |
| Typical training run | 50–100 |
| Fine-tuning a specific domain | 100–300 |

Default: `50`

---

#### Image Size

The input resolution the model will train and infer at (square, in pixels). Must be a multiple of 32.

| Value | Use case |
|---|---|
| `320` | Fast experiments, resource-constrained |
| `640` | Standard — best balance of speed and accuracy |
| `1280` | High-resolution datasets (requires more VRAM) |

Default: `640`

---

#### Batch Size

Number of images processed per gradient update step. Larger batches are faster but require more memory.

| Device | Practical range |
|---|---|
| CPU | 4–16 |
| GPU (8 GB VRAM) | 16–32 |
| GPU (24 GB VRAM) | 64–128 |

If you see **out-of-memory errors** in the log, reduce batch size.
Default: `16`

---

#### Device

| Value | Meaning |
|---|---|
| `cpu` | Use CPU (slow but universally available) |
| `0` | Use GPU 0 (first GPU) |
| `1` | Use GPU 1 (second GPU) |
| `0,1` | Multi-GPU training across GPU 0 and GPU 1 |

Default: `cpu`

> See [Section 11](#11-gpu-training) for GPU setup requirements.

---

#### Export Format

Optional post-training export. The trained `.pt` checkpoint is always saved; this adds an additional deployment-optimised artifact.

| Format | Use case | Requirement |
|---|---|---|
| `— none —` | Skip export (default) | — |
| `onnx` | Cross-platform serving (ONNX Runtime, OpenVINO) | None |
| `engine` | NVIDIA TensorRT (fastest GPU inference) | NVIDIA GPU + TensorRT |
| `torchscript` | Portable PyTorch format | None |
| `coreml` | Apple devices (iOS, macOS) | macOS |
| `saved_model` | TensorFlow / TFLite serving | None |

---

#### Experiment Name

MLflow groups training runs under experiments. Use a descriptive name so you can filter results later.

Examples: `coco_baseline`, `custom_detector_v1`, `yolov10_comparison`

Default: `yolo_training`

---

#### Run Name (Optional)

A human-readable label for this specific run within the experiment.

Examples: `batch32-lr0.01`, `aug-heavy`, `pretrained-yolov8m`

Leave blank to let MLflow auto-generate a name.

---

#### Register Model As (Optional)

If filled in, the best checkpoint (`best.pt`) will be registered in the **MLflow Model Registry** under this name after training. The inference service can then load it automatically.

Example: `yolo_detector`

Leave blank to skip registration.

---

### Example: minimal first run

| Field | Value |
|---|---|
| Dataset YAML | `/data/dataset.yaml` |
| Model | `yolov8n.pt` |
| Epochs | `5` |
| Image Size | `640` |
| Batch Size | `8` |
| Device | `cpu` |
| Export Format | `— none —` |
| Experiment Name | `first_test` |
| Run Name | `sanity-check` |
| Register Model As | _(blank)_ |

---

### Submitting the job

Click **Start Training**.

- The button changes to **Training in progress…** and becomes disabled
- A **job card** appears on the right with a blue `RUNNING` badge
- The log viewer starts showing output within a few seconds

---

## 8. Monitoring Live Log Output

While the job is running the log panel on the right shows the last 50 lines of combined stdout/stderr from the training process, refreshed every 2 seconds.

### What to look for

```
{"time": "...", "level": "INFO", ..., "message": "Starting training: model=yolov8n.pt ..."}
```

Ultralytics prints a training progress table every epoch:

```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
  1/50      0.00G      1.842      2.193      1.321         12        640
  2/50      0.00G      1.721      2.047      1.289         15        640
  ...
```

At the end of training, Ultralytics prints validation metrics:

```
Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
  all        100        312      0.723      0.681      0.741      0.512
```

### Status badges

| Badge | Meaning |
|---|---|
| 🔵 `RUNNING` (pulsing) | Job is in progress |
| 🟢 `DONE` | Training finished successfully |
| 🔴 `FAILED` | An error occurred — check the log |

When the job finishes with `DONE`, a green banner appears with a link to the **Experiments** page.

---

## 9. Viewing Results in MLflow

After training completes:

1. Click **View Experiments →** in the green completion banner, or
2. Navigate to `http://localhost:5000` directly

In the MLflow UI you will find:

- **Parameters** — all hyperparameters logged (model, epochs, imgsz, batch, device)
- **Metrics** — final validation metrics (mAP50, mAP50-95, precision, recall, box/cls/dfl losses)
- **Artifacts** — `weights/best.pt` (best checkpoint) and optionally `exports/` (converted formats)

You can also view a summary of the latest metrics on the **Experiments** page in the AI-Forge UI (`http://localhost:3000/experiments`).

---

## 10. Using the Trained Model for Inference

### Option A — Use the Model Registry (recommended)

If you filled in **Register Model As** during training:

1. Go to `http://localhost:5000` → **Models** tab
2. Find your registered model (e.g., `yolo_detector`)
3. Promote the new version to **Production** (click the version → Stage → Production)
4. Restart the inference service to pick up the new weights:

```bash
docker compose restart inference
```

The inference service (`http://localhost:8000`) will load the Production-stage weights from MLflow automatically.

### Option B — Download the checkpoint manually

```bash
# Copy best.pt from the trainer container
docker exec trainer find /app/runs -name "best.pt" | head -1
# Example output: /app/runs/train/exp/weights/best.pt

docker cp trainer:/app/runs/train/exp/weights/best.pt ./my_best.pt
```

---

## 11. Model Export Options

If you selected an export format, the exported file is available in MLflow under the run's `exports/` artifact directory and also on disk inside the trainer container.

### ONNX

Best for production serving without PyTorch. Compatible with ONNX Runtime, OpenVINO, TensorRT (via trtexec), and most ML serving frameworks.

```bash
# Download the exported ONNX from the trainer container
docker exec trainer find /app/runs -name "*.onnx"
docker cp trainer:/app/runs/train/exp/weights/best.onnx ./model.onnx
```

### TensorRT (`engine`)

Requires an NVIDIA GPU with TensorRT installed inside the container. The export step is slow (several minutes) but the resulting `.engine` file gives the fastest GPU inference.

> If export fails with a TensorRT error, the training results (`.pt` checkpoint and metrics) are still saved. Export failure is non-fatal.

### TorchScript

Portable PyTorch format that can be loaded without the Ultralytics library:

```python
import torch
model = torch.jit.load("best.torchscript")
```

---

## 12. GPU Training

### Requirements

- NVIDIA GPU with CUDA 11.8+ or CUDA 12.x drivers on the host
- NVIDIA Container Toolkit installed: `nvidia-ctk --version`
- Docker daemon configured for GPU access

### Enable GPU in docker-compose.yaml

Add a `deploy` block to the `trainer` service:

```yaml
trainer:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]
```

Rebuild:

```bash
docker compose up -d --build trainer
```

Verify the GPU is visible inside the container:

```bash
docker exec trainer nvidia-smi
```

### GPU device values in the UI

| UI Device field | Meaning |
|---|---|
| `0` | First GPU |
| `1` | Second GPU |
| `0,1` | Data-parallel across both GPUs |

---

## 13. Troubleshooting

### "Dataset YAML not found" or immediate failure

- Verify the path is correct inside the container: `docker exec trainer ls /data`
- Confirm the volume mount in `docker-compose.yaml` includes your data directory
- The path must start with `/` (absolute) — relative paths are not supported

### Job stays on `RUNNING` indefinitely with no log output

- Ultralytics downloads model weights from the internet on first use. This can take 1–5 minutes depending on connection speed.
- Look for a line like `Downloading yolov8n.pt from https://...` in the log viewer.

### `FAILED` badge immediately after submit

- Check the log output for the Python exception message
- Common causes:
  - Incorrect `path:` in `dataset.yaml`
  - Missing or misnamed label files
  - `batch` too large for available RAM/VRAM → reduce batch size
  - Unsupported model name typo

### Out of memory (OOM) error

Reduce batch size or image size in the form. On CPU, batch sizes above 16 with `imgsz=640` can require several GB of RAM.

### Trainer service not reachable (UI shows "Train service unreachable")

```bash
# Check trainer health
docker compose ps trainer
curl http://localhost:6006/health
# Expected: {"status":"ok"}

# Restart if unhealthy
docker compose restart trainer
```

### MLflow not recording metrics

- Ensure `MLFLOW_TRACKING_URI=http://mlflow:5000` is set (already in `docker-compose.yaml`)
- Check that the MLflow service is healthy: `docker compose ps mlflow`
- MLflow requires MinIO and Postgres to be healthy first

### Previous job not visible after trainer restart

The in-memory job store is **cleared on container restart**. Job history is not persisted between restarts. Your training results (metrics, artifacts) remain in MLflow permanently — only the job status/log entries shown in the UI are lost.
