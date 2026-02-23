"""YOLO training pipeline for AI-Forge.

Trains a YOLO model, tracks the experiment with MLflow, and registers the
best checkpoint in the MLflow model registry so the inference service can
load it automatically.

Supported YOLO presets (auto-downloaded by Ultralytics on first use):
  YOLOv8  : yolov8n/s/m/l/x.pt
  YOLOv10 : yolov10n/s/m/b/l/x.pt
  YOLO11  : yolo11n/s/m/l/x.pt

Usage (inside the trainer container):
    python train.py \\
        --model yolov8n.pt \\
        --data /data/dataset.yaml \\
        --epochs 50 \\
        --imgsz 640 \\
        --batch 16 \\
        --experiment "my_experiment" \\
        --run-name "baseline" \\
        --register yolo_detector \\
        --export onnx
"""
import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional

import mlflow
import mlflow.pyfunc

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}',
)
logger = logging.getLogger("train_service")

# ---------------------------------------------------------------------------
# MLflow configuration (from environment, see docker-compose.yaml)
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", MLFLOW_S3_ENDPOINT_URL)

# ---------------------------------------------------------------------------
# Supported YOLO model presets (M6)
# Ultralytics auto-downloads these weights on first use.
# Custom .pt file paths are also accepted — a warning is issued if the path
# does not match a known preset and does not exist locally.
# ---------------------------------------------------------------------------
SUPPORTED_YOLO_PRESETS: List[str] = [
    # YOLOv8 family
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
    # YOLOv10 family
    "yolov10n.pt", "yolov10s.pt", "yolov10m.pt", "yolov10b.pt", "yolov10l.pt", "yolov10x.pt",
    # YOLO11 family (Ultralytics' latest generation)
    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
]

# Formats supported by Ultralytics model.export() (M7)
SUPPORTED_EXPORT_FORMATS: List[str] = [
    "onnx",         # ONNX — cross-platform, widely supported
    "engine",       # TensorRT engine (requires NVIDIA GPU + TensorRT)
    "torchscript",  # TorchScript — portable PyTorch format
    "coreml",       # Core ML — Apple platforms
    "saved_model",  # TensorFlow SavedModel
]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(
    model_weights: str,
    data_yaml: str,
    epochs: int,
    imgsz: int,
    batch: int,
    experiment: str,
    run_name: Optional[str],
    device: str,
    register_name: Optional[str],
    export_format: Optional[str] = None,
) -> None:
    """Run YOLO training and log results to MLflow.

    Args:
        model_weights:  Path or name of the pretrained YOLO weights
                        (e.g. ``yolov8n.pt`` or an absolute path).
        data_yaml:      Path to the dataset YAML file (Ultralytics format).
        epochs:         Number of training epochs.
        imgsz:          Input image size (square side length in pixels).
        batch:          Batch size.
        experiment:     MLflow experiment name.
        run_name:       Optional MLflow run name.
        device:         Training device (``"cpu"``, ``"0"``, ``"0,1"``).
        register_name:  If set, register the best model checkpoint in the
                        MLflow Model Registry under this name.
        export_format:  If set, export the trained model to this format after
                        training (e.g. ``"onnx"``, ``"engine"``).
    """
    # Warn if the weights name is neither a known preset nor a local file path
    if model_weights not in SUPPORTED_YOLO_PRESETS and not Path(model_weights).exists():
        logger.warning(
            "Model '%s' is not a recognised preset and does not exist locally. "
            "Ultralytics will attempt to download it from the internet.",
            model_weights,
        )

    mlflow.set_experiment(experiment)
    logger.info(
        "Starting training: model=%s data=%s epochs=%d imgsz=%d batch=%d export=%s",
        model_weights,
        data_yaml,
        epochs,
        imgsz,
        batch,
        export_format or "none",
    )

    with mlflow.start_run(run_name=run_name) as run:
        # --- Log hyper-parameters -----------------------------------------
        params: dict = {
            "model_weights": model_weights,
            "data_yaml": data_yaml,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "device": device,
        }
        if export_format:
            params["export_format"] = export_format
        mlflow.log_params(params)

        # --- Train -----------------------------------------------------------
        from ultralytics import YOLO  # lazy import — avoids hard dep at test time

        model = YOLO(model_weights)
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project="/app/runs/train",
            name=run_name or "exp",
            exist_ok=True,
        )

        # --- Log per-epoch metrics ------------------------------------------
        metrics_history = results.results_dict if hasattr(results, "results_dict") else {}
        for key, value in metrics_history.items():
            try:
                mlflow.log_metric(key, float(value))
            except (TypeError, ValueError):
                pass

        # --- Log best checkpoint as an artifact ----------------------------
        best_pt = Path(results.save_dir) / "weights" / "best.pt"
        if best_pt.exists():
            mlflow.log_artifact(str(best_pt), artifact_path="weights")
            logger.info("Logged best checkpoint: %s", best_pt)
        else:
            logger.warning("best.pt not found at %s", best_pt)

        # --- Register model in MLflow Model Registry -----------------------
        if register_name and best_pt.exists():
            artifact_uri = f"runs:/{run.info.run_id}/weights/best.pt"
            reg = mlflow.register_model(model_uri=artifact_uri, name=register_name)
            logger.info(
                "Registered model '%s' version %s in MLflow registry.",
                reg.name,
                reg.version,
            )

        # --- Export model to alternate format (M7) --------------------------
        if export_format:
            logger.info("Exporting model to format: %s", export_format)
            try:
                exported_path = model.export(format=export_format)
                if exported_path:
                    mlflow.log_artifact(str(exported_path), artifact_path="exports")
                    logger.info("Exported model logged to MLflow: %s", exported_path)
            except Exception as exc:
                # Export failure is non-fatal — training results are already saved
                logger.warning(
                    "Model export to '%s' failed: %s. "
                    "Ensure the required runtime (e.g. TensorRT) is installed.",
                    export_format,
                    exc,
                )

        logger.info("Training complete. Run ID: %s", run.info.run_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    preset_list = ", ".join(SUPPORTED_YOLO_PRESETS)
    parser = argparse.ArgumentParser(
        description="Train a YOLO model and track the experiment in MLflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help=(
            "Pretrained YOLO weights. Use a preset name for auto-download or an absolute "
            "path to a custom .pt file. Default: yolov8n.pt\n"
            f"Known presets: {preset_list}"
        ),
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to dataset YAML file (Ultralytics format)",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs (default: 50)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument(
        "--experiment",
        default="yolo_training",
        help="MLflow experiment name (default: yolo_training)",
    )
    parser.add_argument("--run-name", default=None, help="MLflow run name (optional)")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Training device: 'cpu', '0', '0,1' (default: cpu)",
    )
    parser.add_argument(
        "--register",
        default=None,
        dest="register_name",
        help="Register best checkpoint in MLflow registry under this name (optional)",
    )
    parser.add_argument(
        "--export",
        default=None,
        dest="export_format",
        choices=SUPPORTED_EXPORT_FORMATS,
        help=(
            "Export format after training (optional). "
            "'engine' requires NVIDIA TensorRT. "
            f"Choices: {', '.join(SUPPORTED_EXPORT_FORMATS)}"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        model_weights=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        experiment=args.experiment,
        run_name=args.run_name,
        device=args.device,
        register_name=args.register_name,
        export_format=args.export_format,
    )
