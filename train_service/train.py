"""YOLO training pipeline for AI-Forge.

Trains a YOLO model, tracks the experiment with MLflow, and registers the
best checkpoint in the MLflow model registry so the inference service can
load it automatically.

Usage (inside the trainer container):
    python train.py \\
        --model yolov8n.pt \\
        --data /data/dataset.yaml \\
        --epochs 50 \\
        --imgsz 640 \\
        --batch 16 \\
        --experiment "my_experiment" \\
        --run-name "baseline"
"""
import argparse
import logging
import os
from pathlib import Path
from typing import Optional

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
) -> None:
    """Run YOLO training and log results to MLflow.

    Args:
        model_weights: Path or name of the pretrained YOLO weights
                       (e.g. ``yolov8n.pt`` or an absolute path).
        data_yaml:     Path to the dataset YAML file (Ultralytics format).
        epochs:        Number of training epochs.
        imgsz:         Input image size (square side length in pixels).
        batch:         Batch size.
        experiment:    MLflow experiment name.
        run_name:      Optional MLflow run name.
        device:        Training device (``"cpu"``, ``"0"``, ``"0,1"``).
        register_name: If set, register the best model checkpoint in the
                       MLflow Model Registry under this name.
    """
    mlflow.set_experiment(experiment)
    logger.info(
        "Starting training: model=%s data=%s epochs=%d imgsz=%d batch=%d",
        model_weights,
        data_yaml,
        epochs,
        imgsz,
        batch,
    )

    with mlflow.start_run(run_name=run_name) as run:
        # --- Log hyper-parameters -----------------------------------------
        mlflow.log_params(
            {
                "model_weights": model_weights,
                "data_yaml": data_yaml,
                "epochs": epochs,
                "imgsz": imgsz,
                "batch": batch,
                "device": device,
            }
        )

        # --- Train -----------------------------------------------------------
        from ultralytics import YOLO  # lazy import — avoids hard dep at test time

        model = YOLO(model_weights)
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project="runs/train",
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

        logger.info("Training complete. Run ID: %s", run.info.run_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a YOLO model and track the experiment in MLflow."
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Pretrained YOLO weights (default: yolov8n.pt)",
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
    )
