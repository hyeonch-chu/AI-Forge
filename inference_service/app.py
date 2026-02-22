"""Inference service for AI-Forge.

Provides a FastAPI-based REST API for object detection using YOLO models
loaded from the MLflow model registry.
"""
import base64
import io
import logging
import os
import time
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}',
)
logger = logging.getLogger("inference_service")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "yolo_detector")
MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI-Forge Inference Service",
    version="1.0.0",
)

_model: Any = None


def get_model() -> Any:
    """Load and cache the detection model from the MLflow registry.

    Returns:
        Loaded pyfunc model, or None if unavailable.
    """
    global _model
    if _model is not None:
        return _model
    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        logger.info("Loading model from MLflow registry: %s", model_uri)
        _model = mlflow.pyfunc.load_model(model_uri)
        logger.info("Model loaded successfully.")
    except Exception as exc:
        logger.warning("MLflow model unavailable (%s); running without a model.", exc)
        _model = None
    return _model


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class DetectRequest(BaseModel):
    """Request body for POST /api/v1/detect."""

    image_base64: str = Field(..., description="Base64-encoded image (JPEG/PNG)")
    options: dict[str, Any] = Field(default_factory=dict, description="Inference options")


class Prediction(BaseModel):
    label: str
    confidence: float
    bbox: list[float] = Field(..., description="[x1, y1, x2, y2] pixel coordinates")


class DetectResponse(BaseModel):
    success: bool
    predictions: list[Prediction]
    metrics: dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def decode_image(image_base64: str) -> Image.Image:
    """Decode a base64-encoded string to a PIL Image.

    Strips an optional data-URL prefix (e.g. ``data:image/png;base64,``).

    Raises:
        HTTPException: 422 for invalid base64 or unsupported image format.
    """
    try:
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]
        image_bytes = base64.b64decode(image_base64)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Invalid image_base64: {exc}",
        ) from exc


def run_inference(
    model: Any,
    image: Image.Image,
    options: dict,
) -> tuple[list[dict], dict]:
    """Run model inference and return (predictions, metrics).

    Args:
        model: MLflow pyfunc model or None.
        image: PIL Image to run inference on.
        options: Additional inference options forwarded to the model.

    Returns:
        Tuple of (predictions list, metrics dict).

    Raises:
        HTTPException: 500 if the model raises an unexpected error.
    """
    t_start = time.perf_counter()
    predictions: list[dict] = []

    if model is not None:
        try:
            import pandas as pd

            input_df = pd.DataFrame([{"image": np.array(image).tolist(), **options}])
            results = model.predict(input_df)
            if isinstance(results, list):
                predictions = results
            elif hasattr(results, "to_dict"):
                predictions = results.to_dict(orient="records")
        except Exception as exc:
            logger.error("Inference failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference failed: {exc}",
            ) from exc
    else:
        logger.warning("No model loaded; returning empty predictions.")

    latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
    metrics: dict[str, Any] = {
        "latency_ms": latency_ms,
        "image_width": image.width,
        "image_height": image.height,
        "num_predictions": len(predictions),
    }
    return predictions, metrics


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", summary="Health check")
def health() -> dict:
    """Return service liveness."""
    return {"status": "ok"}


@app.post(
    "/api/v1/detect",
    response_model=DetectResponse,
    status_code=status.HTTP_200_OK,
    summary="Run object detection on a base64-encoded image",
)
def detect(request: DetectRequest) -> DetectResponse:
    """Decode the image, run MLflow model inference, return structured results."""
    logger.info("POST /api/v1/detect options=%s", request.options)
    image = decode_image(request.image_base64)
    model = get_model()
    raw_preds, metrics = run_inference(model, image, request.options)
    predictions = [Prediction(**p) for p in raw_preds]
    logger.info(
        "Detection complete: num_predictions=%d latency_ms=%s",
        len(predictions),
        metrics.get("latency_ms"),
    )
    return DetectResponse(success=True, predictions=predictions, metrics=metrics)


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for any unhandled exception; returns a safe 500 JSON response."""
    logger.error("Unhandled exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"success": False, "detail": "Internal server error"},
    )
