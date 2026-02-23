"""Tests for inference_service/app.py.

Run inside the inference container:
    docker exec inference pytest tests/ -v

Or locally (requires dependencies installed):
    pytest inference_service/tests/ -v
"""
import base64
import io
from typing import Any
from unittest.mock import MagicMock, patch

import app as app_module  # used to patch auth key constants

import pytest
from fastapi.testclient import TestClient
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_b64_image(width: int = 8, height: int = 8, fmt: str = "PNG") -> str:
    """Create a tiny solid-colour image and return its base64 encoding."""
    img = Image.new("RGB", (width, height), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def client():
    """Return a TestClient with the MLflow model loading patched out."""
    # Patch get_model so tests never reach the real MLflow server
    with patch("app.get_model", return_value=None):
        from app import app  # import after patching

        yield TestClient(app)


@pytest.fixture()
def client_with_model():
    """Return a TestClient backed by a mock pyfunc model."""
    mock_model = MagicMock()
    # Model returns one detection per call
    mock_model.predict.return_value = [
        {"label": "cat", "confidence": 0.92, "bbox": [10.0, 20.0, 100.0, 200.0]}
    ]
    with patch("app.get_model", return_value=mock_model):
        # Reset cached singleton between fixtures
        with patch("app._model", mock_model):
            from app import app

            yield TestClient(app), mock_model


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
class TestHealth:
    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_returns_ok_status(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /api/v1/detect — happy paths
# ---------------------------------------------------------------------------
class TestDetectSuccess:
    def test_returns_200_with_valid_image(self, client: TestClient) -> None:
        payload = {"image_base64": _make_b64_image()}
        resp = client.post("/api/v1/detect", json=payload)
        assert resp.status_code == 200

    def test_response_schema(self, client: TestClient) -> None:
        payload = {"image_base64": _make_b64_image()}
        body = client.post("/api/v1/detect", json=payload).json()
        assert body["success"] is True
        assert isinstance(body["predictions"], list)
        assert isinstance(body["metrics"], dict)

    def test_metrics_fields_present(self, client: TestClient) -> None:
        payload = {"image_base64": _make_b64_image(width=32, height=16)}
        metrics = client.post("/api/v1/detect", json=payload).json()["metrics"]
        assert metrics["image_width"] == 32
        assert metrics["image_height"] == 16
        assert "latency_ms" in metrics
        assert "num_predictions" in metrics

    def test_data_url_prefix_stripped(self, client: TestClient) -> None:
        b64 = _make_b64_image()
        payload = {"image_base64": f"data:image/png;base64,{b64}"}
        resp = client.post("/api/v1/detect", json=payload)
        assert resp.status_code == 200

    def test_options_accepted(self, client: TestClient) -> None:
        payload = {"image_base64": _make_b64_image(), "options": {"conf": 0.5}}
        resp = client.post("/api/v1/detect", json=payload)
        assert resp.status_code == 200

    def test_predictions_populated_when_model_returns_results(
        self, client_with_model: tuple[TestClient, Any]
    ) -> None:
        client, _ = client_with_model
        payload = {"image_base64": _make_b64_image()}
        body = client.post("/api/v1/detect", json=payload).json()
        assert len(body["predictions"]) == 1
        pred = body["predictions"][0]
        assert pred["label"] == "cat"
        assert pred["confidence"] == pytest.approx(0.92)
        assert pred["bbox"] == [10.0, 20.0, 100.0, 200.0]

    def test_num_predictions_matches_prediction_list(
        self, client_with_model: tuple[TestClient, Any]
    ) -> None:
        client, _ = client_with_model
        payload = {"image_base64": _make_b64_image()}
        body = client.post("/api/v1/detect", json=payload).json()
        assert body["metrics"]["num_predictions"] == len(body["predictions"])


# ---------------------------------------------------------------------------
# POST /api/v1/detect — validation errors
# ---------------------------------------------------------------------------
class TestDetectValidation:
    def test_missing_image_base64_returns_422(self, client: TestClient) -> None:
        resp = client.post("/api/v1/detect", json={})
        assert resp.status_code == 422

    def test_invalid_base64_returns_422(self, client: TestClient) -> None:
        resp = client.post("/api/v1/detect", json={"image_base64": "!!!not-valid-base64!!!"})
        assert resp.status_code == 422

    def test_valid_base64_but_not_image_returns_422(self, client: TestClient) -> None:
        garbage = base64.b64encode(b"this is not an image").decode()
        resp = client.post("/api/v1/detect", json={"image_base64": garbage})
        assert resp.status_code == 422

    def test_empty_string_returns_422(self, client: TestClient) -> None:
        resp = client.post("/api/v1/detect", json={"image_base64": ""})
        assert resp.status_code == 422

    def test_no_body_returns_422(self, client: TestClient) -> None:
        resp = client.post("/api/v1/detect")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /api/v1/detect — model inference failure
# ---------------------------------------------------------------------------
class TestDetectInferenceError:
    def test_model_exception_returns_500(self, client: TestClient) -> None:
        broken_model = MagicMock()
        broken_model.predict.side_effect = RuntimeError("GPU OOM")
        with patch("app.get_model", return_value=broken_model):
            payload = {"image_base64": _make_b64_image()}
            resp = client.post("/api/v1/detect", json=payload)
        assert resp.status_code == 500
        assert resp.json()["detail"] == "Inference failed: GPU OOM"


# ---------------------------------------------------------------------------
# L1/L2 — API key authentication tests
# ---------------------------------------------------------------------------
_VALID_ADMIN_KEY = "test-admin-key-abc123"
_VALID_VIEWER_KEY = "test-viewer-key-xyz789"
_WRONG_KEY = "completely-wrong-key"


class TestAuthDisabled:
    """When no keys are configured, all endpoints are open."""

    def test_health_accessible_without_key(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_detect_accessible_without_key_when_auth_disabled(
        self, client: TestClient
    ) -> None:
        # Default fixture has INFERENCE_ADMIN_KEY="" → auth disabled
        payload = {"image_base64": _make_b64_image()}
        resp = client.post("/api/v1/detect", json=payload)
        assert resp.status_code == 200


class TestAdminKeyAuth:
    """When INFERENCE_ADMIN_KEY is set, /detect requires the correct key."""

    def test_detect_returns_401_without_key(self, client: TestClient) -> None:
        with patch.object(app_module, "INFERENCE_ADMIN_KEY", _VALID_ADMIN_KEY):
            payload = {"image_base64": _make_b64_image()}
            resp = client.post("/api/v1/detect", json=payload)
        assert resp.status_code == 401

    def test_detect_returns_401_with_wrong_key(self, client: TestClient) -> None:
        with patch.object(app_module, "INFERENCE_ADMIN_KEY", _VALID_ADMIN_KEY):
            payload = {"image_base64": _make_b64_image()}
            resp = client.post(
                "/api/v1/detect", json=payload, headers={"X-API-Key": _WRONG_KEY}
            )
        assert resp.status_code == 401

    def test_detect_returns_200_with_correct_admin_key(self, client: TestClient) -> None:
        with patch.object(app_module, "INFERENCE_ADMIN_KEY", _VALID_ADMIN_KEY):
            payload = {"image_base64": _make_b64_image()}
            resp = client.post(
                "/api/v1/detect", json=payload, headers={"X-API-Key": _VALID_ADMIN_KEY}
            )
        assert resp.status_code == 200

    def test_health_always_accessible_regardless_of_key(self, client: TestClient) -> None:
        """GET /health must always be reachable — it is used by Docker healthchecks."""
        with patch.object(app_module, "INFERENCE_ADMIN_KEY", _VALID_ADMIN_KEY):
            resp = client.get("/health")
        assert resp.status_code == 200

    def test_error_response_body_mentions_header(self, client: TestClient) -> None:
        with patch.object(app_module, "INFERENCE_ADMIN_KEY", _VALID_ADMIN_KEY):
            payload = {"image_base64": _make_b64_image()}
            body = client.post("/api/v1/detect", json=payload).json()
        assert "X-API-Key" in body["detail"]


class TestViewerKeyRBAC:
    """Viewer key cannot call admin-only endpoints."""

    def test_detect_returns_401_with_viewer_key_when_admin_key_set(
        self, client: TestClient
    ) -> None:
        """A viewer-level key must not be accepted on the admin-only detect endpoint."""
        with (
            patch.object(app_module, "INFERENCE_ADMIN_KEY", _VALID_ADMIN_KEY),
            patch.object(app_module, "INFERENCE_VIEWER_KEY", _VALID_VIEWER_KEY),
        ):
            payload = {"image_base64": _make_b64_image()}
            resp = client.post(
                "/api/v1/detect", json=payload, headers={"X-API-Key": _VALID_VIEWER_KEY}
            )
        assert resp.status_code == 401

    def test_detect_accessible_with_admin_key_when_viewer_also_set(
        self, client: TestClient
    ) -> None:
        with (
            patch.object(app_module, "INFERENCE_ADMIN_KEY", _VALID_ADMIN_KEY),
            patch.object(app_module, "INFERENCE_VIEWER_KEY", _VALID_VIEWER_KEY),
        ):
            payload = {"image_base64": _make_b64_image()}
            resp = client.post(
                "/api/v1/detect", json=payload, headers={"X-API-Key": _VALID_ADMIN_KEY}
            )
        assert resp.status_code == 200
