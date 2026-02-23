"""Tests for the Claude VLM inference backend.

Run inside the inference container:
    docker exec inference pytest tests/ -v

Or locally (requires dependencies installed):
    pytest inference_service/tests/ -v
"""
import io
import json
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from backends.claude_backend import run_claude_inference


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_image(width: int = 8, height: int = 8) -> Image.Image:
    """Create a small solid-colour PIL Image for testing."""
    return Image.new("RGB", (width, height), color=(100, 150, 200))


def _make_claude_response(detections: list) -> MagicMock:
    """Build a minimal mock Anthropic Message whose text is a JSON-serialised list."""
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text=json.dumps(detections))]
    return mock_resp


def _patched_client(return_value):
    """Patch anthropic.Anthropic so no real API calls are made."""
    mock_client = MagicMock()
    mock_client.messages.create.return_value = return_value
    return patch("backends.claude_backend.anthropic.Anthropic", return_value=mock_client)


# ---------------------------------------------------------------------------
# Happy-path detection tests
# ---------------------------------------------------------------------------
class TestRunClaudeInferenceSuccess:
    """Tests for successful Claude API calls with various response shapes."""

    def test_empty_array_returns_empty_list(self) -> None:
        with _patched_client(_make_claude_response([])):
            result = run_claude_inference(_make_image(), {})
        assert result == []

    def test_parses_single_detection(self) -> None:
        detections = [{"label": "cat", "confidence": 0.9, "bbox": [10.0, 20.0, 100.0, 200.0]}]
        with _patched_client(_make_claude_response(detections)):
            result = run_claude_inference(_make_image(), {})
        assert len(result) == 1
        assert result[0]["label"] == "cat"
        assert result[0]["confidence"] == pytest.approx(0.9)
        assert result[0]["bbox"] == [10.0, 20.0, 100.0, 200.0]

    def test_parses_multiple_detections(self) -> None:
        detections = [
            {"label": "dog", "confidence": 0.8, "bbox": [0.0, 0.0, 50.0, 50.0]},
            {"label": "cat", "confidence": 0.7, "bbox": [50.0, 50.0, 100.0, 100.0]},
        ]
        with _patched_client(_make_claude_response(detections)):
            result = run_claude_inference(_make_image(), {})
        assert len(result) == 2
        assert result[0]["label"] == "dog"
        assert result[1]["label"] == "cat"

    def test_confidence_defaults_to_1_when_missing(self) -> None:
        detections = [{"label": "bird", "bbox": [0.0, 0.0, 10.0, 10.0]}]
        with _patched_client(_make_claude_response(detections)):
            result = run_claude_inference(_make_image(), {})
        assert result[0]["confidence"] == pytest.approx(1.0)

    def test_missing_bbox_defaults_to_full_image_dimensions(self) -> None:
        """Detections without bbox should use the full image extent."""
        detections = [{"label": "person", "confidence": 0.99}]
        img = _make_image(width=100, height=80)
        with _patched_client(_make_claude_response(detections)):
            result = run_claude_inference(img, {})
        assert result[0]["bbox"] == [0.0, 0.0, 100.0, 80.0]


# ---------------------------------------------------------------------------
# Response format resilience
# ---------------------------------------------------------------------------
class TestResponseParsing:
    """Tests for handling varying Claude response formats."""

    def test_strips_json_markdown_code_fence(self) -> None:
        """Claude sometimes wraps the JSON array in ```json ... ``` fences."""
        detections = [{"label": "car", "confidence": 0.95, "bbox": [0.0, 0.0, 100.0, 100.0]}]
        fenced = "```json\n" + json.dumps(detections) + "\n```"
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text=fenced)]
        with _patched_client(mock_resp):
            result = run_claude_inference(_make_image(), {})
        assert len(result) == 1
        assert result[0]["label"] == "car"

    def test_strips_plain_code_fence(self) -> None:
        """Handles fences without a language tag (``` ... ```)."""
        detections = [{"label": "truck", "confidence": 0.8, "bbox": [0.0, 0.0, 50.0, 50.0]}]
        fenced = "```\n" + json.dumps(detections) + "\n```"
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text=fenced)]
        with _patched_client(mock_resp):
            result = run_claude_inference(_make_image(), {})
        assert len(result) == 1

    def test_skips_malformed_entries_without_label(self) -> None:
        """Entries missing the required 'label' key are silently skipped."""
        raw = json.dumps(
            [
                {"confidence": 0.9, "bbox": [0.0, 0.0, 10.0, 10.0]},  # no label
                {"label": "valid", "confidence": 0.8, "bbox": [0.0, 0.0, 10.0, 10.0]},
            ]
        )
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text=raw)]
        with _patched_client(mock_resp):
            result = run_claude_inference(_make_image(), {})
        assert len(result) == 1
        assert result[0]["label"] == "valid"

    def test_label_coerced_to_string(self) -> None:
        """Labels that are not strings should be coerced safely."""
        detections = [{"label": 42, "confidence": 0.5, "bbox": [0.0, 0.0, 10.0, 10.0]}]
        with _patched_client(_make_claude_response(detections)):
            result = run_claude_inference(_make_image(), {})
        assert result[0]["label"] == "42"


# ---------------------------------------------------------------------------
# Custom prompt option
# ---------------------------------------------------------------------------
class TestCustomPrompt:
    """Tests that the 'prompt' option is forwarded to Claude."""

    def test_custom_prompt_overrides_default(self) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_claude_response([])
        with patch(
            "backends.claude_backend.anthropic.Anthropic", return_value=mock_client
        ):
            run_claude_inference(_make_image(), {"prompt": "my custom prompt"})

        call_kwargs = mock_client.messages.create.call_args[1]
        messages = call_kwargs["messages"]
        text_block = next(b for b in messages[0]["content"] if b["type"] == "text")
        assert text_block["text"] == "my custom prompt"

    def test_default_prompt_used_when_no_option(self) -> None:
        from backends.claude_backend import _DETECT_PROMPT

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_claude_response([])
        with patch(
            "backends.claude_backend.anthropic.Anthropic", return_value=mock_client
        ):
            run_claude_inference(_make_image(), {})

        call_kwargs = mock_client.messages.create.call_args[1]
        messages = call_kwargs["messages"]
        text_block = next(b for b in messages[0]["content"] if b["type"] == "text")
        assert text_block["text"] == _DETECT_PROMPT


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
class TestErrorHandling:
    """Tests for error paths in run_claude_inference."""

    def test_raises_runtime_error_on_api_failure(self) -> None:
        """Any exception from messages.create is wrapped as RuntimeError."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("rate limit exceeded")
        with patch(
            "backends.claude_backend.anthropic.Anthropic", return_value=mock_client
        ):
            with pytest.raises(RuntimeError, match="Claude API error"):
                run_claude_inference(_make_image(), {})

    def test_raises_runtime_error_on_non_json_response(self) -> None:
        """A non-JSON text response raises RuntimeError with parse error message."""
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text="I cannot detect objects in this image.")]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_resp
        with patch(
            "backends.claude_backend.anthropic.Anthropic", return_value=mock_client
        ):
            with pytest.raises(RuntimeError, match="parse error"):
                run_claude_inference(_make_image(), {})

    def test_raises_runtime_error_on_non_array_json(self) -> None:
        """A JSON object (instead of array) response raises RuntimeError."""
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text='{"error": "unexpected"}')]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_resp
        with patch(
            "backends.claude_backend.anthropic.Anthropic", return_value=mock_client
        ):
            with pytest.raises(RuntimeError, match="Expected a JSON array"):
                run_claude_inference(_make_image(), {})
