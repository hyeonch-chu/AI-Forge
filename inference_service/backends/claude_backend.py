"""Claude VLM inference backend for AI-Forge.

Uses the Anthropic API (claude-sonnet-4-6 by default) as an alternative to the
MLflow-hosted YOLO model. When selected, a base64-encoded image is sent to Claude
with a structured detection prompt, and the JSON response is normalised into the
same Prediction schema used by the MLflow backend.

Environment variables:
    ANTHROPIC_API_KEY  — required; Anthropic API key.
    CLAUDE_MODEL       — optional; defaults to "claude-sonnet-4-6".
"""
import base64
import io
import json
import logging
import os
from typing import Any

import anthropic
from PIL import Image

logger = logging.getLogger("inference_service.claude_backend")

# Model used for vision inference (overridable via env var)
CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

# Default prompt that instructs Claude to return structured JSON detections
_DETECT_PROMPT: str = (
    "Analyze this image and detect all visible objects.\n"
    "Return ONLY a JSON array of detections with no other text:\n"
    "[\n"
    '  {"label": "object name", "confidence": 0.95, "bbox": [x1, y1, x2, y2]}\n'
    "]\n"
    "Where bbox values are approximate pixel coordinates [x1, y1, x2, y2] "
    "measured from the top-left corner (0-indexed).\n"
    "If no objects are detected return an empty array []."
)


def run_claude_inference(image: Image.Image, options: dict[str, Any]) -> list[dict]:
    """Run Claude VLM object detection on a PIL Image.

    Encodes the image as PNG base64 and sends it to the Anthropic messages API
    with a structured detection prompt. The JSON response is parsed and normalised
    into prediction dicts compatible with the ``Prediction`` Pydantic schema.

    Args:
        image: PIL Image to analyse.
        options: Inference options forwarded from the request body.
                 Supports a ``"prompt"`` key to override the default detection prompt.

    Returns:
        List of dicts with ``label``, ``confidence``, and ``bbox`` keys.

    Raises:
        RuntimeError: If the Anthropic API call fails or the response cannot be
                      parsed as a JSON array of detections.
    """
    # Encode image to PNG base64 for the Anthropic messages API
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_b64: str = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Allow callers to supply a custom prompt (e.g. for domain-specific instructions)
    prompt: str = options.get("prompt", _DETECT_PROMPT)

    api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    client = anthropic.Anthropic(api_key=api_key)

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
    except Exception as exc:
        # Wrap any Anthropic API error (network, auth, rate-limit, etc.) as RuntimeError
        raise RuntimeError(f"Claude API error: {exc}") from exc

    raw_text: str = response.content[0].text.strip()

    # Strip markdown code fences that Claude may wrap JSON in (e.g. ```json ... ```)
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        # Remove the opening fence line; remove the closing fence if present
        inner = lines[1:] if len(lines) < 3 else lines[1:-1]
        raw_text = "\n".join(inner)

    try:
        detections = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        logger.error(
            "Claude returned non-JSON response (first 200 chars): %s",
            raw_text[:200],
        )
        raise RuntimeError(f"Claude response parse error: {exc}") from exc

    if not isinstance(detections, list):
        raise RuntimeError(
            f"Expected a JSON array from Claude, got: {type(detections).__name__}"
        )

    # Normalise each detection entry to match the Prediction schema
    predictions: list[dict] = []
    for det in detections:
        try:
            predictions.append(
                {
                    "label": str(det["label"]),
                    "confidence": float(det.get("confidence", 1.0)),
                    # Default bbox covers the full image if Claude omits coordinates
                    "bbox": [
                        float(v)
                        for v in det.get(
                            "bbox",
                            [0.0, 0.0, float(image.width), float(image.height)],
                        )
                    ],
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            # Skip malformed entries rather than failing the whole request
            logger.warning("Skipping malformed detection entry %s: %s", det, exc)

    logger.info("Claude backend returned %d detection(s).", len(predictions))
    return predictions
