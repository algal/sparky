"""Scene capture utility for visual awareness.

Captures frames from CameraWorker, encodes as JPEG, and provides them as
base64 image content blocks compatible with Anthropic's vision API.

This enables the robot to "see" — the LLM can describe what's in front of it,
identify objects, read text, etc.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def capture_scene_image(
    camera_worker: Any,
    max_size: int = 768,
    jpeg_quality: int = 80,
) -> dict | None:
    """Capture current camera frame as an Anthropic-compatible image content block.

    Args:
        camera_worker: CameraWorker instance with get_latest_frame().
        max_size: Max dimension (width or height) for resizing. Keeps aspect ratio.
        jpeg_quality: JPEG compression quality (1-100).

    Returns:
        Anthropic image content block dict, or None if no frame available.
        Format: {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}}
    """
    if camera_worker is None:
        return None

    frame = camera_worker.get_latest_frame()
    if frame is None:
        logger.debug("No frame available from camera worker")
        return None

    return encode_frame_as_image_block(frame, max_size=max_size, jpeg_quality=jpeg_quality)


def encode_frame_as_image_block(
    frame: NDArray[np.uint8],
    max_size: int = 768,
    jpeg_quality: int = 80,
) -> dict:
    """Encode a BGR numpy frame as an Anthropic image content block.

    Args:
        frame: BGR uint8 numpy array from camera.
        max_size: Max dimension for resizing.
        jpeg_quality: JPEG quality.

    Returns:
        Anthropic image content block dict.
    """
    import cv2

    # Resize if needed (keep aspect ratio)
    h, w = frame.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Encode as JPEG
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    success, jpeg_buf = cv2.imencode(".jpg", frame, encode_params)
    if not success:
        raise RuntimeError("Failed to encode frame as JPEG")

    b64_data = base64.b64encode(jpeg_buf.tobytes()).decode("ascii")

    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": b64_data,
        },
    }


def capture_scene_description_prompt(
    camera_worker: Any,
    question: str = "Briefly describe what you see in front of you.",
) -> list[dict] | None:
    """Build a message content array with the current camera frame + question.

    Returns a list of content blocks suitable for Anthropic's messages API:
    [image_block, {"type": "text", "text": question}]

    Returns None if no frame is available.
    """
    image_block = capture_scene_image(camera_worker)
    if image_block is None:
        return None

    return [
        image_block,
        {"type": "text", "text": question},
    ]
