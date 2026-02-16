"""
Tool implementations for Reachy MVP.

Provides time/date tools and scene awareness (look_at_scene) for
the robot's LLM reasoning loop.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Callable, Awaitable, Optional

logger = logging.getLogger(__name__)

# Module-level reference to camera_worker, set by state machine during startup.
# This lets the look_at_scene tool access the camera without import coupling.
_camera_worker: Any = None


def set_camera_worker(camera_worker: Any) -> None:
    """Register the CameraWorker instance for scene awareness tools."""
    global _camera_worker
    _camera_worker = camera_worker


# Tool handler functions

async def get_time() -> str:
    """Get the current time."""
    now = datetime.now()
    return f"The current time is {now.strftime('%I:%M %p')}"


async def get_date() -> str:
    """Get the current date."""
    now = datetime.now()
    return f"Today is {now.strftime('%A, %B %d, %Y')}"


async def look_at_scene(question: str = "Describe what you see.") -> str:
    """Look through the robot's camera and describe the scene.

    Captures the current camera frame, analyzes basic properties, and
    returns a description. For full vision (object recognition, text reading),
    this would pass the image to a vision-capable LLM.

    Note: With OpenClaw provider, this tool is not used (tools handled server-side).
    With Anthropic direct, this provides basic scene awareness.
    """
    if _camera_worker is None:
        return "Camera is not available. I cannot see right now."

    frame = _camera_worker.get_latest_frame()
    if frame is None:
        return "No camera frame available. The camera may still be initializing."

    h, w = frame.shape[:2]

    # Basic scene analysis (no external model needed)
    import numpy as np

    # Average brightness
    gray = np.mean(frame, axis=2)
    avg_brightness = float(np.mean(gray))
    brightness_desc = "dark" if avg_brightness < 60 else "dimly lit" if avg_brightness < 120 else "well lit" if avg_brightness < 200 else "very bright"

    # Dominant color region (rough)
    center_region = frame[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    avg_bgr = np.mean(center_region, axis=(0, 1))
    b, g, r = avg_bgr

    # Face detection status
    offsets = _camera_worker.get_face_tracking_offsets()
    face_detected = any(abs(o) > 0.001 for o in offsets)

    parts = [
        f"I can see a {w}x{h} view that appears {brightness_desc}.",
    ]
    if face_detected:
        parts.append("I can see a person's face in front of me.")
    else:
        parts.append("I don't see anyone directly in front of me right now.")

    return " ".join(parts)


# Anthropic Claude function definitions

TOOL_DEFINITIONS = [
    {
        "name": "get_time",
        "description": "Get the current time",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_date",
        "description": "Get the current date",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "look_at_scene",
        "description": "Look through the robot's camera to see what's in front of you. Use this when the user asks what you can see, who is in front of you, or to describe your surroundings.",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "What to focus on when looking at the scene (default: general description)"
                }
            },
            "required": []
        }
    }
]


# Tool handler registry

TOOL_HANDLERS: Dict[str, Callable[..., Awaitable[str]]] = {
    "get_time": get_time,
    "get_date": get_date,
    "look_at_scene": look_at_scene,
}


def get_tool_handler(tool_name: str) -> Optional[Callable[..., Awaitable[str]]]:
    """Get a tool handler by name."""
    return TOOL_HANDLERS.get(tool_name)
