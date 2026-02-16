"""
Gentle organic actions for Reachy Mini Lite.

These are subtle, natural-looking movements the agent can request via
the OpenClaw node action interface. Designed to be non-dramatic —
stretching, yawning, slight fidgets — to give the robot personality
without startling anyone.

Each action is a Move subclass that can be queued on MovementManager.
"""

from __future__ import annotations

import math
import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from reachy_mini.utils import create_head_pose
from reachy_mini.motion.move import Move

logger = logging.getLogger(__name__)


class GentleStretchMove(Move):
    """Slow head stretch — tilt to one side, then the other, return to center."""

    def __init__(self, duration: float = 3.0, amplitude_deg: float = 12.0):
        self._duration = duration
        self._amplitude = math.radians(amplitude_deg)

    @property
    def duration(self) -> float:
        return self._duration

    def evaluate(self, t: float):
        progress = t / self._duration
        # Smooth sine: tilt right (0→0.33), tilt left (0.33→0.66), return (0.66→1.0)
        roll = self._amplitude * math.sin(2 * math.pi * progress)
        # Small accompanying pitch forward during stretch
        pitch = math.radians(3.0) * math.sin(math.pi * progress)
        head = create_head_pose(0, 0, 0, roll, pitch, 0, degrees=False, mm=False)
        # Antennas spread slightly during stretch
        ant_spread = math.radians(10) * abs(math.sin(2 * math.pi * progress))
        antennas = np.array([ant_spread, -ant_spread], dtype=np.float64)
        return (head, antennas, 0.0)


class GentleYawnMove(Move):
    """Yawn-like motion: slow head tilt back, antennas droop, then recover."""

    def __init__(self, duration: float = 3.5):
        self._duration = duration

    @property
    def duration(self) -> float:
        return self._duration

    def evaluate(self, t: float):
        progress = t / self._duration
        # Head tilts back (pitch up) then returns
        # Peak at ~40% through the yawn
        if progress < 0.4:
            phase = progress / 0.4
            pitch = -math.radians(8) * _smoothstep(phase)
        elif progress < 0.7:
            phase = (progress - 0.4) / 0.3
            pitch = -math.radians(8) * (1.0 - _smoothstep(phase) * 0.5)
        else:
            phase = (progress - 0.7) / 0.3
            pitch = -math.radians(4) * (1.0 - _smoothstep(phase))

        head = create_head_pose(0, 0, 0, 0, pitch, 0, degrees=False, mm=False)
        # Antennas droop during yawn peak
        droop = math.radians(15) * math.sin(math.pi * progress)
        antennas = np.array([-droop, droop], dtype=np.float64)
        return (head, antennas, 0.0)


class GentleNodMove(Move):
    """Gentle nod — as if agreeing or acknowledging something."""

    def __init__(self, duration: float = 1.5, nods: int = 2):
        self._duration = duration
        self._nods = nods

    @property
    def duration(self) -> float:
        return self._duration

    def evaluate(self, t: float):
        progress = t / self._duration
        pitch = math.radians(6) * math.sin(2 * math.pi * self._nods * progress)
        # Fade in and out
        envelope = math.sin(math.pi * progress)
        pitch *= envelope
        head = create_head_pose(0, 0, 0, 0, pitch, 0, degrees=False, mm=False)
        return (head, None, None)


class GentleShakeMove(Move):
    """Gentle head shake — as if saying 'no' or musing."""

    def __init__(self, duration: float = 2.0, shakes: int = 2):
        self._duration = duration
        self._shakes = shakes

    @property
    def duration(self) -> float:
        return self._duration

    def evaluate(self, t: float):
        progress = t / self._duration
        yaw = math.radians(8) * math.sin(2 * math.pi * self._shakes * progress)
        envelope = math.sin(math.pi * progress)
        yaw *= envelope
        head = create_head_pose(0, 0, 0, 0, 0, yaw, degrees=False, mm=False)
        return (head, None, None)


class GentleLookAroundMove(Move):
    """Curious look around — head rotates slowly to one side, pauses, returns."""

    def __init__(self, duration: float = 3.0, direction: str = "left"):
        self._duration = duration
        self._sign = 1.0 if direction == "left" else -1.0

    @property
    def duration(self) -> float:
        return self._duration

    def evaluate(self, t: float):
        progress = t / self._duration
        # Move to side (0→0.4), hold (0.4→0.6), return (0.6→1.0)
        if progress < 0.4:
            phase = _smoothstep(progress / 0.4)
        elif progress < 0.6:
            phase = 1.0
        else:
            phase = 1.0 - _smoothstep((progress - 0.6) / 0.4)

        yaw = self._sign * math.radians(20) * phase
        # Slight pitch down when looking to the side (curious)
        pitch = math.radians(3) * phase
        head = create_head_pose(0, 0, 0, 0, pitch, yaw, degrees=False, mm=False)
        # Antenna tilt in direction of look
        ant = math.radians(8) * phase
        antennas = np.array(
            [self._sign * ant, -self._sign * ant], dtype=np.float64
        )
        return (head, antennas, 0.0)


class GentleAntennaWiggleMove(Move):
    """Quick antenna wiggle — like perking up or reacting to something."""

    def __init__(self, duration: float = 1.2):
        self._duration = duration

    @property
    def duration(self) -> float:
        return self._duration

    def evaluate(self, t: float):
        progress = t / self._duration
        # Fast wiggle with decay
        freq = 3.0
        decay = 1.0 - progress
        wiggle = math.radians(20) * math.sin(2 * math.pi * freq * progress) * decay
        antennas = np.array([wiggle, -wiggle], dtype=np.float64)
        return (None, antennas, None)


def _smoothstep(x: float) -> float:
    """Hermite smoothstep for jump-free transitions."""
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


# ---------------------------------------------------------------------------
# Registry: maps action names to Move constructors
# ---------------------------------------------------------------------------

GENTLE_ACTIONS = {
    "stretch": {
        "description": "Slow gentle stretch — head tilts side to side with antenna spread",
        "create": lambda params: GentleStretchMove(
            duration=params.get("duration", 3.0),
            amplitude_deg=params.get("amplitude", 12.0),
        ),
    },
    "yawn": {
        "description": "Yawn-like motion — head tilts back, antennas droop, then recover",
        "create": lambda params: GentleYawnMove(
            duration=params.get("duration", 3.5),
        ),
    },
    "nod": {
        "description": "Gentle nod — as if agreeing or acknowledging",
        "create": lambda params: GentleNodMove(
            duration=params.get("duration", 1.5),
            nods=params.get("nods", 2),
        ),
    },
    "shake": {
        "description": "Gentle head shake — as if musing or saying no",
        "create": lambda params: GentleShakeMove(
            duration=params.get("duration", 2.0),
            shakes=params.get("shakes", 2),
        ),
    },
    "look_around": {
        "description": "Curious look to one side, pause, return",
        "create": lambda params: GentleLookAroundMove(
            duration=params.get("duration", 3.0),
            direction=params.get("direction", "left"),
        ),
    },
    "antenna_wiggle": {
        "description": "Quick antenna wiggle — perking up or reacting",
        "create": lambda params: GentleAntennaWiggleMove(
            duration=params.get("duration", 1.2),
        ),
    },
}
