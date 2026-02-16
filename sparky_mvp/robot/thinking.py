"""Staged thinking animation that escalates intensity over time.

3-stage animation queued during PROCESSING state:

  Stage 1 (0-2s):  Subtle — gentle asymmetric antenna tilt, no head movement.
  Stage 2 (2-5s):  Medium — slow head pitch/yaw sway added.
  Stage 3 (5s+):   Full   — more pronounced head + antenna movement.

The move runs until explicitly cleared via ``MovementManager.clear_move_queue()``.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from reachy_mini.motion.move import Move
from reachy_mini.utils import create_head_pose


# Stage boundaries (seconds from move start)
STAGE_2_START = 2.0
STAGE_3_START = 5.0

# Stage 1: subtle antenna tilt
S1_ANTENNA_AMPLITUDE = math.radians(8)   # 8 degrees
S1_ANTENNA_FREQ = 0.4                     # Hz

# Stage 2: adds head sway
S2_ANTENNA_AMPLITUDE = math.radians(12)  # 12 degrees
S2_ANTENNA_FREQ = 0.5
S2_HEAD_PITCH_DEG = 2.0                  # degrees peak
S2_HEAD_PITCH_FREQ = 0.3                 # Hz
S2_HEAD_YAW_DEG = 3.0
S2_HEAD_YAW_FREQ = 0.2

# Stage 3: full thinking
S3_ANTENNA_AMPLITUDE = math.radians(18)  # 18 degrees
S3_ANTENNA_FREQ = 0.6
S3_HEAD_PITCH_DEG = 4.0
S3_HEAD_PITCH_FREQ = 0.35
S3_HEAD_YAW_DEG = 5.0
S3_HEAD_YAW_FREQ = 0.25
S3_HEAD_ROLL_DEG = 2.0
S3_HEAD_ROLL_FREQ = 0.15

# Blend duration between stages (seconds)
BLEND_DURATION = 1.0


def _smoothstep(t: float) -> float:
    """Hermite smoothstep: 0→1 over t in [0,1]."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


class ThinkingMove(Move):
    """Continuous thinking animation that escalates over time."""

    @property
    def duration(self) -> float:
        return float("inf")

    def evaluate(
        self, t: float
    ) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        """Sample the thinking animation at time *t* seconds."""

        if t < STAGE_2_START:
            # --- Stage 1: subtle antenna only ---
            ant_amp = S1_ANTENNA_AMPLITUDE
            ant_freq = S1_ANTENNA_FREQ
            pitch_deg = 0.0
            yaw_deg = 0.0
            roll_deg = 0.0

        elif t < STAGE_3_START:
            # --- Stage 2: antenna + head sway, with blend-in ---
            blend = _smoothstep((t - STAGE_2_START) / BLEND_DURATION)
            ant_amp = S1_ANTENNA_AMPLITUDE + blend * (S2_ANTENNA_AMPLITUDE - S1_ANTENNA_AMPLITUDE)
            ant_freq = S1_ANTENNA_FREQ + blend * (S2_ANTENNA_FREQ - S1_ANTENNA_FREQ)
            pitch_deg = blend * S2_HEAD_PITCH_DEG * math.sin(2 * math.pi * S2_HEAD_PITCH_FREQ * t)
            yaw_deg = blend * S2_HEAD_YAW_DEG * math.sin(2 * math.pi * S2_HEAD_YAW_FREQ * t)
            roll_deg = 0.0

        else:
            # --- Stage 3: full thinking, with blend-in ---
            blend = _smoothstep((t - STAGE_3_START) / BLEND_DURATION)
            ant_amp = S2_ANTENNA_AMPLITUDE + blend * (S3_ANTENNA_AMPLITUDE - S2_ANTENNA_AMPLITUDE)
            ant_freq = S2_ANTENNA_FREQ + blend * (S3_ANTENNA_FREQ - S2_ANTENNA_FREQ)
            pitch_deg = (
                S2_HEAD_PITCH_DEG + blend * (S3_HEAD_PITCH_DEG - S2_HEAD_PITCH_DEG)
            ) * math.sin(2 * math.pi * S3_HEAD_PITCH_FREQ * t)
            yaw_deg = (
                S2_HEAD_YAW_DEG + blend * (S3_HEAD_YAW_DEG - S2_HEAD_YAW_DEG)
            ) * math.sin(2 * math.pi * S3_HEAD_YAW_FREQ * t)
            roll_deg = blend * S3_HEAD_ROLL_DEG * math.sin(2 * math.pi * S3_HEAD_ROLL_FREQ * t)

        # Antenna: asymmetric oscillation (one leads, one lags)
        ant_left = ant_amp * math.sin(2 * math.pi * ant_freq * t)
        ant_right = ant_amp * math.sin(2 * math.pi * ant_freq * t + math.pi * 0.6)
        antennas = np.array([ant_left, ant_right], dtype=np.float64)

        # Head pose
        head_pose = create_head_pose(
            x=0, y=0, z=0,
            roll=roll_deg, pitch=pitch_deg, yaw=yaw_deg,
            degrees=True,
        )

        return (head_pose, antennas, 0.0)
