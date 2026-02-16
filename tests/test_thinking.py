"""Unit tests for staged thinking animation."""

import math
import pytest
import numpy as np

from sparky_mvp.robot.thinking import (
    ThinkingMove,
    STAGE_2_START,
    STAGE_3_START,
    S1_ANTENNA_AMPLITUDE,
    S3_ANTENNA_AMPLITUDE,
    _smoothstep,
)


class TestSmoothstep:

    def test_zero(self):
        assert _smoothstep(0.0) == 0.0

    def test_one(self):
        assert _smoothstep(1.0) == 1.0

    def test_half(self):
        assert _smoothstep(0.5) == 0.5

    def test_clamped_below(self):
        assert _smoothstep(-0.5) == 0.0

    def test_clamped_above(self):
        assert _smoothstep(1.5) == 1.0


class TestThinkingMove:

    def test_infinite_duration(self):
        move = ThinkingMove()
        assert move.duration == float("inf")

    def test_evaluate_returns_correct_shape(self):
        move = ThinkingMove()
        head_pose, antennas, body_yaw = move.evaluate(1.0)
        assert head_pose.shape == (4, 4)
        assert antennas.shape == (2,)
        assert isinstance(body_yaw, float)
        assert body_yaw == 0.0

    def test_stage1_no_head_movement(self):
        """Stage 1 (0-2s) should have zero head rotation."""
        move = ThinkingMove()
        # Sample in the middle of stage 1
        head_pose, antennas, _ = move.evaluate(1.0)
        # Extract rotation from pose matrix (3x3 upper-left)
        rot = head_pose[:3, :3]
        # Identity rotation means no head movement
        identity = np.eye(3)
        np.testing.assert_allclose(rot, identity, atol=1e-6)

    def test_stage1_has_antenna_movement(self):
        """Stage 1 should have non-zero antenna values at non-zero time."""
        move = ThinkingMove()
        # Sample at t=0.5 (where sin is non-zero for the given freq)
        _, antennas, _ = move.evaluate(0.5)
        # At least one antenna should be moving
        assert np.any(np.abs(antennas) > 0.01)

    def test_stage1_antenna_bounded(self):
        """Stage 1 antenna amplitude should not exceed S1 limit."""
        move = ThinkingMove()
        max_amp = 0.0
        for t_ms in range(0, int(STAGE_2_START * 1000), 10):
            _, antennas, _ = move.evaluate(t_ms / 1000.0)
            max_amp = max(max_amp, float(np.max(np.abs(antennas))))
        # Should stay within S1 amplitude (with small margin for float precision)
        assert max_amp <= S1_ANTENNA_AMPLITUDE + 0.01

    def test_stage2_has_head_movement(self):
        """Stage 2 (2-5s) should have non-zero head rotation after blend."""
        move = ThinkingMove()
        # Sample well into stage 2 (after blend completes)
        head_pose, _, _ = move.evaluate(4.0)
        rot = head_pose[:3, :3]
        identity = np.eye(3)
        # Should NOT be identity (head is moving)
        assert not np.allclose(rot, identity, atol=1e-4)

    def test_stage3_larger_antenna_than_stage1(self):
        """Stage 3 should have larger antenna amplitude than stage 1."""
        move = ThinkingMove()
        # Collect max amplitudes for each stage
        s1_max = 0.0
        for t_ms in range(0, int(STAGE_2_START * 1000), 10):
            _, antennas, _ = move.evaluate(t_ms / 1000.0)
            s1_max = max(s1_max, float(np.max(np.abs(antennas))))

        s3_max = 0.0
        for t_ms in range(int((STAGE_3_START + 2.0) * 1000), int((STAGE_3_START + 5.0) * 1000), 10):
            _, antennas, _ = move.evaluate(t_ms / 1000.0)
            s3_max = max(s3_max, float(np.max(np.abs(antennas))))

        assert s3_max > s1_max

    def test_smooth_transition_at_stage_boundary(self):
        """Antenna amplitude should change smoothly across stage 2 boundary."""
        move = ThinkingMove()
        prev_amps = None
        for dt_ms in range(-200, 200, 10):
            t = STAGE_2_START + dt_ms / 1000.0
            if t < 0:
                continue
            _, antennas, _ = move.evaluate(t)
            amps = np.abs(antennas)
            if prev_amps is not None:
                # Change between samples should be smooth (no jumps > 0.05 rad)
                diff = float(np.max(np.abs(amps - prev_amps)))
                assert diff < 0.05, f"Jump at t={t:.3f}s: {diff:.4f} rad"
            prev_amps = amps
