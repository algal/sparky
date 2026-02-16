"""
Tests for SpontaneousGestureManager.

Verifies timing, state gating, presence gating, weighted selection,
and emotion selection without requiring the actual recorded emotions
library or robot hardware.
"""

import asyncio
import time
from collections import Counter
from unittest.mock import MagicMock, patch

import pytest

from sparky_mvp.core.spontaneous_gestures import (
    SpontaneousGestureManager,
    SPONTANEOUS_EMOTIONS,
)


def _mock_emotions_on(mgr):
    """Set up mock emotions library on a manager."""
    mock_moves = MagicMock()
    mock_moves.list_moves.return_value = list(SPONTANEOUS_EMOTIONS.keys())
    mgr._recorded_moves = mock_moves
    mgr._valid_emotions = list(SPONTANEOUS_EMOTIONS.keys())
    mgr._valid_weights = list(SPONTANEOUS_EMOTIONS.values())


class TestSpontaneousEmotionsDict:
    def test_emotions_not_empty(self):
        assert len(SPONTANEOUS_EMOTIONS) > 0

    def test_all_keys_are_strings(self):
        for e in SPONTANEOUS_EMOTIONS:
            assert isinstance(e, str)

    def test_all_weights_are_positive(self):
        for name, weight in SPONTANEOUS_EMOTIONS.items():
            assert weight > 0, f"{name} has non-positive weight {weight}"

    def test_known_emotions_present(self):
        assert "curious1" in SPONTANEOUS_EMOTIONS
        assert "thoughtful1" in SPONTANEOUS_EMOTIONS
        assert "tired1" in SPONTANEOUS_EMOTIONS
        assert "serenity1" in SPONTANEOUS_EMOTIONS

    def test_subtle_weighted_higher_than_dramatic(self):
        """Subtle gestures should be more likely than dramatic ones."""
        assert SPONTANEOUS_EMOTIONS["thoughtful1"] > SPONTANEOUS_EMOTIONS["sleep1"]
        assert SPONTANEOUS_EMOTIONS["serenity1"] > SPONTANEOUS_EMOTIONS["exhausted1"]
        assert SPONTANEOUS_EMOTIONS["curious1"] > SPONTANEOUS_EMOTIONS["boredom2"]


class TestWeightedSelection:
    def test_weights_bias_selection(self):
        """Over many draws, high-weight emotions should appear much more often."""
        import random
        random.seed(42)

        names = list(SPONTANEOUS_EMOTIONS.keys())
        weights = list(SPONTANEOUS_EMOTIONS.values())
        draws = random.choices(names, weights=weights, k=10000)
        counts = Counter(draws)

        # thoughtful1 (weight 10) should appear far more than sleep1 (weight 0.5)
        assert counts["thoughtful1"] > counts["sleep1"] * 5


class TestManagerConfig:
    def test_default_config(self):
        mgr = SpontaneousGestureManager(
            queue_move=MagicMock(),
            check_state=lambda: True,
        )
        assert mgr.min_interval_s == 2 * 60
        assert mgr.max_interval_s == 5 * 60
        assert mgr.require_presence is False

    def test_custom_config(self):
        mgr = SpontaneousGestureManager(
            queue_move=MagicMock(),
            check_state=lambda: True,
            config={
                "min_interval_m": 1,
                "max_interval_m": 3,
                "require_presence": True,
            },
        )
        assert mgr.min_interval_s == 60
        assert mgr.max_interval_s == 180
        assert mgr.require_presence is True


class TestManagerTiming:
    def test_next_delay_within_bounds(self):
        mgr = SpontaneousGestureManager(
            queue_move=MagicMock(),
            check_state=lambda: True,
            config={"min_interval_m": 2, "max_interval_m": 5},
        )
        for _ in range(100):
            delay = mgr._next_delay()
            # With ±20% jitter, bounds expand slightly
            assert delay >= 30.0  # Floor
            assert delay <= 5 * 60 * 1.2 + 1  # Max with jitter


class TestManagerStateGating:
    @pytest.mark.asyncio
    async def test_skips_when_not_interactive(self):
        """Gestures should be skipped when check_state returns False."""
        queue_move = MagicMock()
        call_count = 0

        def fake_check_state():
            nonlocal call_count
            call_count += 1
            return False  # Never in right state

        mgr = SpontaneousGestureManager(
            queue_move=queue_move,
            check_state=fake_check_state,
            config={"min_interval_m": 0.01, "max_interval_m": 0.01},
        )
        _mock_emotions_on(mgr)

        task = asyncio.create_task(mgr.run())
        await asyncio.sleep(0.5)
        mgr.stop()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except asyncio.CancelledError:
            pass

        # Should have checked state but never queued a move
        assert call_count > 0
        queue_move.assert_not_called()


class TestManagerPresenceGating:
    @pytest.mark.asyncio
    async def test_skips_when_presence_required_and_absent(self):
        """Gestures should be skipped when presence required but not detected."""
        queue_move = MagicMock()

        mgr = SpontaneousGestureManager(
            queue_move=queue_move,
            check_state=lambda: True,
            check_presence=lambda: False,  # No presence
            config={
                "min_interval_m": 0.01,
                "max_interval_m": 0.01,
                "require_presence": True,
            },
        )
        _mock_emotions_on(mgr)

        task = asyncio.create_task(mgr.run())
        await asyncio.sleep(0.5)
        mgr.stop()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except asyncio.CancelledError:
            pass

        queue_move.assert_not_called()


class TestManagerStop:
    @pytest.mark.asyncio
    async def test_stop_terminates_loop(self):
        mgr = SpontaneousGestureManager(
            queue_move=MagicMock(),
            check_state=lambda: True,
            config={"min_interval_m": 100, "max_interval_m": 200},
        )
        _mock_emotions_on(mgr)

        task = asyncio.create_task(mgr.run())
        await asyncio.sleep(0.1)
        mgr.stop()

        # Should stop within a few seconds (interruptible_sleep polls every 5s)
        await asyncio.wait_for(task, timeout=10.0)

    @pytest.mark.asyncio
    async def test_stops_when_no_valid_emotions(self):
        """Manager exits immediately if no valid emotions found."""
        mgr = SpontaneousGestureManager(
            queue_move=MagicMock(),
            check_state=lambda: True,
        )
        mgr._recorded_moves = MagicMock()
        mgr._valid_emotions = []
        mgr._valid_weights = []

        task = asyncio.create_task(mgr.run())
        await asyncio.wait_for(task, timeout=2.0)
        # Should have returned without error
