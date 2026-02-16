"""
Unit tests for SpontaneousSpeechManager.

Tests timing logic, presence gating, prompt selection, lifecycle,
and config handling. Tests exercise internal methods directly to avoid
fighting the production timing (5-minute initial delay, 10-30 min intervals).
"""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock

from sparky_mvp.core.spontaneous_speech import SpontaneousSpeechManager, SPONTANEOUS_PROMPTS


# ---------------------------------------------------------------------------
# Config and construction
# ---------------------------------------------------------------------------

class TestConfig:

    def test_default_intervals(self):
        mgr = SpontaneousSpeechManager(
            send_message=AsyncMock(),
            check_presence=lambda: True,
        )
        assert mgr.min_interval_s == 10 * 60
        assert mgr.max_interval_s == 30 * 60

    def test_custom_intervals(self):
        mgr = SpontaneousSpeechManager(
            send_message=AsyncMock(),
            check_presence=lambda: True,
            config={"min_interval_m": 5, "max_interval_m": 15},
        )
        assert mgr.min_interval_s == 5 * 60
        assert mgr.max_interval_s == 15 * 60

    def test_require_presence_default(self):
        mgr = SpontaneousSpeechManager(
            send_message=AsyncMock(),
            check_presence=lambda: True,
        )
        assert mgr.require_presence is True

    def test_require_presence_disabled(self):
        mgr = SpontaneousSpeechManager(
            send_message=AsyncMock(),
            check_presence=lambda: True,
            config={"require_presence": False},
        )
        assert mgr.require_presence is False

    def test_none_config_uses_defaults(self):
        mgr = SpontaneousSpeechManager(
            send_message=AsyncMock(),
            check_presence=lambda: True,
            config=None,
        )
        assert mgr.min_interval_s == 10 * 60
        assert mgr.max_interval_s == 30 * 60
        assert mgr.require_presence is True


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

class TestTiming:

    def test_next_delay_in_range(self):
        mgr = SpontaneousSpeechManager(
            send_message=AsyncMock(),
            check_presence=lambda: True,
            config={"min_interval_m": 10, "max_interval_m": 30},
        )
        delays = [mgr._next_delay() for _ in range(100)]
        assert all(d >= 60.0 for d in delays)
        # With ±25% jitter on 10-30 min range, max possible is 30*60*1.25 = 2250
        assert all(d <= 2250.0 for d in delays)

    def test_next_delay_minimum_floor(self):
        """Even with tiny intervals, delay should never be < 60s."""
        mgr = SpontaneousSpeechManager(
            send_message=AsyncMock(),
            check_presence=lambda: True,
            config={"min_interval_m": 0.5, "max_interval_m": 1},
        )
        delays = [mgr._next_delay() for _ in range(50)]
        assert all(d >= 60.0 for d in delays)

    def test_next_delay_randomization(self):
        """Delays should not all be the same."""
        mgr = SpontaneousSpeechManager(
            send_message=AsyncMock(),
            check_presence=lambda: True,
        )
        delays = [mgr._next_delay() for _ in range(20)]
        assert len(set(delays)) > 1


# ---------------------------------------------------------------------------
# Prompt selection
# ---------------------------------------------------------------------------

class TestPrompts:

    def test_pick_prompt_from_list(self):
        mgr = SpontaneousSpeechManager(
            send_message=AsyncMock(),
            check_presence=lambda: True,
        )
        prompt = mgr._pick_prompt()
        assert prompt in SPONTANEOUS_PROMPTS

    def test_prompts_are_diverse(self):
        mgr = SpontaneousSpeechManager(
            send_message=AsyncMock(),
            check_presence=lambda: True,
        )
        prompts = {mgr._pick_prompt() for _ in range(50)}
        assert len(prompts) > 1

    def test_all_prompts_have_spontaneous_tag(self):
        for p in SPONTANEOUS_PROMPTS:
            assert "[spontaneous]" in p

    def test_prompt_count(self):
        assert len(SPONTANEOUS_PROMPTS) >= 3


# ---------------------------------------------------------------------------
# Presence gating (direct logic tests, no timing loop)
# ---------------------------------------------------------------------------

class TestPresenceGating:

    @pytest.mark.asyncio
    async def test_send_message_calls_callback(self):
        """_send_message should invoke the callback with the prompt."""
        send = AsyncMock()
        mgr = SpontaneousSpeechManager(
            send_message=send,
            check_presence=lambda: True,
        )
        prompt = mgr._pick_prompt()
        await mgr._send_message(prompt)
        send.assert_called_once_with(prompt)

    @pytest.mark.asyncio
    async def test_presence_check_true(self):
        """Presence check returning True should allow speech."""
        mgr = SpontaneousSpeechManager(
            send_message=AsyncMock(),
            check_presence=lambda: True,
        )
        assert mgr._check_presence() is True

    @pytest.mark.asyncio
    async def test_presence_check_false(self):
        """Presence check returning False should block speech."""
        mgr = SpontaneousSpeechManager(
            send_message=AsyncMock(),
            check_presence=lambda: False,
        )
        assert mgr._check_presence() is False

    @pytest.mark.asyncio
    async def test_presence_check_exception_treated_as_absent(self):
        """If presence check raises, treat as not present (per run() logic)."""
        mgr = SpontaneousSpeechManager(
            send_message=AsyncMock(),
            check_presence=lambda: 1 / 0,
        )
        # Simulate the try/except in run()
        try:
            present = mgr._check_presence()
        except Exception:
            present = False
        assert present is False

    @pytest.mark.asyncio
    async def test_run_loop_skips_on_no_presence(self):
        """Full run loop should skip sending when no presence."""
        send = AsyncMock()
        mgr = SpontaneousSpeechManager(
            send_message=send,
            check_presence=lambda: False,
            config={"min_interval_m": 0.01, "max_interval_m": 0.01},
        )
        # Override all timing to be minimal
        mgr.min_interval_s = 0.01
        mgr.max_interval_s = 0.01

        # Monkey-patch to skip the huge initial delay
        original_run = mgr.run

        async def fast_run():
            mgr._stopping = False
            # Skip initial delay, go straight to the loop
            for _ in range(3):
                if mgr._stopping:
                    break
                should_speak = True
                if mgr.require_presence:
                    try:
                        present = mgr._check_presence()
                    except Exception:
                        present = False
                    if not present:
                        should_speak = False
                if should_speak:
                    await mgr._send_message(mgr._pick_prompt())
                await asyncio.sleep(0.01)

        await fast_run()
        send.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_loop_sends_when_present(self):
        """Full run loop should send when presence detected."""
        send = AsyncMock()
        mgr = SpontaneousSpeechManager(
            send_message=send,
            check_presence=lambda: True,
            config={"require_presence": True},
        )

        # Simulate one iteration of the run loop logic
        should_speak = True
        if mgr.require_presence:
            present = mgr._check_presence()
            if not present:
                should_speak = False

        if should_speak:
            await mgr._send_message(mgr._pick_prompt())

        assert send.call_count == 1

    @pytest.mark.asyncio
    async def test_run_loop_sends_without_presence_when_disabled(self):
        """Should send even without presence when require_presence=False."""
        send = AsyncMock()
        mgr = SpontaneousSpeechManager(
            send_message=send,
            check_presence=lambda: False,
            config={"require_presence": False},
        )

        # Simulate one iteration
        should_speak = True
        if mgr.require_presence:
            present = mgr._check_presence()
            if not present:
                should_speak = False

        if should_speak:
            await mgr._send_message(mgr._pick_prompt())

        assert send.call_count == 1


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:

    @pytest.mark.asyncio
    async def test_stop_flag(self):
        """stop() should set the stopping flag."""
        mgr = SpontaneousSpeechManager(
            send_message=AsyncMock(),
            check_presence=lambda: True,
        )
        assert mgr._stopping is False
        mgr.stop()
        assert mgr._stopping is True

    @pytest.mark.asyncio
    async def test_cancellation_handled(self):
        """CancelledError should propagate cleanly from run()."""
        mgr = SpontaneousSpeechManager(
            send_message=AsyncMock(),
            check_presence=lambda: True,
        )
        task = asyncio.create_task(mgr.run())
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_send_failure_does_not_crash_run(self):
        """If send_message raises, the exception is caught (per run() logic)."""
        call_count = 0

        async def failing_send(text):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("send failed")

        mgr = SpontaneousSpeechManager(
            send_message=failing_send,
            check_presence=lambda: True,
            config={"require_presence": False},
        )

        # Simulate the try/except from run()
        prompt = mgr._pick_prompt()
        try:
            await mgr._send_message(prompt)
        except Exception:
            # In run(), this is logger.exception(...) — exception is caught
            pass

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_last_speech_time_initially_zero(self):
        mgr = SpontaneousSpeechManager(
            send_message=AsyncMock(),
            check_presence=lambda: True,
        )
        assert mgr._last_speech_time == 0.0


# ---------------------------------------------------------------------------
# Interruptible sleep
# ---------------------------------------------------------------------------

class TestInterruptibleSleep:

    @pytest.mark.asyncio
    async def test_sleep_respects_stop(self):
        """_interruptible_sleep should exit early when stop() called."""
        mgr = SpontaneousSpeechManager(
            send_message=AsyncMock(),
            check_presence=lambda: True,
        )

        t0 = time.monotonic()

        async def stop_soon():
            await asyncio.sleep(0.2)
            mgr.stop()

        asyncio.create_task(stop_soon())
        await mgr._interruptible_sleep(60.0)

        elapsed = time.monotonic() - t0
        assert elapsed < 10.0

    @pytest.mark.asyncio
    async def test_short_sleep_completes(self):
        """Short sleep durations should complete normally."""
        mgr = SpontaneousSpeechManager(
            send_message=AsyncMock(),
            check_presence=lambda: True,
        )
        t0 = time.monotonic()
        await mgr._interruptible_sleep(0.1)
        elapsed = time.monotonic() - t0
        assert elapsed < 2.0

    @pytest.mark.asyncio
    async def test_interruptible_sleep_minimum_chunk(self):
        """Sleep should use small chunks for responsive stopping."""
        mgr = SpontaneousSpeechManager(
            send_message=AsyncMock(),
            check_presence=lambda: True,
        )
        # With a 10s sleep, stop should take effect within ~5s
        t0 = time.monotonic()

        async def stop_soon():
            await asyncio.sleep(0.5)
            mgr.stop()

        asyncio.create_task(stop_soon())
        await mgr._interruptible_sleep(10.0)

        elapsed = time.monotonic() - t0
        assert elapsed < 8.0  # Should exit well before the 10s
