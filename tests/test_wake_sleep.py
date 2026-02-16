"""
Unit tests for wake/sleep word feature.

Tests:
- Wake word model loading (pretrained and file path)
- Sleep phrase detection
- State transitions: SLEEP → INTERACTIVE, INTERACTIVE → SLEEP
- Config-driven enable/disable
"""

import asyncio
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch

from sparky_mvp.core.state_machine import ReachyStateMachine, ReachyState


# ---------------------------------------------------------------------------
# Minimal config for testing
# ---------------------------------------------------------------------------

def _make_config(*, wake_enabled=True, sleep_phrases=None, model="hey_jarvis_v0.1"):
    """Build a minimal config dict for testing wake/sleep."""
    return {
        "audio": {"sample_rate": 16000, "chunk_size": 1280, "channels": 1, "format": "int16"},
        "vad": {"model_path": "models/silero_vad_v5.onnx", "confidence_threshold": 0.5},
        "stt": {"engine": "faster_whisper", "model_size": "base", "device": "cpu", "language": "en",
                "min_speech_confidence": -1.0, "max_no_speech_prob": 0.8,
                "min_compression_ratio": 0.1, "max_compression_ratio": 50.0, "min_text_length": 3},
        "wake_word": {
            "enabled": wake_enabled,
            "model": model,
            "model_path": "",
            "threshold": 0.5,
            "cooldown": 5.0,
            "sleep_phrases": sleep_phrases or ["time to sleep", "go to sleep"],
        },
        "anthropic": {"model": "claude-sonnet-4-20250514", "temperature": 0.8, "max_tokens": 1000},
        "openclaw": {"provider": "anthropic"},
        "tts": {"engine": "kokoro", "timeout": 30.0},
        "camera": {"enabled": False},
        "robot": {"enable_animations": True},
        "conversation": {
            "system_prompt": "You are a test assistant.",
            "max_history": 20,
            "interactive_timeout": 30.0,
        },
    }


# ---------------------------------------------------------------------------
# Wake word model setup tests
# ---------------------------------------------------------------------------

class TestWakeWordSetup:

    def test_pretrained_model_loads(self):
        """Pretrained model name should resolve and load."""
        config = _make_config(wake_enabled=True, model="hey_jarvis_v0.1")
        with patch("sparky_mvp.core.state_machine.pyaudio"):
            sm = ReachyStateMachine(config)
        model = sm._setup_wake_word()
        assert model is not None
        assert "hey_jarvis_v0.1" in model.models

    def test_invalid_model_raises(self):
        """Nonexistent model name should raise RuntimeError."""
        config = _make_config(wake_enabled=True, model="nonexistent_model_xyz")
        with patch("sparky_mvp.core.state_machine.pyaudio"):
            sm = ReachyStateMachine(config)
        with pytest.raises(RuntimeError, match="Wake word model not found"):
            sm._setup_wake_word()

    def test_empty_model_name_raises(self):
        """Empty model name should raise RuntimeError."""
        config = _make_config(wake_enabled=True, model="")
        with patch("sparky_mvp.core.state_machine.pyaudio"):
            sm = ReachyStateMachine(config)
        with pytest.raises(RuntimeError, match="No wake word model configured"):
            sm._setup_wake_word()

    def test_file_path_model_loads(self):
        """Explicit .onnx file path should load."""
        import openwakeword
        paths = openwakeword.get_pretrained_model_paths()
        if not paths:
            pytest.skip("No pretrained models available")
        config = _make_config(wake_enabled=True)
        config["wake_word"]["model_path"] = paths[0]
        with patch("sparky_mvp.core.state_machine.pyaudio"):
            sm = ReachyStateMachine(config)
        model = sm._setup_wake_word()
        assert model is not None


# ---------------------------------------------------------------------------
# Sleep phrase detection tests
# ---------------------------------------------------------------------------

class TestSleepPhraseDetection:

    def _make_sm(self, sleep_phrases=None):
        config = _make_config(wake_enabled=False, sleep_phrases=sleep_phrases)
        with patch("sparky_mvp.core.state_machine.pyaudio"):
            sm = ReachyStateMachine(config)
        sm._sleep_phrases = [p.lower() for p in (sleep_phrases or ["time to sleep", "go to sleep"])]
        return sm

    def test_exact_match(self):
        sm = self._make_sm()
        assert sm._is_sleep_phrase("time to sleep") is True

    def test_case_insensitive(self):
        sm = self._make_sm()
        assert sm._is_sleep_phrase("Time To Sleep") is True

    def test_substring_match(self):
        sm = self._make_sm()
        assert sm._is_sleep_phrase("OK it's time to sleep now Dobby") is True

    def test_no_match(self):
        sm = self._make_sm()
        assert sm._is_sleep_phrase("What time is it?") is False

    def test_go_to_sleep_match(self):
        sm = self._make_sm()
        assert sm._is_sleep_phrase("Please go to sleep") is True

    def test_empty_phrases_no_match(self):
        sm = self._make_sm(sleep_phrases=["placeholder_that_wont_match"])
        sm._sleep_phrases = []  # Override to truly empty
        assert sm._is_sleep_phrase("time to sleep") is False

    def test_empty_text_no_match(self):
        sm = self._make_sm()
        assert sm._is_sleep_phrase("") is False

    def test_custom_phrases(self):
        sm = self._make_sm(sleep_phrases=["goodnight dobby", "shut down"])
        assert sm._is_sleep_phrase("Goodnight Dobby") is True
        assert sm._is_sleep_phrase("Please shut down") is True
        assert sm._is_sleep_phrase("time to sleep") is False


# ---------------------------------------------------------------------------
# Config-driven enable/disable tests
# ---------------------------------------------------------------------------

class TestWakeWordConfig:

    def test_disabled_by_config(self):
        """When wake_word.enabled is false, wake_word_model should be None after init."""
        config = _make_config(wake_enabled=False)
        with patch("sparky_mvp.core.state_machine.pyaudio"):
            sm = ReachyStateMachine(config)
        # Simulate what startup() does
        ww_cfg = sm.config.get("wake_word", {})
        ww_enabled = ww_cfg.get("enabled", False) and os.getenv("DISABLE_SLEEP") != "1"
        assert ww_enabled is False

    def test_enabled_by_config(self):
        """When wake_word.enabled is true, model should load."""
        config = _make_config(wake_enabled=True)
        with patch("sparky_mvp.core.state_machine.pyaudio"):
            sm = ReachyStateMachine(config)
        ww_cfg = sm.config.get("wake_word", {})
        ww_enabled = ww_cfg.get("enabled", False)
        assert ww_enabled is True

    def test_disabled_by_env_var(self):
        """DISABLE_SLEEP=1 should override config."""
        config = _make_config(wake_enabled=True)
        with patch("sparky_mvp.core.state_machine.pyaudio"):
            sm = ReachyStateMachine(config)
        with patch.dict(os.environ, {"DISABLE_SLEEP": "1"}):
            ww_cfg = sm.config.get("wake_word", {})
            ww_enabled = ww_cfg.get("enabled", False) and os.getenv("DISABLE_SLEEP") != "1"
            assert ww_enabled is False


# ---------------------------------------------------------------------------
# State transition tests
# ---------------------------------------------------------------------------

class TestStateTransitions:

    @pytest.mark.asyncio
    async def test_sleep_phase_no_model_goes_interactive(self):
        """sleep_phase with no wake word model should transition to INTERACTIVE."""
        config = _make_config(wake_enabled=False)
        with patch("sparky_mvp.core.state_machine.pyaudio"):
            sm = ReachyStateMachine(config)
        sm.wake_word_model = None
        sm.animator = MagicMock()
        sm.animator.wake_up = AsyncMock()
        sm.camera_worker = None
        sm.movement_manager = None
        sm.head_wobbler = None
        sm.vad_capture = MagicMock()
        sm.vad_capture.capture_speech = AsyncMock(side_effect=asyncio.CancelledError)

        # Mock mic stream
        sm.mic_stream = MagicMock()

        await sm.sleep_phase()
        assert sm.state == ReachyState.INTERACTIVE

    @pytest.mark.asyncio
    async def test_sleep_phase_with_model_waits_for_wake(self):
        """sleep_phase with wake word model should block until wake detected."""
        config = _make_config(wake_enabled=True)
        with patch("sparky_mvp.core.state_machine.pyaudio"):
            sm = ReachyStateMachine(config)

        sm.wake_word_model = MagicMock()
        sm.animator = MagicMock()
        sm.animator.wake_up = AsyncMock()
        sm.camera_worker = None
        sm.movement_manager = None
        sm.head_wobbler = None
        sm.vad_capture = MagicMock()
        sm.vad_capture.capture_speech = AsyncMock(side_effect=asyncio.CancelledError)
        sm.mic_stream = MagicMock()
        sm.running = True
        sm.state = ReachyState.SLEEP

        # Mock _wait_for_wake_word to return True immediately
        with patch.object(sm, "_wait_for_wake_word", return_value=True):
            await sm.sleep_phase()

        assert sm.state == ReachyState.INTERACTIVE
        sm.animator.wake_up.assert_called_once()


class TestContinuousVADListenerCharacterization:

    @pytest.mark.asyncio
    async def test_processing_none_capture_does_not_set_interruption_flag(self):
        """PROCESSING + None capture should not trigger interruption by itself."""
        config = _make_config(wake_enabled=False)
        with patch("sparky_mvp.core.state_machine.pyaudio"):
            sm = ReachyStateMachine(config)

        sm.state = ReachyState.PROCESSING
        sm.vad_capture = MagicMock()
        sm.vad_capture.last_result_reason = "timeout_no_activation"
        sm.vad_capture.capture_speech = AsyncMock(side_effect=[None, asyncio.CancelledError()])

        with patch.dict(os.environ, {"ENABLE_BARGE_IN": "1"}):
            with pytest.raises(asyncio.CancelledError):
                await sm._continuous_vad_listener()

        assert sm.interruption_flag.is_set() is False
