"""
Unit tests for speaker identification module.

Tests embedding extraction, speaker matching, threshold behavior,
and edge cases (short audio, missing enrollments, stereo input).
"""

import io
import json
import struct
import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest

from sparky_mvp.core.speaker_id import SpeakerIdentifier, _wav_bytes_to_float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav_bytes(duration_s: float = 3.0, freq: float = 440.0,
                    sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Generate a synthetic WAV file as bytes."""
    n_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    samples = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        if channels == 2:
            # Duplicate mono to stereo
            stereo = np.column_stack([samples, samples]).flatten()
            wf.writeframes(stereo.tobytes())
        else:
            wf.writeframes(samples.tobytes())
    return buf.getvalue()


def _make_enrollments_file(tmp_path: Path, speakers: dict[str, list[float]]) -> str:
    """Create a temporary enrollments JSON file."""
    path = tmp_path / "enrollments.json"
    with open(path, "w") as f:
        json.dump(speakers, f)
    return str(path)


# ---------------------------------------------------------------------------
# WAV conversion tests
# ---------------------------------------------------------------------------


class TestWavConversion:

    def test_mono_16khz(self):
        wav = _make_wav_bytes(duration_s=1.0, sample_rate=16000, channels=1)
        audio = _wav_bytes_to_float(wav)
        assert audio.dtype == np.float32
        assert len(audio) == 16000
        assert -1.0 <= audio.max() <= 1.0

    def test_stereo_to_mono(self):
        wav = _make_wav_bytes(duration_s=1.0, sample_rate=16000, channels=2)
        audio = _wav_bytes_to_float(wav)
        assert len(audio) == 16000  # mono output

    def test_resample_44100_to_16000(self):
        wav = _make_wav_bytes(duration_s=1.0, sample_rate=44100, channels=1)
        audio = _wav_bytes_to_float(wav)
        # Should be ~16000 samples after resample
        assert abs(len(audio) - 16000) < 100


# ---------------------------------------------------------------------------
# SpeakerIdentifier tests
# ---------------------------------------------------------------------------


class TestSpeakerIdentifier:

    def test_missing_enrollments_file(self, tmp_path):
        si = SpeakerIdentifier(str(tmp_path / "nonexistent.json"))
        assert si.n_speakers == 0
        name, conf = si.identify(_make_wav_bytes())
        assert name == "unknown"
        assert conf == 0.0

    def test_load_enrollments(self, tmp_path):
        emb = [0.1] * 256
        path = _make_enrollments_file(tmp_path, {"alice": emb, "bob": emb})
        si = SpeakerIdentifier(path)
        assert si.n_speakers == 2

    def test_short_audio_returns_unknown(self, tmp_path):
        emb = [0.1] * 256
        path = _make_enrollments_file(tmp_path, {"alice": emb})
        si = SpeakerIdentifier(path)
        # 0.5 second clip — too short
        wav = _make_wav_bytes(duration_s=0.5)
        name, conf = si.identify(wav)
        assert name == "unknown"

    def test_threshold_filtering(self, tmp_path):
        """With a very high threshold, no speaker should match."""
        emb = np.random.randn(256).tolist()
        path = _make_enrollments_file(tmp_path, {"alice": emb})
        si = SpeakerIdentifier(path, threshold=0.99)
        wav = _make_wav_bytes(duration_s=3.0)
        name, _ = si.identify(wav)
        assert name == "unknown"


# ---------------------------------------------------------------------------
# Voice prefix construction
# ---------------------------------------------------------------------------


class TestVoicePrefix:

    def test_prefix_without_speaker(self):
        from sparky_mvp.core.middlewares.openclaw_provider import OpenClawProviderMiddleware
        from unittest.mock import MagicMock

        provider = OpenClawProviderMiddleware(
            gateway_client=MagicMock(),
            session_key="test",
        )
        provider.set_user_message("hello")
        provider.set_speaker_name(None)
        msg = provider._build_voice_message()
        assert "Speaker:" not in msg
        assert "hello" in msg

    def test_prefix_with_speaker(self):
        from sparky_mvp.core.middlewares.openclaw_provider import OpenClawProviderMiddleware
        from unittest.mock import MagicMock

        provider = OpenClawProviderMiddleware(
            gateway_client=MagicMock(),
            session_key="test",
        )
        provider.set_user_message("hello")
        provider.set_speaker_name("alexis")
        msg = provider._build_voice_message()
        assert "Speaker identified by voice recognition: alexis" in msg
        assert "hello" in msg
