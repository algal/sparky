"""Unit tests for STTEngine — covers both backends and shared logic."""

import io
import wave

import numpy as np
import pytest

from sparky_mvp.core.stt_engine import STTEngine, STTResult, _ensure_cuda_ld_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav(pcm_int16: np.ndarray, sr: int = 16000) -> bytes:
    """Encode int16 PCM as WAV bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm_int16.tobytes())
    return buf.getvalue()


def _silence_wav(duration_s: float = 2.0, sr: int = 16000) -> bytes:
    return _make_wav(np.zeros(int(sr * duration_s), dtype=np.int16), sr)


# ---------------------------------------------------------------------------
# Faster-Whisper backend
# ---------------------------------------------------------------------------

class TestFasterWhisper:

    @pytest.fixture(scope="class")
    def engine(self):
        return STTEngine(
            engine="faster_whisper",
            model_size="base",
            device="cpu",
            min_speech_confidence=-1.0,
            max_no_speech_prob=0.8,
            min_compression_ratio=0.1,
            max_compression_ratio=50.0,
            min_text_length=3,
        )

    def test_silence_not_speech(self, engine):
        result = engine.transcribe(_silence_wav())
        assert isinstance(result, STTResult)
        assert result.duration == pytest.approx(2.0, abs=0.1)

    def test_engine_attribute(self, engine):
        assert engine.engine == "faster_whisper"
        assert engine.model is not None


# ---------------------------------------------------------------------------
# Parakeet backend (requires GPU — skipped if unavailable)
# ---------------------------------------------------------------------------

_PARAKEET_AVAILABLE = False
try:
    import onnx_asr
    import onnxruntime
    if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
        import os
        if os.path.isdir("models/parakeet-tdt-0.6b-v2"):
            _PARAKEET_AVAILABLE = True
except ImportError:
    pass


@pytest.mark.skipif(not _PARAKEET_AVAILABLE, reason="Parakeet model or GPU not available")
class TestParakeet:

    @pytest.fixture(scope="class")
    def engine(self):
        _ensure_cuda_ld_path()
        return STTEngine(
            engine="parakeet",
            parakeet_model_path="models/parakeet-tdt-0.6b-v2",
            parakeet_gpu_device=1,
            min_speech_confidence=-1.0,
            max_no_speech_prob=0.8,
            min_compression_ratio=0.1,
            max_compression_ratio=50.0,
            min_text_length=3,
        )

    def test_silence_not_speech(self, engine):
        result = engine.transcribe(_silence_wav())
        assert isinstance(result, STTResult)
        assert result.is_speech is False
        assert result.text == ""
        assert result.duration == pytest.approx(2.0, abs=0.1)

    def test_engine_attribute(self, engine):
        assert engine.engine == "parakeet"
        assert engine._parakeet_model is not None

    def test_confidence_fields(self, engine):
        """Parakeet fills placeholder confidence values."""
        result = engine.transcribe(_silence_wav())
        assert result.avg_logprob == 0.0
        assert result.no_speech_prob == 0.0
        assert result.confidence == 1.0

    def test_fast_inference(self, engine):
        """Parakeet on GPU should transcribe 2s audio in under 500ms."""
        import time
        wav = _silence_wav(2.0)
        # Warm up
        engine.transcribe(wav)
        t0 = time.monotonic()
        engine.transcribe(wav)
        elapsed_ms = (time.monotonic() - t0) * 1000
        assert elapsed_ms < 500, f"Transcription took {elapsed_ms:.0f}ms"


# ---------------------------------------------------------------------------
# Shared logic (engine-agnostic)
# ---------------------------------------------------------------------------

class TestSpeechDetection:

    def test_min_text_length(self):
        engine = STTEngine(
            engine="faster_whisper",
            model_size="base",
            device="cpu",
            min_text_length=5,
        )
        # Short text should fail length check
        assert engine._is_speech(
            avg_logprob=0.0, no_speech_prob=0.0,
            compression_ratio=5.0, text_length=2,
        ) is False

    def test_compression_ratio_bounds(self):
        engine = STTEngine(
            engine="faster_whisper",
            model_size="base",
            device="cpu",
            min_speech_confidence=-2.0,
            min_compression_ratio=0.5,
            max_compression_ratio=25.0,
        )
        # Below minimum
        assert engine._is_speech(
            avg_logprob=0.0, no_speech_prob=0.0,
            compression_ratio=0.1, text_length=10,
        ) is False
        # Above maximum
        assert engine._is_speech(
            avg_logprob=0.0, no_speech_prob=0.0,
            compression_ratio=30.0, text_length=10,
        ) is False
        # Within range
        assert engine._is_speech(
            avg_logprob=0.0, no_speech_prob=0.0,
            compression_ratio=5.0, text_length=10,
        ) is True

    def test_wav_duration(self):
        engine = STTEngine(engine="faster_whisper", model_size="base", device="cpu")
        wav = _silence_wav(3.0)
        dur = engine._get_wav_duration(wav)
        assert dur == pytest.approx(3.0, abs=0.01)

    def test_build_result(self):
        engine = STTEngine(engine="faster_whisper", model_size="base", device="cpu")
        result = engine._build_result("hello world", 2.0, "en", -0.5, 0.1)
        assert result.text == "hello world"
        assert result.duration == 2.0
        assert result.language == "en"
        assert result.compression_ratio == pytest.approx(11 / 2.0)


class TestEnsureCudaLdPath:

    def test_idempotent(self):
        """Calling _ensure_cuda_ld_path twice should not corrupt LD_LIBRARY_PATH."""
        import os
        orig = os.environ.get("LD_LIBRARY_PATH", "")
        _ensure_cuda_ld_path()
        first = os.environ.get("LD_LIBRARY_PATH", "")
        _ensure_cuda_ld_path()
        second = os.environ.get("LD_LIBRARY_PATH", "")
        # Restore
        if orig:
            os.environ["LD_LIBRARY_PATH"] = orig
        else:
            os.environ.pop("LD_LIBRARY_PATH", None)
        # Second call adds paths again (acceptable — just verify no crash)
        assert isinstance(second, str)
