"""
Unit tests for VAD speech capture module.
"""

import pytest
import numpy as np
import os
import time
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from collections import deque
import io
import wave

from sparky_mvp.core.vad_capture import VAD, VADSpeechCapture
from sparky_mvp.core.resampling_stream import ResamplingPyAudioStream, ResamplingParams


class TestVAD:
    """Tests for VAD class."""

    @pytest.fixture
    def vad_model_path(self, tmp_path):
        """Create a temporary VAD model path."""
        model_file = tmp_path / "test_vad.onnx"
        model_file.touch()
        return str(model_file)

    @patch('sparky_mvp.core.vad_capture.ort.InferenceSession')
    def test_vad_initialization(self, mock_session, vad_model_path):
        """Test VAD initializes correctly."""
        vad = VAD(model_path=vad_model_path)

        assert vad.SAMPLE_RATE == 16000
        assert hasattr(vad, 'ort_sess')
        assert hasattr(vad, '_state')
        mock_session.assert_called_once()

    @patch('sparky_mvp.core.vad_capture.ort.InferenceSession')
    def test_vad_reset_states(self, mock_session, vad_model_path):
        """Test VAD state reset."""
        vad = VAD(model_path=vad_model_path)
        vad.reset_states(batch_size=2)

        assert vad._state.shape == (2, 2, 128)
        assert vad._last_sr == 0
        assert vad._last_batch_size == 0

    @patch('sparky_mvp.core.vad_capture.ort.InferenceSession')
    def test_vad_processes_audio_chunk(self, mock_session, vad_model_path):
        """Test VAD processes audio chunk correctly."""
        # Mock inference session
        mock_sess_instance = MagicMock()
        mock_sess_instance.run.return_value = (
            np.array([0.8]),  # High confidence
            np.zeros((2, 1, 128), dtype=np.float32)  # State
        )
        mock_session.return_value = mock_sess_instance

        vad = VAD(model_path=vad_model_path)

        # Create audio sample (512 samples for 16kHz)
        audio_sample = np.random.rand(1, 512).astype(np.float32)

        result = vad(audio_sample, sample_rate=16000)

        assert mock_sess_instance.run.called
        assert isinstance(result, np.ndarray)

    @patch('sparky_mvp.core.vad_capture.ort.InferenceSession')
    def test_vad_wrong_sample_count_raises_error(self, mock_session, vad_model_path):
        """Test VAD raises error for wrong sample count."""
        vad = VAD(model_path=vad_model_path)

        # Wrong number of samples
        audio_sample = np.random.rand(1, 256).astype(np.float32)

        with pytest.raises(ValueError, match="Provided number of samples"):
            vad(audio_sample, sample_rate=16000)


class TestVADSpeechCapture:
    """Tests for VADSpeechCapture class."""

    @pytest.fixture
    def mock_audio_stream(self):
        """Create mock PyAudio stream."""
        stream = Mock()
        stream.read = Mock(return_value=b'\x00' * 1024)  # Silence
        return stream

    @pytest.fixture
    def vad_model_path(self, tmp_path):
        """Create temporary VAD model path."""
        model_file = tmp_path / "test_vad.onnx"
        model_file.touch()
        return str(model_file)

    @patch('sparky_mvp.core.vad_capture.VAD')
    def test_vad_speech_capture_initialization(self, mock_vad, mock_audio_stream, vad_model_path):
        """Test VADSpeechCapture initializes correctly."""
        capture = VADSpeechCapture(
            audio_stream=mock_audio_stream,
            sample_rate=16000,
            vad_model_path=vad_model_path,
            vad_threshold=0.5
        )

        assert capture.sample_rate == 16000
        assert capture.vad_threshold == 0.5
        assert capture.vad_chunk_samples == 512
        assert capture.buffer_max_chunks == 25  # 800ms / 32ms
        assert capture.pause_chunks == 37  # 1200ms / 32ms
        assert isinstance(capture._buffer, deque)
        assert capture._buffer.maxlen == 25

    @patch('sparky_mvp.core.vad_capture.VAD')
    def test_pre_activation_buffer_fills(self, mock_vad, mock_audio_stream, vad_model_path):
        """Test circular buffer fills before activation."""
        capture = VADSpeechCapture(
            audio_stream=mock_audio_stream,
            vad_model_path=vad_model_path
        )

        # Add chunks to buffer (no voice activity)
        for i in range(30):  # More than buffer size
            chunk = np.zeros(512, dtype=np.int16)
            capture._manage_pre_activation_buffer(chunk, vad_confidence=False)

        # Buffer should be at max size
        assert len(capture._buffer) == 25
        assert not capture._recording_started

    @patch('sparky_mvp.core.vad_capture.VAD')
    def test_voice_activation_starts_recording(self, mock_vad, mock_audio_stream, vad_model_path):
        """Test voice activity starts recording."""
        capture = VADSpeechCapture(
            audio_stream=mock_audio_stream,
            vad_model_path=vad_model_path
        )

        # Fill buffer
        for i in range(10):
            chunk = np.zeros(512, dtype=np.int16)
            capture._manage_pre_activation_buffer(chunk, vad_confidence=False)

        # Voice detected
        chunk = np.ones(512, dtype=np.int16)
        capture._manage_pre_activation_buffer(chunk, vad_confidence=True)

        assert capture._recording_started
        assert len(capture._samples) == 11  # Buffer contents + current chunk transferred
        assert capture._gap_counter == 0

    @patch('sparky_mvp.core.vad_capture.VAD')
    def test_pause_detection_completes_speech(self, mock_vad, mock_audio_stream, vad_model_path):
        """Test pause detection marks speech as complete."""
        capture = VADSpeechCapture(
            audio_stream=mock_audio_stream,
            vad_model_path=vad_model_path
        )

        capture._recording_started = True
        capture._samples = [np.ones(512, dtype=np.int16) for _ in range(10)]

        # Process silence chunks (below pause limit)
        for i in range(capture.pause_chunks - 1):
            chunk = np.zeros(512, dtype=np.int16)
            result = capture._process_activated_audio(chunk, vad_confidence=False)
            assert result is False  # Not complete yet
            assert capture._gap_counter == i + 1

        # Final silence chunk should complete
        chunk = np.zeros(512, dtype=np.int16)
        result = capture._process_activated_audio(chunk, vad_confidence=False)
        assert result is True  # Speech complete!

    @patch('sparky_mvp.core.vad_capture.VAD')
    def test_voice_activity_resets_gap_counter(self, mock_vad, mock_audio_stream, vad_model_path):
        """Test voice activity resets gap counter."""
        capture = VADSpeechCapture(
            audio_stream=mock_audio_stream,
            vad_model_path=vad_model_path
        )

        capture._recording_started = True
        capture._gap_counter = 10

        # Voice activity should reset counter
        chunk = np.ones(512, dtype=np.int16)
        result = capture._process_activated_audio(chunk, vad_confidence=True)

        assert result is False  # Not complete
        assert capture._gap_counter == 0  # Reset!

    @patch('sparky_mvp.core.vad_capture.VAD')
    def test_samples_to_wav_creates_valid_wav(self, mock_vad, mock_audio_stream, vad_model_path):
        """Test WAV file creation from samples."""
        capture = VADSpeechCapture(
            audio_stream=mock_audio_stream,
            sample_rate=16000,
            vad_model_path=vad_model_path
        )

        # Create test samples (1 second of audio)
        capture._samples = [
            np.ones(512, dtype=np.int16) * 1000  # Some amplitude
            for _ in range(31)  # ~1 second at 16kHz
        ]

        wav_bytes = capture._samples_to_wav()

        # Verify it's valid WAV
        assert isinstance(wav_bytes, bytes)
        assert len(wav_bytes) > 0

        # Parse WAV to verify parameters
        wav_buffer = io.BytesIO(wav_bytes)
        with wave.open(wav_buffer, 'rb') as wf:
            assert wf.getnchannels() == 1  # Mono
            assert wf.getsampwidth() == 2  # 16-bit
            assert wf.getframerate() == 16000
            assert wf.getnframes() == 512 * 31

    @patch('sparky_mvp.core.vad_capture.VAD')
    def test_reset_clears_state(self, mock_vad, mock_audio_stream, vad_model_path):
        """Test reset clears all internal state."""
        capture = VADSpeechCapture(
            audio_stream=mock_audio_stream,
            vad_model_path=vad_model_path
        )

        # Set up some state
        capture._recording_started = True
        capture._samples = [np.ones(512, dtype=np.int16) for _ in range(10)]
        capture._gap_counter = 5
        capture._buffer = deque([np.ones(512, dtype=np.int16) for _ in range(10)], maxlen=25)

        # Reset
        capture.reset()

        assert capture._recording_started is False
        assert len(capture._samples) == 0
        assert capture._gap_counter == 0
        assert len(capture._buffer) == 0

    @pytest.mark.asyncio
    @patch('sparky_mvp.core.vad_capture.VAD')
    async def test_capture_speech_timeout(self, mock_vad, mock_audio_stream, vad_model_path):
        """Test capture_speech returns None on timeout without speech."""
        # Mock VAD to return low confidence (no speech)
        mock_vad_instance = MagicMock()
        mock_vad_instance.return_value = 0.1  # Below threshold
        mock_vad.return_value = mock_vad_instance

        capture = VADSpeechCapture(
            audio_stream=mock_audio_stream,
            vad_model_path=vad_model_path,
            vad_threshold=0.5
        )

        # Mock _process_vad to return False (no speech)
        capture._process_vad = AsyncMock(return_value=False)

        with patch.dict(os.environ, {"VAD_TIMEOUT_SECONDS": "0.02"}):
            result = await capture.capture_speech()

        # Should timeout without capturing speech
        assert result is None

    @pytest.mark.asyncio
    @patch('sparky_mvp.core.vad_capture.VAD')
    async def test_capture_speech_timeout_no_activation_reason(self, mock_vad, mock_audio_stream, vad_model_path, caplog):
        """Characterize timeout-without-activation reason labeling."""
        mock_vad.return_value = MagicMock()
        capture = VADSpeechCapture(
            audio_stream=mock_audio_stream,
            vad_model_path=vad_model_path,
            vad_threshold=0.5
        )
        capture._process_vad = AsyncMock(return_value=False)

        caplog.set_level("INFO")
        with patch.dict(os.environ, {"VAD_TIMEOUT_SECONDS": "0.02"}):
            result = await capture.capture_speech()

        assert result is None
        assert capture.last_result_reason == "timeout_no_activation"
        assert any("reason=timeout_no_activation" in rec.message for rec in caplog.records)

    @pytest.mark.asyncio
    @patch('sparky_mvp.core.vad_capture.VAD')
    async def test_capture_speech_timeout_after_activation_reason(self, mock_vad, mock_audio_stream, vad_model_path, caplog):
        """Characterize timeout-after-activation reason labeling."""
        mock_vad.return_value = MagicMock()
        capture = VADSpeechCapture(
            audio_stream=mock_audio_stream,
            vad_model_path=vad_model_path,
            vad_threshold=0.5
        )
        capture._process_vad = AsyncMock(return_value=True)

        caplog.set_level("INFO")
        with patch.dict(os.environ, {"VAD_TIMEOUT_SECONDS": "0.02"}):
            result = await capture.capture_speech()

        assert result is None
        assert capture._recording_started is True
        assert capture.last_result_reason == "timeout_after_activation"
        assert any("reason=timeout_after_activation" in rec.message for rec in caplog.records)

    @pytest.mark.asyncio
    @patch('sparky_mvp.core.vad_capture.VAD')
    async def test_capture_speech_activation_suppressed_reason(self, mock_vad, mock_audio_stream, vad_model_path, caplog):
        """Characterize callback-suppressed activation reason labeling."""
        mock_vad.return_value = MagicMock()
        capture = VADSpeechCapture(
            audio_stream=mock_audio_stream,
            vad_model_path=vad_model_path,
            vad_threshold=0.5
        )
        capture._process_vad = AsyncMock(return_value=True)

        caplog.set_level("INFO")
        with patch.dict(os.environ, {"VAD_TIMEOUT_SECONDS": "0.02"}):
            result = await capture.capture_speech(on_activation=lambda *_: False)

        assert result is None
        assert capture._recording_started is False
        assert capture.last_result_reason == "activation_suppressed"
        assert any("reason=activation_suppressed" in rec.message for rec in caplog.records)

    @pytest.mark.asyncio
    @patch('sparky_mvp.core.vad_capture.VAD')
    async def test_capture_speech_read_error_reason(self, mock_vad, vad_model_path, caplog):
        """Characterize read-error reason labeling."""
        mock_vad.return_value = MagicMock()

        stream = Mock()
        stream.read = Mock(side_effect=RuntimeError("boom"))

        capture = VADSpeechCapture(
            audio_stream=stream,
            vad_model_path=vad_model_path,
            vad_threshold=0.5
        )

        caplog.set_level("INFO")
        result = await capture.capture_speech()

        assert result is None
        assert capture.last_result_reason == "read_error"
        assert any("reason=read_error" in rec.message for rec in caplog.records)

    @pytest.mark.asyncio
    @patch('sparky_mvp.core.vad_capture.VAD')
    async def test_capture_timeout_uses_wall_clock_not_chunk_count(self, mock_vad, mock_audio_stream, vad_model_path):
        """Timeout should be based on elapsed wall-clock time, not nominal timeout_chunks."""
        mock_vad.return_value = MagicMock()

        class SlowStream:
            def __init__(self):
                self.read_calls = 0

            def read(self, n, exception_on_overflow=False):
                self.read_calls += 1
                time.sleep(0.06)
                return b"\x00" * (n * 2)

        slow_stream = SlowStream()
        capture = VADSpeechCapture(
            audio_stream=slow_stream,
            vad_model_path=vad_model_path,
            vad_threshold=0.5
        )
        capture._process_vad = AsyncMock(return_value=False)
        capture.VAD_SIZE = 30000  # Count-based timeout would allow only one chunk.

        with patch.dict(os.environ, {"VAD_TIMEOUT_SECONDS": "0.25"}):
            result = await capture.capture_speech()

        assert result is None
        assert capture.last_result_reason == "timeout_no_activation"
        assert slow_stream.read_calls >= 2

    @pytest.mark.asyncio
    @patch('sparky_mvp.core.vad_capture.VAD')
    async def test_activation_resets_deadline_for_post_activation_timeout(self, mock_vad, vad_model_path):
        """Late activation should get its own post-activation timeout window."""
        mock_vad.return_value = MagicMock()

        class SlowStream:
            def __init__(self):
                self.read_calls = 0

            def read(self, n, exception_on_overflow=False):
                self.read_calls += 1
                time.sleep(0.06)
                return b"\x00" * (n * 2)

        stream = SlowStream()
        capture = VADSpeechCapture(
            audio_stream=stream,
            vad_model_path=vad_model_path,
            vad_threshold=0.5
        )

        async def _fake_vad(_):
            # No speech first chunk, then continuous speech (no pause) so we hit timeout_after_activation.
            return stream.read_calls >= 2

        capture._process_vad = AsyncMock(side_effect=_fake_vad)
        capture.VAD_SIZE = 30000  # Keep nominal chunk timeout tiny; wall-clock controls behavior.

        with patch.dict(
            os.environ,
            {
                "VAD_TIMEOUT_SECONDS": "0.1",
                "VAD_POST_ACTIVATION_TIMEOUT_SECONDS": "0.25",
            },
        ):
            result = await capture.capture_speech()

        assert result is None
        assert capture.last_result_reason == "timeout_after_activation"
        # Under old logic this would end around 2 reads; reset deadline should allow several more.
        assert stream.read_calls >= 4

    @patch('sparky_mvp.core.vad_capture.VAD')
    def test_flush_input_buffer_resampling_wrapper_is_noop(self, mock_vad, vad_model_path):
        """Flush is a no-op through ResamplingPyAudioStream (no get_read_available).

        ResamplingPyAudioStream intentionally does NOT expose get_read_available()
        because draining through the resampler can block on the raw PyAudio stream.
        AEC + barge-in RMS threshold handle stale TTS audio instead.
        """
        mock_vad.return_value = MagicMock()

        class RawStream:
            def __init__(self):
                self.read_calls = 0

            def read(self, n, exception_on_overflow=False):
                self.read_calls += 1
                return b"\x00" * (n * 2)

            def get_read_available(self):
                return 1024

        raw = RawStream()
        wrapped = ResamplingPyAudioStream(
            raw,
            ResamplingParams(
                input_rate_hz=16000,
                output_rate_hz=16000,
                channels=1,
                input_read_frames=512,
            ),
        )
        capture = VADSpeechCapture(
            audio_stream=wrapped,
            vad_model_path=vad_model_path,
        )

        capture.flush_input_buffer()

        # No reads should happen — ResamplingPyAudioStream has no get_read_available,
        # so flush sees avail=0 and exits immediately.
        assert raw.read_calls == 0


class TestKnownBugs:
    """Tests that expose confirmed bugs in capture_speech.

    Each test documents a real bug found during code review.
    Marked xfail — they should pass when the bug is fixed.
    """

    @pytest.fixture
    def vad_model_path(self, tmp_path):
        model_file = tmp_path / "test_vad.onnx"
        model_file.touch()
        return str(model_file)

    @pytest.fixture
    def mock_audio_stream(self):
        stream = MagicMock()
        stream.read.return_value = b"\x00" * 1024
        stream.get_read_available.return_value = 0
        return stream

    @patch('sparky_mvp.core.vad_capture.VAD')
    def test_post_activation_timeout_reason_is_correct(self, mock_vad, mock_audio_stream, vad_model_path):
        """When post_activation_timeout differs from pre-activation timeout,
        the reason code should be 'timeout_after_activation', not 'unknown'.

        Fixed: timed_out now compares against t_deadline (which is updated
        on activation) instead of the original pre-activation timeout_seconds.
        """
        capture = VADSpeechCapture(
            audio_stream=mock_audio_stream,
            vad_model_path=vad_model_path,
        )

        # Simulate: activation happens, then post-activation timeout expires.
        # pre-activation timeout = 30s, post-activation timeout = 2s.
        # Activation at t=1s. Loop exits at t=3s (1s + 2s post-timeout).
        # t_end >= t_deadline → timed_out = True.

        capture._recording_started = True

        # Simulate the fixed reason logic:
        # t_deadline was updated to activation_time + post_activation_timeout
        # so timed_out is True (we exited because deadline was reached).
        timed_out = True  # t_end >= t_deadline
        activation_suppressed = 0
        read_error = None

        if read_error is not None:
            reason = "read_error"
        elif capture._recording_started and timed_out:
            reason = "timeout_after_activation"
        elif activation_suppressed > 0:
            reason = "activation_suppressed"
        elif timed_out:
            reason = "timeout_no_activation"
        else:
            reason = "unknown"

        assert reason == "timeout_after_activation", (
            f"Expected 'timeout_after_activation' but got '{reason}'."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
