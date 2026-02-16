"""
Unit tests for AcousticEchoCanceller (NLMS AEC).

Tests signal processing, thread safety, speaker buffering,
WAV decoding, and the AECStream wrapper.
"""

import io
import struct
import threading
import wave

import numpy as np
import pytest

from sparky_mvp.core.echo_canceller import AcousticEchoCanceller, _wav_to_pcm_16k
from sparky_mvp.core.aec_stream import AECStream


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pcm16(samples: np.ndarray) -> bytes:
    """Convert float64 array [-32768, 32767] to int16 PCM bytes."""
    return np.clip(samples, -32768, 32767).astype(np.int16).tobytes()


def _make_sine_pcm(freq_hz: float, duration_s: float, sample_rate: int = 16000) -> bytes:
    """Generate a sine wave as int16 PCM bytes."""
    t = np.arange(int(sample_rate * duration_s)) / sample_rate
    signal = (np.sin(2 * np.pi * freq_hz * t) * 16000).astype(np.int16)
    return signal.tobytes()


def _make_wav(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
    """Wrap PCM int16 bytes in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_default_params(self):
        aec = AcousticEchoCanceller()
        assert aec._frame_size == 160
        assert aec._filter_length == 3200
        assert aec._sample_rate == 16000
        assert aec._mu == 0.3

    def test_custom_params(self):
        aec = AcousticEchoCanceller(frame_size=320, filter_length=4800, mu=0.5)
        assert aec._frame_size == 320
        assert aec._filter_length == 4800
        assert aec._mu == 0.5

    def test_initial_state(self):
        aec = AcousticEchoCanceller()
        assert aec._frames_processed == 0
        assert aec._frames_with_ref == 0
        assert len(aec._speaker_buf) == 0
        assert np.all(aec._w == 0.0)


# ---------------------------------------------------------------------------
# Passthrough (no speaker reference)
# ---------------------------------------------------------------------------

class TestPassthrough:

    def test_no_speaker_ref_passthrough(self):
        """With no speaker reference, mic signal should pass through unchanged."""
        aec = AcousticEchoCanceller(frame_size=160)
        mic = _make_sine_pcm(440, 0.01)  # 160 samples = 10ms at 16kHz
        out = aec.process_mic_chunk(mic)
        assert out == mic

    def test_empty_input_passthrough(self):
        aec = AcousticEchoCanceller(frame_size=160)
        out = aec.process_mic_chunk(b"")
        assert out == b""

    def test_short_input_passthrough(self):
        """Input shorter than one frame should pass through unchanged."""
        aec = AcousticEchoCanceller(frame_size=160)
        short = b"\x00" * 100  # Less than 320 bytes (160 samples * 2)
        out = aec.process_mic_chunk(short)
        assert out == short


# ---------------------------------------------------------------------------
# Echo cancellation
# ---------------------------------------------------------------------------

class TestEchoCancellation:

    def test_echo_reduces_energy(self):
        """When mic = speaker (pure echo), output energy should decrease."""
        aec = AcousticEchoCanceller(frame_size=160, filter_length=1600, mu=0.5)

        # Create a repeated tone as speaker reference
        tone = _make_sine_pcm(300, 0.5)  # 500ms tone
        aec.feed_speaker_pcm(tone)

        # Feed same tone as mic (perfect echo scenario)
        # Process multiple frames to let filter adapt
        total_input_energy = 0.0
        total_output_energy = 0.0
        frame_bytes = 160 * 2
        offset = 0

        while offset + frame_bytes <= len(tone):
            mic_frame = tone[offset:offset + frame_bytes]
            out_frame = aec.process_mic_chunk(mic_frame)

            mic_arr = np.frombuffer(mic_frame, dtype=np.int16).astype(np.float64)
            out_arr = np.frombuffer(out_frame, dtype=np.int16).astype(np.float64)

            total_input_energy += np.sum(mic_arr ** 2)
            total_output_energy += np.sum(out_arr ** 2)
            offset += frame_bytes

        # After adaptation, output energy should be lower than input
        assert total_output_energy < total_input_energy
        assert aec._frames_with_ref > 0

    def test_stats_updated(self):
        aec = AcousticEchoCanceller(frame_size=160)
        silence = b"\x00" * 320
        aec.process_mic_chunk(silence)
        stats = aec.stats
        assert stats["frames_processed"] == 1
        assert stats["frames_with_ref"] == 0

    def test_stats_with_speaker_ref(self):
        aec = AcousticEchoCanceller(frame_size=160)
        silence = b"\x00" * 320
        aec.feed_speaker_pcm(silence)
        aec.process_mic_chunk(silence)
        stats = aec.stats
        assert stats["frames_processed"] == 1
        assert stats["frames_with_ref"] == 1


# ---------------------------------------------------------------------------
# Speaker buffer management
# ---------------------------------------------------------------------------

class TestSpeakerBuffer:

    def test_feed_pcm(self):
        aec = AcousticEchoCanceller()
        pcm = b"\x00" * 640
        aec.feed_speaker_pcm(pcm)
        assert len(aec._speaker_buf) == 640

    def test_feed_wav(self):
        aec = AcousticEchoCanceller()
        pcm = _make_sine_pcm(440, 0.1)
        wav = _make_wav(pcm, 16000)
        aec.feed_speaker_wav(wav)
        # Buffer should have samples (exact count depends on WAV overhead)
        assert len(aec._speaker_buf) > 0

    def test_feed_wav_resample(self):
        """24kHz WAV should be resampled to 16kHz."""
        pcm_24k = _make_sine_pcm(440, 0.1, sample_rate=24000)
        wav = _make_wav(pcm_24k, 24000)
        aec = AcousticEchoCanceller()
        aec.feed_speaker_wav(wav)
        # 0.1s at 16kHz = 1600 samples = 3200 bytes
        expected_bytes = 3200
        # Allow some margin for resampling
        assert abs(len(aec._speaker_buf) - expected_bytes) < 100

    def test_clear(self):
        aec = AcousticEchoCanceller()
        aec.feed_speaker_pcm(b"\x00" * 640)
        assert len(aec._speaker_buf) > 0
        aec.clear()
        assert len(aec._speaker_buf) == 0
        # Filter should also be reset
        assert np.all(aec._w == 0.0)

    def test_speaker_consumed_by_processing(self):
        """Speaker buffer should be consumed frame by frame."""
        aec = AcousticEchoCanceller(frame_size=160)
        # Feed exactly 2 frames of speaker reference
        pcm = b"\x00" * (160 * 2 * 2)  # 2 frames * 2 bytes/sample
        aec.feed_speaker_pcm(pcm)
        assert len(aec._speaker_buf) == 640

        # Process 1 mic frame — should consume 1 speaker frame
        mic = b"\x00" * 320
        aec.process_mic_chunk(mic)
        assert len(aec._speaker_buf) == 320  # 1 frame consumed


# ---------------------------------------------------------------------------
# Multi-frame processing
# ---------------------------------------------------------------------------

class TestMultiFrame:

    def test_multi_frame_chunk(self):
        """process_mic_chunk should handle multi-frame input."""
        aec = AcousticEchoCanceller(frame_size=160)
        # 3 frames = 480 samples = 960 bytes
        mic = b"\x00" * (160 * 3 * 2)
        out = aec.process_mic_chunk(mic)
        assert len(out) == len(mic)
        assert aec._frames_processed == 3

    def test_remainder_passthrough(self):
        """Bytes that don't fill a frame should pass through."""
        aec = AcousticEchoCanceller(frame_size=160)
        # 1.5 frames = 240 samples = 480 bytes
        mic = b"\x00" * (240 * 2)
        out = aec.process_mic_chunk(mic)
        assert len(out) == len(mic)
        assert aec._frames_processed == 1  # Only 1 full frame


# ---------------------------------------------------------------------------
# WAV decode helper
# ---------------------------------------------------------------------------

class TestWavDecode:

    def test_16k_wav_passthrough(self):
        pcm = _make_sine_pcm(440, 0.1, 16000)
        wav = _make_wav(pcm, 16000)
        result = _wav_to_pcm_16k(wav, 16000)
        assert len(result) == len(pcm)

    def test_24k_wav_resampled(self):
        pcm_24k = _make_sine_pcm(440, 0.1, 24000)
        wav = _make_wav(pcm_24k, 24000)
        result = _wav_to_pcm_16k(wav, 16000)
        # 0.1s at 16kHz = 1600 samples = 3200 bytes
        expected = 3200
        assert abs(len(result) - expected) < 100


# ---------------------------------------------------------------------------
# AECStream wrapper
# ---------------------------------------------------------------------------

class MockStream:
    def __init__(self, data: bytes, frame_size: int = 160):
        self._data = data
        self._offset = 0
        self._frame_bytes = frame_size * 2

    def read(self, n_frames, exception_on_overflow=False):
        n_bytes = n_frames * 2
        chunk = self._data[self._offset:self._offset + n_bytes]
        self._offset += n_bytes
        if len(chunk) < n_bytes:
            chunk += b"\x00" * (n_bytes - len(chunk))
        return chunk

    def get_read_available(self):
        return (len(self._data) - self._offset) // 2


class TestAECStream:

    def test_passthrough_without_aec(self):
        data = _make_sine_pcm(440, 0.01)
        stream = MockStream(data)
        aec_stream = AECStream(stream, echo_canceller=None)
        out = aec_stream.read(160)
        assert out == data[:320]

    def test_with_aec(self):
        data = b"\x00" * 320
        stream = MockStream(data)
        aec = AcousticEchoCanceller(frame_size=160)
        aec_stream = AECStream(stream, echo_canceller=aec)
        out = aec_stream.read(160)
        assert len(out) == 320

    def test_get_read_available_proxy(self):
        data = b"\x00" * 640
        stream = MockStream(data)
        aec_stream = AECStream(stream)
        assert aec_stream.get_read_available() == 320  # 640 bytes / 2


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:

    def test_concurrent_feed_and_process(self):
        """Concurrent speaker feed + mic processing shouldn't crash."""
        aec = AcousticEchoCanceller(frame_size=160)
        errors = []

        def feed_speaker():
            try:
                for _ in range(100):
                    aec.feed_speaker_pcm(b"\x00" * 320)
            except Exception as e:
                errors.append(e)

        def process_mic():
            try:
                for _ in range(100):
                    aec.process_mic_chunk(b"\x00" * 320)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=feed_speaker)
        t2 = threading.Thread(target=process_mic)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0
