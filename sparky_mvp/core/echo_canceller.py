"""
Acoustic Echo Cancellation (AEC) for barge-in support.

Removes the robot's own TTS playback from the microphone signal so that
Silero VAD can detect real human speech even while the robot is talking.

Uses an NLMS (Normalized Least-Mean-Squares) adaptive filter in pure numpy.
No external dependencies beyond numpy.

Usage::

    aec = AcousticEchoCanceller(frame_size=160, filter_length=3200)

    # TTS playback thread: feed speaker reference (16kHz int16 mono PCM)
    aec.feed_speaker_wav(wav_bytes)

    # Mic read path: clean each frame
    cleaned = aec.process_mic_chunk(raw_mic_bytes)

Config (in config.yaml)::

    barge_in:
      enabled: true
      aec_enabled: true       # Enable acoustic echo cancellation
      aec_filter_length: 3200 # NLMS filter taps (200ms at 16kHz)
      aec_mu: 0.3             # NLMS step size (0.0-1.0, lower = more stable)
"""

from __future__ import annotations

import io
import logging
import threading
import wave
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _wav_to_pcm_16k(wav_bytes: bytes, target_rate: int = 16000) -> bytes:
    """Decode WAV and resample to 16kHz mono int16 PCM."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    if width == 2:
        samples = np.frombuffer(raw, dtype=np.int16)
    elif width == 4:
        # 32-bit float
        samples = (np.frombuffer(raw, dtype=np.float32) * 32767).astype(np.int16)
    else:
        samples = np.frombuffer(raw, dtype=np.int16)

    # Downmix to mono
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)

    # Resample if needed
    if rate != target_rate:
        from scipy.signal import resample_poly

        g = int(np.gcd(rate, target_rate))
        up = target_rate // g
        down = rate // g
        samples_f = samples.astype(np.float32) / 32768.0
        resampled = resample_poly(samples_f, up, down)
        samples = np.clip(resampled * 32768.0, -32768, 32767).astype(np.int16)

    return samples.tobytes()


class AcousticEchoCanceller:
    """
    Thread-safe AEC using NLMS adaptive filter.

    The speaker reference is buffered and consumed frame-by-frame as the
    microphone signal is processed. When no speaker reference is available
    (robot is silent), mic audio passes through unchanged.
    """

    def __init__(
        self,
        frame_size: int = 160,
        filter_length: int = 3200,
        sample_rate: int = 16000,
        mu: float = 0.3,
    ):
        """
        Args:
            frame_size: Samples per AEC frame (160 = 10ms at 16kHz).
            filter_length: NLMS filter taps (3200 = 200ms echo tail at 16kHz).
            sample_rate: Audio sample rate (must match mic and speaker).
            mu: NLMS step size (0.0-1.0). Lower = more stable, higher = faster adapt.
        """
        self._frame_size = frame_size
        self._filter_length = filter_length
        self._sample_rate = sample_rate
        self._mu = mu

        # NLMS filter weights
        self._w = np.zeros(filter_length, dtype=np.float64)
        # Speaker reference ring buffer (for convolution)
        self._x_hist = np.zeros(filter_length, dtype=np.float64)

        # Thread-safe speaker PCM buffer (bytes, int16 LE)
        self._speaker_buf = bytearray()
        self._lock = threading.Lock()

        # Stats
        self._frames_processed = 0
        self._frames_with_ref = 0

    def feed_speaker_wav(self, wav_bytes: bytes) -> None:
        """Feed speaker WAV (any rate/format) — resampled to 16kHz internally."""
        try:
            pcm = _wav_to_pcm_16k(wav_bytes, self._sample_rate)
            self.feed_speaker_pcm(pcm)
        except Exception:
            logger.debug("AEC: failed to decode speaker WAV", exc_info=True)

    def feed_speaker_pcm(self, pcm_int16: bytes) -> None:
        """Feed raw 16kHz int16 mono PCM from the speaker path."""
        with self._lock:
            self._speaker_buf.extend(pcm_int16)

    def process_mic_chunk(self, mic_pcm: bytes) -> bytes:
        """
        Process an arbitrary-length mic chunk through AEC.

        Returns cleaned PCM bytes (same length as input).
        """
        frame_bytes = self._frame_size * 2  # int16 = 2 bytes/sample
        if len(mic_pcm) < frame_bytes:
            return mic_pcm

        result = bytearray()
        offset = 0

        while offset + frame_bytes <= len(mic_pcm):
            mic_frame = mic_pcm[offset : offset + frame_bytes]
            cleaned = self._process_frame(mic_frame)
            result.extend(cleaned)
            offset += frame_bytes

        # Pass through any remainder (< 1 frame)
        if offset < len(mic_pcm):
            result.extend(mic_pcm[offset:])

        return bytes(result)

    def _process_frame(self, mic_frame: bytes) -> bytes:
        """Process a single frame through the NLMS filter."""
        frame_bytes = self._frame_size * 2
        self._frames_processed += 1

        # Get speaker reference for this frame
        with self._lock:
            if len(self._speaker_buf) >= frame_bytes:
                speaker_data = bytes(self._speaker_buf[:frame_bytes])
                del self._speaker_buf[:frame_bytes]
                has_ref = True
            else:
                speaker_data = None
                has_ref = False

        mic = np.frombuffer(mic_frame, dtype=np.int16).astype(np.float64)

        if not has_ref:
            # No speaker reference — pass through unchanged
            return mic_frame

        self._frames_with_ref += 1
        speaker = np.frombuffer(speaker_data, dtype=np.int16).astype(np.float64)

        # Process sample by sample through NLMS
        output = np.zeros(self._frame_size, dtype=np.float64)
        for i in range(self._frame_size):
            # Shift speaker history
            self._x_hist = np.roll(self._x_hist, 1)
            self._x_hist[0] = speaker[i]

            # Estimate echo
            echo_est = np.dot(self._w, self._x_hist)

            # Subtract echo from mic
            error = mic[i] - echo_est
            output[i] = error

            # Update filter (NLMS)
            norm = np.dot(self._x_hist, self._x_hist) + 1e-6
            self._w += (self._mu * error / norm) * self._x_hist

        # Clip and convert back to int16
        output = np.clip(output, -32768, 32767).astype(np.int16)
        return output.tobytes()

    def clear(self) -> None:
        """Clear speaker buffer and reset filter (on playback cancel)."""
        with self._lock:
            self._speaker_buf.clear()
        # Reset filter to prevent stale adaptation
        self._w[:] = 0.0
        self._x_hist[:] = 0.0

    @property
    def stats(self) -> dict:
        """Return AEC processing stats."""
        return {
            "frames_processed": self._frames_processed,
            "frames_with_ref": self._frames_with_ref,
            "speaker_buf_bytes": len(self._speaker_buf),
            "filter_energy": float(np.dot(self._w, self._w)),
        }
