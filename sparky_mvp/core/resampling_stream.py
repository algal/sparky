"""
Resampling audio stream wrapper for PyAudio.

Why: many USB mics (including Shure MV5) don't support 16kHz natively.
The Reachy MVP pipeline expects 16kHz mono int16 for Silero VAD and Whisper.

This wrapper reads from an underlying PyAudio input stream at its native
sample rate (e.g. 44100) and provides a `.read(n_frames)` method that returns
exactly `n_frames` at the target rate (default 16000).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample_poly


@dataclass(frozen=True)
class ResamplingParams:
    input_rate_hz: int
    output_rate_hz: int = 16000
    channels: int = 1
    input_read_frames: int = 4096


class ResamplingPyAudioStream:
    """
    Wrap a PyAudio input stream and expose a 16kHz-style .read() API.

    The returned bytes are little-endian PCM int16 mono at `output_rate_hz`.
    """

    def __init__(self, stream, params: ResamplingParams):
        self._stream = stream
        self._p = params
        self._buf: deque[np.int16] = deque()

        # Precompute rational ratio for polyphase resampling
        g = int(np.gcd(self._p.input_rate_hz, self._p.output_rate_hz))
        self._up = self._p.output_rate_hz // g
        self._down = self._p.input_rate_hz // g

    def read(self, n_frames: int, exception_on_overflow: bool = False) -> bytes:
        while len(self._buf) < n_frames:
            raw = self._stream.read(self._p.input_read_frames, exception_on_overflow)
            x = np.frombuffer(raw, dtype=np.int16)

            if self._p.channels > 1:
                # If device is opened with >1 channels, downmix to mono.
                # Some USB mics present stereo where one channel may be quieter; averaging is safer.
                x = x.reshape(-1, self._p.channels).astype(np.float32).mean(axis=1).astype(np.int16)

            xf = x.astype(np.float32) / 32768.0
            yf = resample_poly(xf, up=self._up, down=self._down).astype(np.float32)
            y = np.clip(yf * 32768.0, -32768, 32767).astype(np.int16)

            self._buf.extend(y.tolist())

        out = np.fromiter((self._buf.popleft() for _ in range(n_frames)), dtype=np.int16)
        return out.tobytes()
