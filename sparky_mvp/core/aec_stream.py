"""
AEC-wrapped audio stream for barge-in support.

Wraps a ResamplingPyAudioStream (or any stream with .read()) to pass
mic audio through the AcousticEchoCanceller before returning it to
the caller (VAD, wake word, etc.).

When AEC is None, the stream is a transparent passthrough.
"""

from __future__ import annotations

import logging
from typing import Optional, Protocol

logger = logging.getLogger(__name__)


class EchoCanceller(Protocol):
    def process_mic_chunk(self, mic_pcm: bytes) -> bytes: ...


class AECStream:
    """Audio stream wrapper that applies AEC to every read."""

    def __init__(self, stream, echo_canceller: Optional[EchoCanceller] = None):
        self._stream = stream
        self._aec = echo_canceller

    def read(self, n_frames: int, exception_on_overflow: bool = False) -> bytes:
        raw = self._stream.read(n_frames, exception_on_overflow=exception_on_overflow)
        if self._aec is not None:
            return self._aec.process_mic_chunk(raw)
        return raw

    # Proxy methods used by VADSpeechCapture
    def get_read_available(self):
        if hasattr(self._stream, "get_read_available"):
            return self._stream.get_read_available()
        return 0
