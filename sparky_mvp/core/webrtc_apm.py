"""
WebRTC APM (full AEC3) echo canceller wrapper for barge-in.

This uses a small C++ shim (``libwebrtc_apm_shim.so``) that wraps the full
WebRTC AudioProcessing module, giving us AEC3 (the advanced echo canceller)
plus noise suppression — significantly better than AECM for speaker-mic
scenarios like the Reachy robot.

Interface matches the existing ``EchoCanceller`` protocol and is a drop-in
replacement for ``WebRtcAecmEchoCanceller``:
- ``feed_speaker_wav(wav_bytes)`` / ``feed_speaker_pcm(pcm_int16)``
- ``process_mic_chunk(mic_pcm)``
- ``clear()``
"""

from __future__ import annotations

import ctypes
import io
import logging
import threading
import wave
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _load_apm_shim() -> ctypes.CDLL:
    """Load the APM shim shared library from the same directory as this module."""
    shim_path = Path(__file__).parent / "libwebrtc_apm_shim.so"
    if not shim_path.exists():
        raise OSError(
            f"APM shim not found at {shim_path}. "
            f"Run: bash {Path(__file__).parent / 'build_apm_shim.sh'}"
        )
    return ctypes.CDLL(str(shim_path))


def _wav_to_pcm_16k_mono_int16(wav_bytes: bytes, target_rate: int = 16000) -> bytes:
    """Decode WAV and resample to target rate mono int16 PCM (best-effort)."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    if width == 2:
        samples = np.frombuffer(raw, dtype=np.int16)
    elif width == 4:
        samples = (np.frombuffer(raw, dtype=np.float32) * 32767).astype(np.int16)
    else:
        samples = np.frombuffer(raw, dtype=np.int16)

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)

    if rate != target_rate:
        from scipy.signal import resample_poly

        g = int(np.gcd(rate, target_rate))
        up = target_rate // g
        down = rate // g
        samples_f = samples.astype(np.float32) / 32768.0
        resampled = resample_poly(samples_f, up, down)
        samples = np.clip(resampled * 32768.0, -32768, 32767).astype(np.int16)

    return samples.tobytes()


class WebRtcApmEchoCanceller:
    """
    Thread-safe wrapper for WebRTC APM with full AEC3.

    APM operates on 10ms frames:
    - 8kHz:  80 samples
    - 16kHz: 160 samples
    - 32kHz: 320 samples
    - 48kHz: 480 samples
    """

    def __init__(self, sample_rate: int = 16000):
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError(
                f"WebRTC APM supports 8/16/32/48 kHz (got {sample_rate})"
            )

        self.sample_rate = int(sample_rate)
        self.frame_size = self.sample_rate // 100  # 10ms worth of samples
        self._frame_bytes = self.frame_size * 2  # int16 = 2 bytes per sample

        self._lib = _load_apm_shim()
        self._bind()

        self._handle: Optional[ctypes.c_void_p] = None
        self._create(sample_rate)

        self._speaker_buf = bytearray()
        self._lock = threading.Lock()

        self._frames_processed = 0
        self._frames_with_ref = 0
        self._errors = 0

    # -----------------------------------------------------------------
    # Binding / lifecycle
    # -----------------------------------------------------------------

    def _bind(self) -> None:
        lib = self._lib

        lib.apm_create.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.apm_create.restype = ctypes.c_void_p

        lib.apm_process_reverse.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int16),
            ctypes.c_int,
        ]
        lib.apm_process_reverse.restype = ctypes.c_int

        lib.apm_process.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int16),
            ctypes.c_int,
        ]
        lib.apm_process.restype = ctypes.c_int

        lib.apm_reinitialize.argtypes = [ctypes.c_void_p]
        lib.apm_reinitialize.restype = ctypes.c_int

        lib.apm_destroy.argtypes = [ctypes.c_void_p]
        lib.apm_destroy.restype = None

    def _create(self, sample_rate: int) -> None:
        handle_val = self._lib.apm_create(sample_rate, 1)  # mono
        if not handle_val:
            raise RuntimeError("apm_create returned NULL")
        self._handle = ctypes.c_void_p(handle_val)

    def close(self) -> None:
        handle = self._handle
        self._handle = None
        if handle:
            try:
                self._lib.apm_destroy(handle)
            except Exception:
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # -----------------------------------------------------------------
    # Speaker reference feed
    # -----------------------------------------------------------------

    def feed_speaker_wav(self, wav_bytes: bytes) -> None:
        try:
            pcm = _wav_to_pcm_16k_mono_int16(wav_bytes, target_rate=self.sample_rate)
            self.feed_speaker_pcm(pcm)
        except Exception:
            logger.debug("WebRtcApm: failed to decode/resample speaker WAV", exc_info=True)

    def feed_speaker_pcm(self, pcm_int16: bytes) -> None:
        with self._lock:
            self._speaker_buf.extend(pcm_int16)

    # -----------------------------------------------------------------
    # Mic processing
    # -----------------------------------------------------------------

    def process_mic_chunk(self, mic_pcm: bytes) -> bytes:
        if self._handle is None:
            return mic_pcm
        if len(mic_pcm) < self._frame_bytes:
            return mic_pcm

        out = bytearray()
        offset = 0
        while offset + self._frame_bytes <= len(mic_pcm):
            mic_frame = mic_pcm[offset : offset + self._frame_bytes]
            out.extend(self._process_frame(mic_frame))
            offset += self._frame_bytes
        if offset < len(mic_pcm):
            out.extend(mic_pcm[offset:])
        return bytes(out)

    def _process_frame(self, mic_frame: bytes) -> bytes:
        self._frames_processed += 1

        # Consume a 10ms far-end frame if available.
        with self._lock:
            if len(self._speaker_buf) >= self._frame_bytes:
                speaker_frame = bytes(self._speaker_buf[: self._frame_bytes])
                del self._speaker_buf[: self._frame_bytes]
            else:
                speaker_frame = None

        # Convert mic frame to numpy for ctypes interop.
        mic = np.frombuffer(mic_frame, dtype=np.int16).copy()

        if speaker_frame is None:
            # No far-end reference: still run through APM for noise suppression
            # and high-pass filter, but don't feed reverse stream.
            try:
                rc = self._lib.apm_process(
                    self._handle,
                    mic.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                    ctypes.c_int(self.frame_size),
                )
                if rc != 0:
                    self._errors += 1
            except Exception:
                self._errors += 1
            return mic.tobytes()

        self._frames_with_ref += 1
        far = np.frombuffer(speaker_frame, dtype=np.int16)

        # Feed far-end (render/speaker) reference into the APM.
        try:
            rc = self._lib.apm_process_reverse(
                self._handle,
                far.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                ctypes.c_int(self.frame_size),
            )
            if rc != 0:
                self._errors += 1
        except Exception:
            self._errors += 1
            return mic.tobytes()

        # Process near-end (mic/capture) through AEC3 — in-place.
        try:
            rc = self._lib.apm_process(
                self._handle,
                mic.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                ctypes.c_int(self.frame_size),
            )
            if rc != 0:
                self._errors += 1
        except Exception:
            self._errors += 1
        return mic.tobytes()

    def clear(self) -> None:
        """Clear buffered speaker reference and reset the APM state."""
        with self._lock:
            self._speaker_buf.clear()
        # Reset APM internal state by re-initializing.
        if self._handle is not None:
            try:
                self._lib.apm_reinitialize(self._handle)
            except Exception:
                logger.debug("apm_reinitialize failed", exc_info=True)

    @property
    def stats(self) -> dict:
        return {
            "frames_processed": self._frames_processed,
            "frames_with_ref": self._frames_with_ref,
            "speaker_buf_bytes": len(self._speaker_buf),
            "errors": self._errors,
            "impl": "webrtc_apm",
        }
