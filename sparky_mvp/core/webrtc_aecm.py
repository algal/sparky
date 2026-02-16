"""
WebRTC AECM (echo canceller mobile) wrapper for barge-in.

This uses the C ABI exported by Debian's ``libwebrtc-audio-processing-1`` shared
library (symbols like ``WebRtcAecm_Create`` / ``WebRtcAecm_Process``).

Why AECM?
- It is significantly more robust than a from-scratch NLMS filter for real
  acoustic paths (delay drift, nonlinearities), while remaining lightweight.
- It avoids needing C++ headers / a compiled binding to the full WebRTC APM.

Interface matches the existing ``AcousticEchoCanceller`` integration points:
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


def _load_webrtc_aecm_lib() -> ctypes.CDLL:
    """
    Load the shared library that exports WebRtcAecm_* symbols.

    Debian trixie package installs:
      /usr/lib/x86_64-linux-gnu/libwebrtc-audio-processing-1.so.3
    """
    candidates = [
        "/usr/lib/x86_64-linux-gnu/libwebrtc-audio-processing-1.so.3",
        "/usr/lib/aarch64-linux-gnu/libwebrtc-audio-processing-1.so.3",
    ]
    for p in candidates:
        if Path(p).exists():
            return ctypes.CDLL(p)
    # As a last resort, let the dynamic loader try (may work if a -dev package
    # is installed providing an unversioned symlink).
    return ctypes.CDLL("libwebrtc-audio-processing-1.so")


def _wav_to_pcm_16k_mono_int16(wav_bytes: bytes, target_rate: int = 16000) -> bytes:
    """Decode WAV and resample to 16kHz mono int16 PCM (best-effort)."""
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


class WebRtcAecmEchoCanceller:
    """
    Thread-safe wrapper for WebRTC AECM.

    AECM operates on 10ms frames:
    - 8kHz: 80 samples
    - 16kHz: 160 samples
    """

    def __init__(self, sample_rate: int = 16000, ms_in_soundcard_buf: int = 60):
        if sample_rate not in (8000, 16000):
            raise ValueError("WebRtcAecm supports only 8kHz or 16kHz")

        self.sample_rate = int(sample_rate)
        self.frame_size = 80 if self.sample_rate == 8000 else 160
        self._frame_bytes = self.frame_size * 2

        # AECM needs a delay estimate (msInSndCardBuf). Clamp per upstream.
        self.ms_in_soundcard_buf = int(max(0, min(500, ms_in_soundcard_buf)))

        self._lib = _load_webrtc_aecm_lib()
        self._bind()

        self._inst: Optional[ctypes.c_void_p] = None
        self._create_and_init()

        self._speaker_buf = bytearray()
        self._lock = threading.Lock()

        self._frames_processed = 0
        self._frames_with_ref = 0
        self._errors = 0

    # ---------------------------------------------------------------------
    # Binding / lifecycle
    # ---------------------------------------------------------------------

    def _bind(self) -> None:
        lib = self._lib

        # Debian's libwebrtc-audio-processing exports AECM Create() as:
        #   void* WebRtcAecm_Create(void);
        lib.WebRtcAecm_Create.argtypes = []
        lib.WebRtcAecm_Create.restype = ctypes.c_void_p

        lib.WebRtcAecm_Free.argtypes = [ctypes.c_void_p]
        lib.WebRtcAecm_Free.restype = None

        lib.WebRtcAecm_Init.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.WebRtcAecm_Init.restype = ctypes.c_int

        lib.WebRtcAecm_BufferFarend.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int16),
            ctypes.c_int16,
        ]
        lib.WebRtcAecm_BufferFarend.restype = ctypes.c_int

        lib.WebRtcAecm_Process.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int16),  # nearendNoisy
            ctypes.POINTER(ctypes.c_int16),  # nearendClean (nullable)
            ctypes.POINTER(ctypes.c_int16),  # out
            ctypes.c_int16,  # nrOfSamples (80 or 160)
            ctypes.c_int16,  # msInSndCardBuf
        ]
        lib.WebRtcAecm_Process.restype = ctypes.c_int

    def _create_and_init(self) -> None:
        inst_val = self._lib.WebRtcAecm_Create()
        if not inst_val:
            raise RuntimeError("WebRtcAecm_Create returned NULL")
        inst = ctypes.c_void_p(inst_val)

        rc = self._lib.WebRtcAecm_Init(inst, self.sample_rate)
        if rc != 0:
            # Free before raising
            try:
                self._lib.WebRtcAecm_Free(inst)
            except Exception:
                pass
            raise RuntimeError(f"WebRtcAecm_Init failed (rc={rc})")
        self._inst = inst

    def close(self) -> None:
        inst = self._inst
        self._inst = None
        if inst:
            try:
                self._lib.WebRtcAecm_Free(inst)
            except Exception:
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ---------------------------------------------------------------------
    # Speaker reference feed
    # ---------------------------------------------------------------------

    def feed_speaker_wav(self, wav_bytes: bytes) -> None:
        try:
            pcm = _wav_to_pcm_16k_mono_int16(wav_bytes, target_rate=self.sample_rate)
            self.feed_speaker_pcm(pcm)
        except Exception:
            logger.debug("WebRtcAecm: failed to decode/resample speaker WAV", exc_info=True)

    def feed_speaker_pcm(self, pcm_int16: bytes) -> None:
        with self._lock:
            self._speaker_buf.extend(pcm_int16)

    # ---------------------------------------------------------------------
    # Mic processing
    # ---------------------------------------------------------------------

    def process_mic_chunk(self, mic_pcm: bytes) -> bytes:
        if self._inst is None:
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

        # Consume a 10ms far-end frame if available and buffer it into AECM.
        with self._lock:
            if len(self._speaker_buf) >= self._frame_bytes:
                speaker_frame = bytes(self._speaker_buf[: self._frame_bytes])
                del self._speaker_buf[: self._frame_bytes]
            else:
                speaker_frame = None

        mic = np.frombuffer(mic_frame, dtype=np.int16)
        out = np.empty(self.frame_size, dtype=np.int16)

        # If we don't have speaker reference, pass through unchanged.
        if speaker_frame is None:
            out[:] = mic
            return out.tobytes()

        self._frames_with_ref += 1
        far = np.frombuffer(speaker_frame, dtype=np.int16)

        # Buffer far-end into the canceller.
        try:
            rc = self._lib.WebRtcAecm_BufferFarend(
                self._inst,
                far.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                ctypes.c_int16(self.frame_size),
            )
            if rc != 0:
                self._errors += 1
        except Exception:
            self._errors += 1
            out[:] = mic
            return out.tobytes()

        # Process near-end (mic) using AECM.
        try:
            rc = self._lib.WebRtcAecm_Process(
                self._inst,
                mic.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                None,  # nearendClean (optional)
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                ctypes.c_int16(self.frame_size),
                ctypes.c_int16(self.ms_in_soundcard_buf),
            )
            if rc != 0:
                self._errors += 1
        except Exception:
            self._errors += 1
            out[:] = mic
        return out.tobytes()

    def clear(self) -> None:
        """Clear buffered speaker reference and reset the canceller state."""
        with self._lock:
            self._speaker_buf.clear()
        # Reset AECM by re-init (simplest safe reset).
        if self._inst is not None:
            try:
                self._lib.WebRtcAecm_Init(self._inst, self.sample_rate)
            except Exception:
                logger.debug("WebRtcAecm_Init reset failed", exc_info=True)

    @property
    def stats(self) -> dict:
        return {
            "frames_processed": self._frames_processed,
            "frames_with_ref": self._frames_with_ref,
            "speaker_buf_bytes": len(self._speaker_buf),
            "errors": self._errors,
            "impl": "webrtc_aecm",
        }
