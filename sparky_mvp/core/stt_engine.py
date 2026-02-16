"""
Speech-to-Text Engine for Reachy MVP

Supports two backends (configured via ``engine`` parameter):

- ``faster_whisper`` — Faster Whisper (CPU/GPU), Whisper "base" by default.
- ``parakeet`` — NVIDIA Parakeet-TDT-0.6b-v2 via onnx-asr on GPU.
  Requires ``pip install onnx-asr[gpu,hub]`` and a local model directory
  (downloaded automatically on first run via HuggingFace Hub).
"""

from __future__ import annotations

import logging
import io
import os
import wave
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class STTResult:
    """Result of speech-to-text transcription"""

    text: str
    is_speech: bool
    confidence: float
    language: str

    # Detailed metrics for debugging
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float
    duration: float

    def __str__(self):
        return (
            f"STTResult(text='{self.text[:50]}...', is_speech={self.is_speech}, "
            f"confidence={self.confidence:.2f}, language={self.language})"
        )


def _ensure_cuda_ld_path() -> None:
    """Add nvidia pip-package library dirs to LD_LIBRARY_PATH at runtime.

    onnxruntime-gpu needs libcublas, libcudnn etc.  When installed via pip
    (nvidia-cublas-cu12, nvidia-cudnn-cu12, …) they live under
    ``site-packages/nvidia/<pkg>/lib/``.  We add those directories once so
    that onnxruntime can dlopen them.
    """
    nvidia_base = os.path.join(
        os.path.dirname(os.path.abspath(np.__file__)),  # site-packages/numpy
        "..", "nvidia",
    )
    nvidia_base = os.path.normpath(nvidia_base)
    if not os.path.isdir(nvidia_base):
        return
    dirs_to_add: list[str] = []
    for pkg in os.listdir(nvidia_base):
        lib_dir = os.path.join(nvidia_base, pkg, "lib")
        if os.path.isdir(lib_dir):
            dirs_to_add.append(lib_dir)
    if dirs_to_add:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(dirs_to_add) + (":" + existing if existing else "")


class STTEngine:
    """Local speech-to-text engine with noise detection.

    Supports ``faster_whisper`` and ``parakeet`` backends.
    """

    def __init__(
        self,
        engine: str = "faster_whisper",
        model_size: str = "base",
        device: str = "cpu",
        compute_type: Optional[str] = None,
        language: str = "en",
        min_speech_confidence: float = 0.6,
        max_no_speech_prob: float = 0.6,
        min_compression_ratio: float = 0.5,
        max_compression_ratio: float = 2.5,
        min_text_length: int = 3,
        no_speech_threshold: Optional[float] = None,
        parakeet_model_path: Optional[str] = None,
        parakeet_gpu_device: int = 1,
    ):
        self.engine = engine
        self.model_size = model_size
        self.device = device
        self.language = language if language != "auto" else None

        # Noise detection thresholds
        self.min_speech_confidence = min_speech_confidence
        self.max_no_speech_prob = max_no_speech_prob
        self.min_compression_ratio = min_compression_ratio
        self.max_compression_ratio = max_compression_ratio
        self.min_text_length = min_text_length
        self.no_speech_threshold = no_speech_threshold

        self.model = None           # Faster Whisper model
        self._parakeet_model = None  # onnx-asr model

        if engine == "parakeet":
            self._init_parakeet(parakeet_model_path, parakeet_gpu_device)
        else:
            self._init_faster_whisper(model_size, device, compute_type)

    # ------------------------------------------------------------------
    # Backend initialisation
    # ------------------------------------------------------------------

    def _init_faster_whisper(
        self, model_size: str, device: str, compute_type: Optional[str]
    ) -> None:
        from faster_whisper import WhisperModel

        if compute_type is None:
            self.compute_type = "int8" if device == "cpu" else "float16"
        else:
            self.compute_type = compute_type

        logger.info(
            f"Initializing Faster Whisper: {model_size} on {device} "
            f"({self.compute_type})"
        )
        self.model = WhisperModel(
            model_size, device=device, compute_type=self.compute_type,
        )
        logger.info("Faster Whisper ready")

    def _init_parakeet(
        self, model_path: Optional[str], gpu_device: int
    ) -> None:
        _ensure_cuda_ld_path()
        import onnx_asr

        model_path = model_path or "models/parakeet-tdt-0.6b-v2"

        # Auto-download from HuggingFace Hub if local directory is missing
        if not os.path.isdir(model_path):
            logger.info(
                f"Parakeet model not found at {model_path}, downloading …"
            )
            from huggingface_hub import snapshot_download

            os.makedirs(model_path, exist_ok=True)
            snapshot_download(
                "istupakov/parakeet-tdt-0.6b-v2", local_dir=model_path,
            )

        providers = [
            (
                "CUDAExecutionProvider",
                {"device_id": str(gpu_device)},
            ),
            "CPUExecutionProvider",
        ]

        logger.info(
            f"Initializing Parakeet-TDT-0.6b-v2 on GPU {gpu_device} "
            f"from {model_path}"
        )
        self._parakeet_model = onnx_asr.load_model(
            "nemo-parakeet-tdt-0.6b-v2",
            path=model_path,
            providers=providers,
        )
        logger.info("Parakeet ready")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, audio_wav: bytes) -> STTResult:
        """Transcribe WAV audio (16 kHz mono int16) and detect speech vs noise."""
        logger.debug(f"Transcribing audio ({len(audio_wav)} bytes)")
        try:
            if self.engine == "parakeet":
                return self._transcribe_parakeet(audio_wav)
            return self._transcribe_whisper(audio_wav)
        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Backend-specific transcription
    # ------------------------------------------------------------------

    def _transcribe_whisper(self, audio_wav: bytes) -> STTResult:
        audio_file = io.BytesIO(audio_wav)
        duration = self._get_wav_duration(audio_wav)

        assert self.model is not None
        segments, info = self.model.transcribe(
            audio_file,
            language=self.language,
            beam_size=5,
            vad_filter=False,
            word_timestamps=False,
            no_speech_threshold=self.no_speech_threshold,
        )

        full_text = ""
        total_logprob = 0.0
        segment_count = 0
        for segment in segments:
            full_text += segment.text
            total_logprob += segment.avg_logprob
            segment_count += 1

        avg_logprob = total_logprob / segment_count if segment_count > 0 else -1.0
        language_out = info.language
        no_speech_prob = getattr(info, "no_speech_prob", 0.0)

        return self._build_result(
            full_text.strip(), duration, language_out, avg_logprob, no_speech_prob,
        )

    def _transcribe_parakeet(self, audio_wav: bytes) -> STTResult:
        duration = self._get_wav_duration(audio_wav)

        # Parse WAV → float32 numpy array
        with wave.open(io.BytesIO(audio_wav), "rb") as wf:
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        assert self._parakeet_model is not None
        text = self._parakeet_model.recognize(pcm, sample_rate=sr)

        # Parakeet doesn't provide logprob / no_speech_prob.
        # Set to values that always pass those checks — speech detection
        # relies on text_length + compression_ratio + upstream VAD.
        return self._build_result(
            text.strip() if text else "",
            duration,
            self.language or "en",
            avg_logprob=0.0,
            no_speech_prob=0.0,
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _build_result(
        self,
        text: str,
        duration: float,
        language: str,
        avg_logprob: float,
        no_speech_prob: float,
    ) -> STTResult:
        text_length = len(text)
        compression_ratio = text_length / duration if duration > 0 else 0.0

        is_speech = self._is_speech(
            avg_logprob=avg_logprob,
            no_speech_prob=no_speech_prob,
            compression_ratio=compression_ratio,
            text_length=text_length,
        )
        confidence = 1.0 - no_speech_prob

        result = STTResult(
            text=text,
            is_speech=is_speech,
            confidence=confidence,
            language=language,
            avg_logprob=avg_logprob,
            no_speech_prob=no_speech_prob,
            compression_ratio=compression_ratio,
            duration=duration,
        )

        logger.info(
            f"[{self.engine}] is_speech={is_speech}, confidence={confidence:.2f}, "
            f"text_len={text_length}, compression={compression_ratio:.2f}"
        )
        if is_speech:
            logger.info(f"Detected speech: '{text}'")
        else:
            logger.info(f"Detected noise. Text was: '{text}'")

        return result

    def _is_speech(
        self,
        avg_logprob: float,
        no_speech_prob: float,
        compression_ratio: float,
        text_length: int,
    ) -> bool:
        """
        Determine if audio contains actual speech vs background noise.

        Uses multiple heuristics:
        - avg_logprob: How confident the model is (higher = more confident)
        - no_speech_prob: Direct probability of no speech (lower = more speech)
        - compression_ratio: Text length vs audio duration (too low/high = not speech)
        - text_length: Minimum characters required

        Args:
            avg_logprob: Average log probability of transcription
            no_speech_prob: Probability of no speech
            compression_ratio: Characters per second ratio
            text_length: Length of transcribed text

        Returns:
            True if speech detected, False if just noise
        """
        # Check all thresholds
        confidence_ok = avg_logprob > self.min_speech_confidence
        no_speech_ok = no_speech_prob < self.max_no_speech_prob
        compression_ok = (
            self.min_compression_ratio < compression_ratio < self.max_compression_ratio
        )
        length_ok = text_length >= self.min_text_length

        # All conditions must be met for speech
        is_speech = confidence_ok and no_speech_ok and compression_ok and length_ok

        # Debug logging
        if not is_speech:
            reasons = []
            if not confidence_ok:
                reasons.append(
                    f"low confidence (avg_logprob={avg_logprob:.2f} < {self.min_speech_confidence})"
                )
            if not no_speech_ok:
                reasons.append(
                    f"high no_speech_prob ({no_speech_prob:.2f} > {self.max_no_speech_prob})"
                )
            if not compression_ok:
                reasons.append(
                    f"bad compression_ratio ({compression_ratio:.2f} not in "
                    f"[{self.min_compression_ratio}, {self.max_compression_ratio}])"
                )
            if not length_ok:
                reasons.append(
                    f"text too short ({text_length} < {self.min_text_length})"
                )

            logger.debug(f"Noise detected: {', '.join(reasons)}")

        return is_speech

    def _get_wav_duration(self, audio_wav: bytes) -> float:
        """
        Get duration of WAV audio in seconds.

        Args:
            audio_wav: WAV audio bytes

        Returns:
            Duration in seconds
        """
        try:
            with wave.open(io.BytesIO(audio_wav), 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                duration = frames / float(rate)
                return duration
        except Exception as e:
            logger.warning(f"Failed to get WAV duration: {e}")
            return 1.0  # Default fallback
