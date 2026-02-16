"""
Speaker identification from short audio clips.

Uses resemblyzer (GE2E model) to extract speaker embeddings and match
against enrolled speakers via cosine similarity.  Runs on GPU if available.

Enrollment:
    Run tools/enroll_speakers.py to create the embeddings JSON file from
    recorded WAV samples.

Runtime:
    SpeakerIdentifier.identify(audio_bytes) → (speaker_name, confidence)
    Takes the same WAV bytes that VAD produces (16-bit PCM, 16kHz, mono or stereo).
    Returns ("unknown", 0.0) if no speaker exceeds the threshold.
"""

import io
import json
import logging
import wave
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Lazy-loaded to avoid import cost when speaker ID is disabled.
_encoder = None


def _get_encoder():
    """Lazy-load the resemblyzer VoiceEncoder (downloads model on first use)."""
    global _encoder
    if _encoder is None:
        from resemblyzer import VoiceEncoder
        _encoder = VoiceEncoder()
        logger.info("Speaker encoder loaded on %s", _encoder.device)
    return _encoder


def _wav_bytes_to_float(wav_bytes: bytes) -> np.ndarray:
    """Convert WAV bytes to float32 numpy array at 16kHz mono.

    Handles stereo→mono conversion and sample rate conversion if needed.
    """
    with io.BytesIO(wav_bytes) as buf:
        with wave.open(buf, "rb") as wf:
            n_channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

    # Parse to int16
    samples = np.frombuffer(raw, dtype=np.int16)

    # Stereo → mono
    if n_channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1).astype(np.int16)

    # Convert to float32 [-1, 1]
    audio = samples.astype(np.float32) / 32768.0

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        import soxr
        audio = soxr.resample(audio, sample_rate, 16000)

    return audio


class SpeakerIdentifier:
    """Identifies speakers from short audio clips using enrolled embeddings.

    Args:
        enrollments_path: Path to JSON file mapping speaker names to embedding vectors.
        threshold: Minimum cosine similarity to consider a match (0-1).
    """

    def __init__(self, enrollments_path: str, threshold: float = 0.7):
        self.threshold = threshold
        self._speakers: dict[str, np.ndarray] = {}

        path = Path(enrollments_path)
        if not path.exists():
            logger.warning("Speaker enrollments not found: %s", path)
            return

        with open(path) as f:
            data = json.load(f)

        for name, embedding in data.items():
            self._speakers[name] = np.array(embedding, dtype=np.float32)

        logger.info(
            "Loaded %d speaker enrollment(s): %s",
            len(self._speakers),
            ", ".join(self._speakers.keys()),
        )

    @property
    def n_speakers(self) -> int:
        return len(self._speakers)

    def identify(self, wav_bytes: bytes) -> Tuple[str, float]:
        """Identify the speaker from WAV audio bytes.

        Args:
            wav_bytes: WAV-format audio (16-bit PCM, any sample rate, mono or stereo).

        Returns:
            (speaker_name, confidence) where confidence is cosine similarity [0, 1].
            Returns ("unknown", 0.0) if no enrolled speaker exceeds the threshold.
        """
        if not self._speakers:
            return ("unknown", 0.0)

        try:
            audio = _wav_bytes_to_float(wav_bytes)
        except Exception as e:
            logger.warning("Failed to decode audio for speaker ID: %s", e)
            return ("unknown", 0.0)

        # Need at least 1 second of audio for a meaningful embedding
        if len(audio) < 16000:
            logger.debug("Audio too short for speaker ID: %.1fs", len(audio) / 16000)
            return ("unknown", 0.0)

        encoder = _get_encoder()
        embedding = encoder.embed_utterance(audio)

        # Compare against all enrolled speakers
        best_name = "unknown"
        best_score = 0.0

        for name, enrolled_emb in self._speakers.items():
            score = float(np.dot(embedding, enrolled_emb))
            if score > best_score:
                best_score = score
                best_name = name

        if best_score < self.threshold:
            logger.debug(
                "No speaker match (best: %s @ %.3f, threshold: %.3f)",
                best_name, best_score, self.threshold,
            )
            return ("unknown", best_score)

        logger.debug("Speaker identified: %s (%.3f)", best_name, best_score)
        return (best_name, best_score)


def enroll_from_wav(wav_path: str) -> list[float]:
    """Extract a speaker embedding from a WAV file for enrollment.

    Args:
        wav_path: Path to WAV file (any format — will be converted to 16kHz mono).

    Returns:
        Embedding as a list of floats (256-dimensional).
    """
    from resemblyzer import preprocess_wav
    wav = preprocess_wav(Path(wav_path))
    encoder = _get_encoder()
    embedding = encoder.embed_utterance(wav)
    return embedding.tolist()
