"""
Audio Input Processor for Reachy MVP

Handles audio transcription and speech detection using local STT,
separating audio processing concerns from state machine orchestration.

Text-only mode - no audio passthrough to LLM.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

from sparky_mvp.core.stt_engine import STTEngine, STTResult

logger = logging.getLogger(__name__)


@dataclass
class AudioInputResult:
    """
    Result of processing audio input through STT.

    Contains transcribed text along with metadata about whether
    the input should be processed or skipped (e.g., background noise).
    """

    transcribed_text: Optional[str] = None
    should_skip: bool = False
    confidence: float = 0.0
    skip_reason: Optional[str] = None

    def to_message(self) -> Dict[str, str]:
        """
        Convert to conversation message format.

        Returns:
            Message dict with role and content
        """
        return {
            "role": "user",
            "content": self.transcribed_text or ""
        }


class AudioInputProcessor:
    """
    Processes audio input through STT pipeline with noise detection.

    Responsibilities:
    - Transcribe audio using local STT engine
    - Detect speech vs background noise
    - Return structured results for state machine
    """

    def __init__(self, stt_engine: STTEngine):
        """
        Initialize the audio input processor.

        Args:
            stt_engine: Local STT engine instance (required)
        """
        if stt_engine is None:
            raise ValueError("STT engine is required for text-only mode")

        self.stt_engine = stt_engine
        logger.info("AudioInputProcessor initialized with local STT")

    async def process_audio_input(self, audio_bytes: bytes) -> AudioInputResult:
        """
        Process captured audio through STT pipeline.

        Flow:
        1. Transcribe audio using local STT
        2. Check for speech vs noise
        3. Return structured result

        Args:
            audio_bytes: Raw audio data (WAV format)

        Returns:
            AudioInputResult with transcription and metadata
        """
        logger.info("Processing audio with local STT engine")

        try:
            # Transcribe audio in thread executor to avoid blocking async event loop
            # faster-whisper has blocking operations that need to run in a separate thread
            loop = asyncio.get_event_loop()
            stt_result: STTResult = await loop.run_in_executor(
                None,  # Use default executor
                self.stt_engine.transcribe,
                audio_bytes
            )

            # Log full transcription details
            logger.info(
                f"STT transcription complete: is_speech={stt_result.is_speech}, "
                f"confidence={stt_result.confidence:.2f}, "
                f"text='{stt_result.text}'"
            )
            logger.debug(
                f"STT metrics: avg_logprob={stt_result.avg_logprob:.2f}, "
                f"no_speech_prob={stt_result.no_speech_prob:.2f}, "
                f"compression={stt_result.compression_ratio:.2f}, "
                f"duration={stt_result.duration:.2f}s"
            )

            # Check if this is actual speech or just noise
            if not stt_result.is_speech:
                # Background noise detected
                reason = (
                    f"Background noise detected. "
                    f"Metrics: avg_logprob={stt_result.avg_logprob:.2f}, "
                    f"no_speech_prob={stt_result.no_speech_prob:.2f}, "
                    f"compression={stt_result.compression_ratio:.2f}"
                )

                logger.info(f"Skipping input: {reason}")

                return AudioInputResult(
                    should_skip=True,
                    skip_reason=reason,
                    confidence=stt_result.confidence
                )

            # Valid speech detected
            logger.info(f"Speech detected: '{stt_result.text}'")

            return AudioInputResult(
                transcribed_text=stt_result.text,
                confidence=stt_result.confidence
            )

        except Exception as e:
            # STT transcription failed - skip this input
            logger.error(f"STT transcription failed: {e}", exc_info=True)

            return AudioInputResult(
                should_skip=True,
                skip_reason=f"STT failed: {str(e)}",
                confidence=0.0
            )
