"""
Direct TTS middleware — plays audio through ReachyMini's SoundDevice system.

Replaces TestbenchTTSMiddleware by eliminating the HTTP round-trip to the
testbench.  Instead of writing a WAV and POSTing it to the testbench API,
this middleware calls ``media_manager.play_sound()`` directly, which pushes
samples into SoundDevice's output buffer.

Playback completion is detected by polling the audio buffer rather than
estimating WAV duration, so the self-listening gap that plagued the testbench
path is eliminated.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import time
import uuid
import wave
from pathlib import Path
from typing import AsyncIterator, Callable, Optional

import numpy as np

from sparky_mvp.core.streaming import StreamChunk, StreamMiddleware
from sparky_mvp.core.middlewares.testbench_tts import (
    _wav_duration_seconds,
    _split_text_for_tts,
)

logger = logging.getLogger(__name__)


class DirectTTSMiddleware(StreamMiddleware):
    """
    Sentence-level TTS with direct SoundDevice playback.

    Class-level (singleton) queue so multiple pipeline instances don't overlap audio.
    """

    _sentence_queue: Optional[asyncio.Queue[tuple[int, str]]] = None
    _worker_task: Optional[asyncio.Task] = None
    _cancel_generation: int = 0
    _speaking: bool = False
    _media_manager: object | None = None
    _head_wobbler: object | None = None
    _echo_canceller: object | None = None
    _recordings_dir: Path = Path("/tmp/sparky_mvp_tts")

    # Lazy-initialized local TTS engines (class-level singletons)
    _kokoro_pipeline: object | None = None
    _orpheus_model: object | None = None

    def __init__(
        self,
        media_manager: object,
        tts_engine: str = "openai",
        openai_model: str = "tts-1",
        openai_voice: str = "alloy",
        openai_speed: float = 1.0,
        kokoro_voice: str = "af_heart",
        kokoro_speed: float = 1.0,
        kokoro_lang: str = "a",
        orpheus_model: str = "canopylabs/orpheus-3b-0.1-ft",
        orpheus_voice: str = "tara",
        request_timeout_s: float = 30.0,
        inter_sentence_gap_s: float = 0.05,
        max_chunk_chars: int | None = None,
        on_tts_text: Callable[[str], None] | None = None,
        head_wobbler: object | None = None,
        echo_canceller: object | None = None,
        riva_url: str = "http://127.0.0.1:9000/v1/audio/synthesize",
        riva_model: str = "magpie-tts-multilingual",
        riva_voice: str = "Magpie-Multilingual.EN-US.Jason",
        riva_language_code: str = "en-US",
        riva_sample_rate_hz: int = 24000,
    ) -> None:
        self.tts_engine = tts_engine
        self.openai_model = openai_model
        self.openai_voice = openai_voice
        self.openai_speed = openai_speed
        self.kokoro_voice = kokoro_voice
        self.kokoro_speed = kokoro_speed
        self.kokoro_lang = kokoro_lang
        self.orpheus_model_name = orpheus_model
        self.orpheus_voice = orpheus_voice
        self.riva_url = riva_url
        self.riva_model = riva_model
        self.riva_voice = riva_voice
        self.riva_language_code = riva_language_code
        self.riva_sample_rate_hz = int(riva_sample_rate_hz)
        self.request_timeout_s = request_timeout_s
        self.inter_sentence_gap_s = inter_sentence_gap_s
        if max_chunk_chars is None:
            max_chunk_chars = int(os.getenv("TTS_MAX_CHARS") or "160")
        self.max_chunk_chars = max(40, int(max_chunk_chars))
        self.on_tts_text = on_tts_text

        DirectTTSMiddleware._media_manager = media_manager
        DirectTTSMiddleware._head_wobbler = head_wobbler
        DirectTTSMiddleware._echo_canceller = echo_canceller

        if DirectTTSMiddleware._sentence_queue is None:
            DirectTTSMiddleware._sentence_queue = asyncio.Queue()

    async def process(self, stream: AsyncIterator[StreamChunk]) -> AsyncIterator[StreamChunk]:
        if (
            DirectTTSMiddleware._worker_task is None
            or DirectTTSMiddleware._worker_task.done()
        ):
            DirectTTSMiddleware._worker_task = asyncio.create_task(self._worker_loop())

        try:
            gen = DirectTTSMiddleware._cancel_generation
            async for chunk in stream:
                if chunk.type == "sentence" and chunk.content.strip():
                    sentence = chunk.content.strip()
                    pieces = _split_text_for_tts(sentence, self.max_chunk_chars)
                    if len(pieces) == 1:
                        logger.info("TTS enqueue sentence: %r", sentence[:80])
                    else:
                        logger.info(
                            "TTS enqueue sentence pieces=%s max_chars=%s: %r",
                            len(pieces),
                            self.max_chunk_chars,
                            sentence[:80],
                        )
                    for piece in pieces:
                        if self.on_tts_text is not None:
                            try:
                                self.on_tts_text(piece)
                            except Exception:
                                logger.debug("on_tts_text callback failed", exc_info=True)
                        assert DirectTTSMiddleware._sentence_queue is not None
                        await DirectTTSMiddleware._sentence_queue.put((gen, piece))
                yield chunk
        finally:
            q = DirectTTSMiddleware._sentence_queue
            if q is not None:
                await q.join()

    @classmethod
    async def cancel_playback(cls) -> None:
        """Stop scheduling further sentences and clear the audio buffer."""
        cls._cancel_generation += 1

        q = cls._sentence_queue
        if q is not None:
            cleared = 0
            while not q.empty():
                try:
                    q.get_nowait()
                    q.task_done()
                    cleared += 1
                except asyncio.QueueEmpty:
                    break
            if cleared:
                logger.info("Cleared %s queued TTS sentences", cleared)

        # Clear audio buffer to stop current playback
        try:
            mm = cls._media_manager
            if mm is not None and getattr(mm, "audio", None) is not None:
                mm.audio.clear_output_buffer()
                logger.info("Cleared audio output buffer")
        except Exception as e:
            logger.warning("Could not clear audio buffer: %s", e)

        # Clear AEC speaker buffer (stale reference after cancel)
        ec = cls._echo_canceller
        if ec is not None:
            try:
                ec.clear()
            except Exception:
                logger.debug("AEC clear failed", exc_info=True)

        # Reset HeadWobbler to stop residual head movement
        hw = cls._head_wobbler
        if hw is not None:
            try:
                hw.reset()
            except Exception:
                logger.debug("HeadWobbler reset failed", exc_info=True)

        cls._speaking = False
        await asyncio.sleep(0)

    @classmethod
    def is_speaking(cls) -> bool:
        return cls._speaking

    @classmethod
    def _is_audio_buffer_empty(cls) -> bool:
        """Check if the SoundDevice output buffer has drained."""
        try:
            mm = cls._media_manager
            if mm is None or getattr(mm, "audio", None) is None:
                return True
            audio = mm.audio
            lock = getattr(audio, "_output_lock", None)
            buf = getattr(audio, "_output_buffer", None)
            if lock is None or buf is None:
                return True
            with lock:
                return len(buf) == 0
        except Exception:
            return True

    async def _worker_loop(self) -> None:
        assert DirectTTSMiddleware._sentence_queue is not None

        while True:
            gen_at_enqueue, sentence = await DirectTTSMiddleware._sentence_queue.get()
            try:
                if gen_at_enqueue != DirectTTSMiddleware._cancel_generation:
                    continue

                wav_bytes = await self._synthesize_wav(sentence)
                if gen_at_enqueue != DirectTTSMiddleware._cancel_generation:
                    continue

                filepath = self._write_recording(wav_bytes)
                if gen_at_enqueue != DirectTTSMiddleware._cancel_generation:
                    continue

                # Feed audio to HeadWobbler for speech-synchronized head movement
                hw = DirectTTSMiddleware._head_wobbler
                if hw is not None:
                    try:
                        hw.feed_wav(wav_bytes)
                    except Exception:
                        logger.debug("HeadWobbler feed failed", exc_info=True)

                # Feed speaker WAV to AEC for echo cancellation
                ec = DirectTTSMiddleware._echo_canceller
                if ec is not None:
                    try:
                        ec.feed_speaker_wav(wav_bytes)
                    except Exception:
                        logger.debug("AEC speaker feed failed", exc_info=True)

                # Play directly through SoundDevice
                DirectTTSMiddleware._speaking = True
                mm = DirectTTSMiddleware._media_manager
                if mm is not None:
                    try:
                        mm.play_sound(str(filepath))
                    except Exception as e:
                        logger.error("play_sound failed: %s", e)
                        DirectTTSMiddleware._speaking = False
                        continue

                # Wait for audio buffer to drain (accurate completion detection)
                # with a safety timeout based on estimated duration
                duration_s = _wav_duration_seconds(wav_bytes)
                max_wav_s = float(os.getenv("TTS_MAX_WAV_SECONDS") or "30.0")
                if duration_s is None or duration_s <= 0.0:
                    duration_s = max(0.25, min(max_wav_s, len(sentence) / 20.0))
                if duration_s > max_wav_s:
                    duration_s = max_wav_s
                deadline = time.monotonic() + duration_s + 2.0  # safety margin

                while time.monotonic() < deadline:
                    if gen_at_enqueue != DirectTTSMiddleware._cancel_generation:
                        break
                    if self._is_audio_buffer_empty():
                        break
                    await asyncio.sleep(0.05)

                DirectTTSMiddleware._speaking = False

                # Inter-sentence gap
                if self.inter_sentence_gap_s > 0:
                    await asyncio.sleep(self.inter_sentence_gap_s)

            except Exception:
                logger.exception("DirectTTS worker error")
                DirectTTSMiddleware._speaking = False
            finally:
                DirectTTSMiddleware._sentence_queue.task_done()

    async def _synthesize_wav(self, text: str) -> bytes:
        if self.tts_engine == "openai":
            return await self._synthesize_openai(text)
        elif self.tts_engine == "kokoro":
            return await self._synthesize_kokoro(text)
        elif self.tts_engine == "orpheus":
            return await self._synthesize_orpheus(text)
        elif self.tts_engine == "riva":
            return await self._synthesize_riva(text)
        else:
            raise RuntimeError(
                f"Unsupported tts_engine={self.tts_engine!r}"
                " (supported: 'openai', 'kokoro', 'orpheus', 'riva')"
            )

    # ------------------------------------------------------------------
    # OpenAI TTS (cloud)
    # ------------------------------------------------------------------

    async def _synthesize_openai(self, text: str) -> bytes:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (required for openai TTS)")

        def _call() -> bytes:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            resp = client.audio.speech.create(
                model=self.openai_model,
                voice=self.openai_voice,
                input=text,
                response_format="wav",
                speed=self.openai_speed,
                timeout=self.request_timeout_s,
            )
            return resp.read()

        loop = asyncio.get_event_loop()
        t0 = time.time()
        wav_bytes = await loop.run_in_executor(None, _call)
        logger.info(
            "OpenAI TTS: %d bytes in %.2fs", len(wav_bytes), time.time() - t0
        )
        return wav_bytes

    # ------------------------------------------------------------------
    # Kokoro TTS (local, 82M params, 24kHz)
    # ------------------------------------------------------------------

    async def _synthesize_kokoro(self, text: str) -> bytes:
        def _call() -> bytes:
            cls = DirectTTSMiddleware
            if cls._kokoro_pipeline is None:
                from kokoro import KPipeline

                logger.info(
                    "Initializing Kokoro TTS (lang=%s) ...", self.kokoro_lang
                )
                cls._kokoro_pipeline = KPipeline(lang_code=self.kokoro_lang)
                logger.info("Kokoro TTS ready")

            pipeline = cls._kokoro_pipeline
            chunks = []
            for _gs, _ps, audio in pipeline(
                text, voice=self.kokoro_voice, speed=self.kokoro_speed
            ):
                if audio is not None:
                    chunks.append(audio)

            if not chunks:
                raise RuntimeError("Kokoro produced no audio")

            full_audio = np.concatenate(chunks)
            return _numpy_to_wav(full_audio, sample_rate=24000)

        loop = asyncio.get_event_loop()
        t0 = time.time()
        wav_bytes = await loop.run_in_executor(None, _call)
        logger.info(
            "Kokoro TTS: %d bytes in %.2fs", len(wav_bytes), time.time() - t0
        )
        return wav_bytes

    # ------------------------------------------------------------------
    # Orpheus TTS (local, 3B params via vLLM, 24kHz)
    # ------------------------------------------------------------------

    async def _synthesize_orpheus(self, text: str) -> bytes:
        def _call() -> bytes:
            cls = DirectTTSMiddleware
            if cls._orpheus_model is None:
                from sparky_mvp.core.orpheus_engine import OrpheusEngine

                logger.info(
                    "Initializing Orpheus TTS (model=%s) ...",
                    self.orpheus_model_name,
                )
                cls._orpheus_model = OrpheusEngine(
                    model_name=self.orpheus_model_name,
                )

            engine = cls._orpheus_model
            assert engine is not None
            pcm_data = engine.synthesize(text=text, voice=self.orpheus_voice)

            if not pcm_data:
                raise RuntimeError("Orpheus produced no audio")

            return _pcm_to_wav(pcm_data, sample_rate=24000, sample_width=2)

        loop = asyncio.get_event_loop()
        t0 = time.time()
        wav_bytes = await loop.run_in_executor(None, _call)
        logger.info(
            "Orpheus TTS: %d bytes in %.2fs", len(wav_bytes), time.time() - t0
        )
        return wav_bytes

    # ------------------------------------------------------------------
    # NVIDIA Riva / Magpie TTS (HTTP)
    # ------------------------------------------------------------------

    async def _synthesize_riva(self, text: str) -> bytes:
        def _call() -> bytes:
            import requests

            headers = {"Content-Type": "application/json", "Accept": "audio/wav"}

            payload = {
                "text": text,
                "model": self.riva_model,
                "voice_name": self.riva_voice,
                "language_code": self.riva_language_code,
                "sample_rate_hz": self.riva_sample_rate_hz,
            }

            response = requests.post(
                self.riva_url,
                json=payload,
                headers=headers,
                timeout=self.request_timeout_s,
            )
            try:
                response.raise_for_status()
            except Exception as e:
                body = (response.text or "")[:300]
                raise RuntimeError(
                    f"Riva TTS request failed: status={response.status_code} body={body!r}"
                ) from e

            wav_bytes = response.content

            if not wav_bytes.startswith(b"RIFF"):
                raise RuntimeError("Riva TTS did not return WAV audio (expected audio/wav)")
            return wav_bytes

        loop = asyncio.get_event_loop()
        t0 = time.time()
        wav_bytes = await loop.run_in_executor(None, _call)
        logger.info(
            "Riva TTS: %d bytes in %.2fs", len(wav_bytes), time.time() - t0
        )
        return wav_bytes

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _write_recording(self, wav_bytes: bytes) -> Path:
        DirectTTSMiddleware._recordings_dir.mkdir(parents=True, exist_ok=True)
        filename = f"tts_{uuid.uuid4().hex}.wav"
        path = DirectTTSMiddleware._recordings_dir / filename
        path.write_bytes(wav_bytes)
        return path


# ------------------------------------------------------------------
# Audio format conversion utilities
# ------------------------------------------------------------------

def _numpy_to_wav(audio: np.ndarray, sample_rate: int = 24000) -> bytes:
    """Convert float32 numpy audio to WAV bytes (16-bit PCM mono)."""
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    return _pcm_to_wav(pcm.tobytes(), sample_rate=sample_rate, sample_width=2)


def _pcm_to_wav(
    pcm_data: bytes, sample_rate: int = 24000, sample_width: int = 2
) -> bytes:
    """Wrap raw PCM bytes in a WAV container (mono)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()
