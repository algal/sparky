"""
Testbench-backed TTS middleware.

Goal: keep the Reachy Mini testbench as the owner of the speaker device, while our
assistant process remains a stateless client.

Approach (option 1):
- For each sentence chunk, synthesize a short WAV (OpenAI TTS by default).
- Write it to /tmp/reachy_mini_testbench/recordings/
- Ask testbench to play it via POST /api/audio/play/{filename}

Interruption (barge-in):
- We cannot stop a currently-playing WAV via the testbench API (no stop endpoint),
  but we can immediately stop scheduling further sentences by cancelling/clearing
  the queue.
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
from typing import Any, AsyncIterator, Optional, Callable

import aiohttp

from sparky_mvp.core.streaming import StreamChunk, StreamMiddleware

logger = logging.getLogger(__name__)


class TestbenchTTSMiddleware(StreamMiddleware):
    """
    Sentence-level TTS -> testbench playback.

    Class-level (singleton) queue so multiple pipeline instances don't overlap audio.
    """

    _sentence_queue: Optional[asyncio.Queue[tuple[int, str]]] = None
    _worker_task: Optional[asyncio.Task] = None
    _cancel_generation: int = 0
    _speaking_until_monotonic: float = 0.0

    def __init__(
        self,
        testbench_base_url: str = "http://127.0.0.1:8042",
        recordings_dir: str = "/tmp/reachy_mini_testbench/recordings",
        tts_engine: str = "openai",
        openai_model: str = "tts-1",
        openai_voice: str = "alloy",
        openai_speed: float = 1.0,
        request_timeout_s: float = 30.0,
        inter_sentence_gap_s: float = 0.05,
        max_chunk_chars: int | None = None,
        on_tts_text: Callable[[str], None] | None = None,
    ) -> None:
        self.testbench_base_url = testbench_base_url.rstrip("/")
        self.recordings_dir = Path(recordings_dir)
        self.tts_engine = tts_engine
        self.openai_model = openai_model
        self.openai_voice = openai_voice
        self.openai_speed = openai_speed
        self.request_timeout_s = request_timeout_s
        self.inter_sentence_gap_s = inter_sentence_gap_s
        if max_chunk_chars is None:
            max_chunk_chars = int(os.getenv("TTS_MAX_CHARS") or "160")
        self.max_chunk_chars = max(40, int(max_chunk_chars))
        self.on_tts_text = on_tts_text

        if TestbenchTTSMiddleware._sentence_queue is None:
            TestbenchTTSMiddleware._sentence_queue = asyncio.Queue()

    async def process(self, stream: AsyncIterator[StreamChunk]) -> AsyncIterator[StreamChunk]:
        if (
            TestbenchTTSMiddleware._worker_task is None
            or TestbenchTTSMiddleware._worker_task.done()
        ):
            TestbenchTTSMiddleware._worker_task = asyncio.create_task(self._worker_loop())

        try:
            gen = TestbenchTTSMiddleware._cancel_generation
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
                        await TestbenchTTSMiddleware._sentence_queue.put((gen, piece))
                yield chunk
        finally:
            q = TestbenchTTSMiddleware._sentence_queue
            if q is not None:
                # IMPORTANT: `q.empty()` is not sufficient here — the worker may have already dequeued
                # items (so the queue is empty) while `unfinished_tasks` is still non-zero. `join()`
                # correctly waits for all enqueued sentences (including the in-flight one) to finish.
                await q.join()

    @classmethod
    async def cancel_playback(cls) -> None:
        """
        Stop scheduling any further sentences.

        Note: cannot stop an already-playing sentence WAV (testbench has no stop API).
        """
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

        # Let the worker continue; it will observe the generation bump and skip any in-flight work.
        await asyncio.sleep(0)

    @classmethod
    def speaking_until_monotonic(cls) -> float:
        return float(cls._speaking_until_monotonic or 0.0)

    @classmethod
    def is_speaking(cls) -> bool:
        return time.monotonic() < cls.speaking_until_monotonic()

    async def _worker_loop(self) -> None:
        assert TestbenchTTSMiddleware._sentence_queue is not None

        while True:
            gen_at_enqueue, sentence = await TestbenchTTSMiddleware._sentence_queue.get()
            try:
                # Drop any sentence that was enqueued before the most recent cancellation.
                if gen_at_enqueue != TestbenchTTSMiddleware._cancel_generation:
                    continue

                wav_bytes = await self._synthesize_wav(sentence)
                if gen_at_enqueue != TestbenchTTSMiddleware._cancel_generation:
                    continue

                filename = await self._write_recording(wav_bytes)
                if gen_at_enqueue != TestbenchTTSMiddleware._cancel_generation:
                    continue

                duration_s = _wav_duration_seconds(wav_bytes)
                max_wav_s = float(os.getenv("TTS_MAX_WAV_SECONDS") or "30.0")
                if duration_s is None or duration_s <= 0.0:
                    # Defensive fallback: estimate from text length (we should never hard-hang).
                    duration_s = max(0.25, min(max_wav_s, len(sentence) / 20.0))
                    logger.warning(
                        "Could not parse WAV duration; using fallback=%.2fs (len=%s)",
                        duration_s,
                        len(sentence),
                    )
                if duration_s > max_wav_s:
                    logger.warning(
                        "Parsed WAV duration=%.2fs exceeds cap=%.2fs; clamping (possible bad WAV header)",
                        duration_s,
                        max_wav_s,
                    )
                    duration_s = max_wav_s

                # Even if we cancel, we cannot stop an already-playing WAV via testbench API,
                # so we keep "speaking" true until this duration elapses.
                TestbenchTTSMiddleware._speaking_until_monotonic = max(
                    TestbenchTTSMiddleware._speaking_until_monotonic,
                    time.monotonic() + float(duration_s),
                )
                await self._testbench_play(filename)

                # Prevent overlap by waiting roughly for audio duration.
                # (Testbench doesn't expose a playback-complete event.)
                sleep_s = max(0.0, duration_s + self.inter_sentence_gap_s)
                if sleep_s:
                    # Allow quick exit from the inter-sentence wait on cancel.
                    step = 0.1
                    remaining = sleep_s
                    while remaining > 0:
                        if gen_at_enqueue != TestbenchTTSMiddleware._cancel_generation:
                            break
                        await asyncio.sleep(min(step, remaining))
                        remaining -= step
            except Exception:
                logger.exception("TestbenchTTS worker error")
            finally:
                TestbenchTTSMiddleware._sentence_queue.task_done()

    async def _synthesize_wav(self, text: str) -> bytes:
        if self.tts_engine != "openai":
            raise RuntimeError(f"Unsupported tts_engine={self.tts_engine!r} (only 'openai' supported for now)")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (required for openai TTS)")

        def _call_openai() -> bytes:
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
        wav_bytes = await loop.run_in_executor(None, _call_openai)
        logger.info("TTS synthesized %d bytes in %.2fs", len(wav_bytes), time.time() - t0)
        return wav_bytes

    async def _write_recording(self, wav_bytes: bytes) -> str:
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        filename = f"tts_{uuid.uuid4().hex}.wav"
        path = self.recordings_dir / filename
        path.write_bytes(wav_bytes)
        logger.info("Wrote TTS wav: %s", path)
        return filename

    async def _testbench_play(self, filename: str) -> None:
        url = f"{self.testbench_base_url}/api/audio/play/{filename}"
        timeout = aiohttp.ClientTimeout(total=self.request_timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    raise RuntimeError(f"testbench play failed: {resp.status} {body[:200]}")
        logger.info("Testbench play: %s", filename)


def _wav_duration_seconds(wav_bytes: bytes) -> float | None:
    """
    Return duration for a WAV byte string.

    Note: OpenAI's TTS WAV output currently uses placeholder chunk sizes (0xFFFFFFFF)
    for streaming. Python's `wave` module will then report absurd `nframes` values.
    So we parse the header ourselves and fall back to the *actual* byte length.
    """
    try:
        import struct

        if len(wav_bytes) < 44:
            return None
        if wav_bytes[0:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
            return None

        pos = 12
        sample_rate: int | None = None
        block_align: int | None = None
        data_offset: int | None = None
        data_size: int | None = None

        while pos + 8 <= len(wav_bytes):
            chunk_id = wav_bytes[pos : pos + 4]
            chunk_size = struct.unpack_from("<I", wav_bytes, pos + 4)[0]
            pos += 8

            if chunk_id == b"fmt " and chunk_size >= 16 and pos + 16 <= len(wav_bytes):
                _audio_fmt, _channels, sr, _byte_rate, balign, _bps = struct.unpack_from(
                    "<HHIIHH", wav_bytes, pos
                )
                sample_rate = int(sr)
                block_align = int(balign)

            if chunk_id == b"data":
                data_offset = pos
                data_size = int(chunk_size)
                break

            pos += chunk_size
            # Chunks are word-aligned.
            if chunk_size % 2 == 1:
                pos += 1

        if sample_rate is None or block_align is None or sample_rate <= 0 or block_align <= 0:
            return None
        if data_offset is None:
            return None

        # Chunk size may be a placeholder (0xFFFFFFFF) for streaming output; trust actual length.
        if data_size is None or data_size == 0xFFFFFFFF or data_offset + data_size > len(wav_bytes):
            data_size = len(wav_bytes) - data_offset
        if data_size <= 0:
            return None

        return float(data_size) / float(sample_rate * block_align)
    except Exception:
        return None


def _split_text_for_tts(text: str, max_chars: int) -> list[str]:
    """
    Split text into smaller pieces to improve perceived responsiveness and barge-in.

    We prefer to split on punctuation, otherwise on whitespace.
    """
    t = " ".join(text.split())
    if not t:
        return []
    if len(t) <= max_chars:
        return [t]

    punct = (".", "!", "?", ";", ":", ",")
    parts: list[str] = []
    remaining = t
    while remaining:
        if len(remaining) <= max_chars:
            parts.append(remaining)
            break

        window = remaining[: max_chars + 1]
        # Prefer punctuation near the end of the window.
        cut = -1
        for p in punct:
            cut = max(cut, window.rfind(p))
        if cut >= int(max_chars * 0.5):
            cut += 1  # include the punctuation
        else:
            cut = window.rfind(" ")
            if cut < int(max_chars * 0.5):
                cut = max_chars

        piece = remaining[:cut].strip()
        if piece:
            parts.append(piece)
        remaining = remaining[cut:].lstrip()

    return parts
