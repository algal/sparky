"""Converts TTS audio into speech-synchronized head movement offsets.

Copied from reachy_mini_conversation_app/audio/head_wobbler.py (2026-02-08).
Modified: added feed_pcm() for direct PCM feeding (avoids base64 round-trip),
fixed import path for local speech_tapper module.
"""

import time
import queue
import base64
import logging
import threading
import wave
import io
from typing import Tuple
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from sparky_mvp.robot.speech_tapper import HOP_MS, SwayRollRT


SAMPLE_RATE = 24000
MOVEMENT_LATENCY_S = 0.2  # seconds between audio and robot movement
logger = logging.getLogger(__name__)


class HeadWobbler:
    """Converts audio deltas into head movement offsets."""

    def __init__(self, set_speech_offsets: Callable[[Tuple[float, float, float, float, float, float]], None], gain: float = 0.6) -> None:
        """Initialize the head wobbler."""
        self._apply_offsets = set_speech_offsets
        self._gain = gain
        self._base_ts: float | None = None
        self._hops_done: int = 0

        self.audio_queue: "queue.Queue[Tuple[int, int, NDArray[np.int16]]]" = queue.Queue()
        self.sway = SwayRollRT()

        # Synchronization primitives
        self._state_lock = threading.Lock()
        self._sway_lock = threading.Lock()
        self._generation = 0

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def feed(self, delta_b64: str) -> None:
        """Thread-safe: push base64-encoded audio into the consumer queue."""
        buf = np.frombuffer(base64.b64decode(delta_b64), dtype=np.int16).reshape(1, -1)
        with self._state_lock:
            generation = self._generation
        self.audio_queue.put((generation, SAMPLE_RATE, buf))

    def feed_pcm(self, pcm: NDArray[np.int16], sample_rate: int) -> None:
        """Thread-safe: push raw PCM int16 audio into the consumer queue."""
        buf = pcm.reshape(1, -1) if pcm.ndim == 1 else pcm
        with self._state_lock:
            generation = self._generation
        self.audio_queue.put((generation, sample_rate, buf))

    def feed_wav(self, wav_bytes: bytes) -> None:
        """Thread-safe: extract PCM from WAV bytes and push into consumer queue."""
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)
            pcm = np.frombuffer(raw, dtype=np.int16)
            self.feed_pcm(pcm, sr)
        except Exception:
            logger.debug("feed_wav: failed to parse WAV", exc_info=True)

    def start(self) -> None:
        """Start the head wobbler loop in a thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.working_loop, daemon=True)
        self._thread.start()
        logger.debug("Head wobbler started")

    def stop(self) -> None:
        """Stop the head wobbler loop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        logger.debug("Head wobbler stopped")

    def working_loop(self) -> None:
        """Convert audio deltas into head movement offsets."""
        hop_dt = HOP_MS / 1000.0

        logger.debug("Head wobbler thread started")
        while not self._stop_event.is_set():
            queue_ref = self.audio_queue
            try:
                chunk_generation, sr, chunk = queue_ref.get_nowait()  # (gen, sr, data)
            except queue.Empty:
                # avoid while to never exit
                time.sleep(MOVEMENT_LATENCY_S)
                continue

            try:
                with self._state_lock:
                    current_generation = self._generation
                if chunk_generation != current_generation:
                    continue

                if self._base_ts is None:
                    with self._state_lock:
                        if self._base_ts is None:
                            self._base_ts = time.monotonic()

                pcm = np.asarray(chunk).squeeze(0)
                with self._sway_lock:
                    results = self.sway.feed(pcm, sr)

                i = 0
                while i < len(results):
                    with self._state_lock:
                        if self._generation != current_generation:
                            break
                        base_ts = self._base_ts
                        hops_done = self._hops_done

                    if base_ts is None:
                        base_ts = time.monotonic()
                        with self._state_lock:
                            if self._base_ts is None:
                                self._base_ts = base_ts
                                hops_done = self._hops_done

                    target = base_ts + MOVEMENT_LATENCY_S + hops_done * hop_dt
                    now = time.monotonic()

                    if now - target >= hop_dt:
                        lag_hops = int((now - target) / hop_dt)
                        drop = min(lag_hops, len(results) - i - 1)
                        if drop > 0:
                            with self._state_lock:
                                self._hops_done += drop
                                hops_done = self._hops_done
                            i += drop
                            continue

                    if target > now:
                        time.sleep(target - now)
                        with self._state_lock:
                            if self._generation != current_generation:
                                break

                    r = results[i]
                    g = self._gain
                    offsets = (
                        g * r["x_mm"] / 1000.0,
                        g * r["y_mm"] / 1000.0,
                        g * r["z_mm"] / 1000.0,
                        g * r["roll_rad"],
                        g * r["pitch_rad"],
                        g * r["yaw_rad"],
                    )

                    with self._state_lock:
                        if self._generation != current_generation:
                            break

                    self._apply_offsets(offsets)

                    with self._state_lock:
                        self._hops_done += 1
                    i += 1
            finally:
                queue_ref.task_done()
        logger.debug("Head wobbler thread exited")

    def reset(self) -> None:
        """Reset the internal state."""
        with self._state_lock:
            self._generation += 1
            self._base_ts = None
            self._hops_done = 0

        # Drain any queued audio chunks from previous generations
        drained_any = False
        while True:
            try:
                _, _, _ = self.audio_queue.get_nowait()
            except queue.Empty:
                break
            else:
                drained_any = True
                self.audio_queue.task_done()

        with self._sway_lock:
            self.sway.reset()

        # Clear speech offsets to neutral
        self._apply_offsets((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

        if drained_any:
            logger.debug("Head wobbler queue drained during reset")
