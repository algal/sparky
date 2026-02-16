"""
TTS Middleware for streaming text-to-speech integration.

This module provides middleware components that send sentence chunks to a TTS
service as they become available during streaming responses.
"""

import asyncio
import logging
import threading
from typing import AsyncIterator, Optional, Callable, Any
import aiohttp

from sparky_mvp.core.streaming import StreamChunk, StreamMiddleware
from sparky_mvp.robot.animations import RobotAnimator

# Set up logger with appropriate level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Show INFO and above for TTS


class TTSStreamMiddleware(StreamMiddleware):
    """
    Middleware that sends sentence chunks to TTS service and plays them.

    Only processes sentence chunks (complete sentences), not individual text chunks.
    This creates a seamless experience where the robot starts speaking
    as soon as the first sentence is ready, while the AI continues generating.

    Audio playback is queued to prevent overlapping speech.
    Uses class-level (singleton) queue shared across all instances.
    """

    # Class-level (singleton) queue and worker task shared across all instances
    _audio_queue = None
    _playback_task = None
    _media_manager = None
    _current_playback_future = None  # Track current playback for cancellation
    _cancellation_event = None  # Event to signal playback should stop
    _playback_stopped_event = None  # Event to signal playback has actually stopped
    _antenna_wiggling = False  # Track if antennas are currently wiggling
    _pending_fetch_tasks = set()  # Track pending TTS fetch tasks

    def __init__(
        self,
        tts_url: str = "http://localhost:4000/tts",
        timeout: float = 30.0,
        trigger_callback: Optional[Callable[[str], None]] = None,
        media_manager: Optional[Any] = None
    ):
        """
        Initialize TTS middleware.

        Args:
            tts_url: URL of the TTS service endpoint
            timeout: Request timeout in seconds
            trigger_callback: Optional callback function for TTS triggering.
                            If provided, this will be called instead of HTTP requests.
                            Useful for integrating with existing TTS systems.
            media_manager: Robot's media_manager for playing sounds (optional)
        """
        self.tts_url = tts_url
        self.timeout = timeout
        self.trigger_callback = trigger_callback

        # Initialize singleton queue if not already created
        if TTSStreamMiddleware._audio_queue is None:
            TTSStreamMiddleware._audio_queue = asyncio.Queue()
            logger.debug("Created singleton audio queue")

        # Initialize cancellation event
        if TTSStreamMiddleware._cancellation_event is None:
            import threading
            TTSStreamMiddleware._cancellation_event = threading.Event()

        # Initialize playback stopped event
        if TTSStreamMiddleware._playback_stopped_event is None:
            TTSStreamMiddleware._playback_stopped_event = asyncio.Event()
            TTSStreamMiddleware._playback_stopped_event.set()  # Initially stopped

        # Set class-level media_manager (shared across instances)
        if media_manager is not None:
            TTSStreamMiddleware._media_manager = media_manager

        logger.debug(f"TTSStreamMiddleware initialized: url={tts_url}, media_manager={TTSStreamMiddleware._media_manager is not None}")

    async def process(self, stream: AsyncIterator[StreamChunk]) -> AsyncIterator[StreamChunk]:
        """
        Process stream and trigger TTS for sentence chunks only.

        Args:
            stream: Input stream of chunks

        Yields:
            Stream chunks (passed through unchanged)
        """
        logger.debug("TTSStreamMiddleware.process() starting")

        # Start playback worker if not already running (singleton)
        if TTSStreamMiddleware._playback_task is None or TTSStreamMiddleware._playback_task.done():
            logger.debug("Starting singleton audio playback worker")
            TTSStreamMiddleware._playback_task = asyncio.create_task(self._playback_worker())

        try:
            async for chunk in stream:
                logger.debug(f"Received chunk: type={chunk.type}")
                # Only process sentence chunks for TTS
                if chunk.type == "sentence" and chunk.content.strip():
                    logger.info(f"Processing sentence for TTS: {chunk.content[:50]}...")
                    # Queue for TTS (non-blocking)
                    if self.trigger_callback:
                        # Use the provided callback (e.g., for integration with existing TTS)
                        await self._call_callback(chunk.content)
                    else:
                        # Fetch audio from TTS service and queue it
                        # Track the task so we can wait for it in cleanup
                        task = asyncio.create_task(self._fetch_and_queue_audio(chunk.content))
                        TTSStreamMiddleware._pending_fetch_tasks.add(task)
                        task.add_done_callback(TTSStreamMiddleware._pending_fetch_tasks.discard)
                    logger.debug("Queued for playback")
                elif chunk.type != "sentence":
                    # Pass through non-sentence chunks without TTS
                    logger.debug(f"Passing through {chunk.type} chunk (no TTS)")

                # Always yield the chunk immediately (don't wait for audio playback)
                yield chunk

        finally:
            # Stream processing complete - this is the natural signal we're done
            # First, wait for all pending TTS fetch tasks to complete
            if TTSStreamMiddleware._pending_fetch_tasks:
                logger.debug(f"Waiting for {len(TTSStreamMiddleware._pending_fetch_tasks)} pending fetch tasks")
                await asyncio.gather(*TTSStreamMiddleware._pending_fetch_tasks, return_exceptions=True)
                logger.debug("All fetch tasks complete")

            # Then wait for all queued audio to finish playing
            if TTSStreamMiddleware._audio_queue and not TTSStreamMiddleware._audio_queue.empty():
                logger.debug("Waiting for queued audio to finish playing")
                await TTSStreamMiddleware._audio_queue.join()
                logger.debug("All audio playback complete")

            # Now reset antennas to default pose (natural end of response)
            if TTSStreamMiddleware._antenna_wiggling:
                logger.debug("Stopping antenna wiggle animation")
                await RobotAnimator.stop_antenna_wiggle()
                TTSStreamMiddleware._antenna_wiggling = False
                logger.debug("Antenna wiggle stopped")

    async def _call_callback(self, text: str) -> None:
        """
        Call the TTS trigger callback in a non-blocking way.

        Args:
            text: Text to send to TTS
        """
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.trigger_callback, text)
        except Exception as e:
            logger.error(f"TTS callback error: {e}")

    async def _fetch_and_queue_audio(self, text: str) -> None:
        """
        Fetch audio from TTS service and add to playback queue.

        Args:
            text: Text to synthesize
        """
        try:
            logger.debug(f"Fetching audio for: '{text[:30]}...'")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.tts_url,
                    json={"text": text},
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        audio_data = await response.read()
                        logger.debug(f"Received {len(audio_data)} bytes, adding to queue")
                        await TTSStreamMiddleware._audio_queue.put(audio_data)
                    else:
                        logger.warning(f"TTS service returned status {response.status}")
        except Exception as e:
            logger.error(f"TTS fetch error: {e}")

    async def _playback_worker(self) -> None:
        """
        Background worker that plays audio from the queue sequentially.
        Ensures only one audio plays at a time.
        Singleton worker shared across all middleware instances.
        """
        logger.debug("Playback worker started")
        try:
            while True:
                # Wait for audio data from singleton queue
                audio_data = await TTSStreamMiddleware._audio_queue.get()
                logger.debug(f"Playing {len(audio_data)} bytes")

                # Start antenna wiggle if not already wiggling (beginning of speech)
                if not TTSStreamMiddleware._antenna_wiggling:
                    logger.debug("Starting antenna wiggle animation")
                    await RobotAnimator.start_antenna_wiggle()
                    TTSStreamMiddleware._antenna_wiggling = True

                # Signal that playback is starting
                TTSStreamMiddleware._playback_stopped_event.clear()

                # Play the audio (this blocks until done)
                await self._play_audio(audio_data)

                # Signal that playback has stopped
                TTSStreamMiddleware._playback_stopped_event.set()

                # Mark task as done
                TTSStreamMiddleware._audio_queue.task_done()
                logger.debug("Playback complete, waiting for next...")

        except asyncio.CancelledError:
            logger.debug("Worker cancelled")
            # Stop antenna wiggle on cancellation
            if TTSStreamMiddleware._antenna_wiggling:
                logger.debug("Stopping antenna wiggle (cancelled)")
                await RobotAnimator.stop_antenna_wiggle()
                TTSStreamMiddleware._antenna_wiggling = False
            # Make sure stopped event is set
            TTSStreamMiddleware._playback_stopped_event.set()
            raise
        except Exception as e:
            logger.error(f"Worker error: {e}")
            # Stop antenna wiggle on error
            if TTSStreamMiddleware._antenna_wiggling:
                logger.debug("Stopping antenna wiggle (error)")
                await RobotAnimator.stop_antenna_wiggle()
                TTSStreamMiddleware._antenna_wiggling = False
            TTSStreamMiddleware._playback_stopped_event.set()

    @classmethod
    async def cancel_playback(cls):
        """
        Cancel current playback and clear queue (for interruption).
        Waits for playback to actually stop before returning.
        """
        logger.debug("Starting cancellation...")

        # Signal cancellation to stop current playback immediately
        if cls._cancellation_event:
            logger.debug("Setting cancellation event")
            cls._cancellation_event.set()
            logger.debug(f"Event set: {cls._cancellation_event.is_set()}")
        else:
            logger.warning("No cancellation event!")

        # Clear the queue
        queue_items_cleared = 0
        if cls._audio_queue:
            while not cls._audio_queue.empty():
                try:
                    cls._audio_queue.get_nowait()
                    cls._audio_queue.task_done()
                    queue_items_cleared += 1
                except asyncio.QueueEmpty:
                    break
            logger.debug(f"Cleared {queue_items_cleared} items from queue")

        # Cancel the current playback future if using media_manager
        if cls._current_playback_future and not cls._current_playback_future.done():
            logger.debug("Cancelling playback future")
            cls._current_playback_future.cancel()

        # Don't cancel the playback task - let it finish gracefully
        # (it will stop when it sees the cancellation event)

        # Wait for playback to actually stop (with timeout)
        if cls._playback_stopped_event:
            logger.debug("Waiting for playback to stop...")
            try:
                await asyncio.wait_for(cls._playback_stopped_event.wait(), timeout=0.5)
                logger.debug("Playback stopped")
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for playback to stop")

        # Stop antenna wiggle on interruption
        if cls._antenna_wiggling:
            logger.debug("Stopping antenna wiggle (interrupted)")
            await RobotAnimator.stop_antenna_wiggle()
            cls._antenna_wiggling = False

        logger.debug("Cancellation complete")

    async def _call_tts_http(self, text: str) -> None:
        """
        Call TTS service via HTTP asynchronously and play the audio.

        Args:
            text: Text to synthesize and play
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.tts_url,
                    json={"text": text},
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        logger.debug("HTTP 200 OK from TTS service")

                        # Read audio data from response
                        audio_data = await response.read()
                        logger.debug(f"Received {len(audio_data)} bytes of audio data")

                        # Play the audio
                        await self._play_audio(audio_data)
                    else:
                        logger.warning(f"TTS service returned status {response.status}")
        except aiohttp.ClientError as e:
            logger.error(f"TTS service error: {e}")
        except asyncio.TimeoutError:
            logger.warning(f"TTS service timeout after {self.timeout}s")
        except Exception as e:
            logger.error(f"TTS error: {e}")

    async def _play_audio(self, audio_data: bytes) -> None:
        """
        Play audio data using pygame mixer (interruptible) or robot's media_manager.

        Args:
            audio_data: WAV audio data bytes
        """
        import tempfile
        import os

        temp_path = None
        try:
            # Write audio to temporary file
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.wav', delete=False) as f:
                temp_path = f.name
                f.write(audio_data)

            logger.debug(f"Playing audio file: {temp_path}")

            # Clear cancellation flag before starting
            TTSStreamMiddleware._cancellation_event.clear()

            # Use sounddevice for interruptible playback to Reachy speaker
            logger.debug("Using sounddevice (interruptible, Reachy speaker)")
            try:
                import sounddevice as sd
                import soundfile as sf
                import time

                loop = asyncio.get_event_loop()

                def play_with_cancellation_check():
                    """Play audio to Reachy speaker, checking for cancellation."""
                    # Find Reachy Mini Audio device
                    devices = sd.query_devices()
                    reachy_device = None

                    # print(f"[TTS-PLAYER] Searching for Reachy speaker...")
                    for idx, device in enumerate(devices):
                        # logger.debug(f"Device {idx}: {device['name']}")
                        if 'Reachy Mini Audio' in device['name']:
                            reachy_device = idx
                            logger.debug(f"Using Reachy speaker (device {idx})")
                            break

                    if reachy_device is None:
                        logger.warning("Reachy speaker not found, using default")

                    # Load audio file
                    data, samplerate = sf.read(temp_path, dtype='float32')  # Force float32
                    # print(f"[TTS-PLAYER] Loaded audio: {len(data)} samples, {samplerate}Hz")

                    # Determine number of channels
                    channels = data.shape[1] if len(data.shape) > 1 else 1
                    # print(f"[TTS-PLAYER] Audio has {channels} channel(s)")

                    # Start playback to Reachy device
                    if reachy_device is not None:
                        # print(f"[TTS-PLAYER] Creating stream for device {reachy_device}")
                        stream = sd.OutputStream(
                            device=reachy_device,
                            samplerate=samplerate,
                            channels=channels
                        )
                    else:
                        stream = sd.OutputStream(
                            samplerate=samplerate,
                            channels=channels
                        )

                    stream.start()
                    logger.info("[TTS-PLAYER] Playback started")

                    # Play in chunks, checking for cancellation
                    chunk_size = int(samplerate * 0.1)  # 100ms chunks
                    chunks_played = 0
                    total_chunks = (len(data) + chunk_size - 1) // chunk_size

                    for i in range(0, len(data), chunk_size):
                        if TTSStreamMiddleware._cancellation_event.is_set():
                            logger.warning(f"[TTS-PLAYER] CANCELLATION DETECTED! Stopping after {chunks_played}/{total_chunks} chunks")
                            stream.stop()
                            stream.close()
                            return

                        chunk = data[i:i + chunk_size]
                        stream.write(chunk)
                        chunks_played += 1

                    stream.stop()
                    stream.close()
                    logger.info(f"[TTS-PLAYER] Playback completed ({chunks_played}/{total_chunks} chunks)")

                await loop.run_in_executor(None, play_with_cancellation_check)

            except Exception as e:
                logger.warning(f"sounddevice error: {e}, trying fallback")
                # Fallback to system command
                import platform
                loop = asyncio.get_event_loop()
                system = platform.system()

                if system == "Darwin":  # macOS
                    cmd = f"afplay {temp_path}"
                elif system == "Linux":
                    cmd = f"aplay {temp_path}"
                else:
                    logger.error(f"Audio playback not implemented for {system}")
                    if temp_path and os.path.exists(temp_path):
                        os.unlink(temp_path)
                    return

                await loop.run_in_executor(None, lambda: os.system(cmd))

            logger.debug("Audio playback complete")

        except asyncio.CancelledError:
            logger.debug("Playback task cancelled")
            # Cancellation event is already set, playback will stop on next chunk check
            raise
        except Exception as e:
            logger.error(f"Audio playback error: {e}")
        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            TTSStreamMiddleware._current_playback_future = None


class InterruptibleTTSMiddleware(StreamMiddleware):
    """
    TTS middleware that can be interrupted by external signals.

    Only processes sentence chunks (complete sentences), not individual text chunks.
    This integrates with a threading.Event to allow cancellation of the TTS
    stream, useful for wake word interruption during response playback.
    """

    def __init__(
        self,
        tts_url: str = "http://localhost:4000/tts",
        timeout: float = 30.0,
        trigger_callback: Optional[Callable[[str], threading.Thread]] = None,
        cancel_flag: Optional[threading.Event] = None
    ):
        """
        Initialize interruptible TTS middleware.

        Args:
            tts_url: URL of the TTS service endpoint
            timeout: Request timeout in seconds
            trigger_callback: Optional callback that returns a Thread object.
                            Useful for tracking/cancelling TTS playback.
            cancel_flag: Threading event that signals cancellation when set
        """
        self.tts_url = tts_url
        self.timeout = timeout
        self.trigger_callback = trigger_callback
        self.cancel_flag = cancel_flag or threading.Event()
        self.current_thread: Optional[threading.Thread] = None
        logger.debug(f"InterruptibleTTSMiddleware initialized: url={tts_url}, callback={bool(trigger_callback)}")

    async def process(self, stream: AsyncIterator[StreamChunk]) -> AsyncIterator[StreamChunk]:
        """
        Process stream and trigger TTS for sentence chunks, with interrupt support.

        Args:
            stream: Input stream of chunks

        Yields:
            Stream chunks until interrupted or stream completes
        """
        async for chunk in stream:
            # Check if we should cancel
            if self.cancel_flag.is_set():
                logger.warning("TTS stream interrupted by cancel flag")
                break

            # Only process sentence chunks for TTS
            if chunk.type == "sentence" and chunk.content.strip():
                logger.info(f"Processing sentence for TTS (interruptible): {chunk.content[:50]}...")
                # Start TTS
                if self.trigger_callback:
                    await self._call_callback_with_thread(chunk.content)
                else:
                    await self._call_tts_http(chunk.content)
            elif chunk.type != "sentence":
                # Pass through non-sentence chunks without TTS
                logger.debug(f"Passing through {chunk.type} chunk (no TTS)")

            yield chunk

    async def _call_callback_with_thread(self, text: str) -> None:
        """
        Call TTS callback that returns a thread for tracking.

        Args:
            text: Text to send to TTS
        """
        try:
            loop = asyncio.get_event_loop()
            self.current_thread = await loop.run_in_executor(
                None, self.trigger_callback, text
            )
        except Exception as e:
            logger.error(f"TTS callback error: {e}")

    async def _call_tts_http(self, text: str) -> None:
        """
        Call TTS service via HTTP asynchronously.

        Args:
            text: Text to synthesize and play
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.tts_url,
                    json={"text": text},
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        logger.debug("HTTP 200 OK from TTS service")
                    else:
                        logger.warning(f"TTS service returned status {response.status}")
        except aiohttp.ClientError as e:
            logger.error(f"TTS service error: {e}")
        except asyncio.TimeoutError:
            logger.warning(f"TTS service timeout after {self.timeout}s")
        except Exception as e:
            logger.error(f"TTS error: {e}")

    def reset(self) -> None:
        """Reset the cancellation flag for reuse."""
        self.cancel_flag.clear()
        self.current_thread = None
