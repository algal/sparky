"""
State machine orchestrating Reachy's interactive voice assistant loop.

Implements the MVP workflow:
- STARTUP: Initialize components, robot sleeps
- SLEEP: Monitor for wake word
- INTERACTIVE: Listen for user speech with VAD
- PROCESSING: Send to OpenClaw, stream responses
"""

import asyncio
import logging
import os
import signal
import sys
import time
import re
from enum import Enum
from typing import Optional, Dict, Any, Deque, Tuple
from collections import deque
import pyaudio
import numpy as np
import openwakeword
from openwakeword.model import Model

logger = logging.getLogger(__name__)

try:
    from reachy_mini import ReachyMini
except ImportError:
    logger.warning("reachy-mini not installed, using mock mode")
    ReachyMini = None

from sparky_mvp.core.vad_capture import VADSpeechCapture
from sparky_mvp.core.resampling_stream import ResamplingParams, ResamplingPyAudioStream
from sparky_mvp.core.stt_engine import STTEngine
from sparky_mvp.core.audio_input_processor import AudioInputProcessor, AudioInputResult
from sparky_mvp.robot.animations import RobotAnimator
from sparky_mvp.robot.moves import MovementManager
from sparky_mvp.robot.head_wobbler import HeadWobbler
from sparky_mvp.robot.thinking import ThinkingMove
from sparky_mvp.robot.camera_worker import CameraWorker
from sparky_mvp.robot.scrfd_head_tracker import SCRFDHeadTracker
from sparky_mvp.tools.tool_handlers import TOOL_DEFINITIONS, TOOL_HANDLERS, set_camera_worker
from sparky_mvp.core.streaming import StreamPipeline, StreamChunk
from sparky_mvp.core.middlewares import (
    SentenceBufferMiddleware,
    FilterMiddleware,
    OpenClawProviderMiddleware,
)
from sparky_mvp.core.openclaw_gateway_client import (
    OpenClawGatewayClient,
    create_gateway_client_from_config,
)
from sparky_mvp.core.spontaneous_speech import SpontaneousSpeechManager
from sparky_mvp.core.spontaneous_gestures import SpontaneousGestureManager
from sparky_mvp.core.echo_canceller import AcousticEchoCanceller
from sparky_mvp.core.aec_stream import AECStream
from sparky_mvp.core.middlewares.tts import TTSStreamMiddleware
from sparky_mvp.core.middlewares.testbench_tts import TestbenchTTSMiddleware
from sparky_mvp.core.middlewares.direct_tts import DirectTTSMiddleware


class ReachyState(Enum):
    """States for Reachy's operation."""
    STARTUP = "startup"
    SLEEP = "sleep"
    INTERACTIVE = "interactive"
    PROCESSING = "processing"


class ReachyStateMachine:
    """
    Main orchestrator for Reachy MVP voice assistant.

    Manages the full lifecycle: startup → sleep → wake → listen → process → respond
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the state machine with configuration.

        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.state = ReachyState.STARTUP

        # Initialize audio
        self.audio = pyaudio.PyAudio()
        self.mic_stream: Optional[Any] = None

        # Initialize wake word detector
        self.wake_word_model: Optional[Model] = None

        # Initialize robot (or mock)
        self.mini: Optional[ReachyMini] = None
        self.use_mock_robot = ReachyMini is None

        # Initialize components (will be set up in startup)
        self.vad_capture: Optional[VADSpeechCapture] = None
        self.llm_client = None
        self.gateway_client: Optional[OpenClawGatewayClient] = None
        self.stt_engine: Optional[STTEngine] = None
        self.audio_processor: Optional[AudioInputProcessor] = None
        self.animator: Optional[RobotAnimator] = None
        self.movement_manager: Optional[MovementManager] = None
        self.head_wobbler: Optional[HeadWobbler] = None
        self.camera_worker: Optional[CameraWorker] = None
        self.node_client = None  # OpenClaw node (camera.snap)
        self._node_task: Optional[asyncio.Task] = None
        self._spontaneous_manager: Optional[SpontaneousSpeechManager] = None
        self._spontaneous_task: Optional[asyncio.Task] = None
        self._gesture_manager: Optional[SpontaneousGestureManager] = None
        self._gesture_task: Optional[asyncio.Task] = None
        self._echo_canceller: Optional[object] = None
        self.speaker_identifier = None  # Speaker voice ID

        # Provider selection: "openclaw" or "anthropic"
        self._provider = config.get("openclaw", {}).get("provider", "anthropic")

        # Conversation state
        self.messages = [
            {"role": "system", "content": config["conversation"]["system_prompt"]}
        ]
        self.max_history = config["conversation"]["max_history"]

        # Control flags
        self.interruption_flag = asyncio.Event()
        self.running = True

        # Current audio being processed
        self.current_audio: Optional[bytes] = None

        # Queue for speech audio from VAD
        self.speech_queue: asyncio.Queue[bytes] = asyncio.Queue()

        # Task for continuous VAD
        self.vad_task: Optional[asyncio.Task] = None

        # Middleware pipeline (will be built in startup)
        self.pipeline: Optional[StreamPipeline] = None
        self._tts_controller = None

        # Echo-guard: recent text that we asked TTS to speak (to avoid responding to ourselves)
        self._recent_tts_text: Deque[Tuple[float, str]] = deque(maxlen=200)
        self._last_bargein_ignore_log_monotonic: float = 0.0

    def _find_reachy_microphone(self) -> int:
        """
        Find Reachy's microphone device index.

        Returns:
            Device index for Reachy's microphone

        Raises:
            RuntimeError: If microphone not found
        """
        for i in range(self.audio.get_device_count()):
            dev_info = self.audio.get_device_info_by_index(i)
            if 'reachy' in dev_info['name'].lower():
                logger.info(f"Found Reachy microphone: {dev_info['name']}")
                return i

        raise RuntimeError("Reachy microphone not found. Is the robot connected?")

    def _iter_input_devices(self):
        for i in range(self.audio.get_device_count()):
            dev_info = self.audio.get_device_info_by_index(i)
            if int(dev_info.get("maxInputChannels", 0)) > 0:
                yield i, dev_info

    def _pick_input_device_index(self) -> int:
        mic_index = os.getenv("MIC_INDEX")
        if mic_index:
            return int(mic_index)

        needle = (os.getenv("MIC_CONTAINS") or "shure").lower()
        matches = []
        for idx, info in self._iter_input_devices():
            name = str(info.get("name", "")).lower()
            if needle in name:
                matches.append(idx)

        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(f"Multiple input devices match MIC_CONTAINS={needle!r}: {matches}. Set MIC_INDEX.")

        try:
            return self._find_reachy_microphone()
        except Exception:
            pass

        default = self.audio.get_default_input_device_info()
        return int(default["index"])

    def _setup_audio_stream(self) -> Any:
        """
        Set up PyAudio stream for microphone input.

        Returns:
            PyAudio stream object
        """
        device_index = self._pick_input_device_index()
        dev = self.audio.get_device_info_by_index(device_index)
        logger.info("Using microphone: index=%s name=%r", device_index, dev.get("name"))

        input_rate = int(float(dev.get("defaultSampleRate") or 44100))
        target_rate = int(self.config["audio"]["sample_rate"])
        max_in_ch = int(dev.get("maxInputChannels") or 1)
        channels = 2 if max_in_ch >= 2 else 1

        frames_per_buffer = max(int(self.config["audio"]["chunk_size"]), 16384)
        raw = self.audio.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=input_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=frames_per_buffer,
        )

        self._raw_mic_stream = raw
        return ResamplingPyAudioStream(
            raw,
            ResamplingParams(
                input_rate_hz=input_rate,
                output_rate_hz=target_rate,
                channels=channels,
            ),
        )

    def _setup_wake_word(self) -> Model:
        """
        Set up OpenWakeWord model.

        Supports both explicit .onnx file paths and pretrained model names
        (e.g. "hey_jarvis_v0.1", "alexa_v0.1").

        Returns:
            OpenWakeWord model instance
        """
        ww_cfg = self.config.get("wake_word", {})

        # Legacy model_path takes precedence if set
        model_path = ww_cfg.get("model_path", "")
        if model_path and os.path.exists(model_path):
            logger.info("Loading wake word model from path: %s", model_path)
            return Model(wakeword_model_paths=[model_path])

        # Otherwise resolve pretrained model by name
        model_name = ww_cfg.get("model", "hey_jarvis_v0.1")
        if not model_name:
            raise RuntimeError("No wake word model configured (set wake_word.model or wake_word.model_path)")

        # Check if it's a file path
        if os.path.exists(model_name):
            logger.info("Loading wake word model from path: %s", model_name)
            return Model(wakeword_model_paths=[model_name])

        # Try to find in pretrained models
        pretrained = openwakeword.get_pretrained_model_paths()
        for p in pretrained:
            basename = os.path.basename(p).replace(".onnx", "")
            if basename == model_name or basename.startswith(model_name):
                logger.info("Loading pretrained wake word model: %s (%s)", model_name, p)
                return Model(wakeword_model_paths=[p])

        raise RuntimeError(
            f"Wake word model not found: {model_name!r}. "
            f"Available pretrained: {[os.path.basename(p).replace('.onnx','') for p in pretrained]}"
        )

    async def _continuous_vad_listener(self) -> None:
        """
        Continuously listen for speech during AWAKE state.

        Yields speech audio bytes via queue when pause detected.
        Runs throughout INTERACTIVE and PROCESSING phases.
        """
        _listener_iter = 0
        _last_heartbeat = time.monotonic()
        while self.state in [ReachyState.INTERACTIVE, ReachyState.PROCESSING]:
            try:
                _listener_iter += 1
                now = time.monotonic()
                if now - _last_heartbeat > 15.0:
                    logger.info(
                        "VAD listener heartbeat: iter=%d state=%s tts_speaking=%s",
                        _listener_iter, self.state.name, self._tts_is_speaking(),
                    )
                    _last_heartbeat = now

                enable_barge_in = os.getenv("ENABLE_BARGE_IN") == "1"

                # Default posture (MVP stability): half-duplex.
                # While PROCESSING, do not listen/capture at all unless explicitly enabled.
                # This avoids self-interrupts from speaker leakage between sentence chunks.
                if self.state == ReachyState.PROCESSING and not enable_barge_in:
                    await asyncio.sleep(0.05)
                    continue

                # If we cannot stop the current WAV anyway, default posture is:
                # - don't try to barge-in while speaking (avoids self-interrupt + echo),
                # - don't even run the VAD capture loop while speaking (saves CPU/log spam).
                if (
                    self.state == ReachyState.PROCESSING
                    and self._tts_is_speaking()
                    and os.getenv("BARGE_IN_WHILE_SPEAKING") != "1"
                ):
                    await asyncio.sleep(0.05)
                    continue

                # If we are in INTERACTIVE but still speaking (tail of a WAV we can't stop),
                # don't capture "user speech" from our own speaker leakage.
                if self.state == ReachyState.INTERACTIVE and self._tts_is_speaking():
                    await asyncio.sleep(0.05)
                    continue

                def _on_activation(rms: float, peak: float):
                    """
                    Speech onset callback — fires when VAD first detects voice.

                    In INTERACTIVE: orient head toward speaker (at speech start, not end).
                    In PROCESSING: barge-in interrupt logic.
                    """
                    if self.state == ReachyState.INTERACTIVE:
                        # Orient to speaker at speech onset, not after capture completes
                        if self.movement_manager is not None:
                            self.movement_manager.orient_to_speaker()
                        return None
                    if self.state != ReachyState.PROCESSING:
                        return None
                    if not enable_barge_in:
                        # Barge-in is disabled; never interrupt processing from background VAD.
                        return False
                    if self.interruption_flag.is_set():
                        return None

                    if self._tts_is_speaking():
                        if os.getenv("BARGE_IN_WHILE_SPEAKING") != "1":
                            # Avoid log spam; this fires ~30 times/sec on speaker leakage.
                            now = time.monotonic()
                            if (os.getenv("BARGE_IN_DEBUG") == "1") or (now - self._last_bargein_ignore_log_monotonic > 1.0):
                                logger.info(
                                    "Ignoring activation during speaking (barge-in disabled while speaking): rms=%.4f peak=%.4f",
                                    rms,
                                    peak,
                                )
                                self._last_bargein_ignore_log_monotonic = now
                            return False
                        rms_thr = float(os.getenv("BARGE_IN_SPEAKING_RMS") or "0.15")
                        if rms < rms_thr:
                            now = time.monotonic()
                            if (os.getenv("BARGE_IN_DEBUG") == "1") or (now - self._last_bargein_ignore_log_monotonic > 1.0):
                                logger.info(
                                    "Ignoring activation during speaking (below rms threshold): rms=%.4f peak=%.4f thr=%.4f",
                                    rms,
                                    peak,
                                    rms_thr,
                                )
                                self._last_bargein_ignore_log_monotonic = now
                            return False

                    logger.info("Speech start detected during processing - interrupting (early)")
                    self.interruption_flag.set()
                    return True

                # Capture speech (blocks until pause detected or timeout)
                logger.debug("VAD listener: calling capture_speech (iter=%d state=%s)", _listener_iter, self.state.name)
                assert self.vad_capture is not None
                audio_bytes = await self.vad_capture.capture_speech(on_activation=_on_activation)
                logger.debug("VAD listener: capture_speech returned (iter=%d bytes=%s)", _listener_iter, len(audio_bytes) if audio_bytes else 0)

                if audio_bytes:
                    # Speech detected - put in queue for processing
                    logger.info("VAD listener: captured %d bytes, queuing", len(audio_bytes))
                    await self.speech_queue.put(audio_bytes)
                else:
                    reason = getattr(self.vad_capture, "last_result_reason", None)
                    logger.info(
                        "VAD listener: capture_speech returned None (state=%s reason=%s)",
                        self.state.name,
                        reason,
                    )

            except asyncio.CancelledError:
                logger.info("VAD listener cancelled")
                raise
            except Exception as e:
                logger.error(f"VAD listener error: {e}", exc_info=True)
                await asyncio.sleep(0.1)

        logger.warning("VAD listener exited while loop (state=%s)", self.state.name)

    def _build_pipeline(self) -> StreamPipeline:
        """
        Build the streaming middleware pipeline.

        Pipeline order (outermost to innermost):
        TTS → Filter → Sentence → Tool → Provider

        Returns:
            Configured StreamPipeline instance
        """
        import logging
        logger = logging.getLogger(__name__)

        # Build middleware list (INNERMOST first!)
        # Pipeline chains like: middlewares[-1].process(middlewares[-2].process(...middlewares[0].process(stream)))
        # So: first in list = innermost (processes input first), last in list = outermost (yields to caller)
        middlewares = []

        if self._provider == "openclaw":
            # OpenClaw provider (INNERMOST - SOURCE) — Gateway handles tools server-side
            oc_cfg = self.config.get("openclaw", {})
            provider = OpenClawProviderMiddleware(
                gateway_client=self.gateway_client,
                session_key=oc_cfg.get("session_key", "agent:main:reachy"),
                timeout_ms=oc_cfg.get("timeout_ms", 60000),
            )
            middlewares.append(provider)
        else:
            logger.error("only openclaw provider is currently supported")

        # Sentence buffer middleware - accumulates text into sentences
        middlewares.append(SentenceBufferMiddleware())

        # Filter middleware - removes empty text/sentence chunks only
        # (Don't filter tool_call, tool_notification, finish, etc. which have empty content)
        middlewares.append(
            FilterMiddleware(
                lambda c: (
                    c.type not in ["text", "sentence"]  # Pass through non-text chunks
                    or bool(c.content.strip())  # For text/sentence, check if non-empty
                )
            )
        )

        # TTS middleware (OUTERMOST) - processes sentence chunks, sends to TTS service
        # Priority: TTS_DISABLE > TTS_TARGET=testbench > TTS_TARGET=service_url > direct (default)
        tts_config = self.config.get("tts", {})
        tts_target = (os.getenv("TTS_TARGET") or "").strip().lower()
        if os.getenv("TTS_DISABLE") == "1":
            logger.warning("TTS disabled via TTS_DISABLE=1")
            self._tts_controller = None
        elif tts_target == "testbench":
            logger.info("TTS enabled: target=testbench")
            middlewares.append(
                TestbenchTTSMiddleware(
                    testbench_base_url=os.getenv("TESTBENCH_URL") or "http://127.0.0.1:8042",
                    recordings_dir=os.getenv("TESTBENCH_RECORDINGS_DIR") or "/tmp/reachy_mini_testbench/recordings",
                    tts_engine=os.getenv("TTS_ENGINE") or "openai",
                    openai_model=os.getenv("OPENAI_TTS_MODEL") or "tts-1",
                    openai_voice=os.getenv("OPENAI_TTS_VOICE") or "alloy",
                    openai_speed=float(os.getenv("OPENAI_TTS_SPEED") or "1.0"),
                    request_timeout_s=float(tts_config.get("timeout", 30.0)),
                    on_tts_text=self._note_tts_text,
                )
            )
            self._tts_controller = TestbenchTTSMiddleware
        elif tts_target == "service_url" and tts_config.get("service_url"):
            logger.info(f"TTS enabled: {tts_config['service_url']}")
            media_manager = self.mini.media_manager if self.mini else None
            middlewares.append(
                TTSStreamMiddleware(
                    tts_url=tts_config["service_url"],
                    timeout=tts_config.get("timeout", 30.0),
                    media_manager=media_manager,
                )
            )
            self._tts_controller = TTSStreamMiddleware
        elif self.mini is not None and getattr(self.mini, "media", None) is not None:
            # Direct playback via SoundDevice (default — no testbench needed)
            tts_engine = os.getenv("TTS_ENGINE") or tts_config.get("engine", "openai")
            logger.info("TTS enabled: target=direct engine=%s", tts_engine)
            middlewares.append(
                DirectTTSMiddleware(
                    media_manager=self.mini.media,
                    tts_engine=tts_engine,
                    openai_model=os.getenv("OPENAI_TTS_MODEL") or tts_config.get("openai_model", "tts-1"),
                    openai_voice=os.getenv("OPENAI_TTS_VOICE") or tts_config.get("openai_voice", "alloy"),
                    openai_speed=float(os.getenv("OPENAI_TTS_SPEED") or tts_config.get("openai_speed", 1.0)),
                    kokoro_voice=tts_config.get("kokoro_voice", "af_heart"),
                    kokoro_speed=float(tts_config.get("kokoro_speed", 1.0)),
                    kokoro_lang=tts_config.get("kokoro_lang", "a"),
                    orpheus_model=tts_config.get("orpheus_model", "canopylabs/orpheus-tts-0.1-finetune-prod"),
                    orpheus_voice=tts_config.get("orpheus_voice", "tara"),
                    riva_url=tts_config.get("riva_url", "http://127.0.0.1:9000/v1/audio/synthesize"),
                    riva_model=tts_config.get("riva_model", "magpie-tts-multilingual"),
                    riva_voice=tts_config.get("riva_voice", "Magpie-Multilingual.EN-US.Jason"),
                    riva_language_code=tts_config.get("riva_language_code", "en-US"),
                    riva_sample_rate_hz=int(tts_config.get("riva_sample_rate_hz", 24000)),
                    request_timeout_s=float(tts_config.get("timeout", 30.0)),
                    on_tts_text=self._note_tts_text,
                    head_wobbler=self.head_wobbler,
                    echo_canceller=self._echo_canceller,
                )
            )
            self._tts_controller = DirectTTSMiddleware
        else:
            logger.warning("TTS disabled (no robot media available)")
            self._tts_controller = None

        logger.info(f"Pipeline built with {len(middlewares)} middlewares")
        return StreamPipeline(middlewares)

    async def _speak_announcement(self) -> None:
        """
        Generate and speak a sarcastic GLaDOS startup announcement.
        Uses the LLM to create a dynamic response each time.
        """
        import logging
        logger = logging.getLogger(__name__)

        try:
            # Prompt for GLaDOS to generate a sarcastic awakening comment
            prompt = "Great with an oh, hi. then Generate a short one sentence, sarcastic observation about being rudely awoken by the user turning me on. Keep it under 20 words."

            logger.debug(f"Prompt: {prompt}")

            # Build a fresh pipeline for the announcement
            announcement_pipeline = self._build_pipeline()

            # Get the provider (first middleware) and set it to text mode
            provider = announcement_pipeline.middlewares[0]
            provider.set_user_message(prompt)

            # Process through pipeline
            logger.info("Speaking announcement...")

            async def empty_stream():
                """Empty stream for provider."""
                if False:
                    yield

            # Collect the response (will be spoken via TTS)
            announcement_text = []
            async for chunk in announcement_pipeline.process(empty_stream()):
                if chunk.type == "text" or chunk.type == "sentence":
                    announcement_text.append(chunk.content)
                elif chunk.type == "error":
                    logger.warning(f"Pipeline error during announcement: {chunk.content}")
                # Don't break - let stream finish naturally to avoid cancel scope errors

            full_announcement = "".join(announcement_text)
            if full_announcement:
                logger.info(f"GLaDOS: {full_announcement}")

            # Wait for audio queue to finish playing
            # Wait for singleton audio queue to be empty
            if TTSStreamMiddleware._audio_queue:
                logger.debug("Waiting for audio playback to complete...")
                await TTSStreamMiddleware._audio_queue.join()
                logger.debug("Audio playback complete")

        except Exception as e:
            logger.error(f"Announcement generation error: {e}", exc_info=True)

    async def startup(self) -> None:
        """
        Initialize all components and prepare for operation.

        Transitions to INTERACTIVE state after completion (for debugging).
        """
        logger.info("Reachy Voice Assistant Starting...")

        # Initialize robot
        if self.use_mock_robot:
            logger.warning("Running in MOCK mode (reachy-mini not installed)")
            self.mini = None
        else:
            logger.info("Initializing robot...")
            # Choose media backend: "default" if camera enabled, "default_no_video" otherwise
            camera_config = self.config.get("camera", {})
            camera_enabled = camera_config.get("enabled", False)
            if camera_enabled:
                media_backend = os.getenv("REACHY_MEDIA_BACKEND") or "default"
            else:
                media_backend = os.getenv("REACHY_MEDIA_BACKEND") or "default_no_video"
            try:
                self.mini = ReachyMini(media_backend=media_backend)
                logger.info("Robot initialized (media_backend=%s)", media_backend)
            except Exception as e:
                logger.warning(f"Robot init failed (falling back to MOCK mode): {e}")
                self.mini = None
                self.use_mock_robot = True

        # Initialize animator (enabled is determined automatically based on mini being None or not)
        self.animator = RobotAnimator(self.mini)

        # Initialize CameraWorker + face tracking (if camera enabled)
        if self.mini is not None:
            camera_config = self.config.get("camera", {})
            # head_tracking mode: "continuous", "orient_on_speech", or "none"
            # Backward compat: old face_tracking bool maps to continuous/none
            self._head_tracking_mode = camera_config.get("head_tracking", None)
            if self._head_tracking_mode is None:
                # Legacy: face_tracking: true → continuous, false → none
                self._head_tracking_mode = "continuous" if camera_config.get("face_tracking", False) else "none"
            if self._head_tracking_mode not in ("continuous", "orient_on_speech", "none"):
                logger.warning("Unknown head_tracking mode %r, defaulting to continuous", self._head_tracking_mode)
                self._head_tracking_mode = "continuous"

            use_face_tracking = self._head_tracking_mode != "none"

            if camera_config.get("enabled", False) and use_face_tracking:
                head_tracker = SCRFDHeadTracker(
                    confidence_threshold=camera_config.get("confidence_threshold", 0.3),
                    gpu_device=camera_config.get("gpu_device", 1),
                    det_size=tuple(camera_config.get("det_size", [640, 640])),
                )
                self.camera_worker = CameraWorker(
                    reachy_mini=self.mini,
                    head_tracker=head_tracker,
                )
                logger.info("CameraWorker created (SCRFD face tracking on GPU %d, mode=%s)", camera_config.get("gpu_device", 1), self._head_tracking_mode)
            elif camera_config.get("enabled", False):
                # Camera enabled but no face tracking — still capture frames for scene awareness
                self.camera_worker = CameraWorker(
                    reachy_mini=self.mini,
                    head_tracker=None,
                )
                logger.info("CameraWorker created (no face tracking)")

            # Register camera_worker for scene awareness tool
            if self.camera_worker is not None:
                set_camera_worker(self.camera_worker)

        # Initialize MovementManager (100Hz control loop with breathing, listening freeze)
        if self.mini is not None:
            self.movement_manager = MovementManager(
                current_robot=self.mini,
                camera_worker=self.camera_worker,
                head_tracking_mode=self._head_tracking_mode,
            )
            logger.info("MovementManager created (head_tracking=%s)", self._head_tracking_mode)

            # Initialize HeadWobbler (speech-synchronized head movement)
            self.head_wobbler = HeadWobbler(
                set_speech_offsets=self.movement_manager.set_speech_offsets
            )
            logger.info("HeadWobbler created")

        # Initialize audio stream
        logger.info("Setting up audio stream...")
        self.mic_stream = self._setup_audio_stream()

        # Initialize AEC (if barge-in with AEC enabled)
        bi_cfg = self.config.get("barge_in", {})
        if bi_cfg.get("aec_enabled", False):
            aec_impl = str(bi_cfg.get("aec_impl", "nlms")).strip().lower()
            sample_rate = int(self.config.get("audio", {}).get("sample_rate", 16000))
            frame_size = sample_rate // 100 if sample_rate % 100 == 0 else 160

            if aec_impl in ("webrtc_aecm", "webrtc", "aecm"):
                if sample_rate not in (8000, 16000):
                    logger.warning(
                        "WebRTC AECM supports only 8kHz/16kHz (got %s). Falling back to NLMS.",
                        sample_rate,
                    )
                    aec_impl = "nlms"
                else:
                    try:
                        from sparky_mvp.core.webrtc_aecm import WebRtcAecmEchoCanceller

                        ms_buf = int(bi_cfg.get("aec_ms_in_soundcard_buf", 60))
                        self._echo_canceller = WebRtcAecmEchoCanceller(
                            sample_rate=sample_rate,
                            ms_in_soundcard_buf=ms_buf,
                        )
                        logger.info(
                            "AEC initialized (WebRTC AECM sample_rate=%d ms_in_soundcard_buf=%d)",
                            sample_rate,
                            ms_buf,
                        )
                    except Exception as e:
                        logger.warning("Failed to init WebRTC AECM; falling back to NLMS: %s", e)
                        aec_impl = "nlms"

            if self._echo_canceller is None and aec_impl in ("webrtc_apm", "apm"):
                if sample_rate not in (8000, 16000, 32000, 48000):
                    logger.warning(
                        "WebRTC APM supports 8/16/32/48 kHz (got %s). Falling back to NLMS.",
                        sample_rate,
                    )
                    aec_impl = "nlms"
                else:
                    try:
                        from sparky_mvp.core.webrtc_apm import WebRtcApmEchoCanceller

                        self._echo_canceller = WebRtcApmEchoCanceller(
                            sample_rate=sample_rate,
                        )
                        logger.info(
                            "AEC initialized (WebRTC APM/AEC3 sample_rate=%d)",
                            sample_rate,
                        )
                    except Exception as e:
                        logger.warning("Failed to init WebRTC APM; falling back to NLMS: %s", e)
                        aec_impl = "nlms"

            if self._echo_canceller is None and aec_impl not in ("nlms", "", "webrtc_aecm", "webrtc", "aecm", "webrtc_apm", "apm"):
                logger.warning("Unknown barge_in.aec_impl=%r; falling back to NLMS.", aec_impl)
                aec_impl = "nlms"

            if self._echo_canceller is None:
                filter_length = int(bi_cfg.get("aec_filter_length", 3200))
                mu = float(bi_cfg.get("aec_mu", 0.3))
                self._echo_canceller = AcousticEchoCanceller(
                    frame_size=frame_size,
                    filter_length=filter_length,
                    sample_rate=sample_rate,
                    mu=mu,
                )
                logger.info(
                    "AEC initialized (NLMS sample_rate=%d frame_size=%d filter_length=%d mu=%.2f)",
                    sample_rate,
                    frame_size,
                    filter_length,
                    mu,
                )

            # Wrap mic stream with AEC processing
            self.mic_stream = AECStream(self.mic_stream, self._echo_canceller)

        # Initialize wake word (if enabled in config)
        ww_cfg = self.config.get("wake_word", {})
        ww_enabled = ww_cfg.get("enabled", False) and os.getenv("DISABLE_SLEEP") != "1"
        if ww_enabled:
            logger.info("Loading wake word model...")
            try:
                self.wake_word_model = self._setup_wake_word()
            except Exception as e:
                logger.warning(f"Wake word disabled: {e}")
                self.wake_word_model = None
        else:
            logger.info("Wake word disabled (wake_word.enabled=%s, DISABLE_SLEEP=%s)",
                        ww_cfg.get("enabled", False), os.getenv("DISABLE_SLEEP"))
            self.wake_word_model = None

        # Load sleep phrases for voice-triggered sleep
        self._sleep_phrases = [p.lower() for p in ww_cfg.get("sleep_phrases", [])]

        # Initialize VAD capture
        logger.info("Initializing VAD...")
        self.vad_capture = VADSpeechCapture(
            audio_stream=self.mic_stream,
            sample_rate=self.config["audio"]["sample_rate"],
            vad_model_path=self.config["vad"]["model_path"],
            vad_threshold=self.config["vad"]["confidence_threshold"]
        )

        # Initialize LLM provider
        if self._provider == "openclaw":
            logger.info("Connecting to OpenClaw Gateway...")
            self.gateway_client = create_gateway_client_from_config(self.config)
            await self.gateway_client.connect()
            logger.info("OpenClaw Gateway connected (protocol=%s)", self.gateway_client._protocol_version)
        else:
            logger.error("Only openclaw provider is supported")

        # Initialize STT engine (required for text-only mode)
        logger.info("Initializing local STT engine...")
        stt_config = self.config.get("stt", {})
        stt_engine_name = stt_config.get("engine", "faster_whisper")
        self.stt_engine = STTEngine(
            engine=stt_engine_name,
            model_size=stt_config.get("model_size", "base"),
            device=stt_config.get("device", "cpu"),
            language=stt_config.get("language", "en"),
            min_speech_confidence=stt_config.get("min_speech_confidence", 0.6),
            max_no_speech_prob=stt_config.get("max_no_speech_prob", 0.6),
            min_compression_ratio=stt_config.get("min_compression_ratio", 0.5),
            max_compression_ratio=stt_config.get("max_compression_ratio", 2.5),
            min_text_length=stt_config.get("min_text_length", 3),
            parakeet_model_path=stt_config.get("parakeet_model_path"),
            parakeet_gpu_device=stt_config.get("parakeet_gpu_device", 1),
        )
        logger.info(f"STT engine ready ({stt_engine_name})")

        # Initialize speaker identification (if enrollment file exists)
        speaker_cfg = self.config.get("speaker_id", {})
        enrollments_path = speaker_cfg.get(
            "enrollments_path", "models/speaker_enrollments.json"
        )
        if os.path.exists(enrollments_path):
            from sparky_mvp.core.speaker_id import SpeakerIdentifier
            self.speaker_identifier = SpeakerIdentifier(
                enrollments_path=enrollments_path,
                threshold=speaker_cfg.get("threshold", 0.7),
            )
            logger.info("Speaker identification ready (%d enrolled)", self.speaker_identifier.n_speakers)
        else:
            logger.info("Speaker ID disabled (no enrollments at %s)", enrollments_path)

        # Initialize audio input processor
        logger.info("Initializing audio input processor...")
        self.audio_processor = AudioInputProcessor(stt_engine=self.stt_engine)

        # Build middleware pipeline
        logger.info("Building middleware pipeline...")
        self.pipeline = self._build_pipeline()

        # Barge-in: auto-enable when AEC is active, or via env var
        bi_cfg = self.config.get("barge_in", {})
        bi_enabled = bi_cfg.get("enabled", False)
        if os.getenv("ENABLE_BARGE_IN") == "1":
            bi_enabled = True
        if bi_enabled:
            os.environ["ENABLE_BARGE_IN"] = "1"
            if self._echo_canceller is not None:
                os.environ["BARGE_IN_WHILE_SPEAKING"] = "1"
        enable_barge_in = os.getenv("ENABLE_BARGE_IN") == "1"
        barge_in_while_speaking = os.getenv("BARGE_IN_WHILE_SPEAKING") == "1"
        logger.info(
            "Barge-in: enabled=%s while_speaking=%s aec=%s",
            enable_barge_in,
            barge_in_while_speaking,
            self._echo_canceller is not None,
        )

        # Register OpenClaw node client (stays running through sleep/wake cycles)
        if self._provider == "openclaw" and (self.camera_worker is not None or self.movement_manager is not None):
            from sparky_mvp.core.openclaw_node_client import OpenClawNodeClient
            oc_cfg = self.config.get("openclaw", {})
            assert self.gateway_client is not None
            self.node_client = OpenClawNodeClient(
                gateway_url=oc_cfg.get("gateway_ws_url", "ws://127.0.0.1:18789"),
                token=self.gateway_client.token,
                camera_worker=self.camera_worker,
                movement_manager=self.movement_manager,
            )
            await self.node_client.connect()
            self._node_task = asyncio.create_task(self.node_client.run())
            logger.info("OpenClaw node client started (camera.snap)")

        # Transition: if wake word enabled, start in SLEEP; otherwise wake up and go interactive
        if self.wake_word_model is not None:
            logger.info("Wake word enabled — starting in SLEEP state (say wake word to interact)")
            await self.animator.go_to_sleep()
            self.state = ReachyState.SLEEP
        else:
            # Full wake: animation + subsystems + interactive
            logger.info("Wake word disabled — starting in INTERACTIVE state")
            await self._wake_up_and_go_interactive()

    async def sleep_phase(self) -> None:
        """
        Monitor for wake word while robot sleeps.

        Transitions to INTERACTIVE when wake word detected.
        If no wake word model available, transitions immediately.
        """
        if self.wake_word_model is None:
            # No wake word — go straight to interactive
            logger.info("No wake word model — skipping sleep, going to interactive")
            await self._wake_up_and_go_interactive()
            return

        logger.info("Sleeping... waiting for wake word")

        # Run wake word detection in thread executor (blocking)
        loop = asyncio.get_event_loop()
        wake_detected = await loop.run_in_executor(
            None, self._wait_for_wake_word
        )

        if wake_detected and self.running:
            logger.info("Wake word detected!")
            await self._wake_up_and_go_interactive()

    async def _wake_up_and_go_interactive(self) -> None:
        """Common wake-up sequence: animation + restart subsystems + go interactive."""
        assert self.animator is not None
        await self.animator.wake_up()

        # Restart CameraWorker after wake
        if self.camera_worker is not None:
            self.camera_worker.start()
            logger.info("CameraWorker restarted after wake")

        # Restart MovementManager after wake
        if self.movement_manager is not None:
            self.movement_manager.start()
            logger.info("MovementManager restarted after wake")

        # Restart HeadWobbler after wake
        if self.head_wobbler is not None:
            self.head_wobbler.start()
            logger.info("HeadWobbler restarted after wake")

        await asyncio.sleep(1)

        # Start continuous VAD listener
        logger.info("Starting continuous VAD listener...")
        self.vad_task = asyncio.create_task(self._continuous_vad_listener())

        # Restart spontaneous speech manager (if configured)
        ss_cfg = self.config.get("spontaneous_speech", {})
        if ss_cfg.get("enabled", False):
            presence_recency_s = float(ss_cfg.get("presence_recency_s", 30.0))
            self._spontaneous_manager = SpontaneousSpeechManager(
                send_message=self._send_spontaneous_message,
                check_presence=lambda: self._check_user_presence(presence_recency_s),
                config=ss_cfg,
            )
            self._spontaneous_task = asyncio.create_task(self._spontaneous_manager.run())
            logger.info("SpontaneousSpeechManager restarted after wake")

        # Restart spontaneous gesture manager (if configured and movement available)
        sg_cfg = self.config.get("spontaneous_gestures", {})
        if sg_cfg.get("enabled", False) and self.movement_manager is not None:
            presence_recency_s = float(sg_cfg.get("presence_recency_s", 30.0))
            self._gesture_manager = SpontaneousGestureManager(
                queue_move=self.movement_manager.queue_move,
                check_state=lambda: self.state == ReachyState.INTERACTIVE,
                check_presence=lambda: self._check_user_presence(presence_recency_s) if sg_cfg.get("require_presence", False) else True,
                config=sg_cfg,
            )
            self._gesture_task = asyncio.create_task(self._gesture_manager.run())
            logger.info("SpontaneousGestureManager restarted after wake")

        self.state = ReachyState.INTERACTIVE

    def _wait_for_wake_word(self) -> bool:
        """
        Blocking wake word detection (runs in thread executor).

        Returns:
            True if wake word detected, False if interrupted
        """
        threshold = self.config["wake_word"]["threshold"]

        # Audio format for wake word
        n_samples = 1280  # 80ms at 16kHz

        # Reset model prediction scores and enforce cooldown.
        # reset() clears the score buffer but NOT the raw audio preprocessing
        # pipeline, so stale audio (e.g. "go to sleep") can still produce a
        # false trigger.  We keep feeding audio during the cooldown to flush
        # the pipeline, but ignore detections until it expires.
        assert self.wake_word_model is not None
        self.wake_word_model.reset()
        cooldown_s = self.config["wake_word"].get("cooldown", 5.0)
        cooldown_until = time.monotonic() + cooldown_s

        try:
            while self.running:
                # Read audio chunk (bytes)
                assert self.mic_stream is not None
                audio_bytes = self.mic_stream.read(n_samples, exception_on_overflow=False)

                # Convert bytes to numpy array (int16 format from pyaudio.paInt16)
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16)

                # Run wake word prediction (always, to keep model state in sync)
                prediction = self.wake_word_model.predict(audio_data)

                # Ignore detections during post-sleep cooldown
                if time.monotonic() < cooldown_until:
                    continue

                # Check for wake word (first model in the list)
                model_name = list(prediction.keys())[0]
                score = prediction[model_name]

                if score >= threshold:
                    return True

        except Exception as e:
            logger.error(f"Wake word detection error: {e}")
            return False

        return False

    async def interactive_phase(self) -> None:
        """
        Wait for VAD to detect speech (via queue).

        VAD runs continuously in background, this phase just waits
        for speech to appear in the queue.

        Transitions to:
        - PROCESSING if speech captured
        - SLEEP if timeout or no speech (only when wake word is enabled)
        """
        logger.info("Listening for your request...")

        # Freeze antennas while listening
        if self.movement_manager is not None:
            self.movement_manager.set_listening(True)

        timeout = self.config["conversation"]["interactive_timeout"]

        try:
            # Wait for speech from continuous VAD listener
            audio_bytes = await asyncio.wait_for(
                self.speech_queue.get(),
                timeout=timeout
            )

            # Shutting down — skip state transitions, let cleanup handle posture
            if not self.running:
                if self.movement_manager is not None:
                    self.movement_manager.set_listening(False)
                return

            if audio_bytes:
                logger.info(f"Captured speech: {len(audio_bytes)} bytes")
                self.current_audio = audio_bytes
                # Orient already fired at speech onset via _on_activation
                # Unfreeze antennas, breathing resumes during processing
                if self.movement_manager is not None:
                    self.movement_manager.set_listening(False)
                self.state = ReachyState.PROCESSING
            else:
                logger.info("No speech detected")
                # If wake-word is disabled, sleeping would just create confusing "reset" motion.
                if self.wake_word_model is not None and os.getenv("DISABLE_SLEEP") != "1":
                    await self._return_to_sleep()
                else:
                    self.state = ReachyState.INTERACTIVE

        except asyncio.TimeoutError:
            logger.info("Timeout waiting for speech")
            # Shutting down — skip state transitions, let cleanup handle posture
            if not self.running:
                if self.movement_manager is not None:
                    self.movement_manager.set_listening(False)
                return
            # If wake-word is disabled, don't transition to SLEEP (there's nothing to wake us).
            if self.wake_word_model is not None and os.getenv("DISABLE_SLEEP") != "1":
                await self._return_to_sleep()
            else:
                self.state = ReachyState.INTERACTIVE

        except Exception as e:
            logger.error(f"Error in interactive phase: {e}")
            if self.wake_word_model is not None and os.getenv("DISABLE_SLEEP") != "1":
                await self._return_to_sleep()
            else:
                self.state = ReachyState.INTERACTIVE

    async def processing_phase(self) -> None:
        """
        Process user request through Anthropic Claude + middleware pipeline.

        VAD continues listening in background. If speech detected, interruption_flag
        will be set and this phase will cancel and restart.

        Handles:
        - Sending transcribed text to Claude
        - Streaming responses
        - Tool calls with animations
        - Interruption via VAD speech detection

        Transitions to:
        - INTERACTIVE if completed or interrupted
        - SLEEP if error occurs
        """
        logger.info("Processing request...")

        # Flush any lingering audio from previous response
        await self._cancel_tts_playback()

        # Reset HeadWobbler for new turn (clear residual offsets, drain stale audio)
        if self.head_wobbler is not None:
            self.head_wobbler.reset()

        # Start staged thinking animation (escalates intensity over time)
        if self.movement_manager is not None:
            self.movement_manager.clear_move_queue()
            self.movement_manager.queue_move(ThinkingMove())

        # Clear interrupt flag for this new processing session
        self.interruption_flag.clear()

        try:
            # Process request with interrupt monitoring
            processing_task = asyncio.create_task(self._process_request())

            # Wait for completion or interruption
            while not processing_task.done():
                if self.interruption_flag.is_set():
                    # Cancel processing
                    logger.info("Interruption detected - cancelling processing")
                    processing_task.cancel()
                    # Don't block the state machine on slow cancellation/cleanup.
                    # (Cancellation may be delayed by in-flight network calls.)
                    asyncio.create_task(self._await_task_done(processing_task, label="processing_task(cancelled)"))

                    # Abort server-side generation (best-effort, non-blocking)
                    if self.gateway_client and self.gateway_client.last_run_id:
                        await self.gateway_client.chat_abort(self.gateway_client.last_run_id)

                    # Clear flag and go back to INTERACTIVE to process new speech
                    await self._cancel_tts_playback()
                    if self.movement_manager is not None:
                        self.movement_manager.clear_move_queue()
                    self.interruption_flag.clear()
                    logger.warning("Interrupted by new speech!")
                    self.state = ReachyState.INTERACTIVE
                    return

                await asyncio.sleep(0.05)

            # Completed normally — stop thinking animation (breathing resumes via idle)
            if self.state == ReachyState.SLEEP:
                # _process_request detected a sleep phrase and already handled transition
                logger.info("Sleep phrase handled during processing — staying in SLEEP")
                return

            if self.movement_manager is not None:
                self.movement_manager.clear_move_queue()

            logger.info("Response complete")
            self.state = ReachyState.INTERACTIVE

        except Exception as e:
            logger.error(f"Error during processing: {e}")
            import traceback
            traceback.print_exc()
            await self._return_to_sleep()

    async def _await_task_done(self, task: asyncio.Task, *, label: str, timeout_s: float = 5.0) -> None:
        """
        Await a task completion in the background to prevent "Task exception was never retrieved".

        This is used for interruption paths where we don't want to block the main state machine.
        """
        t0 = time.time()
        try:
            await asyncio.wait_for(task, timeout=timeout_s)
        except asyncio.TimeoutError:
            logger.warning("%s still not finished after %.1fs", label, time.time() - t0)
        except asyncio.CancelledError:
            # If the awaiter is cancelled, just exit; the original task is already cancelled.
            pass
        except Exception:
            logger.exception("%s ended with error", label)

    async def _process_request(self) -> None:
        """
        Process transcribed text through middleware pipeline.

        The pipeline now includes the provider middleware which calls Anthropic Claude,
        so we don't call the LLM externally anymore.
        """
        import logging
        logger = logging.getLogger(__name__)

        # Process audio input (transcription, noise detection, fallback)
        assert self.audio_processor is not None
        result: AudioInputResult = await self.audio_processor.process_audio_input(
            self.current_audio
        )

        # Check if we should skip this input (noise detected or error)
        if result.should_skip:
            logger.info(f"Skipping input: {result.skip_reason}")
            # Return early without adding to history or processing
            # This will complete _process_request() and return to INTERACTIVE state
            return

        if result.transcribed_text and self._looks_like_echo_of_tts(result.transcribed_text):
            logger.warning("Skipping input: probable echo of our own TTS: %r", result.transcribed_text[:120])
            return

        # Check for sleep phrase (voice-triggered sleep) — only when wake word is active
        if self.wake_word_model is not None and result.transcribed_text and self._is_sleep_phrase(result.transcribed_text):
            logger.info("Sleep phrase detected: %r — returning to sleep", result.transcribed_text)
            await self._return_to_sleep()
            return

        # Identify speaker from voice (runs in parallel with STT, ~4ms on GPU)
        speaker_name = None
        if self.speaker_identifier and self.current_audio:
            speaker_name, speaker_conf = self.speaker_identifier.identify(self.current_audio)
            if speaker_name != "unknown":
                logger.info("Speaker: %s (confidence=%.3f)", speaker_name, speaker_conf)
            else:
                speaker_name = None

        # Log transcribed text
        logger.info(f"Using transcribed text: '{result.transcribed_text}'")

        # Add user message to conversation history
        self.messages.append(result.to_message())

        # Trim history if needed
        if len(self.messages) > self.max_history:
            # Keep system message + recent messages
            self.messages = [self.messages[0]] + self.messages[-(self.max_history-1):]

        # Rebuild pipeline for this request with updated conversation history
        logger.info(
            f"Processing request: text mode, "
            f"history={len(self.messages)} messages"
        )

        # Create fresh pipeline with current state
        request_pipeline = self._build_pipeline()

        # Update the provider (FIRST middleware now) with transcribed text
        # Note: We only support text mode now (no audio to LLM)
        provider = request_pipeline.middlewares[0]  # Provider is now at index 0 (innermost)
        provider.set_user_message(result.transcribed_text)
        if hasattr(provider, "set_speaker_name"):
            provider.set_speaker_name(speaker_name)

        # Process through middleware pipeline with empty input stream
        # (Provider middleware ignores input stream - it's a source)
        accumulated_response = []

        async def empty_stream():
            """Empty stream for provider (it ignores input)."""
            if False:
                yield

        logger.debug("Starting pipeline iteration...")

        try:
            async for chunk in request_pipeline.process(empty_stream()):
                logger.debug(f"Received chunk type={chunk.type}, content='{chunk.content[:50] if chunk.content else ''}'")

                # Handle different chunk types
                if chunk.type == "tool_notification":
                    # Trigger animation for tool execution
                    logger.info(f"Tool notification: {chunk.tool_name}")
                    assert self.animator is not None
                    await self.animator.tool_animation(chunk.tool_name)

                elif chunk.type == "text" and chunk.content:
                    # Accumulate and print text chunks (if any make it through)
                    accumulated_response.append(chunk.content)
                    print(chunk.content, end="", flush=True)

                elif chunk.type == "sentence" and chunk.content:
                    # Accumulate and print sentence chunks
                    accumulated_response.append(chunk.content)
                    print(chunk.content, end="", flush=True)
                    logger.debug(f"Sentence complete: {chunk.content[:50]}...")

                elif chunk.type == "tts":
                    # TTS audio chunk (could be played back if needed)
                    logger.debug(f"TTS audio chunk: {len(chunk.audio_data)} bytes")

                elif chunk.type == "error":
                    # Error occurred in pipeline
                    logger.error(f"Pipeline error: {chunk.content}")

                elif chunk.type == "finish":
                    logger.info(f"Stream finished: reason={chunk.finish_reason}")
                    print()  # New line after response

        except asyncio.CancelledError:
            # Task was cancelled (due to interruption)
            logger.warning("[INTERRUPT] Request processing cancelled!")
            logger.info("[INTERRUPT] Cleaning up TTS playback")
            logger.info("[INTERRUPT] Calling cancel_tts_playback()")
            await self._cancel_tts_playback()
            logger.info("[INTERRUPT] Playback cancellation complete - safe to proceed")
            raise  # Re-raise to propagate cancellation

        finally:
            # Add assistant response to history (even if interrupted)
            if accumulated_response:
                response_text = "".join(accumulated_response)
                self.messages.append({
                    "role": "assistant",
                    "content": response_text
                })
                logger.debug(f"Response added to history: {len(response_text)} chars")

    # FUTURE: Re-enable interrupt monitoring with dedicated audio stream
    # to avoid corrupting VAD stream. See plan: sequential-frolicking-russell.md
    # Methods _monitor_interruption() and _check_wake_word_once() removed.

    async def _return_to_sleep(self) -> None:
        """Helper to transition back to sleep state."""
        # Stop spontaneous speech during sleep
        if self._spontaneous_manager is not None:
            self._spontaneous_manager.stop()
        if self._spontaneous_task is not None:
            self._spontaneous_task.cancel()
            try:
                await self._spontaneous_task
            except asyncio.CancelledError:
                pass
            self._spontaneous_task = None

        # Stop spontaneous gestures during sleep
        if self._gesture_manager is not None:
            self._gesture_manager.stop()
        if self._gesture_task is not None:
            self._gesture_task.cancel()
            try:
                await self._gesture_task
            except asyncio.CancelledError:
                pass
            self._gesture_task = None

        # Cancel continuous VAD listener
        if self.vad_task and not self.vad_task.done():
            logger.info("Cancelling continuous VAD task")
            self.vad_task.cancel()
            try:
                await self.vad_task
            except asyncio.CancelledError:
                pass
            self.vad_task = None

        # Stop HeadWobbler before MovementManager (joins thread)
        if self.head_wobbler is not None:
            self.head_wobbler.stop()

        # Stop CameraWorker before MovementManager
        if self.camera_worker is not None:
            self.camera_worker.stop()

        # Stop MovementManager before sleep (blocking ~2s, joins thread + goto neutral)
        if self.movement_manager is not None:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.movement_manager.stop)

        assert self.animator is not None
        await self.animator.go_to_sleep()
        self.state = ReachyState.SLEEP

    async def run(self) -> None:
        """
        Main event loop.

        Runs the state machine indefinitely until shutdown.
        """
        # Install SIGINT handler that triggers graceful shutdown
        # (asyncio.run's default handler cancels all tasks immediately,
        # which skips our cleanup and leaves the robot in whatever posture)
        loop = asyncio.get_event_loop()
        loop.add_signal_handler(signal.SIGINT, self._handle_sigint)

        try:
            # Initial startup
            await self.startup()

            # Main loop
            while self.running:
                if self.state == ReachyState.SLEEP:
                    await self.sleep_phase()

                elif self.state == ReachyState.INTERACTIVE:
                    await self.interactive_phase()

                elif self.state == ReachyState.PROCESSING:
                    await self.processing_phase()

                # Small yield to event loop
                await asyncio.sleep(0.01)

        except KeyboardInterrupt:
            logger.info("Shutting down (KeyboardInterrupt)...")
            self.running = False

        except asyncio.CancelledError:
            logger.info("Shutting down (cancelled)...")
            self.running = False

        finally:
            await self.cleanup()

    def _handle_sigint(self) -> None:
        """Handle SIGINT gracefully — stop the run loop so cleanup runs."""
        if not self.running:
            # Second SIGINT — force exit (but still try sleep posture)
            logger.warning("Second SIGINT — forcing exit")
            raise KeyboardInterrupt
        logger.info("SIGINT received — shutting down gracefully...")
        self.running = False
        # Unblock interactive_phase if waiting on speech_queue.get()
        try:
            self.speech_queue.put_nowait(None)
        except Exception:
            pass

    async def cleanup(self) -> None:
        """
        Clean up resources before shutdown.
        """
        logger.info("Cleaning up...")

        # Cancel VAD task if running
        if self.vad_task and not self.vad_task.done():
            logger.info("Cancelling VAD task")
            self.vad_task.cancel()
            try:
                await self.vad_task
            except asyncio.CancelledError:
                pass

        # Stop spontaneous speech manager
        if self._spontaneous_manager is not None:
            self._spontaneous_manager.stop()
        if self._spontaneous_task is not None:
            self._spontaneous_task.cancel()
            try:
                await self._spontaneous_task
            except asyncio.CancelledError:
                pass

        # Stop spontaneous gesture manager
        if self._gesture_manager is not None:
            self._gesture_manager.stop()
        if self._gesture_task is not None:
            self._gesture_task.cancel()
            try:
                await self._gesture_task
            except asyncio.CancelledError:
                pass

        # Stop OpenClaw node client
        if self.node_client:
            await self.node_client.stop()
        if self._node_task:
            self._node_task.cancel()
            try:
                await self._node_task
            except asyncio.CancelledError:
                pass

        # Close Gateway connection
        if self.gateway_client:
            await self.gateway_client.close()

        # Close audio stream
        raw = getattr(self, "_raw_mic_stream", None)
        if raw:
            try:
                raw.stop_stream()
            except Exception:
                pass
            try:
                raw.close()
            except Exception:
                pass
            self._raw_mic_stream = None

        # Terminate audio
        if self.audio:
            self.audio.terminate()

        # Stop HeadWobbler before MovementManager
        if self.head_wobbler is not None:
            self.head_wobbler.stop()

        # Stop CameraWorker before MovementManager
        if self.camera_worker is not None:
            self.camera_worker.stop()

        # Stop MovementManager before sleep
        if self.movement_manager is not None:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.movement_manager.stop)

        # Put robot to sleep (skip if already sleeping from _return_to_sleep)
        if not self.use_mock_robot and self.animator and self.state != ReachyState.SLEEP:
            await self.animator.go_to_sleep()

        logger.info("Cleanup complete")

    async def _cancel_tts_playback(self) -> None:
        controller = self._tts_controller
        if controller is None:
            return
        cancel = getattr(controller, "cancel_playback", None)
        if cancel is None:
            return
        await cancel()

    def _tts_is_speaking(self) -> bool:
        controller = self._tts_controller
        if controller is None:
            return False
        is_speaking = getattr(controller, "is_speaking", None)
        if is_speaking is None:
            return False
        try:
            return bool(is_speaking())
        except Exception:
            return False

    def _note_tts_text(self, text: str) -> None:
        t = _normalize_text_for_echo(text)
        if not t:
            return
        now = time.monotonic()
        self._recent_tts_text.append((now, t))
        # Prune by time window
        window_s = float(os.getenv("ECHO_GUARD_WINDOW_S") or "12.0")
        cutoff = now - window_s
        while self._recent_tts_text and self._recent_tts_text[0][0] < cutoff:
            self._recent_tts_text.popleft()

    def _is_sleep_phrase(self, transcribed_text: str) -> bool:
        """Check if transcribed text contains a sleep phrase."""
        if not self._sleep_phrases:
            return False
        text_lower = transcribed_text.lower().strip()
        for phrase in self._sleep_phrases:
            if phrase in text_lower:
                return True
        return False

    def _check_user_presence(self, recency_s: float = 30.0) -> bool:
        """Check if a user face has been detected recently."""
        if self.camera_worker is None:
            return True  # No camera — assume present
        t = self.camera_worker.last_face_detected_time
        if t is None:
            return False
        return (time.time() - t) < recency_s

    async def _send_spontaneous_message(self, prompt: str) -> None:
        """Send a spontaneous message through the chat pipeline (used by SpontaneousSpeechManager)."""
        # Only speak when in INTERACTIVE state (don't interrupt processing or sleep)
        if self.state != ReachyState.INTERACTIVE:
            logger.debug("Spontaneous speech skipped — not in INTERACTIVE state (state=%s)", self.state)
            return

        logger.info("Sending spontaneous message: %s", prompt[:80])

        # Temporarily transition to PROCESSING
        self.state = ReachyState.PROCESSING

        # Start thinking animation
        if self.movement_manager is not None:
            self.movement_manager.clear_move_queue()
            self.movement_manager.queue_move(ThinkingMove())

        # Reset HeadWobbler
        if self.head_wobbler is not None:
            self.head_wobbler.reset()

        try:
            # Build pipeline and send message
            pipeline = self._build_pipeline()
            provider = pipeline.middlewares[0]
            provider.set_user_message(prompt)

            accumulated = []

            async def empty_stream():
                if False:
                    yield

            async for chunk in pipeline.process(empty_stream()):
                if chunk.type in ("text", "sentence") and chunk.content:
                    accumulated.append(chunk.content)
                elif chunk.type == "error":
                    logger.warning("Pipeline error during spontaneous speech: %s", chunk.content)

            if accumulated:
                response_text = "".join(accumulated)
                self.messages.append({"role": "user", "content": prompt})
                self.messages.append({"role": "assistant", "content": response_text})
                logger.info("Spontaneous speech complete: %s", response_text[:120])

        except Exception:
            logger.exception("Spontaneous speech pipeline failed")
        finally:
            # Stop thinking animation and return to INTERACTIVE
            if self.movement_manager is not None:
                self.movement_manager.clear_move_queue()
            self.state = ReachyState.INTERACTIVE

    def _looks_like_echo_of_tts(self, transcribed_text: str) -> bool:
        window_s = float(os.getenv("ECHO_GUARD_WINDOW_S") or "12.0")
        min_jaccard = float(os.getenv("ECHO_GUARD_JACCARD") or "0.78")
        min_overlap = int(os.getenv("ECHO_GUARD_MIN_OVERLAP") or "6")
        min_chars = int(os.getenv("ECHO_GUARD_MIN_CHARS") or "8")

        user_norm = _normalize_text_for_echo(transcribed_text)
        user_words = _words_for_echo(user_norm)
        if not user_norm:
            return False

        now = time.monotonic()
        # Only guard shortly after we spoke something.
        if not self._recent_tts_text:
            return False

        for ts, tts_norm in reversed(self._recent_tts_text):
            if now - ts > window_s:
                break
            tts_words = _words_for_echo(tts_norm)
            if not tts_words:
                continue

            # Short-utterance echo (e.g., "oh wonderful") should still be blocked if it matches.
            if len(user_norm) >= min_chars and user_norm in tts_norm:
                return True
            inter = user_words & tts_words
            # Adjust overlap requirement for short utterances.
            required_overlap = min(min_overlap, max(1, min(len(user_words), len(tts_words))))
            if len(inter) < required_overlap:
                continue

            union = user_words | tts_words
            j = (len(inter) / len(union)) if union else 0.0

            # Containment helps when transcript is a subset/superset with some extra words.
            contain_user = len(inter) / max(1, len(user_words))
            contain_tts = len(inter) / max(1, len(tts_words))

            if j >= min_jaccard or contain_user >= min_jaccard or contain_tts >= min_jaccard:
                return True

        return False


_ECHO_RE = re.compile(r"[^a-z0-9]+")


def _normalize_text_for_echo(text: str) -> str:
    t = (text or "").lower().strip()
    t = _ECHO_RE.sub(" ", t)
    return " ".join(t.split())


def _words_for_echo(norm_text: str) -> set[str]:
    # Skip very short tokens which contribute mostly noise.
    return {w for w in norm_text.split() if len(w) >= 3}
