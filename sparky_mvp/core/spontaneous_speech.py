"""
Spontaneous speech manager — periodic, rare, volunteered speech.

When enabled, the robot occasionally speaks unprompted on a random topic,
but only when the user is present (face detected via CameraWorker).

Timing is randomized: a base interval (10-30 min) with additional jitter
so it never feels perfectly periodic.

The speech is sent through the normal OpenClaw chat pipeline (chat.send)
so the agent produces a contextually appropriate response that gets TTS'd.

Config (in config.yaml):
    spontaneous_speech:
      enabled: true
      min_interval_m: 10     # minimum minutes between spontaneous speech
      max_interval_m: 30     # maximum minutes between spontaneous speech
      require_presence: true  # only speak when face detected

Toggle: set spontaneous_speech.enabled to false to disable entirely.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Prompts that encourage the agent to say something interesting and brief.
# Mix of contextual/personal, interest-driven, and gentle presence.
# Interest-driven prompts reference INTERESTS.md in the agent's workspace.
SPONTANEOUS_PROMPTS = [
    # Contextual — reference the moment, recent activity, time of day
    "[spontaneous] Someone is nearby but it's been quiet. Say something connected "
    "to what's actually happening — the time of day, something from recent "
    "conversation, or what you imagine they might be up to. One or two sentences, "
    "as if thinking aloud.",
    # Interest-driven — framed as genuine thought, not trivia
    "[spontaneous] Something from your current interests (INTERESTS.md) has been "
    "on your mind. Share the thought briefly — not as a fact to deliver, but as "
    "something you've been turning over. One sentence, naturally. If it connects "
    "to something recent, even better.",
    # Curiosity about them — ask, don't tell
    "[spontaneous] Someone is around. Ask them something you're genuinely curious "
    "about — what they're working on, what they think about something, how "
    "something went. One short question. Be a housemate, not an interviewer.",
    # Interest-driven, connection-making
    "[spontaneous] You've noticed a connection between something from your current "
    "interests (INTERESTS.md) and something from recent conversation or the "
    "current situation. Share it — one sentence, the way you'd mention something "
    "that just occurred to you.",
    # Gentle presence — minimal
    "[spontaneous] Someone is nearby. Make a brief, natural remark — a small "
    "observation, a passing thought, or just acknowledge the moment. One sentence. "
    "You're a presence in the room, not a notification.",
]


class SpontaneousSpeechManager:
    """
    Manages periodic spontaneous speech with randomized timing and presence gating.

    Usage::

        manager = SpontaneousSpeechManager(
            send_message=my_send_fn,
            check_presence=my_presence_fn,
            config=config.get("spontaneous_speech", {}),
        )
        task = asyncio.create_task(manager.run())
        ...
        manager.stop()
    """

    def __init__(
        self,
        send_message: Callable[[str], Any],
        check_presence: Callable[[], bool],
        config: dict | None = None,
    ):
        """
        Args:
            send_message: Async callable that sends a message through the chat pipeline.
                         Signature: async def send_message(text: str) -> None
            check_presence: Callable that returns True if the user is present (face detected).
            config: Config dict with min_interval_m, max_interval_m, require_presence.
        """
        cfg = config or {}
        self.min_interval_s = float(cfg.get("min_interval_m", 10)) * 60.0
        self.max_interval_s = float(cfg.get("max_interval_m", 30)) * 60.0
        self.require_presence = cfg.get("require_presence", True)
        self._send_message = send_message
        self._check_presence = check_presence
        self._stopping = False
        self._last_speech_time: float = 0.0

    def _next_delay(self) -> float:
        """Compute next randomized delay (seconds)."""
        base = random.uniform(self.min_interval_s, self.max_interval_s)
        # Add up to ±25% jitter on top
        jitter = base * random.uniform(-0.25, 0.25)
        return max(60.0, base + jitter)  # Never less than 1 minute

    def _pick_prompt(self) -> str:
        """Pick a random spontaneous speech prompt."""
        return random.choice(SPONTANEOUS_PROMPTS)

    async def run(self) -> None:
        """Main loop: sleep, check presence, speak."""
        logger.info(
            "SpontaneousSpeechManager started (interval=%.0f-%.0fm, presence=%s)",
            self.min_interval_s / 60,
            self.max_interval_s / 60,
            self.require_presence,
        )

        # Initial delay before first spontaneous speech
        # Use half the normal interval, with a floor of the configured minimum
        initial_delay = max(self.min_interval_s, self._next_delay() * 0.5)
        logger.info("First spontaneous speech in ~%.0f minutes", initial_delay / 60)

        try:
            await self._interruptible_sleep(initial_delay)

            while not self._stopping:
                # Check if we should speak
                should_speak = True

                if self.require_presence:
                    try:
                        present = self._check_presence()
                    except Exception:
                        present = False
                    if not present:
                        logger.debug("Spontaneous speech skipped — no presence detected")
                        should_speak = False

                if should_speak:
                    prompt = self._pick_prompt()
                    logger.info("Spontaneous speech triggered: %s", prompt[:60])
                    try:
                        await self._send_message(prompt)
                        self._last_speech_time = time.monotonic()
                    except Exception:
                        logger.exception("Spontaneous speech send failed")

                # Wait for next interval
                delay = self._next_delay()
                logger.debug("Next spontaneous speech in ~%.0f minutes", delay / 60)
                await self._interruptible_sleep(delay)

        except asyncio.CancelledError:
            logger.info("SpontaneousSpeechManager cancelled")
            raise

        logger.info("SpontaneousSpeechManager stopped")

    async def _interruptible_sleep(self, seconds: float) -> None:
        """Sleep in small increments so stop() takes effect quickly."""
        end = time.monotonic() + seconds
        while not self._stopping and time.monotonic() < end:
            remaining = end - time.monotonic()
            await asyncio.sleep(min(5.0, remaining))

    def stop(self) -> None:
        """Signal the manager to stop."""
        self._stopping = True
