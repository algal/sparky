"""
Spontaneous gesture manager — periodic ambient idle gestures.

When enabled, the robot occasionally performs a gentle recorded emotion
from the Pollen Robotics emotions library, giving it organic "aliveness"
even when no conversation is happening.

Only fires when:
- The robot is in INTERACTIVE state (not sleeping, not processing)
- Optionally, a face is detected (presence gating, same as spontaneous speech)

These gestures are silent and non-dramatic — they represent a being
that fidgets, looks around, yawns, and thinks while idle.

Config (in config.yaml):
    spontaneous_gestures:
      enabled: true
      min_interval_m: 2       # minimum minutes between gestures
      max_interval_m: 5       # maximum minutes between gestures
      require_presence: false  # gestures are fine even when alone
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Recorded emotions suitable for spontaneous/ambient use.
# These are non-reactive — a being just existing.
#
# Weights control relative probability. Higher = more common.
# Subtle, short gestures are weighted high (common background texture).
# Dramatic, long gestures are weighted low (rare, delightful surprises).
SPONTANEOUS_EMOTIONS: dict[str, float] = {
    # Common — subtle ambient texture
    "thoughtful1":  10.0,   #  5.9s — looking up, searching for an idea
    "thoughtful2":  10.0,   #  5.5s — same family
    "serenity1":    10.0,   #  4.6s — calm, inner peace
    "curious1":      8.0,   # 11.8s — looking around at surroundings
    "lost1":         6.0,   #  8.1s — idle musing, unsure what to do
    # Occasional — noticeable but not startling
    "tired1":        4.0,   #  7.5s — yawning
    "boredom1":      3.0,   # 15.7s — looking around sleepily
    "lonely1":       3.0,   # 10.2s — when no one is around
    # Rare — dramatic, long, delightful when they happen
    "boredom2":      1.0,   # 14.2s — boredom with snoring
    "exhausted1":    0.5,   # 18.3s — falling asleep after long uptime
    "sleep1":        0.5,   # 19.8s — nodding off (the rarest)
}


class SpontaneousGestureManager:
    """Periodically queues ambient recorded emotions on the MovementManager.

    Usage::

        manager = SpontaneousGestureManager(
            queue_move=movement_manager.queue_move,
            check_state=lambda: state_machine.state == ReachyState.INTERACTIVE,
            config=config.get("spontaneous_gestures", {}),
        )
        task = asyncio.create_task(manager.run())
        ...
        manager.stop()
    """

    def __init__(
        self,
        queue_move: Callable,
        check_state: Callable[[], bool],
        check_presence: Optional[Callable[[], bool]] = None,
        config: dict | None = None,
    ):
        """
        Args:
            queue_move: Callable that queues a Move on the MovementManager.
                        Signature: queue_move(move: Move) -> None
            check_state: Returns True if the robot is in a state where
                         gestures are appropriate (INTERACTIVE).
            check_presence: Optional callable returning True if user is present.
            config: Config dict with min_interval_m, max_interval_m, require_presence.
        """
        cfg = config or {}
        self.min_interval_s = float(cfg.get("min_interval_m", 2)) * 60.0
        self.max_interval_s = float(cfg.get("max_interval_m", 5)) * 60.0
        self.require_presence = cfg.get("require_presence", False)
        self._queue_move = queue_move
        self._check_state = check_state
        self._check_presence = check_presence
        self._stopping = False
        self._recorded_moves = None  # Lazy-loaded

    def _load_emotions(self):
        """Lazy-load the recorded emotions library."""
        if self._recorded_moves is not None:
            return
        try:
            from reachy_mini.motion.recorded_move import RecordedMoves
            self._recorded_moves = RecordedMoves(
                "pollen-robotics/reachy-mini-emotions-library"
            )
            available = set(self._recorded_moves.list_moves())
            valid = {e: w for e, w in SPONTANEOUS_EMOTIONS.items() if e in available}
            missing = [e for e in SPONTANEOUS_EMOTIONS if e not in available]
            if missing:
                logger.warning(
                    "Spontaneous emotions not found in library: %s", missing
                )
            self._valid_emotions = list(valid.keys())
            self._valid_weights = list(valid.values())
            logger.info(
                "Loaded %d spontaneous emotions from library (%d available)",
                len(valid), len(available),
            )
        except Exception:
            logger.exception("Failed to load recorded emotions library")
            self._recorded_moves = None
            self._valid_emotions = []
            self._valid_weights = []

    def _next_delay(self) -> float:
        """Compute next randomized delay (seconds)."""
        base = random.uniform(self.min_interval_s, self.max_interval_s)
        jitter = base * random.uniform(-0.2, 0.2)
        floor = min(30.0, self.min_interval_s)  # Don't exceed configured min
        return max(floor, base + jitter)

    async def run(self) -> None:
        """Main loop: sleep, check conditions, perform gesture."""
        self._load_emotions()

        if not self._valid_emotions:
            logger.warning("No valid spontaneous emotions — gesture manager disabled")
            return

        logger.info(
            "SpontaneousGestureManager started (interval=%.0f-%.0fm, "
            "presence=%s, emotions=%d)",
            self.min_interval_s / 60,
            self.max_interval_s / 60,
            self.require_presence,
            len(self._valid_emotions),
        )

        # Initial delay
        initial_delay = self._next_delay() * 0.5
        await self._interruptible_sleep(initial_delay)

        while not self._stopping:
            should_gesture = True

            # Check robot state (must be INTERACTIVE)
            try:
                if not self._check_state():
                    logger.debug("Spontaneous gesture skipped — wrong state")
                    should_gesture = False
            except Exception:
                should_gesture = False

            # Check presence if required
            if should_gesture and self.require_presence and self._check_presence:
                try:
                    if not self._check_presence():
                        logger.debug("Spontaneous gesture skipped — no presence")
                        should_gesture = False
                except Exception:
                    should_gesture = False

            if should_gesture:
                emotion_name = random.choices(
                    self._valid_emotions, weights=self._valid_weights, k=1
                )[0]
                try:
                    assert self._recorded_moves is not None
                    move = self._recorded_moves.get(emotion_name)
                    self._queue_move(move)
                    logger.info(
                        "Spontaneous gesture: %s (%.1fs)",
                        emotion_name, move.duration,
                    )
                except Exception:
                    logger.exception(
                        "Failed to queue spontaneous gesture: %s", emotion_name
                    )

            delay = self._next_delay()
            logger.debug("Next spontaneous gesture in ~%.0f minutes", delay / 60)
            await self._interruptible_sleep(delay)

    async def _interruptible_sleep(self, seconds: float) -> None:
        """Sleep in small increments so stop() takes effect quickly."""
        end = time.monotonic() + seconds
        while not self._stopping and time.monotonic() < end:
            remaining = end - time.monotonic()
            await asyncio.sleep(min(5.0, remaining))

    def stop(self) -> None:
        """Signal the manager to stop."""
        self._stopping = True
