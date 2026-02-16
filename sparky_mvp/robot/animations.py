"""
Robot animations for Reachy Mini.

Provides async wrappers around blocking Reachy Mini animations,
allowing them to be used in async event loops without blocking.

Note: Continuous animations (antenna wiggle during TTS) are deprecated
in favour of MovementManager (see sparky_mvp.robot.moves), which owns
all set_target() calls via a 100 Hz control loop. The wake_up() and
go_to_sleep() wrappers here are still used for lifecycle transitions.
"""

import asyncio
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)

try:
    from reachy_mini import ReachyMini
except ImportError:
    # Mock for testing environments without reachy_mini
    ReachyMini = None


class RobotAnimator:
    """
    Manages Reachy Mini animations for different events.

    Wraps blocking robot animations in async executors to prevent
    blocking the main async event loop.

    Singleton-like antenna state management for speaking animations.
    """

    # Class-level (singleton) antenna animation state
    _antenna_wiggle_task = None
    _antenna_stop_event = None
    _saved_antenna_position = None
    _mini_instance = None
    _reset_lock = None  # Lock to ensure reset completes before new movements

    def __init__(self, mini: Optional[object] = None):
        """
        Initialize robot animator.

        Args:
            mini: ReachyMini instance, or None for mock mode
        """
        self.mini = mini
        self.enabled = mini is not None

        # Set class-level mini instance for singleton methods
        if mini is not None:
            RobotAnimator._mini_instance = mini

        # Initialize singleton stop event
        if RobotAnimator._antenna_stop_event is None:
            RobotAnimator._antenna_stop_event = asyncio.Event()
            RobotAnimator._antenna_stop_event.set()  # Initially stopped

        # Initialize singleton reset lock
        if RobotAnimator._reset_lock is None:
            RobotAnimator._reset_lock = asyncio.Lock()

    async def antenna_wiggle(self, duration: float = 0.5) -> None:
        """
        Quick antenna wiggle animation (for interruption).

        Args:
            duration: Duration per antenna movement in seconds
        """
        if not self.enabled:
            logger.debug("[MOCK] Antenna wiggle animation")
            await asyncio.sleep(duration * 3)  # Simulate duration
            return

        # Run blocking animation in thread executor
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._blocking_wiggle, duration)

    def _blocking_wiggle(self, duration: float) -> None:
        """
        Blocking antenna wiggle implementation.

        Runs in thread executor to avoid blocking async loop.

        Args:
            duration: Duration per movement
        """
        try:
            # Pattern from main.py:302-304
            self.mini.goto_target(antennas=[0.5, -0.5], duration=duration)
            self.mini.goto_target(antennas=[-0.5, 0.5], duration=duration)
            self.mini.goto_target(antennas=[0, 0], duration=duration)
        except Exception as e:
            logger.error(f"Antenna wiggle error: {e}")

    async def tool_animation(self, tool_name: str) -> None:
        """
        Animation based on tool type.

        Tool animations disabled - will be re-enabled later.
        """
        pass

    async def wake_up(self) -> None:
        """
        Wake up animation.

        Uses robot's built-in wake_up() method.
        """
        if not self.enabled:
            logger.debug("[MOCK] Wake up animation")
            await asyncio.sleep(2.0)
            return

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.mini.wake_up)

    async def go_to_sleep(self) -> None:
        """
        Sleep animation.

        Uses robot's built-in goto_sleep() method.
        """
        if not self.enabled:
            logger.debug("[MOCK] Go to sleep animation")
            await asyncio.sleep(2.0)
            return

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.mini.goto_sleep)

    # Singleton antenna animation methods

    @classmethod
    async def start_antenna_wiggle(cls) -> None:
        """
        Start continuous antenna wiggle animation (singleton).

        Waits for any in-progress reset to complete before starting.
        Saves current antenna position and starts background wiggle task.
        Used during TTS playback to show the robot is speaking.
        """
        if cls._mini_instance is None:
            logger.debug("No robot instance available for antenna wiggle")
            return

        # Stop any existing wiggle task (without reset - no lock needed)
        if cls._antenna_wiggle_task is not None:
            logger.info("Stopping existing wiggle task")
            cls._antenna_stop_event.set()
            try:
                await asyncio.wait_for(cls._antenna_wiggle_task, timeout=1.0)
            except asyncio.TimeoutError:
                cls._antenna_wiggle_task.cancel()
                try:
                    await cls._antenna_wiggle_task
                except asyncio.CancelledError:
                    pass
            cls._antenna_wiggle_task = None

        # Wait for any in-progress reset to complete (ensures movements queue properly)
        async with cls._reset_lock:
            logger.debug("Acquired reset lock, starting wiggle animation")

            # Note: We don't save antenna position because:
            # 1. ReachyMini API doesn't expose direct antenna position reading
            # 2. We always reset to neutral [0, 0] position anyway when stopping
            cls._saved_antenna_position = [0, 0]  # Will reset to neutral

            # Clear stop event (antenna wiggle is active)
            cls._antenna_stop_event.clear()

            # Start background wiggle task
            cls._antenna_wiggle_task = asyncio.create_task(cls._antenna_wiggle_loop())
            logger.info("Started antenna wiggle animation")

    @classmethod
    async def stop_antenna_wiggle(cls) -> None:
        """
        Stop antenna wiggle animation and return to default neutral pose (singleton).

        Signals the wiggle loop to stop and waits for it to complete,
        then BLOCKS while returning head and antennas to neutral default position.
        The reset is blocking - no new movements can start until it completes.
        """
        if cls._antenna_wiggle_task is None:
            logger.debug("No antenna wiggle task to stop")
            return

        logger.info("Stopping antenna wiggle animation")

        # Signal stop event
        cls._antenna_stop_event.set()

        # Wait for wiggle task to complete (with timeout)
        try:
            await asyncio.wait_for(cls._antenna_wiggle_task, timeout=1.0)
            logger.info("Antenna wiggle task stopped")
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for antenna wiggle to stop, cancelling")
            cls._antenna_wiggle_task.cancel()
            try:
                await cls._antenna_wiggle_task
            except asyncio.CancelledError:
                pass

        cls._antenna_wiggle_task = None

        # BLOCKING RESET: Acquire lock to prevent any new movements during reset
        async with cls._reset_lock:
            logger.info("Acquired reset lock - BLOCKING reset to neutral pose")

            # Return to default neutral pose (head and antennas)
            if cls._mini_instance:
                try:
                    logger.info("Returning to default neutral pose")
                    loop = asyncio.get_event_loop()

                    # Get default neutral head pose
                    from reachy_mini.utils import create_head_pose
                    default_head_pose = create_head_pose()

                    # Return to neutral with smooth motion (BLOCKING - holds lock)
                    await loop.run_in_executor(
                        None,
                        lambda: cls._mini_instance.goto_target(
                            head=default_head_pose,
                            antennas=[0.0, 0.0],
                            duration=0.5
                        )
                    )
                    logger.info("Returned to default pose - releasing lock")
                except Exception as e:
                    logger.warning(f"Could not return to default pose: {e}")

            cls._saved_antenna_position = None

    @classmethod
    async def _antenna_wiggle_loop(cls) -> None:
        """
        Continuous antenna wiggle loop (runs in background).

        Uses set_target() with explicit head pose to prevent head wilting.
        Smooth sine wave pattern for gentle antenna movement during speech.
        """
        import numpy as np
        import time

        logger.info("Antenna wiggle loop started")
        loop = asyncio.get_event_loop()

        try:
            # Get current head pose to lock it in place
            current_head_pose = cls._mini_instance.get_current_head_pose()

            start_time = time.time()

            while not cls._antenna_stop_event.is_set():
                t = time.time() - start_time

                # Sine wave pattern for smooth movement (frequency: 0.5 Hz, amplitude: 20 degrees)
                antennas_offset = np.deg2rad(20 * np.sin(2 * np.pi * 0.5 * t))

                # Set target with explicit head pose to prevent wilting
                await loop.run_in_executor(
                    None,
                    lambda offset=antennas_offset: cls._mini_instance.set_target(
                        head=current_head_pose,
                        antennas=(offset, offset)
                    )
                )

                # Small sleep to control update rate (30 Hz)
                await asyncio.sleep(0.033)

        except asyncio.CancelledError:
            logger.info("Antenna wiggle loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in antenna wiggle loop: {e}")
        finally:
            logger.info("Antenna wiggle loop ended")


# Helper function to create animator

def create_animator(mini: Optional[object] = None) -> RobotAnimator:
    """
    Create a RobotAnimator instance.

    Args:
        mini: ReachyMini instance, or None for mock mode

    Returns:
        RobotAnimator instance
    """
    return RobotAnimator(mini=mini)
