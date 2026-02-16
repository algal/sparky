#!/usr/bin/env python3
"""Standalone smoke test for MovementManager on a real Reachy Mini.

Sequence: wake -> breathing 5s -> listening freeze 3s -> unfreeze 5s -> stop -> sleep

Run from scaffold root:
    python tools/movement_smoke_test.py
"""

import logging
import time

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from reachy_mini import ReachyMini
from sparky_mvp.robot.moves import MovementManager


def main() -> None:
    logger.info("Connecting to Reachy Mini...")
    mini = ReachyMini(media_backend="no_media")

    logger.info("Waking up...")
    mini.wake_up()
    time.sleep(1.0)

    logger.info("Creating MovementManager...")
    mm = MovementManager(current_robot=mini)

    logger.info("Starting MovementManager (breathing should begin)...")
    mm.start()
    time.sleep(5.0)

    logger.info("Setting listening=True (antennas should freeze)...")
    mm.set_listening(True)
    time.sleep(3.0)

    logger.info("Setting listening=False (antennas should blend back)...")
    mm.set_listening(False)
    time.sleep(5.0)

    logger.info("Stopping MovementManager (goto neutral ~2s)...")
    mm.stop()

    logger.info("Going to sleep...")
    mini.goto_sleep()

    logger.info("Done.")


if __name__ == "__main__":
    main()
