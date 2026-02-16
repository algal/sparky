#!/usr/bin/env python3
"""Smoke test for HeadWobbler with MovementManager on a real robot.

Generates synthetic speech audio (sine wave at speech frequencies) and
feeds it through HeadWobbler -> MovementManager to verify head movement.

Usage:
    PYTHONPATH=. .venv/bin/python3 tools/head_wobbler_smoke_test.py
"""

import time
import numpy as np

from reachy_mini import ReachyMini
from sparky_mvp.robot.moves import MovementManager
from sparky_mvp.robot.head_wobbler import HeadWobbler


def generate_speech_tone(duration_s: float = 2.0, sample_rate: int = 24000) -> np.ndarray:
    """Generate a sine wave that simulates speech loudness for testing."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), dtype=np.float32)
    # Mix of frequencies typical in speech (200Hz fundamental + harmonics)
    signal = (
        0.3 * np.sin(2 * np.pi * 200 * t)
        + 0.2 * np.sin(2 * np.pi * 400 * t)
        + 0.1 * np.sin(2 * np.pi * 800 * t)
    )
    # Amplitude modulation at ~3Hz (syllable rate)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    signal *= envelope
    # Scale to int16
    pcm = (signal * 16000).astype(np.int16)
    return pcm


def main():
    print("Initializing robot...")
    mini = ReachyMini(media_backend="default_no_video")

    print("Creating MovementManager...")
    mm = MovementManager(current_robot=mini)

    print("Creating HeadWobbler...")
    hw = HeadWobbler(set_speech_offsets=mm.set_speech_offsets)

    # Wake up
    print("Waking up...")
    mini.goto_target(mini.get_default_posture(), duration=2.0, wait=True)
    time.sleep(0.5)

    # Start both
    print("Starting MovementManager + HeadWobbler...")
    mm.start()
    hw.start()

    # Phase 1: Idle breathing (no speech input)
    print("\n[Phase 1] Idle breathing - 3s (no head wobble)")
    time.sleep(3.0)

    # Phase 2: Feed synthetic speech audio
    print("[Phase 2] Feeding synthetic speech audio - 4s (expect head wobble)")
    pcm = generate_speech_tone(duration_s=4.0, sample_rate=24000)
    hw.feed_pcm(pcm, sample_rate=24000)
    time.sleep(5.0)  # 4s audio + 1s for wobbler to finish

    # Phase 3: Reset and verify head returns to neutral
    print("[Phase 3] Reset HeadWobbler - 2s (head should return to neutral)")
    hw.reset()
    time.sleep(2.0)

    # Phase 4: Feed another burst
    print("[Phase 4] Second speech burst - 2s")
    pcm2 = generate_speech_tone(duration_s=2.0, sample_rate=24000)
    hw.feed_pcm(pcm2, sample_rate=24000)
    time.sleep(3.0)

    # Cleanup
    print("\nStopping HeadWobbler...")
    hw.stop()
    print("Stopping MovementManager...")
    mm.stop()

    print("Going to sleep...")
    mini.goto_target(mini.get_default_posture(), duration=2.0, wait=True)
    print("Done!")


if __name__ == "__main__":
    main()
