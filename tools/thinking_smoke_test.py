#!/usr/bin/env python3
"""Smoke test for staged thinking animation on a real robot.

Runs through all 3 stages so you can visually verify the escalation.

Usage:
    PYTHONPATH=. .venv/bin/python3 tools/thinking_smoke_test.py
"""

import time

from reachy_mini import ReachyMini
from sparky_mvp.robot.moves import MovementManager
from sparky_mvp.robot.thinking import ThinkingMove


def main():
    print("Initializing robot...")
    mini = ReachyMini(media_backend="default_no_video")

    print("Creating MovementManager...")
    mm = MovementManager(current_robot=mini)

    # Wake up
    print("Waking up...")
    mini.goto_target(mini.get_default_posture(), duration=2.0, wait=True)
    time.sleep(0.5)

    # Start MovementManager
    print("Starting MovementManager...")
    mm.start()

    # Let it breathe for a moment
    print("\n[Breathing idle] 3s")
    time.sleep(3.0)

    # Queue thinking move
    print("[Thinking started] queuing ThinkingMove...")
    mm.clear_move_queue()
    mm.queue_move(ThinkingMove())

    print("[Stage 1: Subtle] 0-2s — gentle antenna tilt")
    time.sleep(2.0)

    print("[Stage 2: Medium] 2-5s — head sway added")
    time.sleep(3.0)

    print("[Stage 3: Full]   5s+ — pronounced movement")
    time.sleep(5.0)

    # Clear thinking (simulates response complete)
    print("\n[Response complete] clearing move queue...")
    mm.clear_move_queue()

    print("[Breathing resumes] 4s")
    time.sleep(4.0)

    # One more cycle
    print("\n[Second thinking cycle] 6s")
    mm.clear_move_queue()
    mm.queue_move(ThinkingMove())
    time.sleep(6.0)

    print("[Clear] stopping...")
    mm.clear_move_queue()
    time.sleep(1.0)

    # Cleanup
    print("\nStopping MovementManager...")
    mm.stop()

    print("Going to sleep...")
    mini.goto_target(mini.get_default_posture(), duration=2.0, wait=True)
    print("Done!")


if __name__ == "__main__":
    main()
