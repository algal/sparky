#!/usr/bin/env python3
"""Smoke test for orient-on-speech face tracking mode.

Tests the orient_to_speaker() mechanism in isolation — no HeadWobbler,
no TTS, no VAD. Just MovementManager + CameraWorker.

Sequence:
  0. Pre-check: verify camera returns frames and SCRFD detects a face
  1. Wake, start CameraWorker + MovementManager (orient_on_speech mode)
  2. Breathe for 5s — robot should NOT track your face
  3. Trigger orient_to_speaker() — robot should snap to your face position
  4. Hold for 5s — robot should stay oriented
  5. Trigger again — robot should re-orient to current face position
  6. Hold for 5s
  7. Stop, sleep

Stand in front of the robot and move around between triggers to see the effect.

Run from scaffold root:
    .venv/bin/python tools/orient_on_speech_smoke_test.py

Compare with continuous mode:
    .venv/bin/python tools/orient_on_speech_smoke_test.py --continuous
"""

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def fmt_offsets(offsets):
    """Format a 6-tuple of offsets for display."""
    labels = ("x", "y", "z", "roll", "pitch", "yaw")
    return " ".join(f"{l}={v:+.4f}" for l, v in zip(labels, offsets))


def precheck_camera_and_scrfd(mini, tracker):
    """Verify camera returns frames and SCRFD can detect a face.

    Returns True if a face was detected, False otherwise.
    Aborts the test early if the camera is broken.
    """
    logger.info("=== PRE-CHECK: camera + SCRFD ===")

    # 1. Can we get a frame?
    logger.info("Grabbing a frame from robot camera...")
    frame = None
    for attempt in range(10):
        frame = mini.media.get_frame()
        if frame is not None:
            break
        time.sleep(0.2)

    if frame is None:
        logger.error("FAILED: camera returned no frames after 2s. Is the camera connected?")
        return False

    logger.info("Got frame: shape=%s dtype=%s", frame.shape, frame.dtype)

    # 2. Can SCRFD detect a face in this frame?
    logger.info("Running SCRFD on captured frame (stand in front of the camera!)...")
    eye_center, roll = tracker.get_head_position(frame)

    if eye_center is not None:
        logger.info(
            "FACE DETECTED: eye_center=(%.3f, %.3f) roll=%.1f°",
            eye_center[0], eye_center[1], float(roll) * 180 / 3.14159,
        )
    else:
        logger.warning("No face detected in pre-check frame. Trying a few more...")
        # Try several more frames over 3 seconds
        detected = False
        for i in range(30):
            time.sleep(0.1)
            frame = mini.media.get_frame()
            if frame is not None:
                eye_center, roll = tracker.get_head_position(frame)
                if eye_center is not None:
                    logger.info(
                        "FACE DETECTED on attempt %d: eye_center=(%.3f, %.3f) roll=%.1f°",
                        i + 1, eye_center[0], eye_center[1],
                        float(roll) * 180 / 3.14159,
                    )
                    detected = True
                    break
        if not detected:
            logger.error(
                "FAILED: SCRFD did not detect a face in 30 attempts over 3s. "
                "Make sure you are standing in front of the robot with your face visible."
            )
            return False

    # 3. Verify CameraWorker pipeline produces offsets
    logger.info("Pre-check: starting CameraWorker for 3s to verify offset pipeline...")
    from sparky_mvp.robot.camera_worker import CameraWorker

    cw = CameraWorker(reachy_mini=mini, head_tracker=tracker)
    cw.start()

    detected_count = 0
    last_offsets = None
    for i in range(75):  # 3s at 25Hz
        time.sleep(0.04)
        offsets = cw.get_face_tracking_offsets()
        if any(abs(o) > 0.001 for o in offsets):
            detected_count += 1
            last_offsets = offsets

    cw.stop()

    if detected_count > 0:
        logger.info(
            "CameraWorker produced %d/75 non-zero offset polls. Last: [%s]",
            detected_count, fmt_offsets(last_offsets),
        )
        logger.info("PRE-CHECK PASSED")
        return True
    else:
        logger.error(
            "FAILED: CameraWorker ran for 3s but never produced non-zero offsets. "
            "SCRFD detected a face directly but CameraWorker pipeline may have a bug."
        )
        return False


def poll_offsets(cw, mm, duration_s, label, trigger_at=None):
    """Poll and log offsets for a duration.

    Args:
        cw: CameraWorker
        mm: MovementManager
        duration_s: how long to poll
        label: phase label for logging
        trigger_at: if set, call orient_to_speaker() this many seconds in
    """
    start = time.time()
    triggered = False
    last_log = 0

    while time.time() - start < duration_s:
        elapsed = time.time() - start

        # Trigger orient if requested
        if trigger_at is not None and not triggered and elapsed >= trigger_at:
            logger.info("[%s] >>> TRIGGERING orient_to_speaker() <<<", label)
            mm.orient_to_speaker()
            triggered = True

        # Log offsets every 0.5s
        if elapsed - last_log >= 0.5:
            cam_offsets = cw.get_face_tracking_offsets()
            mm_offsets = mm.state.face_tracking_offsets
            has_face = any(abs(o) > 0.001 for o in cam_offsets)
            logger.info(
                "[%s] t=%.1fs  camera=[%s]  mm=[%s]  face=%s",
                label,
                elapsed,
                fmt_offsets(cam_offsets),
                fmt_offsets(mm_offsets),
                "YES" if has_face else "no",
            )
            last_log = elapsed

        time.sleep(0.04)


def main():
    parser = argparse.ArgumentParser(description="Orient-on-speech smoke test")
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Use continuous mode instead of orient_on_speech (for comparison)",
    )
    parser.add_argument(
        "--skip-precheck",
        action="store_true",
        help="Skip the camera/SCRFD pre-check",
    )
    args = parser.parse_args()

    mode = "continuous" if args.continuous else "orient_on_speech"

    from reachy_mini import ReachyMini
    from sparky_mvp.robot.camera_worker import CameraWorker
    from sparky_mvp.robot.moves import MovementManager
    from sparky_mvp.robot.scrfd_head_tracker import SCRFDHeadTracker

    gpu_device = int(os.environ.get("SCRFD_GPU", "1"))

    logger.info("Connecting to Reachy Mini (with camera)...")
    mini = ReachyMini(media_backend="default")

    logger.info("Waking up...")
    mini.wake_up()
    time.sleep(1.5)

    logger.info("Initializing SCRFD tracker (GPU %d)...", gpu_device)
    tracker = SCRFDHeadTracker(confidence_threshold=0.3, gpu_device=gpu_device)

    # Pre-check: verify camera and face detection work
    if not args.skip_precheck:
        ok = precheck_camera_and_scrfd(mini, tracker)
        if not ok:
            logger.error("Pre-check failed. Fix the issue above before testing orient-on-speech.")
            mini.goto_sleep()
            return
        # Small gap to let user re-position
        logger.info("Pre-check done. Starting orient test in 2s...")
        time.sleep(2.0)

    logger.info("Creating CameraWorker...")
    cw = CameraWorker(reachy_mini=mini, head_tracker=tracker)

    logger.info("Creating MovementManager (head_tracking_mode=%s)...", mode)
    mm = MovementManager(
        current_robot=mini,
        camera_worker=cw,
        head_tracking_mode=mode,
    )

    logger.info("Starting CameraWorker...")
    cw.start()

    # Wait for SCRFD to warm up and detect a face
    logger.info("Waiting for face detection to stabilize (3s)...")
    for i in range(75):
        time.sleep(0.04)
        offsets = cw.get_face_tracking_offsets()
        if i % 25 == 0:
            has_face = any(abs(o) > 0.001 for o in offsets)
            logger.info("  t=%.1fs camera=[%s] face=%s", i * 0.04, fmt_offsets(offsets), "YES" if has_face else "no")

    logger.info("Starting MovementManager (breathing begins)...")
    mm.start()

    try:
        if mode == "orient_on_speech":
            logger.info("=== Phase 1: Breathing only, NO tracking (5s) ===")
            logger.info("Stand in front of the robot. It should NOT follow your face.")
            poll_offsets(cw, mm, 5.0, "breathe")

            logger.info("=== Phase 2: ORIENT trigger + hold (5s) ===")
            logger.info("Robot should snap to your current face position.")
            poll_offsets(cw, mm, 5.0, "orient1", trigger_at=0.5)

            logger.info("=== Phase 3: Move to a different position, then second ORIENT (5s) ===")
            logger.info("Move to a different spot before the trigger fires at t=2s.")
            poll_offsets(cw, mm, 5.0, "orient2", trigger_at=2.0)

        else:
            logger.info("=== Continuous tracking for 15s ===")
            logger.info("Robot should continuously follow your face.")
            poll_offsets(cw, mm, 15.0, "continuous")

    except KeyboardInterrupt:
        logger.info("Interrupted")

    logger.info("Stopping MovementManager...")
    mm.stop()

    logger.info("Stopping CameraWorker...")
    cw.stop()

    logger.info("Going to sleep...")
    mini.goto_sleep()

    logger.info("Done.")


if __name__ == "__main__":
    main()
