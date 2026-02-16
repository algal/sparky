#!/usr/bin/env python3
"""Smoke test for SCRFD face detection and camera worker.

Usage (standalone — no robot needed for SCRFD test):
    .venv/bin/python tools/face_tracking_smoke_test.py

Note: Do NOT set CUDA_VISIBLE_DEVICES — the tracker uses device_id=1 via
ONNX provider options. Setting CUDA_VISIBLE_DEVICES=1 remaps GPU 1 → device 0,
causing device_id=1 to fail and fall back to CPU.

With robot (face tracking + movement):
    .venv/bin/python tools/face_tracking_smoke_test.py --robot
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_scrfd_standalone():
    """Test SCRFD face detection on a synthetic image."""
    print("\n=== SCRFD Face Detection (standalone) ===")

    from sparky_mvp.robot.scrfd_head_tracker import SCRFDHeadTracker

    gpu_device = int(os.environ.get("SCRFD_GPU", "1"))
    print(f"  GPU device: {gpu_device}")

    print("  Initializing SCRFD (first call downloads model ~30MB)...")
    t0 = time.time()
    tracker = SCRFDHeadTracker(
        confidence_threshold=0.3,
        gpu_device=gpu_device,
    )

    # Create a blank test image (no face expected)
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    eye_center, roll = tracker.get_head_position(blank)
    init_time = time.time() - t0
    print(f"  Cold init + first inference: {init_time:.2f}s")
    print(f"  Blank image result: eye_center={eye_center}, roll={roll}")
    assert eye_center is None, "Should detect no face in blank image"

    # Warm inference timing (blank image, but measures overhead)
    times = []
    for _ in range(20):
        t0 = time.time()
        tracker.get_head_position(blank)
        times.append(time.time() - t0)

    avg_ms = np.mean(times) * 1000
    print(f"  Warm inference (blank 640x480): {avg_ms:.1f}ms avg ({len(times)} runs)")

    print("  SCRFD standalone test PASSED")
    return tracker


def test_with_robot():
    """Test face tracking with real robot camera."""
    print("\n=== Face Tracking with Robot Camera ===")

    from reachy_mini import ReachyMini
    from sparky_mvp.robot.camera_worker import CameraWorker
    from sparky_mvp.robot.scrfd_head_tracker import SCRFDHeadTracker

    gpu_device = int(os.environ.get("SCRFD_GPU", "1"))
    tracker = SCRFDHeadTracker(confidence_threshold=0.3, gpu_device=gpu_device)

    print("  Initializing robot (with camera)...")
    mini = ReachyMini(media_backend="default")

    print("  Waking up robot (lifting head)...")
    mini.wake_up()
    time.sleep(2.0)  # Let wake animation complete

    print("  Creating CameraWorker...")
    cw = CameraWorker(reachy_mini=mini, head_tracker=tracker)

    print("  Starting CameraWorker...")
    cw.start()

    print("  Waiting for frames...")
    time.sleep(1.0)

    frame = cw.get_latest_frame()
    if frame is not None:
        print(f"  Got frame: shape={frame.shape}, dtype={frame.dtype}")
    else:
        print("  WARNING: No frame received after 1s")

    # Poll face tracking for 10 seconds
    print("  Face tracking for 10s (move your face in front of the camera)...")
    start = time.time()
    detected_count = 0
    total_polls = 0

    while time.time() - start < 10.0:
        offsets = cw.get_face_tracking_offsets()
        total_polls += 1
        if any(abs(o) > 0.001 for o in offsets):
            detected_count += 1
            if detected_count % 20 == 1:
                print(
                    f"  Face detected: x={offsets[0]:.3f} y={offsets[1]:.3f} z={offsets[2]:.3f} "
                    f"r={offsets[3]:.3f} p={offsets[4]:.3f} yaw={offsets[5]:.3f}"
                )
        time.sleep(0.04)

    print(f"  Detection rate: {detected_count}/{total_polls} polls had face offsets")

    print("  Stopping CameraWorker...")
    cw.stop()

    print("  Putting robot to sleep...")
    mini.goto_sleep()

    print("  Robot camera test complete")


def test_scene_capture():
    """Test scene capture utility (requires robot camera)."""
    print("\n=== Scene Capture ===")

    from reachy_mini import ReachyMini
    from sparky_mvp.robot.camera_worker import CameraWorker
    from sparky_mvp.robot.scene_capture import capture_scene_image

    print("  Initializing robot (with camera)...")
    mini = ReachyMini(media_backend="default")

    print("  Waking up robot (lifting head)...")
    mini.wake_up()
    time.sleep(2.0)

    cw = CameraWorker(reachy_mini=mini, head_tracker=None)
    cw.start()
    time.sleep(1.0)

    image_block = capture_scene_image(cw, max_size=512, jpeg_quality=75)
    cw.stop()
    mini.goto_sleep()

    if image_block is not None:
        data_len = len(image_block["source"]["data"])
        # Rough size estimate: base64 is ~4/3 of raw size
        raw_kb = data_len * 3 / 4 / 1024
        print(f"  Captured image: base64 len={data_len} (~{raw_kb:.0f} KB)")
        print(f"  Media type: {image_block['source']['media_type']}")
        print("  Scene capture PASSED")
    else:
        print("  WARNING: No frame captured")


def main():
    parser = argparse.ArgumentParser(description="Face tracking smoke test")
    parser.add_argument("--robot", action="store_true", help="Test with real robot camera")
    parser.add_argument("--scene", action="store_true", help="Test scene capture")
    args = parser.parse_args()

    # Always run standalone SCRFD test
    test_scrfd_standalone()

    if args.robot:
        test_with_robot()

    if args.scene:
        test_scene_capture()

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
