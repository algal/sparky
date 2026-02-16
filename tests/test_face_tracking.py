"""Tests for SCRFD head tracker and CameraWorker."""

import threading
import time

import numpy as np
import pytest


class TestSCRFDHeadTracker:
    """Test SCRFDHeadTracker interface and face selection logic."""

    def test_constructor_stores_params(self):
        from sparky_mvp.robot.scrfd_head_tracker import SCRFDHeadTracker

        tracker = SCRFDHeadTracker(
            confidence_threshold=0.5,
            gpu_device=0,
            det_size=(320, 320),
        )
        assert tracker.confidence_threshold == 0.5
        assert tracker.gpu_device == 0
        assert tracker.det_size == (320, 320)
        assert tracker._app is None  # Lazy init

    def test_select_best_face_empty(self):
        from sparky_mvp.robot.scrfd_head_tracker import SCRFDHeadTracker

        tracker = SCRFDHeadTracker()
        assert tracker._select_best_face([]) is None

    def test_select_best_face_below_threshold(self):
        from sparky_mvp.robot.scrfd_head_tracker import SCRFDHeadTracker

        tracker = SCRFDHeadTracker(confidence_threshold=0.8)

        # Mock face with low confidence
        class MockFace:
            det_score = 0.3
            bbox = np.array([10, 10, 100, 100])

        assert tracker._select_best_face([MockFace()]) is None

    def test_select_best_face_single(self):
        from sparky_mvp.robot.scrfd_head_tracker import SCRFDHeadTracker

        tracker = SCRFDHeadTracker(confidence_threshold=0.3)

        class MockFace:
            det_score = 0.9
            bbox = np.array([10, 10, 100, 100])

        assert tracker._select_best_face([MockFace()]) == 0

    def test_select_best_face_picks_largest_confident(self):
        from sparky_mvp.robot.scrfd_head_tracker import SCRFDHeadTracker

        tracker = SCRFDHeadTracker(confidence_threshold=0.3)

        class SmallFace:
            det_score = 0.9
            bbox = np.array([10, 10, 30, 30])  # 20x20 = 400

        class BigFace:
            det_score = 0.85
            bbox = np.array([10, 10, 210, 210])  # 200x200 = 40000

        # Big face should win due to area bonus
        result = tracker._select_best_face([SmallFace(), BigFace()])
        assert result == 1  # BigFace index


class TestCameraWorker:
    """Test CameraWorker thread safety and interface."""

    def test_initial_offsets_are_zero(self):
        from sparky_mvp.robot.camera_worker import CameraWorker

        # Pass None as reachy_mini (won't be used in test)
        cw = CameraWorker(reachy_mini=None, head_tracker=None)
        offsets = cw.get_face_tracking_offsets()
        assert offsets == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def test_get_latest_frame_initially_none(self):
        from sparky_mvp.robot.camera_worker import CameraWorker

        cw = CameraWorker(reachy_mini=None, head_tracker=None)
        assert cw.get_latest_frame() is None

    def test_thread_safe_frame_storage(self):
        from sparky_mvp.robot.camera_worker import CameraWorker

        cw = CameraWorker(reachy_mini=None, head_tracker=None)

        # Manually set a frame (simulating what working_loop does)
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_frame[0, 0, 0] = 42  # marker

        with cw.frame_lock:
            cw.latest_frame = test_frame

        # Get frame returns a copy
        result = cw.get_latest_frame()
        assert result is not None
        assert result[0, 0, 0] == 42
        assert result is not test_frame  # Should be a copy

    def test_set_head_tracking_enabled(self):
        from sparky_mvp.robot.camera_worker import CameraWorker

        cw = CameraWorker(reachy_mini=None, head_tracker=None)
        assert cw.is_head_tracking_enabled is True

        cw.set_head_tracking_enabled(False)
        assert cw.is_head_tracking_enabled is False

        cw.set_head_tracking_enabled(True)
        assert cw.is_head_tracking_enabled is True

    def test_thread_safe_offset_access(self):
        """Concurrent reads/writes to face_tracking_offsets don't crash."""
        from sparky_mvp.robot.camera_worker import CameraWorker

        cw = CameraWorker(reachy_mini=None, head_tracker=None)
        errors = []

        def writer():
            for i in range(100):
                with cw.face_tracking_lock:
                    cw.face_tracking_offsets = [
                        float(i), 0.0, 0.0, 0.0, 0.0, 0.0,
                    ]
                time.sleep(0.001)

        def reader():
            for _ in range(100):
                try:
                    offsets = cw.get_face_tracking_offsets()
                    assert len(offsets) == 6
                except Exception as e:
                    errors.append(e)
                time.sleep(0.001)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"Thread safety errors: {errors}"


class TestSceneCapture:
    """Test scene capture utility functions."""

    def test_capture_returns_none_without_camera(self):
        from sparky_mvp.robot.scene_capture import capture_scene_image

        assert capture_scene_image(None) is None

    def test_capture_returns_none_with_no_frame(self):
        from sparky_mvp.robot.scene_capture import capture_scene_image

        class MockCameraWorker:
            def get_latest_frame(self):
                return None

        assert capture_scene_image(MockCameraWorker()) is None

    def test_encode_frame_produces_valid_block(self):
        from sparky_mvp.robot.scene_capture import encode_frame_as_image_block

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        block = encode_frame_as_image_block(frame, max_size=320, jpeg_quality=50)

        assert block["type"] == "image"
        assert block["source"]["type"] == "base64"
        assert block["source"]["media_type"] == "image/jpeg"
        assert len(block["source"]["data"]) > 0

    def test_encode_frame_respects_max_size(self):
        """Encoded image should not exceed max_size in either dimension."""
        import base64
        import io

        from sparky_mvp.robot.scene_capture import encode_frame_as_image_block

        # Large frame
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        block = encode_frame_as_image_block(frame, max_size=512)

        # Decode and check dimensions
        import cv2

        jpg_bytes = base64.b64decode(block["source"]["data"])
        arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        h, w = decoded.shape[:2]
        assert max(h, w) <= 512 + 1  # +1 for rounding
