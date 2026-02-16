"""Camera worker thread with frame buffering and face tracking.

Copied from reachy_mini_conversation_app (2026-02-08) for independent use
in the Reachy MVP scaffold. Source:
  forks/reachy_mini_conversation_app/src/reachy_mini_conversation_app/camera_worker.py

Provides:
- 25Hz camera polling with thread-safe frame buffering
- Face tracking integration with smooth interpolation back to neutral
- Latest frame always available for scene awareness tools

The head_tracker must implement:
    get_head_position(img) -> (eye_center[-1,1], roll)
Both SCRFDHeadTracker and the conversation app's YOLO HeadTracker satisfy this.
"""

import time
import logging
import threading
from typing import Any, List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini
from reachy_mini.utils.interpolation import linear_pose_interpolation


logger = logging.getLogger(__name__)


class CameraWorker:
    """Thread-safe camera worker with frame buffering and face tracking."""

    def __init__(self, reachy_mini: ReachyMini, head_tracker: Any = None) -> None:
        self.reachy_mini = reachy_mini
        self.head_tracker = head_tracker

        # Thread-safe frame storage
        self.latest_frame: NDArray[np.uint8] | None = None
        self.frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Face tracking state
        self.is_head_tracking_enabled = True
        self.face_tracking_offsets: List[float] = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]  # x, y, z, roll, pitch, yaw
        self.face_tracking_lock = threading.Lock()

        # Face tracking timing (smooth interpolation back to neutral when face lost)
        self.last_face_detected_time: float | None = None
        self.interpolation_start_time: float | None = None
        self.interpolation_start_pose: NDArray[np.float32] | None = None
        self.face_lost_delay = 2.0  # seconds before starting interpolation
        self.interpolation_duration = 1.0  # seconds to interpolate back to neutral

        # Track state changes
        self.previous_head_tracking_state = self.is_head_tracking_enabled

    def get_latest_frame(self) -> NDArray[np.uint8] | None:
        """Get the latest frame (thread-safe). Returns a copy in BGR format."""
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def get_face_tracking_offsets(
        self,
    ) -> Tuple[float, float, float, float, float, float]:
        """Get current face tracking offsets (thread-safe)."""
        with self.face_tracking_lock:
            offsets = self.face_tracking_offsets
            return (offsets[0], offsets[1], offsets[2], offsets[3], offsets[4], offsets[5])

    def set_head_tracking_enabled(self, enabled: bool) -> None:
        """Enable/disable head tracking."""
        self.is_head_tracking_enabled = enabled
        logger.info("Head tracking %s", "enabled" if enabled else "disabled")

    def start(self) -> None:
        """Start the camera worker loop in a thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.working_loop, daemon=True)
        self._thread.start()
        logger.debug("Camera worker started")

    def stop(self) -> None:
        """Stop the camera worker loop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        logger.debug("Camera worker stopped")

    def working_loop(self) -> None:
        """Camera worker main loop — polls frames and computes face tracking offsets.

        Ported from conversation app camera_worker() with identical logic.
        """
        logger.debug("Starting camera working loop")

        neutral_pose = np.eye(4)  # Neutral pose (identity matrix)
        self.previous_head_tracking_state = self.is_head_tracking_enabled

        while not self._stop_event.is_set():
            try:
                current_time = time.time()

                # Get frame from robot camera
                frame = self.reachy_mini.media.get_frame()

                if frame is not None:
                    # Thread-safe frame storage
                    with self.frame_lock:
                        self.latest_frame = frame

                    # Check if face tracking was just disabled
                    if (
                        self.previous_head_tracking_state
                        and not self.is_head_tracking_enabled
                    ):
                        self.last_face_detected_time = current_time
                        self.interpolation_start_time = None
                        self.interpolation_start_pose = None

                    self.previous_head_tracking_state = self.is_head_tracking_enabled

                    # Handle face tracking if enabled and head tracker available
                    if self.is_head_tracking_enabled and self.head_tracker is not None:
                        eye_center, _ = self.head_tracker.get_head_position(frame)

                        if eye_center is not None:
                            # Face detected — immediately track
                            self.last_face_detected_time = current_time
                            self.interpolation_start_time = None

                            # Convert normalized [-1,1] coords to pixel coords
                            h, w, _ = frame.shape
                            eye_center_norm = (eye_center + 1) / 2
                            eye_center_pixels = [
                                np.clip(eye_center_norm[0] * w, 1, w - 1),
                                np.clip(eye_center_norm[1] * h, 1, h - 1),
                            ]

                            # Compute head pose to look at target (without moving)
                            target_pose = self.reachy_mini.look_at_image(
                                eye_center_pixels[0],
                                eye_center_pixels[1],
                                duration=0.0,
                                perform_movement=False,
                            )

                            # Extract translation and rotation
                            translation = target_pose[:3, 3]
                            rotation = R.from_matrix(
                                target_pose[:3, :3]
                            ).as_euler("xyz", degrees=False)

                            # Scale down (smaller FOV compensation)
                            translation *= 0.6
                            rotation *= 0.6

                            with self.face_tracking_lock:
                                self.face_tracking_offsets = [
                                    translation[0],
                                    translation[1],
                                    translation[2],
                                    rotation[0],
                                    rotation[1],
                                    rotation[2],
                                ]

                    # Handle smooth interpolation back to neutral when face lost
                    if self.last_face_detected_time is not None:
                        time_since_face_lost = (
                            current_time - self.last_face_detected_time
                        )

                        if time_since_face_lost >= self.face_lost_delay:
                            if self.interpolation_start_time is None:
                                self.interpolation_start_time = current_time
                                with self.face_tracking_lock:
                                    current_translation = (
                                        self.face_tracking_offsets[:3]
                                    )
                                    current_rotation_euler = (
                                        self.face_tracking_offsets[3:]
                                    )
                                    pose_matrix = np.eye(4, dtype=np.float32)
                                    pose_matrix[:3, 3] = current_translation
                                    pose_matrix[:3, :3] = R.from_euler(
                                        "xyz",
                                        current_rotation_euler,
                                    ).as_matrix()
                                    self.interpolation_start_pose = pose_matrix

                            # Interpolation progress (0 to 1)
                            elapsed = (
                                current_time - self.interpolation_start_time
                            )
                            t = min(1.0, elapsed / self.interpolation_duration)

                            interpolated_pose = linear_pose_interpolation(
                                self.interpolation_start_pose,
                                neutral_pose,
                                t,
                            )

                            translation = interpolated_pose[:3, 3]
                            rotation = R.from_matrix(
                                interpolated_pose[:3, :3]
                            ).as_euler("xyz", degrees=False)

                            with self.face_tracking_lock:
                                self.face_tracking_offsets = [
                                    translation[0],
                                    translation[1],
                                    translation[2],
                                    rotation[0],
                                    rotation[1],
                                    rotation[2],
                                ]

                            if t >= 1.0:
                                self.last_face_detected_time = None
                                self.interpolation_start_time = None
                                self.interpolation_start_pose = None

                # ~25Hz polling (same as conversation app)
                time.sleep(0.04)

            except Exception as e:
                logger.error("Camera worker error: %s", e)
                time.sleep(0.1)

        logger.debug("Camera worker thread exited")
