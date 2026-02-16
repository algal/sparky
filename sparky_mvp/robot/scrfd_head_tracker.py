"""SCRFD face detection head tracker using insightface.

Implements the same HeadTracker interface as the YOLO tracker from the
conversation app (get_head_position → (eye_center[-1,1], roll)) but uses
SCRFD for faster, more accurate face detection on GPU.

SCRFD achieves ~3-5ms inference on RTX 3090 vs ~15-20ms for YOLO.
Provides 5-point facial landmarks (eyes, nose, mouth corners) which
enable eye-center tracking and roll angle estimation.

Source model: buffalo_sc (SCRFD-500M detection only — no recognition).
Auto-downloads from insightface model zoo on first use (~30MB).
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class SCRFDHeadTracker:
    """Face detection head tracker using SCRFD via insightface.

    Same interface as the YOLO HeadTracker from the conversation app:
        get_head_position(img) -> (eye_center[-1,1], roll)

    Lazy-initializes the ONNX model on first call to avoid import-time GPU allocation.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.3,
        gpu_device: int = 1,
        det_size: Tuple[int, int] = (640, 640),
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.gpu_device = gpu_device
        self.det_size = det_size
        self._app = None  # Lazy-initialized

    def _ensure_model(self) -> None:
        """Lazy-initialize the SCRFD model on first use."""
        if self._app is not None:
            return

        from insightface.app import FaceAnalysis

        logger.info(
            "Initializing SCRFD face detection (GPU %d, det_size=%s)...",
            self.gpu_device,
            self.det_size,
        )

        # Use buffalo_sc (SCRFD-500M detection only — lightweight, no recognition)
        self._app = FaceAnalysis(
            name="buffalo_sc",
            providers=[
                (
                    "CUDAExecutionProvider",
                    {"device_id": str(self.gpu_device)},
                ),
                ("CPUExecutionProvider", {}),
            ],
        )
        self._app.prepare(ctx_id=0, det_size=self.det_size)
        logger.info("SCRFD face detection ready")

    def _select_best_face(self, faces: list) -> int | None:
        """Select the best face based on confidence and area.

        Uses the same weighted scoring as the YOLO tracker:
        score = confidence * 0.7 + normalized_area * 0.3

        Returns:
            Index of best face or None if no valid faces.
        """
        if not faces:
            return None

        # Filter by confidence
        valid = [
            (i, f)
            for i, f in enumerate(faces)
            if f.det_score >= self.confidence_threshold
        ]
        if not valid:
            return None

        if len(valid) == 1:
            return valid[0][0]

        # Score by confidence * 0.7 + normalized area * 0.3
        areas = []
        for _, f in valid:
            bbox = f.bbox
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            areas.append(area)

        max_area = max(areas) if areas else 1.0
        best_idx = valid[0][0]
        best_score = -1.0

        for (i, f), area in zip(valid, areas):
            score = f.det_score * 0.7 + (area / max_area) * 0.3
            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx

    def get_head_position(
        self, img: NDArray[np.uint8]
    ) -> Tuple[NDArray[np.float32] | None, float | None]:
        """Get head position from face detection.

        Same interface as YOLO HeadTracker.

        Args:
            img: Input image (BGR, from camera).

        Returns:
            Tuple of (eye_center [-1,1] normalized coords, roll_angle in radians).
            Returns (None, None) if no face detected.
        """
        self._ensure_model()

        h, w = img.shape[:2]

        try:
            assert self._app is not None
            faces = self._app.get(img)

            face_idx = self._select_best_face(faces)
            if face_idx is None:
                logger.debug("No face detected above confidence threshold")
                return None, None

            face = faces[face_idx]
            bbox = face.bbox  # [x1, y1, x2, y2]

            # Use eye keypoints if available, otherwise fall back to bbox center
            kps = getattr(face, "kps", None)
            if kps is not None and len(kps) >= 2:
                # kps[0] = left eye, kps[1] = right eye
                eye_center_px = (kps[0] + kps[1]) / 2.0
                # Compute roll from eye angle
                dy = kps[1][1] - kps[0][1]
                dx = kps[1][0] - kps[0][0]
                roll = float(np.arctan2(dy, dx))
            else:
                eye_center_px = np.array(
                    [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]
                )
                roll = 0.0

            # Normalize to [-1, 1]
            norm_x = (eye_center_px[0] / w) * 2.0 - 1.0
            norm_y = (eye_center_px[1] / h) * 2.0 - 1.0

            if face.det_score is not None:
                logger.debug(
                    "Face detected: conf=%.2f center=(%.2f, %.2f) roll=%.1f°",
                    face.det_score,
                    norm_x,
                    norm_y,
                    np.degrees(roll),
                )

            return np.array([norm_x, norm_y], dtype=np.float32), roll

        except Exception as e:
            logger.error("SCRFD detection error: %s", e)
            return None, None
