from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np

from .geometry import (
    DEFAULT_ASSUMED_IPD_MM,
    DEFAULT_CAMERA_HORIZONTAL_FOV_DEG,
    DEFAULT_MOUTH_WIDE_OPEN_RATIO,
    MouthDetection,
    Point,
    mouth_detection_from_points,
)


LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291
UPPER_INNER_LIP = 13
LOWER_INNER_LIP = 14
LEFT_PUPIL = 468
RIGHT_PUPIL = 473
DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


class MouthDetector(AbstractContextManager["MouthDetector"]):
    """MediaPipe Face Landmarker wrapper that extracts a mouth target."""

    def __init__(
        self,
        *,
        min_detection_confidence: float = 0.65,
        model_path: str | Path | None = None,
        wide_open_threshold: float = DEFAULT_MOUTH_WIDE_OPEN_RATIO,
        assumed_ipd_mm: float = DEFAULT_ASSUMED_IPD_MM,
        camera_horizontal_fov_deg: float = DEFAULT_CAMERA_HORIZONTAL_FOV_DEG,
    ) -> None:
        if wide_open_threshold <= 0:
            raise ValueError("wide_open_threshold must be greater than zero")
        if assumed_ipd_mm <= 0:
            raise ValueError("assumed_ipd_mm must be greater than zero")
        if not 0 < camera_horizontal_fov_deg < 180:
            raise ValueError("camera_horizontal_fov_deg must be between 0 and 180")

        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "mediapipe is required for mouth detection. Install with `uv sync`."
            ) from exc

        model_file = ensure_model(model_path)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_file)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._timestamp_ms = 0
        self._wide_open_threshold = wide_open_threshold
        self._assumed_ipd_mm = assumed_ipd_mm
        self._camera_horizontal_fov_deg = camera_horizontal_fov_deg

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        self._landmarker.close()

    def detect(self, frame_bgr: np.ndarray) -> MouthDetection | None:
        import cv2
        import mediapipe as mp

        frame_height, frame_width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame_rgb),
        )
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)
        self._timestamp_ms += 33

        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        return mouth_detection_from_points(
            left_corner=_landmark_to_point(
                landmarks[LEFT_MOUTH_CORNER], frame_width, frame_height
            ),
            right_corner=_landmark_to_point(
                landmarks[RIGHT_MOUTH_CORNER], frame_width, frame_height
            ),
            upper_lip=_landmark_to_point(
                landmarks[UPPER_INNER_LIP], frame_width, frame_height
            ),
            lower_lip=_landmark_to_point(
                landmarks[LOWER_INNER_LIP], frame_width, frame_height
            ),
            frame_width=frame_width,
            frame_height=frame_height,
            wide_open_threshold=self._wide_open_threshold,
            left_pupil=_landmark_to_point_or_none(
                landmarks, LEFT_PUPIL, frame_width, frame_height
            ),
            right_pupil=_landmark_to_point_or_none(
                landmarks, RIGHT_PUPIL, frame_width, frame_height
            ),
            assumed_ipd_mm=self._assumed_ipd_mm,
            camera_horizontal_fov_deg=self._camera_horizontal_fov_deg,
        )


def _landmark_to_point(landmark: Any, frame_width: int, frame_height: int) -> Point:
    return Point(landmark.x * frame_width, landmark.y * frame_height)


def _landmark_to_point_or_none(
    landmarks: list[Any], index: int, frame_width: int, frame_height: int
) -> Point | None:
    if index >= len(landmarks):
        return None
    return _landmark_to_point(landmarks[index], frame_width, frame_height)


def default_model_path() -> Path:
    return Path.home() / ".cache" / "snack-launcher" / "face_landmarker.task"


def ensure_model(model_path: str | Path | None = None) -> Path:
    path = Path(model_path).expanduser() if model_path is not None else default_model_path()
    if path.exists():
        return path

    if model_path is not None:
        raise FileNotFoundError(f"Face landmarker model not found: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(DEFAULT_MODEL_URL, path)
    return path


def iter_camera_detections(
    *,
    camera_index: int = 0,
    mirror: bool = True,
    max_frames: int | None = None,
    model_path: str | Path | None = None,
    wide_open_threshold: float = DEFAULT_MOUTH_WIDE_OPEN_RATIO,
    assumed_ipd_mm: float = DEFAULT_ASSUMED_IPD_MM,
    camera_horizontal_fov_deg: float = DEFAULT_CAMERA_HORIZONTAL_FOV_DEG,
) -> Iterator[tuple[np.ndarray, MouthDetection | None]]:
    import cv2

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}.")

    try:
        with MouthDetector(
            model_path=model_path,
            wide_open_threshold=wide_open_threshold,
            assumed_ipd_mm=assumed_ipd_mm,
            camera_horizontal_fov_deg=camera_horizontal_fov_deg,
        ) as detector:
            frame_count = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Camera returned no frame.")

                if mirror:
                    frame = cv2.flip(frame, 1)

                yield frame, detector.detect(frame)

                frame_count += 1
                if max_frames is not None and frame_count >= max_frames:
                    return
    finally:
        capture.release()


def detection_to_payload(detection: MouthDetection | None) -> dict[str, object]:
    if detection is None:
        return {"detected": False}

    payload = asdict(detection)
    offset_x, offset_y = detection.aim_offset
    payload.update(
        {
            "detected": True,
            "aim_offset": {"x": offset_x, "y": offset_y},
            "mouth_wide_open": detection.is_wide_open,
        }
    )
    return payload
