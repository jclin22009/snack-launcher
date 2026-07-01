from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager
from dataclasses import asdict
from pathlib import Path
from urllib.request import urlretrieve
from typing import Any

import numpy as np

from .geometry import MouthDetection, Point, mouth_detection_from_points


LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291
UPPER_INNER_LIP = 13
LOWER_INNER_LIP = 14
DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


class MouthDetector(AbstractContextManager["MouthDetector"]):
    """MediaPipe Face Mesh wrapper that extracts a mouth target from frames."""

    def __init__(
        self,
        *,
        min_detection_confidence: float = 0.65,
        model_path: str | Path | None = None,
    ) -> None:
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
        )


def _landmark_to_point(landmark: Any, frame_width: int, frame_height: int) -> Point:
    return Point(landmark.x * frame_width, landmark.y * frame_height)


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
) -> Iterator[tuple[np.ndarray, MouthDetection | None]]:
    import cv2

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}.")

    try:
        with MouthDetector(model_path=model_path) as detector:
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
            "mouth_open": detection.is_open,
        }
    )
    return payload
