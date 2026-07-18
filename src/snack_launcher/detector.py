from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager
from dataclasses import asdict
from pathlib import Path
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


LEFT_MOUTH_CORNER = 48
RIGHT_MOUTH_CORNER = 54
UPPER_OUTER_LIP = (50, 51, 52)
LOWER_OUTER_LIP = (56, 57, 58)
RIGHT_EYE = (36, 37, 38, 39, 40, 41)
LEFT_EYE = (42, 43, 44, 45, 46, 47)
DEFAULT_MODEL_URL = (
    "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/"
    "lbfmodel.yaml"
)


class MouthDetector(AbstractContextManager["MouthDetector"]):
    """OpenCV 68-point face-landmark wrapper that extracts a mouth target.

    Eye centers are used as a pupil-position proxy for the coarse distance
    estimate. They are not direct iris/pupil measurements.
    """

    def __init__(
        self,
        *,
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
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "opencv-contrib-python is required for mouth detection. Install with `uv sync`."
            ) from exc
        if not hasattr(cv2, "face"):
            raise RuntimeError(
                "opencv-contrib-python is required; the installed OpenCV lacks cv2.face."
            )

        model_file = ensure_model(model_path)
        self._facemark = cv2.face.createFacemarkLBF()
        self._facemark.loadModel(str(model_file))
        self._face_detector = cv2.CascadeClassifier(
            str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        )
        if self._face_detector.empty():
            raise RuntimeError("OpenCV's bundled frontal-face detector could not be loaded.")
        self._wide_open_threshold = wide_open_threshold
        self._assumed_ipd_mm = assumed_ipd_mm
        self._camera_horizontal_fov_deg = camera_horizontal_fov_deg

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        return None

    def detect(self, frame_bgr: np.ndarray) -> MouthDetection | None:
        import cv2

        frame_height, frame_width = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )
        if len(faces) == 0:
            return None

        largest_face = max(faces, key=lambda face: int(face[2]) * int(face[3]))
        fitted, landmarks = self._facemark.fit(
            frame_bgr, np.asarray([largest_face], dtype=np.int32)
        )
        if not fitted or len(landmarks) == 0:
            return None

        points = np.asarray(landmarks[0], dtype=float).reshape(-1, 2)
        if len(points) != 68:
            return None
        return _detection_from_landmarks(
            points,
            frame_width=frame_width,
            frame_height=frame_height,
            wide_open_threshold=self._wide_open_threshold,
            assumed_ipd_mm=self._assumed_ipd_mm,
            camera_horizontal_fov_deg=self._camera_horizontal_fov_deg,
        )


def _detection_from_landmarks(
    points: np.ndarray,
    *,
    frame_width: int,
    frame_height: int,
    wide_open_threshold: float,
    assumed_ipd_mm: float,
    camera_horizontal_fov_deg: float,
) -> MouthDetection:
    """Convert OpenCV's 68 landmarks into the public detection shape.

    LBF's inner-lip landmarks collapse toward the teeth on very open mouths.
    Averaging the central outer-lip landmarks tracks the full opening more
    reliably and reduces single-point jitter.
    """
    return mouth_detection_from_points(
        left_corner=_point_from_xy(points[LEFT_MOUTH_CORNER]),
        right_corner=_point_from_xy(points[RIGHT_MOUTH_CORNER]),
        upper_lip=_mean_point(points, UPPER_OUTER_LIP),
        lower_lip=_mean_point(points, LOWER_OUTER_LIP),
        frame_width=frame_width,
        frame_height=frame_height,
        wide_open_threshold=wide_open_threshold,
        left_pupil=_mean_point(points, LEFT_EYE),
        right_pupil=_mean_point(points, RIGHT_EYE),
        assumed_ipd_mm=assumed_ipd_mm,
        camera_horizontal_fov_deg=camera_horizontal_fov_deg,
    )


def _point_from_xy(point: np.ndarray) -> Point:
    return Point(float(point[0]), float(point[1]))


def _mean_point(points: np.ndarray, indices: tuple[int, ...]) -> Point:
    return _point_from_xy(points[list(indices)].mean(axis=0))


def default_model_path() -> Path:
    return Path.home() / ".cache" / "snack-launcher" / "lbfmodel.yaml"


def ensure_model(model_path: str | Path | None = None) -> Path:
    path = Path(model_path).expanduser() if model_path is not None else default_model_path()
    if path.exists():
        return path

    if model_path is not None:
        raise FileNotFoundError(f"Face landmark model not found: {path}")

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
