from __future__ import annotations

from dataclasses import dataclass
from math import hypot, radians, tan


DEFAULT_MOUTH_WIDE_OPEN_RATIO = 0.6
DEFAULT_ASSUMED_IPD_MM = 63.0
MACBOOK_PRO_13_2022_CAMERA_PRESET = "macbook-pro-13-2022"
CAMERA_PRESET_HORIZONTAL_FOV_DEG = {
    # Apple publishes the camera resolution, but not lens intrinsics. This is
    # a practical 60-degree horizontal-FOV approximation for the 720p camera.
    MACBOOK_PRO_13_2022_CAMERA_PRESET: 60.0,
}
DEFAULT_CAMERA_PRESET = MACBOOK_PRO_13_2022_CAMERA_PRESET
DEFAULT_CAMERA_HORIZONTAL_FOV_DEG = CAMERA_PRESET_HORIZONTAL_FOV_DEG[
    DEFAULT_CAMERA_PRESET
]


@dataclass(frozen=True)
class Point:
    """A 2D point in image pixel coordinates."""

    x: float
    y: float


@dataclass(frozen=True)
class MouthDetection:
    """Mouth target data from one video frame."""

    center: Point
    left_corner: Point
    right_corner: Point
    upper_lip: Point
    lower_lip: Point
    openness_ratio: float
    frame_width: int
    frame_height: int
    wide_open_threshold: float = DEFAULT_MOUTH_WIDE_OPEN_RATIO
    distance_estimate_m: float | None = None

    @property
    def aim_offset(self) -> tuple[float, float]:
        """Return center offset from frame center, normalized to -1..1-ish."""

        return (
            (self.center.x - self.frame_width / 2) / (self.frame_width / 2),
            (self.center.y - self.frame_height / 2) / (self.frame_height / 2),
        )

    @property
    def is_wide_open(self) -> bool:
        """Whether the mouth has reached the configured wide-open threshold."""

        return self.openness_ratio >= self.wide_open_threshold


def distance(a: Point, b: Point) -> float:
    return hypot(a.x - b.x, a.y - b.y)


def midpoint(a: Point, b: Point) -> Point:
    return Point((a.x + b.x) / 2, (a.y + b.y) / 2)


def estimate_distance_m(
    *,
    pupil_distance_px: float,
    frame_width: int,
    assumed_ipd_mm: float = DEFAULT_ASSUMED_IPD_MM,
    camera_horizontal_fov_deg: float = DEFAULT_CAMERA_HORIZONTAL_FOV_DEG,
) -> float | None:
    """Estimate camera distance from apparent pupil separation in one frame."""

    _validate_distance_estimation_config(
        assumed_ipd_mm=assumed_ipd_mm,
        camera_horizontal_fov_deg=camera_horizontal_fov_deg,
    )
    if pupil_distance_px <= 0 or frame_width <= 0:
        return None

    focal_length_px = (frame_width / 2) / tan(radians(camera_horizontal_fov_deg / 2))
    return focal_length_px * assumed_ipd_mm / pupil_distance_px / 1000


def camera_preset_horizontal_fov_deg(camera_preset: str) -> float:
    """Return the configured horizontal FOV for a named camera preset."""

    try:
        return CAMERA_PRESET_HORIZONTAL_FOV_DEG[camera_preset]
    except KeyError as exc:
        raise ValueError(f"unknown camera preset: {camera_preset}") from exc


def mouth_detection_from_points(
    *,
    left_corner: Point,
    right_corner: Point,
    upper_lip: Point,
    lower_lip: Point,
    frame_width: int,
    frame_height: int,
    wide_open_threshold: float = DEFAULT_MOUTH_WIDE_OPEN_RATIO,
    left_pupil: Point | None = None,
    right_pupil: Point | None = None,
    assumed_ipd_mm: float = DEFAULT_ASSUMED_IPD_MM,
    camera_horizontal_fov_deg: float = DEFAULT_CAMERA_HORIZONTAL_FOV_DEG,
) -> MouthDetection:
    if wide_open_threshold <= 0:
        raise ValueError("wide_open_threshold must be greater than zero")
    _validate_distance_estimation_config(
        assumed_ipd_mm=assumed_ipd_mm,
        camera_horizontal_fov_deg=camera_horizontal_fov_deg,
    )

    mouth_width = max(distance(left_corner, right_corner), 1.0)
    mouth_height = distance(upper_lip, lower_lip)
    center = midpoint(upper_lip, lower_lip)
    distance_estimate = None
    if left_pupil is not None and right_pupil is not None:
        distance_estimate = estimate_distance_m(
            pupil_distance_px=distance(left_pupil, right_pupil),
            frame_width=frame_width,
            assumed_ipd_mm=assumed_ipd_mm,
            camera_horizontal_fov_deg=camera_horizontal_fov_deg,
        )

    return MouthDetection(
        center=center,
        left_corner=left_corner,
        right_corner=right_corner,
        upper_lip=upper_lip,
        lower_lip=lower_lip,
        openness_ratio=mouth_height / mouth_width,
        frame_width=frame_width,
        frame_height=frame_height,
        wide_open_threshold=wide_open_threshold,
        distance_estimate_m=distance_estimate,
    )


def _validate_distance_estimation_config(
    *, assumed_ipd_mm: float, camera_horizontal_fov_deg: float
) -> None:
    if assumed_ipd_mm <= 0:
        raise ValueError("assumed_ipd_mm must be greater than zero")
    if not 0 < camera_horizontal_fov_deg < 180:
        raise ValueError("camera_horizontal_fov_deg must be between 0 and 180")
