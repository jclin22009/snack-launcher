from __future__ import annotations

from dataclasses import dataclass
from math import hypot


DEFAULT_MOUTH_WIDE_OPEN_RATIO = 0.15


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


def mouth_detection_from_points(
    *,
    left_corner: Point,
    right_corner: Point,
    upper_lip: Point,
    lower_lip: Point,
    frame_width: int,
    frame_height: int,
    wide_open_threshold: float = DEFAULT_MOUTH_WIDE_OPEN_RATIO,
) -> MouthDetection:
    if wide_open_threshold <= 0:
        raise ValueError("wide_open_threshold must be greater than zero")

    mouth_width = max(distance(left_corner, right_corner), 1.0)
    mouth_height = distance(upper_lip, lower_lip)
    center = midpoint(upper_lip, lower_lip)

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
    )
