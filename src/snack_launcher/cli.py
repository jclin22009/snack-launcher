from __future__ import annotations

import argparse
import json
from typing import Sequence

from .detector import detection_to_payload, iter_camera_detections
from .geometry import (
    CAMERA_PRESET_HORIZONTAL_FOV_DEG,
    DEFAULT_ASSUMED_IPD_MM,
    DEFAULT_CAMERA_HORIZONTAL_FOV_DEG,
    DEFAULT_CAMERA_PRESET,
    DEFAULT_MOUTH_WIDE_OPEN_RATIO,
    MouthDetection,
    camera_preset_horizontal_fov_deg,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track a mouth target from a webcam for snack launcher aiming."
    )
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Run without displaying the camera overlay.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print one JSON detection payload per frame.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many frames. Useful for smoke tests.",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Do not horizontally mirror the webcam image.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to an OpenCV LBF face-landmark model file.",
    )
    parser.add_argument(
        "--mouth-wide-open-ratio",
        type=_positive_value,
        default=DEFAULT_MOUTH_WIDE_OPEN_RATIO,
        help=(
            "Mouth openness ratio required for mouth_wide_open "
            f"(default: {DEFAULT_MOUTH_WIDE_OPEN_RATIO})."
        ),
    )
    parser.add_argument(
        "--assumed-ipd-mm",
        type=_positive_value,
        default=DEFAULT_ASSUMED_IPD_MM,
        help=(
            "Assumed interpupillary distance used for the distance estimate "
            f"in millimeters (default: {DEFAULT_ASSUMED_IPD_MM})."
        ),
    )
    parser.add_argument(
        "--camera-preset",
        choices=tuple(CAMERA_PRESET_HORIZONTAL_FOV_DEG),
        default=DEFAULT_CAMERA_PRESET,
        help=(
            "Camera calibration preset used for distance estimation "
            f"(default: {DEFAULT_CAMERA_PRESET})."
        ),
    )
    parser.add_argument(
        "--camera-horizontal-fov-deg",
        type=_horizontal_fov,
        default=None,
        help=(
            "Override the selected preset's horizontal field of view for the "
            "distance estimate, in degrees."
        ),
    )
    return parser


def _positive_value(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def _horizontal_fov(value: str) -> float:
    field_of_view = _positive_value(value)
    if field_of_view >= 180:
        raise argparse.ArgumentTypeError("must be less than 180 degrees")
    return field_of_view


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    camera_horizontal_fov_deg = (
        args.camera_horizontal_fov_deg
        if args.camera_horizontal_fov_deg is not None
        else camera_preset_horizontal_fov_deg(args.camera_preset)
    )

    if not args.no_window:
        import cv2

    for frame, detection in iter_camera_detections(
        camera_index=args.camera,
        mirror=not args.no_mirror,
        max_frames=args.max_frames,
        model_path=args.model,
        wide_open_threshold=args.mouth_wide_open_ratio,
        assumed_ipd_mm=args.assumed_ipd_mm,
        camera_horizontal_fov_deg=camera_horizontal_fov_deg,
    ):
        if args.json:
            print(json.dumps(detection_to_payload(detection)), flush=True)

        if args.no_window:
            continue

        _draw_overlay(frame, detection)
        cv2.imshow("snack mouth detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if not args.no_window:
        cv2.destroyAllWindows()

    return 0


def _draw_overlay(frame: object, detection: MouthDetection | None) -> None:
    import cv2

    if detection is None:
        cv2.putText(
            frame,
            "No mouth detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        return

    center = (round(detection.center.x), round(detection.center.y))
    left = (round(detection.left_corner.x), round(detection.left_corner.y))
    right = (round(detection.right_corner.x), round(detection.right_corner.y))
    upper = (round(detection.upper_lip.x), round(detection.upper_lip.y))
    lower = (round(detection.lower_lip.x), round(detection.lower_lip.y))
    status = "WIDE OPEN" if detection.is_wide_open else "not wide open"
    offset_x, offset_y = detection.aim_offset

    cv2.line(frame, left, right, (255, 180, 0), 2)
    cv2.line(frame, upper, lower, (255, 180, 0), 2)
    cv2.circle(frame, center, 9, (0, 255, 0), 2)
    cv2.drawMarker(frame, center, (0, 255, 0), cv2.MARKER_CROSS, 28, 2)
    cv2.putText(
        frame,
        (
            f"mouth {status} ratio={detection.openness_ratio:.2f} "
            f"threshold={detection.wide_open_threshold:.2f}"
        ),
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
    )
    distance_text = (
        f"distance estimate={detection.distance_estimate_m:.2f} m"
        if detection.distance_estimate_m is not None
        else "distance estimate unavailable"
    )
    cv2.putText(
        frame,
        distance_text,
        (20, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"aim offset x={offset_x:+.2f} y={offset_y:+.2f}",
        (20, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
    )
