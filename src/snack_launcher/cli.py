from __future__ import annotations

import argparse
import json
from typing import Sequence

from .detector import detection_to_payload, iter_camera_detections
from .geometry import MouthDetection


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
        help="Path to a MediaPipe face_landmarker.task model bundle.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.no_window:
        import cv2

    for frame, detection in iter_camera_detections(
        camera_index=args.camera,
        mirror=not args.no_mirror,
        max_frames=args.max_frames,
        model_path=args.model,
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
    status = "OPEN" if detection.is_open else "closed"
    offset_x, offset_y = detection.aim_offset

    cv2.line(frame, left, right, (255, 180, 0), 2)
    cv2.line(frame, upper, lower, (255, 180, 0), 2)
    cv2.circle(frame, center, 9, (0, 255, 0), 2)
    cv2.drawMarker(frame, center, (0, 255, 0), cv2.MARKER_CROSS, 28, 2)
    cv2.putText(
        frame,
        f"mouth {status} ratio={detection.openness_ratio:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
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
