import numpy as np

from snack_launcher.detector import _detection_from_landmarks, detection_to_payload
from snack_launcher.geometry import Point, mouth_detection_from_points


def test_payload_reports_wide_open_using_configured_threshold() -> None:
    detection = mouth_detection_from_points(
        left_corner=Point(0, 0),
        right_corner=Point(100, 0),
        upper_lip=Point(50, 40),
        lower_lip=Point(50, 60),
        frame_width=100,
        frame_height=100,
        wide_open_threshold=0.25,
    )

    payload = detection_to_payload(detection)

    assert payload["openness_ratio"] == 0.2
    assert payload["wide_open_threshold"] == 0.25
    assert payload["mouth_wide_open"] is False
    assert payload["distance_estimate_m"] is None
    assert "mouth_open" not in payload


def test_lbf_detection_uses_outer_lips_for_wide_open_mouth() -> None:
    points = np.zeros((68, 2), dtype=float)
    points[48] = (0, 50)
    points[54] = (100, 50)
    points[[50, 51, 52]] = (50, 10)
    points[[56, 57, 58]] = (50, 80)
    # LBF can collapse these inner-lip points toward the upper teeth.
    points[62] = (50, 40)
    points[66] = (50, 55)
    points[[36, 37, 38, 39, 40, 41]] = (25, 20)
    points[[42, 43, 44, 45, 46, 47]] = (75, 20)

    detection = _detection_from_landmarks(
        points,
        frame_width=100,
        frame_height=100,
        wide_open_threshold=0.6,
        assumed_ipd_mm=63,
        camera_horizontal_fov_deg=60,
    )

    assert detection.upper_lip == Point(50, 10)
    assert detection.lower_lip == Point(50, 80)
    assert detection.openness_ratio == 0.7
    assert detection.is_wide_open is True
