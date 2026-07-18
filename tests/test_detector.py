from snack_launcher.detector import detection_to_payload
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
