from snack_launcher.geometry import Point, mouth_detection_from_points


def test_mouth_detection_computes_center_openness_and_offset() -> None:
    detection = mouth_detection_from_points(
        left_corner=Point(280, 240),
        right_corner=Point(360, 240),
        upper_lip=Point(320, 230),
        lower_lip=Point(320, 250),
        frame_width=640,
        frame_height=480,
    )

    assert detection.center == Point(320, 240)
    assert detection.openness_ratio == 0.25
    assert detection.aim_offset == (0.0, 0.0)
    assert detection.is_wide_open


def test_mouth_detection_treats_small_opening_as_closed() -> None:
    detection = mouth_detection_from_points(
        left_corner=Point(100, 100),
        right_corner=Point(200, 100),
        upper_lip=Point(150, 99),
        lower_lip=Point(150, 101),
        frame_width=300,
        frame_height=200,
    )

    assert detection.openness_ratio == 0.02
    assert not detection.is_wide_open


def test_mouth_detection_uses_adjustable_wide_open_threshold() -> None:
    detection = mouth_detection_from_points(
        left_corner=Point(100, 100),
        right_corner=Point(200, 100),
        upper_lip=Point(150, 90),
        lower_lip=Point(150, 110),
        frame_width=300,
        frame_height=200,
        wide_open_threshold=0.25,
    )

    assert detection.openness_ratio == 0.2
    assert not detection.is_wide_open
