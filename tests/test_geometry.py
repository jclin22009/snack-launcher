import pytest

from snack_launcher.geometry import (
    Point,
    estimate_distance_m,
    mouth_detection_from_points,
)


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
    assert not detection.is_wide_open


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


def test_estimate_distance_uses_pupil_separation_and_camera_fov() -> None:
    estimate = estimate_distance_m(
        pupil_distance_px=100,
        frame_width=1000,
        assumed_ipd_mm=60,
        camera_horizontal_fov_deg=90,
    )

    assert estimate == pytest.approx(0.3)


def test_mouth_detection_has_no_distance_without_usable_pupils() -> None:
    detection = mouth_detection_from_points(
        left_corner=Point(100, 100),
        right_corner=Point(200, 100),
        upper_lip=Point(150, 90),
        lower_lip=Point(150, 110),
        frame_width=1000,
        frame_height=200,
        left_pupil=Point(100, 50),
        right_pupil=Point(100, 50),
    )

    assert detection.distance_estimate_m is None


@pytest.mark.parametrize(
    ("assumed_ipd_mm", "camera_horizontal_fov_deg"),
    [(0, 60), (63, 0), (63, 180)],
)
def test_estimate_distance_rejects_invalid_configuration(
    assumed_ipd_mm: float, camera_horizontal_fov_deg: float
) -> None:
    with pytest.raises(ValueError):
        estimate_distance_m(
            pupil_distance_px=100,
            frame_width=1000,
            assumed_ipd_mm=assumed_ipd_mm,
            camera_horizontal_fov_deg=camera_horizontal_fov_deg,
        )
