import pytest

from snack_launcher import cli
from snack_launcher.geometry import (
    DEFAULT_ASSUMED_IPD_MM,
    DEFAULT_CAMERA_HORIZONTAL_FOV_DEG,
    DEFAULT_CAMERA_PRESET,
)


def test_parser_uses_distance_estimate_defaults() -> None:
    args = cli.build_parser().parse_args([])

    assert args.assumed_ipd_mm == DEFAULT_ASSUMED_IPD_MM
    assert args.camera_preset == DEFAULT_CAMERA_PRESET
    assert args.camera_horizontal_fov_deg is None


@pytest.mark.parametrize(
    "arguments",
    [
        ["--assumed-ipd-mm", "0"],
        ["--camera-horizontal-fov-deg", "0"],
        ["--camera-horizontal-fov-deg", "180"],
    ],
)
def test_parser_rejects_invalid_distance_estimate_settings(arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(arguments)


def test_main_passes_distance_estimate_settings_to_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    def fake_iter_camera_detections(**kwargs: object):
        observed.update(kwargs)
        return iter(())

    monkeypatch.setattr(cli, "iter_camera_detections", fake_iter_camera_detections)

    assert (
        cli.main(
            [
                "--no-window",
                "--assumed-ipd-mm",
                "61.5",
                "--camera-horizontal-fov-deg",
                "72",
            ]
        )
        == 0
    )
    assert observed["assumed_ipd_mm"] == 61.5
    assert observed["camera_horizontal_fov_deg"] == 72.0


def test_main_uses_the_selected_camera_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    def fake_iter_camera_detections(**kwargs: object):
        observed.update(kwargs)
        return iter(())

    monkeypatch.setattr(cli, "iter_camera_detections", fake_iter_camera_detections)

    assert cli.main(["--no-window"]) == 0
    assert observed["camera_horizontal_fov_deg"] == DEFAULT_CAMERA_HORIZONTAL_FOV_DEG
