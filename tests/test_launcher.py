from snack_launcher.launcher import (
    SINGLE_SHOT_DURATION_S,
    SINGLE_SHOT_THROTTLE,
    home_gimbal,
    single_shot,
    set_pan_angle,
    set_tilt_angle,
    zero_pan_servo,
    zero_tilt_servo,
)


class FakeServo:
    def __init__(self) -> None:
        self.throttle: float | None = None


class FakePositionalServo:
    def __init__(self) -> None:
        self.angle: float | None = None
        self.actuation_range: int | None = None
        self.pulse_width_range: tuple[int, int] | None = None

    def set_pulse_width_range(self, min_pulse: int, max_pulse: int) -> None:
        self.pulse_width_range = (min_pulse, max_pulse)


class FakeKit:
    def __init__(self) -> None:
        self.continuous_servo = [FakeServo() for _ in range(16)]
        self.servo = [FakePositionalServo() for _ in range(16)]


def test_single_shot_uses_calibrated_channel_zero_settings() -> None:
    kit = FakeKit()
    delays: list[float] = []

    single_shot(kit, sleep=delays.append)

    assert delays == [SINGLE_SHOT_DURATION_S]
    assert kit.continuous_servo[0].throttle is None
    assert kit.continuous_servo[1].throttle is None
    assert SINGLE_SHOT_THROTTLE == -0.35


def test_zero_tilt_servo_moves_channel_one_to_home_degrees() -> None:
    kit = FakeKit()

    zero_tilt_servo(kit)

    assert kit.servo[1].pulse_width_range == (1000, 2000)
    assert kit.servo[1].actuation_range == 100
    assert kit.servo[1].angle == 0


def test_zero_pan_servo_moves_channel_two_to_center_degrees() -> None:
    kit = FakeKit()

    zero_pan_servo(kit)

    assert kit.servo[2].pulse_width_range == (1000, 2000)
    assert kit.servo[2].actuation_range == 100
    assert kit.servo[2].angle == 50


def test_home_gimbal_homes_positional_servos_and_stops_firing_servo() -> None:
    kit = FakeKit()
    kit.continuous_servo[0].throttle = -0.35

    home_gimbal(kit)

    assert kit.servo[1].angle == 0
    assert kit.servo[2].angle == 50
    assert kit.continuous_servo[0].throttle is None


def test_set_tilt_angle_uses_calibrated_travel() -> None:
    kit = FakeKit()

    set_tilt_angle(45, kit)

    assert kit.servo[1].actuation_range == 100
    assert kit.servo[1].angle == 45


def test_set_tilt_angle_rejects_out_of_range_values() -> None:
    kit = FakeKit()

    try:
        set_tilt_angle(91, kit)
    except ValueError as error:
        assert "between 0 and 90" in str(error)
    else:
        raise AssertionError("Expected an out-of-range angle to fail")


def test_set_pan_angle_uses_a_signed_centered_coordinate_system() -> None:
    kit = FakeKit()

    set_pan_angle(-50, kit)
    assert kit.servo[2].angle == 0

    set_pan_angle(0, kit)
    assert kit.servo[2].angle == 50

    set_pan_angle(50, kit)
    assert kit.servo[2].angle == 100


def test_set_pan_angle_rejects_out_of_range_values() -> None:
    kit = FakeKit()

    try:
        set_pan_angle(51, kit)
    except ValueError as error:
        assert "between -50 and 50" in str(error)
    else:
        raise AssertionError("Expected an out-of-range angle to fail")
