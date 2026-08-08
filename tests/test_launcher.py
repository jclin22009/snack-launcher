from snack_launcher.launcher import (
    SINGLE_SHOT_DURATION_S,
    SINGLE_SHOT_THROTTLE,
    single_shot,
    set_vert_angle,
    zero_servo,
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


def test_zero_servo_moves_channel_one_to_zero_degrees() -> None:
    kit = FakeKit()

    zero_servo(kit)

    assert kit.servo[1].pulse_width_range == (1000, 2000)
    assert kit.servo[1].actuation_range == 100
    assert kit.servo[1].angle == 0


def test_set_vert_angle_uses_calibrated_travel() -> None:
    kit = FakeKit()

    set_vert_angle(45, kit)

    assert kit.servo[1].actuation_range == 100
    assert kit.servo[1].angle == 45


def test_set_vert_angle_rejects_out_of_range_values() -> None:
    kit = FakeKit()

    try:
        set_vert_angle(91, kit)
    except ValueError as error:
        assert "between 0 and 90" in str(error)
    else:
        raise AssertionError("Expected an out-of-range angle to fail")
